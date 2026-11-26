import copy
import pickle

import pytest
import torch

from torch_geometric.data import Batch, Data, OWLData, OWLSchema
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.nn import GCNConv
from torch_geometric.testing import withPackage


def get_owl_data() -> OWLData:
    data = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        y=torch.tensor([0, 1, 1, 0]),
    )
    schema = OWLSchema(
        base_iri='http://example.org/test#',
        node_names=['a', 'b', 'c', 'd'],
        feature_names=['Red', 'Blue'],
    )
    return OWLData.from_data(data, schema)


def test_owl_schema():
    schema = OWLSchema(base_iri='http://example.org/test')
    assert schema.base_iri == 'http://example.org/test#'
    assert schema.iri('Person') == 'http://example.org/test#Person'
    assert schema.iri('my class') == 'http://example.org/test#my_class'
    assert schema.iri('http://other.org/x#Y') == 'http://other.org/x#Y'
    assert schema.node_iri(3) == 'http://example.org/test#node_3'
    assert schema.relation_iri(0) == 'http://example.org/test#related_to'
    assert schema.relation_iri(1, 2) == 'http://example.org/test#relation_1'
    assert schema.encoding(0, torch.tensor([0.0, 1.0])) == 'class'
    assert schema.encoding(0, torch.tensor([0.0, 0.5])) == 'numeric'

    schema = OWLSchema(feature_names=['a',
                                      'b'], feature_encoding={'b': 'ignore'},
                       target_classes=['T'], ignore_classes=['U'])
    assert schema.encoding(1) == 'ignore'
    assert schema.encoding(0, torch.tensor([2.0])) == 'numeric'
    assert schema.excluded_classes() == [
        'http://example.org/pyg#T',
        'http://example.org/pyg#U',
    ]

    assert OWLSchema.cast(dict(node_class='Paper')).node_class == 'Paper'
    with pytest.raises(ValueError, match="absolute IRI"):
        OWLSchema(base_iri='example')


def test_owl_data():
    data = get_owl_data()
    assert isinstance(data, Data)
    assert data.num_nodes == 4
    assert data.node_iri[0] == 'http://example.org/test#a'
    assert data.schema.feature_names == ['Red', 'Blue']
    assert data.source is None
    assert 'schema' not in data.keys()
    assert data.index_of(['http://example.org/test#c']).tolist() == [2]
    with pytest.raises(KeyError, match="unknown individual"):
        data.index_of('http://example.org/test#z')

    # The schema is preserved, but not part of the attribute storage:
    for out in [
            copy.copy(data),
            copy.deepcopy(data),
            data.to('cpu'),
            pickle.loads(pickle.dumps(data)),
    ]:
        assert isinstance(out, OWLData)
        assert out.schema.feature_names == ['Red', 'Blue']

    # IRIs are node-level attributes:
    out = data.subgraph(torch.tensor([1, 3]))
    assert out.node_iri == [
        'http://example.org/test#b', 'http://example.org/test#d'
    ]
    assert out.schema.node_names == ['a', 'b', 'c', 'd']

    batch = Batch.from_data_list([data, out])
    assert batch.num_nodes == 6
    assert batch.node_iri[1] == out.node_iri


def test_owl_data_set_labels():
    data = get_owl_data()
    data.set_labels_from_examples(
        positive=['http://example.org/test#a'],
        negative=['http://example.org/test#c', 'http://example.org/test#d'],
    )
    assert data.y.tolist() == [1, -1, 0, 0]
    assert data.train_mask.tolist() == [True, False, True, True]


def test_owl_data_with_mask_explainer():
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(2, 8)
            self.conv2 = GCNConv(8, 2)

        def forward(self, x, edge_index):
            return self.conv2(self.conv1(x, edge_index).relu(), edge_index)

    data = get_owl_data()
    explainer = Explainer(
        model=GCN(),
        algorithm=GNNExplainer(epochs=2),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
    explanation = explainer(data.x, data.edge_index, index=1)
    assert explanation.node_mask.size() == data.x.size()
    assert explanation.edge_mask.size() == (data.num_edges, )


@withPackage('rdflib')
def test_owl_data_from_owl(tmp_path):
    data = get_owl_data()
    path = data.save_owl(str(tmp_path / 'graph.owl'))

    out = OWLData.from_owl(path, target_classes=['Red'])
    assert out.source == path
    assert out.node_iri == data.node_iri
    assert out.y.tolist() == [1, 0, 1, 0]
    assert 'http://example.org/test#Red' not in out.schema.feature_names

    # Re-using the data object keeps its schema and source:
    out = OWLData.from_data(out)
    assert out.source == path
    assert out.schema.target_classes == ['http://example.org/test#Red']


@withPackage('rdflib')
def test_owl_hetero_data():
    from torch_geometric.data import HeteroData, OWLHeteroData

    BASE = 'http://example.org/test#'

    data = HeteroData()
    data['author'].x = torch.randn(2, 4)
    data['paper'].num_nodes = 3
    data['author', 'writes', 'paper'].edge_index = torch.tensor([[0, 1],
                                                                 [1, 2]])

    schema = OWLSchema(base_iri=BASE, node_names={'paper': ['a', 'b', 'c']})
    data = OWLHeteroData.from_data(data, schema)
    assert data['author'].node_iri == [BASE + 'author_0', BASE + 'author_1']
    assert data['paper'].node_iri == [BASE + 'a', BASE + 'b', BASE + 'c']
    assert data.index_of([BASE + 'c', BASE + 'a'], 'paper').tolist() == [2, 0]
    with pytest.raises(KeyError, match="of node type 'paper'"):
        data.index_of(BASE + 'author_0', 'paper')

    # The schema is not part of the attribute storage:
    data.schema = dict(base_iri='http://example.org/other#')
    assert data.schema.base_iri == 'http://example.org/other#'
    assert 'schema' not in data._global_store
    assert copy.copy(data).schema is data.schema
    assert pickle.loads(pickle.dumps(data)).schema.base_iri == \
        data.schema.base_iri

    with pytest.raises(ValueError, match="OWLHeteroData.from_owl"):
        OWLData.from_owl(data.to_owl(), node_types=['author'])
    out = OWLHeteroData.from_owl(data.to_owl(), node_types=['author', 'paper'])
    assert out['paper'].num_nodes == 3
