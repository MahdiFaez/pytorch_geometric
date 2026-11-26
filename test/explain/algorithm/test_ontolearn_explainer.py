import pytest
import torch

from torch_geometric.data import Data, OWLData, OWLSchema
from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.algorithm import OntolearnExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.nn import GCNConv
from torch_geometric.testing import withPackage

BASE = 'http://example.org/test#'


class GCN(torch.nn.Module):
    def __init__(self, out_channels: int = 2):
        super().__init__()
        self.conv1 = GCNConv(2, 8)
        self.conv2 = GCNConv(8, out_channels)

    def forward(self, x, edge_index):
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index)


def get_data() -> Data:
    # Nodes with an even index are red, nodes with an odd index blue. The
    # planted concept of class 1 is "Red":
    num_nodes = 12
    red = torch.arange(num_nodes) % 2 == 0
    x = torch.stack([red, ~red], dim=1).float()
    row = torch.arange(num_nodes)
    col = (row + 1) % num_nodes
    edge_index = torch.cat([torch.stack([row, col]),
                            torch.stack([col, row])], dim=1)
    return Data(x=x, edge_index=edge_index, y=red.long())


def train(model: torch.nn.Module, data: Data) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        torch.nn.functional.cross_entropy(out, data.y).backward()
        optimizer.step()
    return model


def get_schema() -> OWLSchema:
    return OWLSchema(base_iri=BASE, relation_names=['next'],
                     feature_names=['Red', 'Blue'])


def get_explainer(algorithm, explanation_type='phenomenon', model=None):
    return Explainer(
        model=model or GCN(),
        algorithm=algorithm,
        explanation_type=explanation_type,
        node_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )


@withPackage('ontolearn', 'rdflib')
@pytest.mark.parametrize('learner', ['EvoLearner', 'CELOE'])
def test_ontolearn_explainer(learner):
    data = get_data()
    algorithm = OntolearnExplainer(learner, schema=get_schema(),
                                   num_hypotheses=2, max_runtime=2)
    explainer = get_explainer(algorithm)

    explanation = explainer(data.x, data.edge_index, target=data.y, index=0)
    assert isinstance(explanation, Explanation)
    # Learners may return any equivalent concept, e.g., `Red` or `¬Blue`:
    assert 'Red' in explanation.concept or 'Blue' in explanation.concept
    assert len(explanation.concepts) <= 2
    assert explanation.concept_class == 1
    assert explanation.concept_accuracy == 1.0
    assert explanation.concept_f1 == 1.0
    assert explanation.node_mask.size() == (data.num_nodes, 1)
    assert explanation.node_mask.view(-1).tolist() == data.y.float().tolist()

    # Results are cached:
    explanation = explainer(data.x, data.edge_index, target=data.y, index=2)
    assert explanation.node_mask.view(-1).tolist() == data.y.float().tolist()
    assert len(algorithm._cache) == 1


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_with_owl_data(tmp_path):
    data = OWLData.from_data(get_data(), get_schema())
    data = OWLData.from_owl(data.save_owl(str(tmp_path / 'graph.owl')),
                            target_classes=['Red'])
    assert BASE + 'Red' not in data.schema.feature_names

    # The target class `Red` is not available to the concept learner,
    # but its complement is equivalent on this graph:
    algorithm = OntolearnExplainer('EvoLearner', data=data,
                                   example_index=torch.arange(8),
                                   ignore_classes=['Blue'], max_runtime=2)
    explainer = get_explainer(algorithm)
    x = torch.randn(data.num_nodes, 2)
    explanation = explainer(x, data.edge_index, target=data.y, index=0)
    assert 'Red' not in explanation.concept
    assert 'Blue' not in explanation.concept
    assert explanation.concept_class == 1

    # The class of class 0 is explained when indexing a node of class 0:
    explanation = explainer(x, data.edge_index, target=data.y, index=1)
    assert explanation.concept_class == 0

    with pytest.raises(ValueError, match="one class at a time"):
        explainer(x, data.edge_index, target=data.y, index=torch.arange(2))
    with pytest.raises(ValueError, match="does not match"):
        explainer(x[:3], data.edge_index, target=data.y[:3], index=0)

    algorithm = OntolearnExplainer(data=data, ignore_classes=['Green'])
    explainer = get_explainer(algorithm)
    with pytest.raises(ValueError, match="Could not find the OWL class"):
        explainer(x, data.edge_index, target=data.y, index=0)


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_model_explanation():
    data = get_data()
    algorithm = OntolearnExplainer('CELOE', schema=get_schema(),
                                   target_class=0, max_examples=3, seed=0,
                                   max_runtime=1)
    torch.manual_seed(12345)
    model = train(GCN(), data)
    explainer = get_explainer(algorithm, 'model', model)
    explanation = explainer(data.x, data.edge_index)
    assert explanation.concept_class == 0
    assert explanation.prediction.size() == (data.num_nodes, 2)
    assert explanation.target.tolist() == data.y.tolist()
    assert explanation.concept_accuracy == 1.0
    mask = explanation.node_mask.view(-1)
    assert mask.tolist() == (data.y == 0).float().tolist()


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_examples():
    data = get_data()
    algorithm = OntolearnExplainer(schema=get_schema(), balanced=True, seed=0,
                                   max_examples=4)
    algorithm.connect(
        dict(explanation_type='phenomenon', node_mask_type='object'),
        ModelConfig('binary_classification', 'node', 'raw'),
    )
    y = data.y.clone()
    y[:3] = 1  # Five positive and six negative examples.
    y[3] = -1  # Unlabeled.
    pos, neg = algorithm._get_examples(y, cls=1)
    assert pos.numel() == neg.numel() == 4
    assert 3 not in pos.tolist() + neg.tolist()
    assert algorithm._get_class(y, index=None) == 1

    with pytest.raises(ValueError, match="at least one positive"):
        algorithm._get_examples(torch.zeros_like(y), cls=1)


def test_ontolearn_explainer_supports():
    with pytest.raises(ValueError, match="does not support the learner"):
        OntolearnExplainer('DLLearner')
    with pytest.raises(ValueError, match="needs to be positive"):
        OntolearnExplainer(num_hypotheses=0)

    model_config = dict(mode='multiclass_classification', task_level='node',
                        return_type='raw')
    with pytest.raises(ValueError, match="not support the given explanation"):
        Explainer(GCN(), OntolearnExplainer(), explanation_type='model',
                  node_mask_type='attributes', model_config=model_config)
    with pytest.raises(ValueError, match="not support the given explanation"):
        Explainer(GCN(), OntolearnExplainer(), explanation_type='model',
                  node_mask_type='object', edge_mask_type='object',
                  model_config=model_config)
    with pytest.raises(ValueError, match="not support the given explanation"):
        Explainer(GCN(), OntolearnExplainer(), explanation_type='model',
                  node_mask_type='object',
                  model_config=dict(model_config, task_level='graph'))
    with pytest.raises(ValueError, match="not support the given explanation"):
        Explainer(
            GCN(1), OntolearnExplainer(), explanation_type='model',
            node_mask_type='object', model_config=dict(mode='regression',
                                                       task_level='node',
                                                       return_type='raw'))

    assert str(OntolearnExplainer('celoe')) == 'OntolearnExplainer(CELOE)'


class ConstantModel(torch.nn.Module):
    def forward(self, x, edge_index, **kwargs):
        return torch.zeros(x.size(0), 2)


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_edge_attr():
    # A ring, where only edges from even to odd nodes are double bonds. The
    # planted concept of class 1 thus is `∃ hasOutgoingEdge.DoubleBond`:
    num_nodes = 12
    row = torch.arange(num_nodes)
    col = (row + 1) % num_nodes
    edge_index = torch.cat([torch.stack([row, col]),
                            torch.stack([col, row])], dim=1)
    double = torch.cat([row % 2 == 0, torch.zeros(num_nodes, dtype=bool)])
    edge_attr = torch.stack([~double, double], dim=1).float()
    x = torch.ones(num_nodes, 1)
    y = (row % 2 == 0).long()

    schema = OWLSchema(base_iri=BASE, relation_names=['bond'],
                       feature_names=['Atom'],
                       edge_feature_names=['SingleBond', 'DoubleBond'])
    algorithm = OntolearnExplainer('EvoLearner', schema=schema, seed=0,
                                   max_runtime=5)
    explainer = get_explainer(algorithm, model=ConstantModel())
    explanation = explainer(x, edge_index, target=y, index=0,
                            edge_attr=edge_attr)
    assert explanation.concept_accuracy == 1.0
    assert explanation.node_mask.view(-1).tolist() == y.float().tolist()


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_name_collisions():
    # Two relations and two data properties share their local names, as in
    # AIFB (`swrc:type` and `merge:type`):
    data = get_data()
    data.edge_type = torch.arange(data.edge_index.size(1)) % 2
    data.x = torch.cat([data.x, torch.rand(data.num_nodes, 2)], dim=1)
    schema = OWLSchema(
        base_iri=BASE,
        relation_names=['http://a.org/x#next', 'http://b.org/y#next'],
        feature_names=[
            'Red', 'Blue', 'http://a.org/x#weight', 'http://b.org/y#weight'
        ],
        feature_encoding=['class', 'class', 'numeric', 'numeric'],
    )
    algorithm = OntolearnExplainer('EvoLearner', schema=schema, seed=0,
                                   max_runtime=2)
    explainer = get_explainer(algorithm, model=ConstantModel())
    explanation = explainer(data.x, data.edge_index, target=data.y, index=0,
                            edge_type=data.edge_type)
    assert explanation.concept_accuracy == 1.0

    # Fitting again rebuilds the primitives of EvoLearner:
    explanation = explainer(data.x, data.edge_index, target=data.y, index=1,
                            edge_type=data.edge_type)
    assert explanation.concept_class == 0
    assert explanation.concept_accuracy == 1.0

    # The patch of DEAP is only active during learning:
    from deap import gp
    assert gp.PrimitiveSetTyped.addPrimitive.__module__ == 'deap.gp'


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_urn_iris():
    data = get_data()
    schema = OWLSchema(base_iri='urn:example:graph#')
    explainer = get_explainer(OntolearnExplainer(schema=schema))
    with pytest.raises(ValueError, match="contain a '/'"):
        explainer(data.x, data.edge_index, target=data.y, index=0)
