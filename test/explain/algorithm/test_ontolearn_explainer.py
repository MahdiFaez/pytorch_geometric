import re

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


def get_explainer(algorithm, explanation_type='phenomenon', model=None,
                  edge_mask_type=None):
    return Explainer(
        model=model or GCN(),
        algorithm=algorithm,
        explanation_type=explanation_type,
        node_mask_type='object',
        edge_mask_type=edge_mask_type,
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
                  node_mask_type='object',
                  model_config=dict(model_config, task_level='edge'))
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


class HeteroConstantModel(torch.nn.Module):
    def forward(self, x_dict, edge_index_dict, **kwargs):
        return torch.zeros(x_dict['author'].size(0), 2)


def get_hetero_data():
    from torch_geometric.data import HeteroData

    # Author `i` writes paper `i`, and papers with an even index are theory
    # papers. The planted concept of class 1 is `∃ writes.Theory`:
    num_nodes = 8
    data = HeteroData()
    data['author'].x = torch.ones(num_nodes, 1)
    theory = (torch.arange(num_nodes) % 2 == 0).float().view(-1, 1)
    data['paper'].x = torch.cat([theory, 1 - theory], dim=1)
    data['author', 'writes',
         'paper'].edge_index = torch.arange(num_nodes).repeat(2, 1)
    data['author'].y = theory.view(-1).long()
    return data


@withPackage('ontolearn', 'rdflib')
@pytest.mark.parametrize('use_data', [False, True])
def test_ontolearn_explainer_hetero(use_data, tmp_path):
    from torch_geometric.data import OWLHeteroData
    from torch_geometric.explain import HeteroExplanation

    data = get_hetero_data()
    schema = OWLSchema(
        base_iri=BASE, feature_names={
            'author': ['Author'],
            'paper': ['Theory', 'Systems'],
        })
    kwargs = dict(schema=schema)
    if use_data:
        path = OWLHeteroData.from_data(data, schema).save_owl(
            str(tmp_path / 'graph.owl'))
        kwargs = dict(data=OWLHeteroData.from_owl(path, ['author', 'paper']))

    algorithm = OntolearnExplainer('EvoLearner', node_type='author', seed=0,
                                   ignore_classes=['Systems'], max_runtime=5,
                                   **kwargs)
    explainer = get_explainer(algorithm, model=HeteroConstantModel())
    explanation = explainer(data.x_dict, data.edge_index_dict,
                            target=data['author'].y, index=0)
    assert isinstance(explanation, HeteroExplanation)
    assert explanation.concept_class == 1
    assert explanation.concept_accuracy == 1.0
    assert 'writes' in explanation.concept
    mask = explanation['author'].node_mask.view(-1)
    assert mask.tolist() == data['author'].y.float().tolist()
    assert explanation['paper'].node_mask.size() == (8, 1)


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_hetero_errors():
    data = get_hetero_data()
    explainer = get_explainer(OntolearnExplainer(),
                              model=HeteroConstantModel())
    with pytest.raises(ValueError, match="requires 'node_type'"):
        explainer(data.x_dict, data.edge_index_dict, target=data['author'].y,
                  index=0)

    explainer = get_explainer(OntolearnExplainer(node_type='venue'),
                              model=HeteroConstantModel())
    with pytest.raises(ValueError, match="'venue' is not part"):
        explainer(data.x_dict, data.edge_index_dict, target=data['author'].y,
                  index=0)

    homogeneous = OWLData.from_data(get_data(), get_schema())
    explainer = get_explainer(
        OntolearnExplainer(data=homogeneous, node_type='author'),
        model=HeteroConstantModel())
    with pytest.raises(ValueError, match="to be heterogeneous"):
        explainer(data.x_dict, data.edge_index_dict, target=data['author'].y,
                  index=0)


MOL = 'http://example.org/mol#'


def get_molecules() -> list:
    # Molecules of even index hold a nitrogen atom bonded to an oxygen atom
    # (the planted concept of class 1), molecules of odd index hold an oxygen
    # atom that is only bonded to carbon atoms:
    generator = torch.Generator().manual_seed(0)
    data_list = []
    for i in range(16):
        num_nodes = 4 + i % 3
        atoms = torch.randint(0, 2, (num_nodes, ), generator=generator)
        if i % 2 == 0:
            atoms[0], atoms[1] = 1, 2
        else:
            atoms[:4] = torch.tensor([0, 2, 0, 0])
        row = torch.arange(num_nodes - 1)
        col = row + 1
        edge_index = torch.cat(
            [torch.stack([row, col]),
             torch.stack([col, row])], dim=1)
        x = torch.nn.functional.one_hot(atoms, 3).float()
        data_list.append(
            Data(x=x, edge_index=edge_index,
                 y=torch.tensor([int(i % 2 == 0)])))
    return data_list


def get_molecule_schema() -> OWLSchema:
    return OWLSchema(base_iri=MOL, node_class='Atom', graph_class='Molecule',
                     has_node_name='hasAtom', relation_names=['bond'],
                     feature_names=['C', 'N', 'O'])


class NitroModel(torch.nn.Module):
    # Predicts class 1 for graphs with a bond from nitrogen to oxygen:
    def forward(self, x, edge_index, batch):
        row, col = edge_index
        score = torch.zeros(int(batch.max()) + 1)
        score.scatter_add_(0, batch[row], x[row, 1] * x[col, 2])
        return torch.stack([-score, score],
                           dim=-1) - 0.5 * torch.tensor([-1.0, 1.0])


def get_graph_explainer(algorithm, explanation_type='phenomenon',
                        edge_mask_type='object'):
    return Explainer(
        model=NitroModel(),
        algorithm=algorithm,
        explanation_type=explanation_type,
        node_mask_type='object',
        edge_mask_type=edge_mask_type,
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_graph():
    from torch_geometric.data import Batch
    from torch_geometric.data.owl_data import PYG_HAS_NODE

    batch = Batch.from_data_list(get_molecules())
    algorithm = OntolearnExplainer('CELOE', schema=get_molecule_schema(),
                                   max_runtime=3)
    explainer = get_graph_explainer(algorithm)
    explanation = explainer(batch.x, batch.edge_index, target=batch.y, index=0,
                            batch=batch.batch)

    assert isinstance(explanation, Explanation)
    assert 'hasAtom' in explanation.concept and 'N' in explanation.concept
    assert explanation.concept_class == 1
    assert explanation.concept_accuracy == 1.0
    assert explanation.graph_mask.size() == (batch.num_graphs, 1)
    assert explanation.graph_mask.view(-1).tolist() == batch.y.float().tolist()
    assert bool(explanation.concept_agreement.all())

    # The witnesses are the nitrogen atoms of the positive graphs (and the
    # oxygen atoms they are bonded to, if the concept mentions them):
    witnesses = explanation.node_mask.view(-1).bool()
    assert explanation.node_mask.size() == (batch.num_nodes, 1)
    assert batch.y[batch.batch[witnesses]].tolist() == [1] * int(
        witnesses.sum())
    assert batch.x[witnesses, 0].sum() == 0  # No carbon atoms.
    nitrogen = (batch.x[:, 1] == 1) & (batch.y[batch.batch] == 1)
    assert bool(witnesses[nitrogen & (batch.x[:, 1] == 1)].any())
    assert explanation.edge_mask.size() == (batch.num_edges, )
    row, col = batch.edge_index[:, explanation.edge_mask.bool()]
    assert bool(witnesses[row].all()) and bool(witnesses[col].all())

    # `pyg:hasNode` holds no assertions (graphs use `hasAtom`), so it is not
    # searched:
    properties = [p.str for p in algorithm._kb.get_object_properties()]
    assert MOL + 'hasAtom' in properties and PYG_HAS_NODE not in properties

    # Results (and witnesses) are cached, and `index` selects the class:
    explanation = explainer(batch.x, batch.edge_index, target=batch.y, index=2,
                            batch=batch.batch)
    assert len(algorithm._cache) == 1
    explanation = explainer(batch.x, batch.edge_index, target=batch.y, index=1,
                            batch=batch.batch)
    assert explanation.concept_class == 0
    assert explanation.graph_mask.view(-1).tolist() == (
        batch.y == 0).float().tolist()


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_graph_model_and_data(tmp_path):
    from torch_geometric.data import Batch
    from torch_geometric.utils import from_owl, to_owl

    # A graph dataset read from OWL, explained via the model's predictions,
    # without edge masks and with subsampled examples:
    path = str(tmp_path / 'molecules.owl')
    to_owl(get_molecules(), get_molecule_schema()).serialize(path, 'xml')
    data_list = from_owl(
        path, graph_level=True, add_inverse_edges=False,
        schema=OWLSchema(base_iri=MOL, graph_class='Molecule'))
    batch = Batch.from_data_list(data_list)
    algorithm = OntolearnExplainer('CELOE', data=data_list, target_class=1,
                                   max_examples=6, balanced=True, seed=0,
                                   max_runtime=3)
    explainer = get_graph_explainer(algorithm, 'model', edge_mask_type=None)
    explanation = explainer(batch.x[:, :3], batch.edge_index,
                            batch=batch.batch)
    assert explanation.target.tolist() == [1, 0] * 8
    assert explanation.concept_class == 1
    assert explanation.concept_accuracy == 1.0
    assert 'edge_mask' not in explanation
    assert explanation.graph_mask.size() == (16, 1)
    assert algorithm._kb.path == path  # The original ontology is used.

    with pytest.raises(ValueError, match="number of graphs in 'data'"):
        explainer(batch.x[:, :3], batch.edge_index,
                  batch=torch.zeros(batch.num_nodes, dtype=torch.long))
    with pytest.raises(ValueError, match="number of nodes in 'data'"):
        other = Batch.from_data_list(data_list[:15] + data_list[14:15])
        assert other.num_nodes != batch.num_nodes
        small = other
        explainer(small.x[:, :3], small.edge_index, batch=small.batch)

    algorithm = OntolearnExplainer(data=data_list[0])
    explainer = get_graph_explainer(algorithm)
    with pytest.raises(ValueError, match="graph dataset"):
        explainer(batch.x, batch.edge_index, target=explanation.target,
                  batch=batch.batch)
    with pytest.raises(ValueError, match="non-empty list"):
        OntolearnExplainer(data=[])


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_graph_witnesses():
    from owlapy.class_expression import (
        OWLClass,
        OWLObjectComplementOf,
        OWLObjectIntersectionOf,
        OWLObjectMaxCardinality,
        OWLObjectMinCardinality,
        OWLObjectSomeValuesFrom,
        OWLObjectUnionOf,
    )
    from owlapy.iri import IRI
    from owlapy.owl_property import OWLObjectInverseOf, OWLObjectProperty

    from torch_geometric.data import Batch

    data_list = get_molecules()[:2]
    batch = Batch.from_data_list(data_list)
    algorithm = OntolearnExplainer('CELOE', schema=get_molecule_schema(),
                                   max_runtime=1)
    explainer = get_graph_explainer(algorithm)
    explainer(batch.x, batch.edge_index, target=batch.y, index=0,
              batch=batch.batch)

    def cls(name):
        return OWLClass(IRI.create(MOL + name))

    has_atom = OWLObjectProperty(IRI.create(MOL + 'hasAtom'))
    bond = OWLObjectProperty(IRI.create(MOL + 'bond'))
    nitrogen = OWLObjectSomeValuesFrom(has_atom, cls('N'))
    nitro = OWLObjectSomeValuesFrom(
        has_atom,
        OWLObjectIntersectionOf(
            [cls('N'), OWLObjectSomeValuesFrom(bond, cls('O'))]))
    graphs = algorithm._example_iri

    def witnesses(expression):
        individuals, _ = algorithm._get_witnesses(expression, graphs)
        index = {iri: i for i, iri in enumerate(algorithm._node_iri)}
        return sorted(index[iri] for iri in individuals if iri in index)

    n = batch.x[:, 1].nonzero().view(-1).tolist()
    row, col = batch.edge_index
    bonded = (batch.x[row, 1] == 1) & (batch.x[col, 2] == 1)  # N-O bonds.
    nitro_atoms = sorted(set(row[bonded].tolist() + col[bonded].tolist()))
    assert 0 in nitro_atoms and 1 in nitro_atoms
    assert witnesses(nitrogen) == n
    assert witnesses(nitro) == nitro_atoms
    assert witnesses(OWLObjectMinCardinality(1, has_atom, cls('N'))) == n
    # Negation and upper bounds have no witnesses:
    assert witnesses(OWLObjectComplementOf(nitrogen)) == []
    assert witnesses(OWLObjectMaxCardinality(1, has_atom, cls('N'))) == []
    # Unions only collect the witnesses of the operands that hold:
    union = OWLObjectUnionOf([nitro, cls('C')])
    assert witnesses(union) == nitro_atoms
    # Inverse properties are followed backwards:
    inverse = OWLObjectSomeValuesFrom(
        has_atom, OWLObjectSomeValuesFrom(OWLObjectInverseOf(bond), cls('N')))
    assert 1 in witnesses(inverse) and 0 in witnesses(inverse)


class FixedLearner:
    # A concept learner that always returns the same class expression:
    def __init__(self, expression):
        self.expression = expression

    def fit(self, problem):
        return self

    def best_hypotheses(self, n=1):
        return [self.expression]


def owl(name):
    from owlapy.class_expression import OWLClass
    from owlapy.iri import IRI
    from owlapy.owl_property import OWLObjectProperty
    if name[0].isupper():
        return OWLClass(IRI.create(BASE + name))
    return OWLObjectProperty(IRI.create(BASE + name))


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_witnesses():
    from owlapy.class_expression import OWLObjectSomeValuesFrom

    # A directed path in which every third node is red. Class 1 holds the
    # nodes whose successor is red, *i.e.*, `∃ next.Red`:
    num_nodes = 12
    red = torch.arange(num_nodes) % 3 == 0
    x = torch.stack([red, ~red], dim=1).float()
    edge_index = torch.stack([torch.arange(11), torch.arange(1, 12)])
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[:-1] = red[1:].long()

    concept = OWLObjectSomeValuesFrom(owl('next'), owl('Red'))
    algorithm = OntolearnExplainer(FixedLearner(concept), schema=get_schema(),
                                   target_class=1)
    explainer = get_explainer(algorithm, model=ConstantModel(),
                              edge_mask_type='object')

    # The witness of node 2 is its red successor 3, connected by edge 2:
    explanation = explainer(x, edge_index, target=y, index=2)
    assert explanation.concept == '∃ next.Red'
    assert explanation.node_mask.view(-1).tolist() == y.float().tolist()
    assert explanation.witness_mask.size() == (num_nodes, 1)
    assert explanation.witness_mask.view(-1).nonzero().view(-1).tolist() == [3]
    assert explanation.edge_mask.nonzero().view(-1).tolist() == [2]
    assert bool(explanation.concept_agreement.all())

    # Without an index, the witnesses of all instances are returned:
    explanation = explainer(x, edge_index, target=y)
    witnesses = explanation.witness_mask.view(-1).nonzero().view(-1)
    assert witnesses.tolist() == [3, 6, 9]
    assert explanation.edge_mask.nonzero().view(-1).tolist() == [2, 5, 8]

    # Nodes that are no instances have no witnesses, and disagreements with
    # the target are reported:
    target = y.clone()
    target[0] = 1
    explanation = explainer(x, edge_index, target=target, index=0)
    assert int(explanation.witness_mask.sum()) == 0
    assert explanation.concept_agreement.tolist() == [False] + [True] * 11

    # Without edge masks:
    explainer = get_explainer(algorithm, model=ConstantModel())
    explanation = explainer(x, edge_index, target=y, index=2)
    assert 'edge_mask' not in explanation
    assert int(explanation.witness_mask.sum()) == 1


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_witnesses_reified_edges():
    from owlapy.class_expression import OWLObjectSomeValuesFrom
    from owlapy.iri import IRI
    from owlapy.owl_property import OWLObjectProperty

    from torch_geometric.data.owl_data import PYG_HAS_OUTGOING_EDGE

    # A ring in which edges from even to odd nodes are double bonds:
    num_nodes = 12
    row = torch.arange(num_nodes)
    col = (row + 1) % num_nodes
    edge_index = torch.cat([torch.stack([row, col]),
                            torch.stack([col, row])], dim=1)
    double = torch.cat([row % 2 == 0, torch.zeros(num_nodes, dtype=bool)])
    edge_attr = torch.stack([~double, double], dim=1).float()
    y = (row % 2 == 0).long()

    schema = OWLSchema(base_iri=BASE, relation_names=['bond'],
                       feature_names=['Atom'],
                       edge_feature_names=['SingleBond', 'DoubleBond'])
    outgoing = OWLObjectProperty(IRI.create(PYG_HAS_OUTGOING_EDGE))
    concept = OWLObjectSomeValuesFrom(outgoing, owl('DoubleBond'))
    algorithm = OntolearnExplainer(FixedLearner(concept), schema=schema)
    explainer = get_explainer(algorithm, model=ConstantModel(),
                              edge_mask_type='object')
    explanation = explainer(torch.ones(num_nodes, 1), edge_index, target=y,
                            index=4, edge_attr=edge_attr)

    # The witness of node 4 is its (reified) double bond to node 5:
    assert explanation.node_mask.view(-1).tolist() == y.float().tolist()
    assert int(explanation.witness_mask.sum()) == 0  # An edge, not a node.
    assert explanation.edge_mask.nonzero().view(-1).tolist() == [4]


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_witnesses_hetero():
    from owlapy.class_expression import OWLObjectSomeValuesFrom

    from torch_geometric.explain import HeteroExplanation

    data = get_hetero_data()
    schema = OWLSchema(
        base_iri=BASE, feature_names={
            'author': ['Author'],
            'paper': ['Theory', 'Systems'],
        })
    concept = OWLObjectSomeValuesFrom(owl('writes'), owl('Theory'))
    algorithm = OntolearnExplainer(FixedLearner(concept), schema=schema,
                                   node_type='author')
    explainer = get_explainer(algorithm, model=HeteroConstantModel(),
                              edge_mask_type='object')
    explanation = explainer(data.x_dict, data.edge_index_dict,
                            target=data['author'].y, index=2)

    # The witness of author 2 is the theory paper 2 they write:
    assert isinstance(explanation, HeteroExplanation)
    edge_type = ('author', 'writes', 'paper')
    assert int(explanation['author'].witness_mask.sum()) == 0
    witnesses = explanation['paper'].witness_mask.view(-1).nonzero()
    assert witnesses.view(-1).tolist() == [2]
    assert explanation[edge_type].edge_mask.nonzero().view(-1).tolist() == [2]
    assert bool(explanation['author'].concept_agreement.all())


@withPackage('ontolearn', 'rdflib')
def test_ontolearn_explainer_witness_edge_types():
    from owlapy.class_expression import OWLObjectSomeValuesFrom

    # Node 0 is related to the red node 1 via `next` and `likes`, with
    # inverse edge types (`next` = 0, `next⁻` = 1, `likes` = 2):
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 0]])
    edge_type = torch.tensor([0, 2, 1])
    schema = OWLSchema(base_iri=BASE, relation_names=['next', 'likes'],
                       inverse_edge_types=True, feature_names=['Red', 'Blue'])
    concept = OWLObjectSomeValuesFrom(owl('next'), owl('Red'))
    algorithm = OntolearnExplainer(FixedLearner(concept), schema=schema)
    explainer = get_explainer(algorithm, model=ConstantModel(),
                              edge_mask_type='object')
    explanation = explainer(x, edge_index, target=torch.tensor([1, 0]),
                            index=0, edge_type=edge_type)

    # Only the `next` assertion (and its inverse edge) is a witness edge:
    assert explanation.witness_mask.view(-1).tolist() == [0.0, 1.0]
    assert explanation.edge_mask.tolist() == [1.0, 0.0, 1.0]


@withPackage('ontolearn', 'rdflib')
@pytest.mark.parametrize('learner', ['EvoLearner', 'CELOE'])
def test_ontolearn_explainer_card_limit(learner):
    # Hubs 0 to 3 have 12 successors, hubs 4 to 7 have 11 successors. The
    # planted concept `≥ 12 next.⊤` exceeds Ontolearn's default limit of 10:
    row = torch.arange(8).repeat_interleave(torch.tensor([12] * 4 + [11] * 4))
    col = torch.arange(8, 8 + row.numel())
    num_nodes = 8 + row.numel()
    y = torch.full((num_nodes, ), -1)
    y[:4], y[4:8] = 1, 0
    x = torch.ones(num_nodes, 1)
    schema = OWLSchema(base_iri=BASE, relation_names=['next'],
                       feature_names=['Thing'])

    def explain(**kwargs):
        algorithm = OntolearnExplainer(learner, schema=schema, **kwargs)
        explainer = get_explainer(algorithm, model=ConstantModel())
        return algorithm, explainer(x, torch.stack([row, col]), target=y,
                                    index=0)

    algorithm, explanation = explain(card_limit=15, max_runtime=2)
    concept_learner = algorithm._concept_learner
    if learner == 'EvoLearner':
        assert concept_learner.card_limit == 15
    else:
        assert concept_learner.operator.card_limit == 15
    # Learners may return any equivalent concept, e.g., `≥ 12 next.Thing` or
    # `¬(≤ 11 next.Thing)`, but each needs a number larger than 10:
    numbers = re.findall(r'[≥≤] (\d+)', explanation.concept)
    assert max(int(number) for number in numbers) > 10
    assert explanation.concept_accuracy == 1.0

    # With the default limit, the hubs cannot be separated:
    _, explanation = explain(max_runtime=1)
    assert explanation.concept_accuracy < 1.0


@withPackage('ontolearn')
def test_ontolearn_explainer_card_limit_errors():
    with pytest.raises(ValueError, match="needs to be positive"):
        OntolearnExplainer(card_limit=0)
    with pytest.raises(ValueError, match="already initialized learner"):
        OntolearnExplainer(FixedLearner(None), card_limit=5)
    with pytest.raises(ValueError, match="Pass either"):
        OntolearnExplainer('CELOE', card_limit=5, refinement_operator=None)
    with pytest.raises(ValueError, match="not supported for the learner"):
        OntolearnExplainer(card_limit=5)._get_learner_kwargs(object, None)

    # Without a limit, the learner's own arguments are passed as they are:
    algorithm = OntolearnExplainer('CELOE', max_runtime=1)
    assert algorithm._get_learner_kwargs(object, None) == dict(max_runtime=1)
