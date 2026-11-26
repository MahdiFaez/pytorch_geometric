import pytest
import torch

from torch_geometric.data import Data, OWLData, OWLSchema
from torch_geometric.testing import withPackage
from torch_geometric.utils import from_owl, to_owl

BASE = 'http://example.org/test#'


def get_data() -> Data:
    x = torch.tensor([
        [1.0, 0.0, 30.0],
        [0.0, 1.0, 40.0],
        [1.0, 1.0, 50.0],
    ])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_type = torch.tensor([0, 0, 1, 1])
    y = torch.tensor([0, 1, 1])
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)


def get_schema(**kwargs) -> OWLSchema:
    return OWLSchema(
        base_iri=BASE,
        node_class='Person',
        node_names=['alice', 'bob', 'carol'],
        relation_names=['knows', 'likes'],
        feature_names=['Student', 'Teacher', 'age'],
        class_hierarchy={
            'Student': 'Role',
            'Teacher': 'Role'
        },
        **kwargs,
    )


@withPackage('rdflib')
def test_to_owl():
    from rdflib import Literal, URIRef
    from rdflib.namespace import OWL, RDF, RDFS, XSD

    graph = to_owl(get_data(), get_schema())

    def iri(name: str) -> URIRef:
        return URIRef(BASE + name)

    assert (iri('alice'), RDF.type, OWL.NamedIndividual) in graph
    assert (iri('alice'), RDF.type, iri('Person')) in graph
    assert (iri('knows'), RDF.type, OWL.ObjectProperty) in graph
    assert (iri('alice'), iri('knows'), iri('bob')) in graph
    assert (iri('bob'), iri('likes'), iri('carol')) in graph
    assert (iri('alice'), iri('likes'), iri('bob')) not in graph

    # Binary features are encoded as classes, only true facts are asserted:
    assert (iri('Student'), RDF.type, OWL.Class) in graph
    assert (iri('alice'), RDF.type, iri('Student')) in graph
    assert (iri('bob'), RDF.type, iri('Student')) not in graph
    assert (iri('Student'), RDFS.subClassOf, iri('Role')) in graph

    # Continuous features are encoded as numeric data properties:
    assert (iri('age'), RDF.type, OWL.DatatypeProperty) in graph
    assert (iri('carol'), iri('age'), Literal(50.0,
                                              datatype=XSD.double)) in graph

    # Labels are not added by default:
    assert not any('label_' in str(obj) for obj in graph.objects())


@withPackage('rdflib')
def test_to_owl_labels_and_defaults():
    from rdflib import URIRef
    from rdflib.namespace import RDF

    data = Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=3,
                y=torch.tensor([0, 1, -1]))
    graph = to_owl(
        data,
        OWLSchema(base_iri=BASE, include_labels=True,
                  label_names=['Neg', 'Pos']))
    assert (URIRef(BASE + 'node_0'), URIRef(BASE + 'related_to'),
            URIRef(BASE + 'node_1')) in graph
    assert (URIRef(BASE + 'node_1'), RDF.type, URIRef(BASE + 'Pos')) in graph
    assert len(list(graph.subjects(RDF.type, URIRef(BASE + 'Neg')))) == 1

    with pytest.raises(ValueError, match="relation names"):
        to_owl(get_data(), OWLSchema(relation_names=['knows']))
    with pytest.raises(ValueError, match="feature names"):
        to_owl(get_data(), OWLSchema(feature_names=['a']))
    with pytest.raises(ValueError, match="Unknown feature encoding"):
        OWLSchema(feature_encoding='one_hot')


@withPackage('rdflib')
def test_to_owl_feature_encodings():
    from rdflib import Literal, URIRef
    from rdflib.namespace import RDF, XSD

    data = get_data()
    schema = get_schema(feature_encoding=['boolean', 'ignore', 'numeric'])
    graph = to_owl(data, schema)

    alice, student = URIRef(BASE + 'alice'), URIRef(BASE + 'Student')
    assert (alice, student, Literal(True, datatype=XSD.boolean)) in graph
    assert (alice, RDF.type, student) not in graph
    assert len(list(graph.triples((None, URIRef(BASE + 'Teacher'), None))))\
        == 0


@withPackage('rdflib')
def test_from_owl():
    graph = to_owl(get_data(), get_schema())
    data = from_owl(graph, target_classes=['Teacher'])

    assert isinstance(data, OWLData)
    assert data.num_nodes == 3
    assert data.node_iri == [BASE + 'alice', BASE + 'bob', BASE + 'carol']
    assert data.y.tolist() == [0, 1, 1]

    # Relations keep their order, inverse edges follow `Entities`:
    assert data.schema.relation_names == [BASE + 'knows', BASE + 'likes']
    assert data.schema.inverse_edge_types
    edges = set(
        zip(data.edge_index[0].tolist(), data.edge_index[1].tolist(),
            data.edge_type.tolist()))
    assert edges == {(0, 1, 0), (1, 0, 1), (1, 0, 0), (0, 1, 1), (1, 2, 2),
                     (2, 1, 3), (2, 1, 2), (1, 2, 3)}

    # The target class is never a node feature. Columns of the original
    # feature matrix come first, inferred types follow:
    names = data.schema.feature_names
    assert names == [
        BASE + 'Student', BASE + 'age', BASE + 'Person', BASE + 'Role'
    ]
    assert data.schema.target_classes == [BASE + 'Teacher']
    assert data.x[:, names.index(BASE + 'Student')].tolist() == [1, 0, 1]
    assert data.x[:, names.index(BASE + 'age')].tolist() == [30, 40, 50]
    # `bob` is only a `Role` via `Teacher`, which would leak the label:
    assert data.x[:, names.index(BASE + 'Role')].tolist() == [1, 0, 1]


@withPackage('rdflib')
def test_from_owl_options(tmp_path):
    data = OWLData.from_data(get_data(), get_schema())
    path = data.save_owl(str(tmp_path / 'graph.owl'))

    out = from_owl(path, features=(), add_inverse_edges=False)
    assert out.source == path
    assert out.x is None
    assert out.num_nodes == 3
    assert out.edge_type.tolist() == [0, 0, 1, 1]
    assert not out.schema.inverse_edge_types

    with pytest.warns(UserWarning, match="more than one target class"):
        out = from_owl(path, target_classes=['Student', 'Teacher'],
                       ignore_classes=['Person'])
    assert out.y.tolist() == [0, 1, 0]
    assert BASE + 'Person' not in out.schema.feature_names
    assert out.schema.ignore_classes == [BASE + 'Person']

    with pytest.raises(ValueError, match="Could not find the OWL class"):
        from_owl(path, target_classes=['Unknown'])
    with pytest.raises(ValueError, match="Unknown feature type"):
        from_owl(path, features=['identity'])


@withPackage('rdflib')
def test_owl_round_trip():
    data = OWLData.from_data(get_data(), get_schema())
    out = from_owl(data.to_owl())
    out = from_owl(out.to_owl())  # Converting twice is lossless.

    assert out.node_iri == data.node_iri
    assert out.edge_index.size(1) == 2 * data.edge_index.size(1)
    forward = out.edge_type % 2 == 0
    assert set(map(tuple, out.edge_index[:, forward].t().tolist())) == set(
        map(tuple,
            data.edge_index.t().tolist()))


@withPackage('rdflib')
def test_from_owl_natural_order():
    data = Data(edge_index=torch.tensor([[0, 11], [11, 0]]), num_nodes=12)
    graph = to_owl(data, OWLSchema(base_iri=BASE), annotate_order=False)

    out = from_owl(graph)
    assert out.node_iri == [BASE + f'node_{i}' for i in range(12)]


@withPackage('rdflib')
def test_owl_round_trip_order():
    # Node names, relation names and feature names that are not sorted:
    data = get_data()
    schema = get_schema()
    schema.node_names = ['carol', 'alice', 'bob']
    schema.relation_names = ['likes', 'knows']
    schema.feature_names = ['Teacher', 'Student', 'age']

    out = from_owl(to_owl(data, schema))
    assert out.node_iri == [BASE + name for name in schema.node_names]
    assert out.schema.relation_names == [
        BASE + name for name in schema.relation_names
    ]
    assert out.schema.feature_names[:3] == [
        BASE + name for name in schema.feature_names
    ]
    assert torch.equal(out.x[:, :3], data.x)

    forward = out.edge_type % 2 == 0
    out_edges = torch.cat([out.edge_index, out.edge_type.view(1, -1) // 2])
    edges = torch.cat([data.edge_index, data.edge_type.view(1, -1)])
    assert sorted(out_edges[:, forward].t().tolist()) == sorted(
        edges.t().tolist())


@withPackage('rdflib')
def test_owl_edge_attr():
    from rdflib import URIRef
    from rdflib.namespace import RDF

    from torch_geometric.data.owl_data import PYG_EDGE, PYG_HAS_OUTGOING_EDGE

    # A parallel edge (0 -> 1) and edge features (a bond type and a length):
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 1, 2, 0]])
    edge_attr = torch.tensor([
        [1.0, 0.0, 1.5],
        [0.0, 1.0, 1.2],
        [1.0, 0.0, 1.4],
        [0.0, 1.0, 1.3],
    ])
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    schema = OWLSchema(
        base_iri=BASE, relation_names=['bond'],
        edge_feature_names=['SingleBond', 'DoubleBond', 'length'])

    graph = to_owl(data, schema)
    edge = URIRef(BASE + 'edge_1')
    assert (edge, RDF.type, URIRef(PYG_EDGE)) in graph
    assert (edge, RDF.type, URIRef(BASE + 'DoubleBond')) in graph
    assert (URIRef(BASE + 'node_0'), URIRef(PYG_HAS_OUTGOING_EDGE),
            edge) in graph
    assert (URIRef(BASE + 'node_0'), URIRef(BASE + 'bond'),
            URIRef(BASE + 'node_1')) in graph

    out = from_owl(graph, add_inverse_edges=False)
    assert out.num_nodes == 3  # Edge individuals are no nodes.
    assert out.x.size(1) == 1  # Only `Node`, no `pyg:Edge` or edge features.
    assert out.schema.reify_edges
    assert out.schema.edge_feature_names == [
        BASE + 'SingleBond', BASE + 'DoubleBond', BASE + 'length'
    ]
    assert out.schema.edge_feature_encoding == ['class', 'class', 'numeric']
    assert torch.equal(out.edge_index, edge_index)
    assert torch.allclose(out.edge_attr, edge_attr)

    # Inverse edges share the features of their forward edge:
    out = from_owl(graph)
    assert out.edge_index.size(1) == 8
    forward = out.edge_type == 0
    assert torch.allclose(out.edge_attr[forward], edge_attr)
    assert torch.allclose(out.edge_attr[~forward].sum(0), edge_attr.sum(0))

    # Converting twice is lossless:
    out = from_owl(out.to_owl(), add_inverse_edges=False)
    assert torch.equal(out.edge_index, edge_index)
    assert torch.allclose(out.edge_attr, edge_attr)

    # Parallel edges are preserved even without edge features:
    data = Data(edge_index=edge_index, num_nodes=3)
    out = from_owl(to_owl(data, OWLSchema(base_iri=BASE)),
                   add_inverse_edges=False)
    assert torch.equal(out.edge_index, edge_index)
    assert out.edge_attr is None

    data = Data(edge_index=edge_index[:, 1:], num_nodes=3)
    graph = to_owl(data, OWLSchema(base_iri=BASE))
    assert len(list(graph.subjects(RDF.type, URIRef(PYG_EDGE)))) == 0

    with pytest.raises(ValueError, match="one row per edge"):
        to_owl(
            Data(edge_index=edge_index, edge_attr=edge_attr[:2], num_nodes=3))
    with pytest.raises(ValueError, match="edge feature names"):
        to_owl(Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3),
               OWLSchema(edge_feature_names=['length']))


@withPackage('rdflib')
def test_from_owl_skips_builtin_classes():
    from rdflib import URIRef
    from rdflib.namespace import OWL, RDF, RDFS, XSD

    datatype = URIRef(str(OWL) + 'Datatype')  # Used in, *e.g.*, AIFB.
    graph = to_owl(get_data(), get_schema())
    graph.add((datatype, RDF.type, RDFS.Class))
    graph.add((URIRef(BASE + 'myString'), RDF.type, datatype))
    graph.add((XSD.string, RDF.type, OWL.NamedIndividual))
    graph.add((URIRef(BASE + 'Role'), RDFS.subClassOf, OWL.Thing))

    out = from_owl(graph)
    assert out.num_nodes == 3  # Datatypes are no individuals.
    names = out.schema.feature_names
    assert str(datatype) not in names
    assert str(OWL.Thing) not in names
