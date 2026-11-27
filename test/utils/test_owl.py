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


def get_hetero_data():
    from torch_geometric.data import HeteroData
    from torch_geometric.transforms import ToUndirected

    data = HeteroData()
    data['author'].x = torch.tensor([[1.0, 0.0, 30.0], [0.0, 1.0, 40.0]])
    data['paper'].x = torch.tensor([[1.0], [0.0], [1.0]])
    data['author', 'writes', 'paper'].edge_index = torch.tensor([
        [0, 0, 1],
        [0, 1, 2],
    ])
    data['author', 'writes', 'paper'].edge_attr = torch.tensor([1., 2., 3.])
    data['paper', 'cites', 'paper'].edge_index = torch.tensor([[0, 1], [1, 2]])
    return ToUndirected()(data)


def get_hetero_schema(**kwargs) -> OWLSchema:
    return OWLSchema(
        base_iri=BASE,
        node_class='Thing',
        inverse_edge_types=True,
        feature_names={
            'author': ['Senior', 'Junior', 'age'],
            'paper': ['Theory'],
        },
        edge_feature_names={'writes': ['position']},
        **kwargs,
    )


@withPackage('rdflib')
def test_hetero_to_owl():
    from rdflib import Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD

    graph = to_owl(get_hetero_data(), get_hetero_schema())

    def iri(name: str) -> URIRef:
        return URIRef(BASE + name)

    assert (iri('author_0'), RDF.type, iri('author')) in graph
    assert (iri('paper_2'), RDF.type, iri('paper')) in graph
    assert (iri('author'), RDFS.subClassOf, iri('Thing')) in graph
    assert (iri('author_0'), RDF.type, iri('Senior')) in graph
    assert (iri('author_1'), iri('age'), Literal(40.0,
                                                 datatype=XSD.double)) in graph
    assert (iri('paper_0'), RDF.type, iri('Theory')) in graph
    assert (iri('author_1'), iri('writes'), iri('paper_2')) in graph
    assert (iri('paper_0'), iri('cites'), iri('paper_1')) in graph
    # Reverse edge types are implied by their forward edge type:
    assert len(list(graph.triples((None, iri('rev_writes'), None)))) == 0

    with pytest.raises(ValueError, match="encodings of heterogeneous"):
        to_owl(get_hetero_data(), OWLSchema(feature_encoding=['class']))
    with pytest.raises(ValueError, match="of node type 'paper'"):
        to_owl(get_hetero_data(), OWLSchema(feature_names={'paper': []}))


@withPackage('rdflib')
def test_hetero_from_owl():
    from torch_geometric.data import OWLHeteroData

    data = get_hetero_data()
    graph = to_owl(data, get_hetero_schema())
    out = from_owl(graph, node_types=['author', 'paper'])
    assert isinstance(out, OWLHeteroData)
    assert set(out.node_types) == {'author', 'paper'}
    assert set(out.edge_types) == {
        ('author', 'writes', 'paper'),
        ('paper', 'rev_writes', 'author'),
        ('paper', 'cites', 'paper'),
        ('paper', 'rev_cites', 'paper'),
    }
    assert out['author'].node_iri == [BASE + 'author_0', BASE + 'author_1']
    assert out.schema.node_type_names == {
        'author': BASE + 'author',
        'paper': BASE + 'paper',
    }
    assert out.schema.relation_names == {
        'writes': BASE + 'writes',
        'cites': BASE + 'cites',
    }

    # Node type classes and their super classes are no features:
    assert out.schema.feature_names['author'][:3] == [
        BASE + 'Senior', BASE + 'Junior', BASE + 'age'
    ]
    assert torch.equal(out['author'].x[:, :3], data['author'].x)
    assert torch.equal(out['paper'].x[:, :1], data['paper'].x)

    store = out['author', 'writes', 'paper']
    assert torch.equal(store.edge_index, data['author', 'writes',
                                              'paper'].edge_index)
    assert store.edge_attr.view(-1).tolist() == [1.0, 2.0, 3.0]
    store = out['paper', 'rev_writes', 'author']
    assert store.edge_index.tolist() == [[0, 1, 2], [0, 0, 1]]
    assert store.edge_attr.view(-1).tolist() == [1.0, 2.0, 3.0]
    assert out.schema.edge_feature_names == {'writes': [BASE + 'position']}

    # Converting twice is lossless:
    out2 = from_owl(out.to_owl(), node_types=['author', 'paper'])
    for node_type in out.node_types:
        assert out2[node_type].node_iri == out[node_type].node_iri
        assert torch.equal(out2[node_type].x, out[node_type].x)
    for edge_type in out.edge_types:
        assert torch.equal(out2[edge_type].edge_index,
                           out[edge_type].edge_index)

    # Labels are set for every node type:
    out = from_owl(graph, node_types=['author', 'paper'],
                   target_classes=['Theory'])
    assert out['paper'].y.tolist() == [1, 0, 1]
    assert out['author'].y.tolist() == [0, 0]
    # `Theory` was the only feature of papers:
    assert 'x' not in out['paper']
    assert 'paper' not in out.schema.feature_names

    with pytest.warns(UserWarning, match="belong to none of the node types"):
        out = from_owl(graph, node_types=['paper'], add_inverse_edges=False)
    assert out.node_types == ['paper']
    assert out.edge_types == [('paper', 'cites', 'paper')]

    with pytest.warns(UserWarning, match="more than one node type"):
        from_owl(graph, node_types=['author', 'Thing', 'paper'])
    with pytest.raises(ValueError, match="duplicated node types"):
        from_owl(graph, node_types=['author', 'author'])


def get_graph_dataset() -> list:
    # Two "molecules": C-N-C with a double bond C=N, and a single N atom.
    g1 = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
        graph_attr=torch.tensor([[2.5]]),
        y=torch.tensor([1]),
    )
    g2 = Data(
        x=torch.tensor([[0.0, 1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
        graph_attr=torch.tensor([[1.0]]),
        y=torch.tensor([0]),
    )
    return [g1, g2]


def get_graph_schema(**kwargs) -> OWLSchema:
    return OWLSchema(
        base_iri=BASE,
        node_class='Atom',
        graph_class='Molecule',
        has_node_name='hasAtom',
        feature_names=['C', 'N'],
        edge_feature_names=['Double'],
        graph_feature_names=['logP'],
        label_names=['Inactive', 'Active'],
        **kwargs,
    )


@withPackage('rdflib')
def test_graph_dataset_to_owl():
    from rdflib import Literal, URIRef
    from rdflib.namespace import OWL, RDF, RDFS, XSD

    from torch_geometric.data.owl_data import PYG_GRAPH, PYG_HAS_NODE

    def iri(name: str) -> URIRef:
        return URIRef(BASE + name)

    graph = to_owl(get_graph_dataset(), get_graph_schema())

    # Every graph is an individual linked to its nodes:
    for name in ['graph_0', 'graph_1']:
        assert (iri(name), RDF.type, URIRef(PYG_GRAPH)) in graph
        assert (iri(name), RDF.type, iri('Molecule')) in graph
    assert (iri('hasAtom'), RDF.type, OWL.ObjectProperty) in graph
    assert (iri('hasAtom'), RDFS.subPropertyOf, URIRef(PYG_HAS_NODE)) in graph
    members = {(str(s), str(o))
               for s, o in graph.subject_objects(iri('hasAtom'))}
    assert members == {(BASE + 'graph_0', BASE + f'node_{i}')
                       for i in range(3)
                       } | {(BASE + 'graph_1', BASE + 'node_3')}

    # Node offsets are applied, and nodes and edges are converted as usual:
    assert (iri('node_3'), RDF.type, iri('N')) in graph
    assert (iri('node_1'), RDF.type, iri('Atom')) in graph
    assert (iri('node_2'), iri('related_to'), iri('node_1')) in graph
    assert (iri('edge_0'), RDF.type, iri('Double')) in graph

    # Graph features are asserted on the graph individuals:
    assert (iri('graph_0'), iri('logP'), Literal(2.5,
                                                 datatype=XSD.double)) in graph

    # Graph labels are not added by default, and only on graphs otherwise:
    assert (iri('Active'), RDF.type, OWL.Class) not in graph
    graph = to_owl(get_graph_dataset(), get_graph_schema(include_labels=True))
    assert (iri('graph_0'), RDF.type, iri('Active')) in graph
    assert (iri('graph_1'), RDF.type, iri('Inactive')) in graph
    assert not any(
        (iri(f'node_{i}'), RDF.type, iri('Active')) in graph for i in range(4))

    # Without a name, graphs are linked via `pyg:hasNode`:
    graph = to_owl(get_graph_dataset(), OWLSchema(base_iri=BASE))
    assert (iri('graph_1'), URIRef(PYG_HAS_NODE), iri('node_3')) in graph
    assert (None, RDFS.subPropertyOf, None) not in graph


@withPackage('rdflib')
def test_graph_dataset_inputs():
    from torch_geometric.data import Batch

    data_list = get_graph_dataset()
    schema = get_graph_schema()
    expected = len(to_owl(data_list, schema))

    # A batch, a tuple and a list of `OWLData` (whose schema is used) give
    # the same ontology:
    assert len(to_owl(Batch.from_data_list(data_list), schema)) == expected
    assert len(to_owl(tuple(data_list), schema)) == expected
    owl_list = [OWLData.from_data(data, schema) for data in data_list]
    for i, data in enumerate(owl_list):
        data.node_iri = [f'{BASE}m{i}_{j}' for j in range(data.num_nodes)]
        data.graph_iri = f'{BASE}m{i}'
    graph = to_owl(owl_list)
    assert len(graph) == expected
    out = from_owl(graph, schema=OWLSchema(base_iri=BASE), graph_level=True)
    assert [data.graph_iri for data in out] == [BASE + 'm0', BASE + 'm1']
    assert out[0].node_iri == [BASE + f'm0_{j}' for j in range(3)]

    with pytest.raises(ValueError, match="at least one graph"):
        to_owl([], schema)
    with pytest.raises(ValueError, match="'graph_iri' to hold 2 IRIs"):
        batch = Batch.from_data_list(data_list)
        batch.graph_iri = [BASE + 'm0']
        to_owl(batch, schema)
    with pytest.raises(ValueError, match="Expected 1 graph feature names"):
        schema = get_graph_schema()
        schema.graph_feature_names = ['a', 'b']
        to_owl(data_list, schema)


@withPackage('rdflib')
def test_graph_dataset_round_trip():
    from torch_geometric.data import Batch

    data_list = get_graph_dataset()
    graph = to_owl(data_list, get_graph_schema(include_labels=True))
    schema = OWLSchema(base_iri=BASE, graph_class='Molecule')
    out = from_owl(graph, schema=schema, target_classes=['Active'],
                   ignore_classes=['Inactive', 'Atom'], graph_level=True,
                   add_inverse_edges=False)
    assert [item.y.tolist() for item in out] == [[1], [0]]

    # Converting twice is lossless (labels are not written back):
    out = from_owl(to_owl(out), schema=schema, graph_level=True,
                   add_inverse_edges=False)

    assert len(out) == 2
    for data, item in zip(data_list, out):
        assert isinstance(item, OWLData)
        assert 'y' not in item
        assert torch.equal(item.x, data.x)
        assert torch.equal(item.graph_attr, data.graph_attr)
        edges = torch.cat([data.edge_index, data.edge_attr.t()]).t().tolist()
        out_edges = torch.cat([item.edge_index, item.edge_attr.t()]).t()
        assert sorted(out_edges.tolist()) == sorted(edges)
    assert out[0].graph_iri == BASE + 'graph_0'
    assert out[1].node_iri == [BASE + 'node_3']

    schema = out[0].schema
    assert schema.graph_class == BASE + 'Molecule'
    assert schema.has_node_name == BASE + 'hasAtom'
    assert schema.feature_names == [BASE + 'C', BASE + 'N']
    assert schema.graph_feature_names == [BASE + 'logP']

    batch = Batch.from_data_list(out)
    assert batch.num_graphs == 2 and batch.x.size() == (4, 2)
    assert batch.graph_iri == [BASE + 'graph_0', BASE + 'graph_1']


@withPackage('rdflib')
def test_from_owl_graph_level_errors():
    from rdflib import URIRef
    from rdflib.namespace import RDF

    from torch_geometric.data.owl_data import PYG_GRAPH, PYG_HAS_NODE

    def iri(name: str) -> URIRef:
        return URIRef(BASE + name)

    graph = to_owl(get_graph_dataset(), OWLSchema(base_iri=BASE))
    with pytest.raises(ValueError, match="cannot be combined"):
        from_owl(graph, graph_level=True, node_types=['Node'])
    with pytest.raises(ValueError, match="graph_level=True"):
        OWLData.from_owl(graph, graph_level=True)

    with pytest.raises(ValueError, match="Could not find any graph"):
        from_owl(to_owl(get_data(), get_schema()), graph_level=True)

    # Nodes that belong to no graph are dropped:
    graph.add((iri('lonely'), RDF.type, iri('Node')))
    with pytest.warns(UserWarning, match="1 individual"):
        out = from_owl(graph, graph_level=True)
    assert sum(data.num_nodes for data in out) == 4

    # A node in two graphs:
    graph.add((iri('graph_1'), URIRef(PYG_HAS_NODE), iri('node_0')))
    with pytest.raises(ValueError, match="more than one graph"):
        from_owl(graph, graph_level=True)

    # An edge between two graphs:
    graph = to_owl(get_graph_dataset(), OWLSchema(base_iri=BASE))
    graph.add((iri('node_0'), iri('related_to'), iri('node_3')))
    with pytest.raises(ValueError, match="different graphs"):
        from_owl(graph, graph_level=True)

    # Graphs without `pyg:Graph` are found via the schema:
    graph = to_owl(get_graph_dataset(), get_graph_schema())
    graph.remove((None, RDF.type, URIRef(PYG_GRAPH)))
    out = from_owl(graph, schema=OWLSchema(base_iri=BASE,
                                           graph_class='Molecule'),
                   graph_level=True)
    assert [data.num_nodes for data in out] == [3, 1]
