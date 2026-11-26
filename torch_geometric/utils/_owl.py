import os.path as osp
import warnings
from collections import defaultdict
from decimal import Decimal
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.typing import EdgeType, NodeType

SchemaType = Optional[Union['torch_geometric.data.OWLSchema', Dict[str, Any]]]


def _as_matrix(x: Tensor, name: str) -> Tensor:
    x = x.detach().cpu()
    x = x.view(-1, 1) if x.dim() == 1 else x
    if x.dim() != 2:
        raise ValueError(f"Expected '{name}' to be a two-dimensional matrix "
                         f"(got {x.dim()} dimensions)")
    return x


def _check_names(names: Optional[List[str]], x: Tensor, name: str) -> None:
    if names is not None and len(names) != x.size(1):
        raise ValueError(f"Expected {x.size(1)} {name} names (got "
                         f"{len(names)})")


class _OWLWriter:
    r"""Writes nodes, edges and features to an :class:`rdflib.Graph`."""
    def __init__(
        self,
        schema: 'torch_geometric.data.OWLSchema',
        ontology_iri: Optional[str],
        annotate_order: bool,
    ):
        import rdflib
        from rdflib import URIRef
        from rdflib.namespace import OWL, RDF

        from torch_geometric.data.owl_data import PYG_INDEX, PYG_NAMESPACE

        self.schema = schema
        self.annotate_order = annotate_order
        self.graph = rdflib.Graph()
        self.graph.bind('owl', OWL)
        self.graph.bind('', rdflib.Namespace(schema.base_iri))
        self.graph.bind('pyg', rdflib.Namespace(PYG_NAMESPACE))

        onto = URIRef(ontology_iri or schema.base_iri.rstrip('#/'))
        self.graph.add((onto, RDF.type, OWL.Ontology))

        self.index = URIRef(PYG_INDEX)
        if annotate_order:
            self.graph.add((self.index, RDF.type, OWL.AnnotationProperty))

        self.node_entities: Set[Any] = set()
        self.edge_entities: Set[Any] = set()
        self.num_edges = 0

    def annotate(self, entity: Any, index: int) -> None:
        from rdflib import Literal
        if self.annotate_order:
            self.graph.add((entity, self.index, Literal(index)))

    def declare_class(self, iri: str) -> Any:
        from rdflib import URIRef
        from rdflib.namespace import OWL, RDF
        node = URIRef(iri)
        self.graph.add((node, RDF.type, OWL.Class))
        return node

    def add_individuals(self, iris: List[str],
                        classes: List[Any]) -> List[Any]:
        from rdflib import URIRef
        from rdflib.namespace import OWL, RDF

        individuals = [URIRef(iri) for iri in iris]
        if (len(set(individuals)) != len(individuals)
                or len(self.node_entities & set(individuals)) > 0):
            raise ValueError("Found duplicated node IRIs")
        for i, individual in enumerate(individuals):
            self.graph.add((individual, RDF.type, OWL.NamedIndividual))
            for cls in classes:
                self.graph.add((individual, RDF.type, cls))
            self.annotate(individual, i)
        self.node_entities |= set(individuals)
        return individuals

    def add_property(self, iri: str, index: int) -> Any:
        from rdflib import URIRef
        from rdflib.namespace import OWL, RDF
        prop = URIRef(iri)
        self.graph.add((prop, RDF.type, OWL.ObjectProperty))
        self.annotate(prop, index)
        return prop

    def add_features(
        self,
        entities: List[Any],
        x: Tensor,
        iri_of: Callable[[int], str],
        encoding_of: Callable[[int, Tensor], str],
    ) -> None:
        from rdflib import Literal, URIRef
        from rdflib.namespace import OWL, RDF, RDFS, XSD

        threshold = self.schema.feature_threshold
        for j in range(x.size(1)):
            column = x[:, j]
            encoding = encoding_of(j, column)
            if encoding == 'ignore':
                continue

            iri = URIRef(iri_of(j))
            self.annotate(iri, j)
            if encoding == 'class':
                self.declare_class(str(iri))
                index = (column > threshold).nonzero().view(-1)
                for i in index.tolist():
                    self.graph.add((entities[i], RDF.type, iri))
                continue

            self.graph.add((iri, RDF.type, OWL.DatatypeProperty))
            if encoding == 'boolean':
                self.graph.add((iri, RDFS.range, XSD.boolean))
                values = (column > threshold).tolist()
                for i, value in enumerate(values):
                    self.graph.add((entities[i], iri,
                                    Literal(bool(value),
                                            datatype=XSD.boolean)))
            else:  # encoding == 'numeric':
                self.graph.add((iri, RDFS.range, XSD.double))
                for i, value in enumerate(column.tolist()):
                    if value == value:  # Skip NaN values.
                        self.graph.add((entities[i], iri,
                                        Literal(float(value),
                                                datatype=XSD.double)))

    def add_edges(
        self,
        src: List[Any],
        dst: List[Any],
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor],
        properties: List[Any],
        iri_of: Callable[[int], str],
        encoding_of: Callable[[int, Tensor], str],
    ) -> None:
        r"""Adds object property assertions, and reifies edges if they hold
        features or if there are parallel edges.
        """
        from rdflib import URIRef
        from rdflib.namespace import OWL, RDF

        from torch_geometric.data.owl_data import (
            PYG_EDGE,
            PYG_HAS_INCOMING_EDGE,
            PYG_HAS_OUTGOING_EDGE,
            PYG_RELATION,
        )

        edges = torch.stack([edge_index[0], edge_index[1], edge_type], dim=0)
        unique_edges = torch.unique(edges, dim=1)
        for s, d, r in unique_edges.t().tolist():
            self.graph.add((src[s], properties[r], dst[d]))

        reify = self.schema.reify_edges
        if reify is None:
            reify = (edge_attr is not None
                     or unique_edges.size(1) < edges.size(1))
        if not reify:
            return

        edge_class = self.declare_class(PYG_EDGE)
        outgoing = URIRef(PYG_HAS_OUTGOING_EDGE)
        incoming = URIRef(PYG_HAS_INCOMING_EDGE)
        relation = URIRef(PYG_RELATION)
        self.graph.add((outgoing, RDF.type, OWL.ObjectProperty))
        self.graph.add((incoming, RDF.type, OWL.ObjectProperty))
        self.graph.add((relation, RDF.type, OWL.AnnotationProperty))

        entities = []
        for s, d, r in edges.t().tolist():
            edge = URIRef(self.schema.edge_iri(self.num_edges))
            self.graph.add((edge, RDF.type, OWL.NamedIndividual))
            self.graph.add((edge, RDF.type, edge_class))
            self.graph.add((src[s], outgoing, edge))
            self.graph.add((dst[d], incoming, edge))
            self.graph.add((edge, relation, properties[r]))
            self.annotate(edge, self.num_edges)
            self.num_edges += 1
            entities.append(edge)
        self.edge_entities |= set(entities)

        if edge_attr is not None:
            self.add_features(entities, edge_attr, iri_of, encoding_of)

    def add_labels(self, individuals: List[Any], y: Any) -> None:
        from rdflib.namespace import RDF
        if (not isinstance(y, Tensor) or y.dim() != 1
                or y.numel() != len(individuals) or y.is_floating_point()):
            return
        labels = y.detach().cpu().tolist()
        label_classes = {
            label: self.declare_class(self.schema.label_iri(label))
            for label in set(labels) if label >= 0
        }
        for i, label in enumerate(labels):
            if label >= 0:
                self.graph.add(
                    (individuals[i], RDF.type, label_classes[label]))

    def finish(self) -> Any:
        from rdflib.namespace import RDFS

        if len(self.edge_entities & self.node_entities) > 0:
            raise ValueError("Found edge IRIs that collide with node IRIs "
                             "(rename the nodes via 'node_names')")

        hierarchy = self.schema.class_hierarchy or {}
        for child, parents in hierarchy.items():
            child_node = self.declare_class(self.schema.iri(child))
            for parent in [parents] if isinstance(parents, str) else parents:
                parent_node = self.declare_class(self.schema.iri(parent))
                self.graph.add((child_node, RDFS.subClassOf, parent_node))
        return self.graph


def to_owl(
    data: Union['torch_geometric.data.Data',
                'torch_geometric.data.HeteroData'],
    schema: SchemaType = None,
    ontology_iri: Optional[str] = None,
    annotate_order: bool = True,
) -> Any:
    r"""Converts a :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData` instance to an OWL ontology,
    given as an :class:`rdflib.Graph`.

    * Every node becomes an :obj:`owl:NamedIndividual` (named via
      :obj:`data.node_iri` or the schema).
    * Every edge type becomes an :obj:`owl:ObjectProperty`, and every edge
      an object property assertion.
    * Every column of :obj:`data.x` becomes an OWL class or a data property,
      depending on :obj:`schema.feature_encoding`.
    * If the graph holds edge features :obj:`data.edge_attr` or parallel
      edges (or if :obj:`schema.reify_edges=True`), every edge is
      additionally reified as an individual of class :obj:`pyg:Edge` that
      holds the edge features (encoded via
      :obj:`schema.edge_feature_encoding`). Otherwise, duplicate edges are
      removed, since OWL assertions form a set.
    * :obj:`schema.class_hierarchy` is added via :obj:`rdfs:subClassOf`.
    * Node labels :obj:`data.y` are **not** added, unless
      :obj:`schema.include_labels=True`. Otherwise, a concept learner would
      trivially rediscover the label.

    For heterogeneous graphs, every node type becomes an OWL class (a sub
    class of :obj:`schema.node_class`) whose individuals are the nodes of
    that type, and every relation :obj:`rel` of an edge type
    :obj:`(src, rel, dst)` becomes an object property. If
    :obj:`schema.inverse_edge_types=True`, edge types
    :obj:`(dst, "rev_" + rel, src)` (as added by
    :class:`~torch_geometric.transforms.ToUndirected`) are implied by their
    forward edge type and not written.

    The result can be written to disk via :meth:`rdflib.Graph.serialize`.

    .. code-block:: python

        from torch_geometric.data import OWLSchema
        from torch_geometric.utils import to_owl

        schema = OWLSchema(relation_names=['knows'],
                           feature_names=['Student', 'age'],
                           feature_encoding=['class', 'numeric'])
        graph = to_owl(data, schema)
        graph.serialize('graph.owl', format='xml')

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The graph data object.
        schema (OWLSchema or Dict[str, Any], optional): The schema that maps
            the graph to OWL. If set to :obj:`None`, the schema of an
            :class:`~torch_geometric.data.OWLData` or
            :class:`~torch_geometric.data.OWLHeteroData` object is used, or a
            default schema otherwise. (default: :obj:`None`)
        ontology_iri (str, optional): The IRI of the ontology. If set to
            :obj:`None`, it is derived from :obj:`schema.base_iri`.
            (default: :obj:`None`)
        annotate_order (bool, optional): If set to :obj:`True`, annotates
            individuals, relations and features with their index via the
            annotation property :obj:`pyg:index`, so that
            :meth:`~torch_geometric.utils.from_owl` restores the original
            order. Annotations have no logical meaning and are ignored by
            reasoners and concept learners. (default: :obj:`True`)
    """
    from torch_geometric.data import HeteroData, OWLData, OWLHeteroData
    from torch_geometric.data.owl_data import to_owl_schema

    if schema is None and isinstance(data, (OWLData, OWLHeteroData)):
        schema = data.schema
    owl_schema = to_owl_schema(schema)
    writer = _OWLWriter(owl_schema, ontology_iri, annotate_order)

    if isinstance(data, HeteroData):
        _hetero_to_owl(data, writer)
        return writer.finish()

    # Individuals #############################################################
    num_nodes = data.num_nodes or 0
    if 'node_iri' in data:
        node_iri = list(data.node_iri)
        if len(node_iri) != num_nodes:
            raise ValueError(f"Expected 'node_iri' to hold {num_nodes} IRIs "
                             f"(got {len(node_iri)})")
    else:
        node_iri = [owl_schema.node_iri(i) for i in range(num_nodes)]

    classes = []
    if owl_schema.node_class is not None:
        classes.append(
            writer.declare_class(owl_schema.iri(owl_schema.node_class)))
    individuals = writer.add_individuals(node_iri, classes)

    # Object properties #######################################################
    edge_index = data.edge_index
    edge_attr = data.edge_attr if 'edge_attr' in data else None
    if edge_index is not None and edge_index.numel() > 0:
        edge_index = edge_index.detach().cpu()
        if 'edge_type' in data and data.edge_type is not None:
            edge_type = data.edge_type.detach().cpu().view(-1)
        else:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
        if isinstance(edge_attr, Tensor):
            edge_attr = _as_matrix(edge_attr, 'edge_attr')
            if edge_attr.size(0) != edge_type.numel():
                raise ValueError(f"Expected 'edge_attr' to hold one row per "
                                 f"edge (got shape {tuple(edge_attr.size())})")
            names = owl_schema.edge_feature_names
            _check_names(names if isinstance(names, list) else None, edge_attr,
                         'edge feature')
        else:
            edge_attr = None

        if owl_schema.inverse_edge_types:
            # Inverse edges are implied by their forward assertion:
            mask = edge_type % 2 == 0
            edge_index, edge_type = edge_index[:, mask], edge_type[mask] // 2
            edge_attr = None if edge_attr is None else edge_attr[mask]

        num_relations = int(edge_type.max()) + 1 if edge_type.numel() else 0
        relation_names = owl_schema.relation_names
        if (isinstance(relation_names, list)
                and num_relations > len(relation_names)):
            raise ValueError(f"Found {num_relations} relations but only "
                             f"{len(relation_names)} relation names")
        if isinstance(relation_names, list):
            num_relations = max(num_relations, len(relation_names))

        properties = [
            writer.add_property(owl_schema.relation_iri(r, num_relations), r)
            for r in range(num_relations)
        ]
        writer.add_edges(individuals, individuals, edge_index, edge_type,
                         edge_attr, properties, owl_schema.edge_feature_iri,
                         owl_schema.edge_encoding)

    # Features ################################################################
    x = data.x if 'x' in data else None
    if isinstance(x, Tensor):
        x = _as_matrix(x, 'x')
        names = owl_schema.feature_names
        _check_names(names if isinstance(names, list) else None, x, 'feature')
        writer.add_features(individuals, x, owl_schema.feature_iri,
                            owl_schema.encoding)

    # Labels ##################################################################
    if owl_schema.include_labels and 'y' in data:
        writer.add_labels(individuals, data.y)

    return writer.finish()


def _hetero_to_owl(
    data: 'torch_geometric.data.HeteroData',
    writer: _OWLWriter,
) -> None:
    from rdflib.namespace import RDFS

    from torch_geometric.data.owl_data import _select

    schema = writer.schema
    node_class = None
    if schema.node_class is not None:
        node_class = writer.declare_class(schema.iri(schema.node_class))

    # Individuals and features ################################################
    individuals: Dict[NodeType, List[Any]] = {}
    for node_type in data.node_types:
        store = data[node_type]
        num_nodes = store.num_nodes or 0
        if 'node_iri' in store:
            node_iri = list(store.node_iri)
            if len(node_iri) != num_nodes:
                raise ValueError(f"Expected 'node_iri' of node type "
                                 f"'{node_type}' to hold {num_nodes} IRIs "
                                 f"(got {len(node_iri)})")
        else:
            node_iri = [
                schema.node_iri(i, node_type) for i in range(num_nodes)
            ]

        cls = writer.declare_class(schema.node_type_iri(node_type))
        if node_class is not None and cls != node_class:
            writer.graph.add((cls, RDFS.subClassOf, node_class))
        individuals[node_type] = writer.add_individuals(node_iri, [cls])

        x = store.get('x')
        if isinstance(x, Tensor):
            x = _as_matrix(x, 'x')
            _check_names(_select(schema.feature_names, node_type), x,
                         f"feature (of node type '{node_type}')")
            writer.add_features(
                individuals[node_type], x,
                partial(schema.feature_iri, node_type=node_type),
                partial(schema.encoding, node_type=node_type))

        if schema.include_labels and 'y' in store:
            writer.add_labels(individuals[node_type], store.y)

    # Object properties #######################################################
    edge_types: List[EdgeType] = []
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        if (schema.inverse_edge_types and rel.startswith('rev_')
                and (dst, rel[4:], src) in data.edge_types):
            continue  # Implied by the forward edge type.
        if data[edge_type].get('edge_index') is not None:
            edge_types.append(edge_type)

    relations = list(dict.fromkeys(rel for _, rel, _ in edge_types))
    properties = {
        rel: writer.add_property(schema.relation_iri(rel), r)
        for r, rel in enumerate(relations)
    }

    for edge_type in edge_types:
        src, rel, dst = edge_type
        store = data[edge_type]
        edge_index = store.edge_index.detach().cpu()
        edge_attr = store.get('edge_attr')
        if isinstance(edge_attr, Tensor):
            edge_attr = _as_matrix(edge_attr, 'edge_attr')
            if edge_attr.size(0) != edge_index.size(1):
                raise ValueError(f"Expected 'edge_attr' of edge type "
                                 f"{edge_type} to hold one row per edge "
                                 f"(got shape {tuple(edge_attr.size())})")
            _check_names(_select(schema.edge_feature_names, rel), edge_attr,
                         f"edge feature (of relation '{rel}')")
        else:
            edge_attr = None

        writer.add_edges(individuals[src], individuals[dst], edge_index,
                         torch.zeros(edge_index.size(1), dtype=torch.long),
                         edge_attr, [properties[rel]],
                         partial(schema.edge_feature_iri, relation=rel),
                         partial(schema.edge_encoding, relation=rel))


class _OWLReader:
    r"""Reads the individuals, relations and features of an ontology."""
    def __init__(
        self,
        source: Any,
        schema: SchemaType,
        format: Optional[str],
        features: Sequence[str],
    ):
        import rdflib
        from rdflib import URIRef
        from rdflib.namespace import OWL, RDF, RDFS
        from rdflib.util import guess_format

        from torch_geometric.data import OWLSchema
        from torch_geometric.data.owl_data import (
            PYG_EDGE,
            PYG_HAS_INCOMING_EDGE,
            PYG_HAS_OUTGOING_EDGE,
            PYG_INDEX,
            PYG_RELATION,
            is_absolute_iri,
            to_owl_schema,
        )

        for feature in features:
            if feature not in {'types', 'data_properties'}:
                raise ValueError(f"Unknown feature type '{feature}' (expected "
                                 f"'types' or 'data_properties')")
        self.features = features

        self.path: Optional[str] = None
        if isinstance(source, rdflib.Graph):
            graph = source
        else:
            path = str(source)
            if format is None:
                format = guess_format(path)
                if format is None and path.endswith('.owl'):
                    format = 'xml'
            graph = rdflib.Graph()
            graph.parse(path, format=format)
            self.path = osp.abspath(path) if osp.exists(path) else path
        self.graph = graph

        if schema is None:  # Derive the namespace from the ontology header:
            ontologies = sorted(
                str(onto) for onto in graph.subjects(RDF.type, OWL.Ontology)
                if isinstance(onto, URIRef) and is_absolute_iri(str(onto)))
            if len(ontologies) > 0:
                schema = OWLSchema(base_iri=ontologies[0])
        self.schema = to_owl_schema(schema)

        self.pyg_index = URIRef(PYG_INDEX)
        self.pyg_relation = URIRef(PYG_RELATION)
        self.pyg_edge = URIRef(PYG_EDGE)
        self.outgoing = URIRef(PYG_HAS_OUTGOING_EDGE)
        self.incoming = URIRef(PYG_HAS_INCOMING_EDGE)

        # Collect declared entities ###########################################
        uris = self.uris
        classes = uris(graph.subjects(RDF.type, OWL.Class))
        classes |= uris(graph.subjects(RDF.type, RDFS.Class))
        classes |= uris(graph.objects(None, RDFS.subClassOf))
        self.classes = {cls for cls in classes if not self.is_builtin(cls)}
        self.object_properties = uris(
            graph.subjects(RDF.type, OWL.ObjectProperty))
        self.data_properties = uris(
            graph.subjects(RDF.type, OWL.DatatypeProperty))
        schema_entities = (self.classes | self.object_properties
                           | self.data_properties)
        schema_entities |= uris(
            graph.subjects(RDF.type, OWL.AnnotationProperty))
        schema_entities |= uris(graph.subjects(RDF.type, OWL.Ontology))
        self.schema_entities = schema_entities

        edge_individuals = uris(graph.subjects(RDF.type, self.pyg_edge))
        edge_individuals |= uris(graph.objects(None, self.outgoing))
        edge_individuals |= uris(graph.objects(None, self.incoming))
        self.edge_individuals = edge_individuals

        individuals = uris(graph.subjects(RDF.type, OWL.NamedIndividual))
        for subj, obj in graph.subject_objects(RDF.type):
            if isinstance(subj, URIRef) and obj in self.classes:
                individuals.add(subj)
        for prop in self.object_properties:
            for subj, obj in graph.subject_objects(prop):
                individuals |= uris([subj, obj])
        individuals -= schema_entities | edge_individuals
        self.individuals = {
            node
            for node in individuals if not self.is_builtin(node)
        }

        # Class hierarchy (asserted) ##########################################
        self.parents: Dict[Any, Set[Any]] = defaultdict(set)
        for child, parent in graph.subject_objects(RDFS.subClassOf):
            if isinstance(child, URIRef) and isinstance(parent, URIRef):
                self.parents[child].add(parent)

    @staticmethod
    def uris(nodes: Any) -> Set[Any]:
        from rdflib import URIRef
        return {node for node in nodes if isinstance(node, URIRef)}

    @staticmethod
    def is_builtin(iri: Any) -> bool:  # RDF/RDFS/OWL/XSD/PyG vocabulary.
        from rdflib.namespace import OWL, RDF, RDFS, XSD

        from torch_geometric.data.owl_data import PYG_NAMESPACE
        return str(iri).startswith(
            (str(RDF), str(RDFS), str(OWL), str(XSD), PYG_NAMESPACE))

    def ordered(self, entities: Any) -> List[Any]:
        r"""Orders entities by their :obj:`pyg:index` annotation first, and
        by the natural order of their IRIs afterwards.
        """
        from rdflib import Literal

        from torch_geometric.data.owl_data import natural_key

        entities = set(entities)
        index: Dict[Any, int] = {}
        for entity in entities:
            values = [
                v.toPython()
                for v in self.graph.objects(entity, self.pyg_index)
                if isinstance(v, Literal)
            ]
            if len(values) == 1 and isinstance(values[0], int):
                index[entity] = values[0]
        if len(set(index.values())) != len(index):
            index = {}  # Ambiguous annotations (*e.g.*, merged ontologies).
        head = sorted(index.keys(), key=lambda entity: index[entity])
        tail = sorted(entities - set(index.keys()),
                      key=lambda entity: natural_key(str(entity)))
        return head + tail

    def closure(self, types: Set[Any]) -> Set[Any]:
        r"""Returns :obj:`types` and all their (asserted) super classes."""
        out, stack = set(types), list(types)
        while len(stack) > 0:
            for parent in self.parents.get(stack.pop(), ()):
                if parent not in out and parent in self.classes:
                    out.add(parent)
                    stack.append(parent)
        return out

    def resolve(self, names: Optional[Sequence[str]]) -> List[Any]:
        r"""Resolves class names (IRIs or local names) to classes."""
        from rdflib import URIRef

        from torch_geometric.data.owl_data import local_name

        out = []
        for name in names or []:
            iri = URIRef(self.schema.iri(name))
            if iri not in self.classes:
                matches = [c for c in self.classes if local_name(c) == name]
                if len(matches) != 1:
                    raise ValueError(f"Could not find the OWL class '{name}'")
                iri = matches[0]
            out.append(iri)
        return out

    def asserted_types(self, entities: List[Any]) -> List[Set[Any]]:
        from rdflib.namespace import RDF

        index = {entity: i for i, entity in enumerate(entities)}
        out: List[Set[Any]] = [set() for _ in entities]
        for subj, obj in self.graph.subject_objects(RDF.type):
            if subj in index and obj in self.classes:
                out[index[subj]].add(obj)
        return out

    def get_features(
        self,
        entities: List[Any],
        types: List[Set[Any]],
        excluded: Set[Any],
    ) -> Tuple[Optional[Tensor], List[str], List[str]]:
        r"""Returns the feature matrix of :obj:`entities`, its column names
        and its encodings.
        """
        from rdflib import Literal

        index = {entity: i for i, entity in enumerate(entities)}
        columns: Dict[Any, Tensor] = {}
        encodings: Dict[Any, str] = {}

        if 'types' in self.features:
            feature_types = [
                self.closure(t - excluded) - excluded for t in types
            ]
            for cls in set().union(*feature_types):
                columns[cls] = torch.zeros(len(entities))
                encodings[cls] = 'class'
            for i, entity_types in enumerate(feature_types):
                for cls in entity_types:
                    columns[cls][i] = 1.0

        if 'data_properties' in self.features:
            values: Dict[Any, Dict[int, Any]] = defaultdict(dict)
            for subj, pred, obj in self.graph:
                if subj in index and isinstance(obj, Literal):
                    if pred in self.data_properties or not self.is_builtin(
                            pred):
                        value = obj.toPython()
                        if isinstance(value, (bool, int, float, Decimal)):
                            values[pred].setdefault(index[subj], value)

            for prop, prop_values in values.items():
                if (prop in self.schema_entities
                        and prop not in self.data_properties):
                    continue  # Skip annotation properties.
                column = torch.zeros(len(entities))
                for i, value in prop_values.items():
                    column[i] = float(value)
                columns[prop] = column
                is_bool = all(
                    isinstance(v, bool) for v in prop_values.values())
                encodings[prop] = 'boolean' if is_bool else 'numeric'

        order = self.ordered(columns.keys())
        if len(order) == 0:
            return None, [], []
        x = torch.stack([columns[key] for key in order], dim=1)
        return x, [str(key) for key in order], [encodings[k] for k in order]

    def get_edge_features(
        self,
        rows: List[Tuple[int, Any, int, Any]],
        excluded: Set[Any],
    ) -> Tuple[Optional[Tensor], List[str], List[str]]:
        r"""Returns the features of the (reified) edges in :obj:`rows`.
        Edges without an individual hold no features.
        """
        edges = [row[3] for row in rows if row[3] is not None]
        if len(edges) == 0:
            return None, [], []
        types = [t - {self.pyg_edge} for t in self.asserted_types(edges)]
        attr, names, encodings = self.get_features(edges, types, excluded)
        if attr is None:
            return None, [], []
        out = torch.zeros(len(rows), attr.size(1))
        out[torch.tensor([row[3] is not None for row in rows])] = attr
        return out, names, encodings

    def get_rows(
        self,
        node_index: Dict[Any, int],
    ) -> List[Tuple[int, Any, int, Any]]:
        r"""Returns all edges between nodes in :obj:`node_index` as rows of
        (source, relation, target, edge individual or :obj:`None`).
        """
        rows: List[Tuple[int, Any, int, Any]] = []
        reified = set()
        for edge in self.ordered(self.edge_individuals):
            srcs = [
                n for n in self.graph.subjects(self.outgoing, edge)
                if n in node_index
            ]
            dsts = [
                n for n in self.graph.subjects(self.incoming, edge)
                if n in node_index
            ]
            if len(srcs) == 0 or len(dsts) == 0:
                continue  # Edge from or to a node that is not converted.
            if len(srcs) != 1 or len(dsts) != 1:
                raise ValueError(f"Expected the reified edge '{edge}' to "
                                 f"have exactly one source and one target "
                                 f"node (got {len(srcs)} sources and "
                                 f"{len(dsts)} targets)")
            rels = self.uris(self.graph.objects(edge, self.pyg_relation))
            if len(rels) == 0:  # Fall back to the asserted object property:
                rels = {
                    p
                    for p in self.graph.predicates(srcs[0], dsts[0])
                    if not self.is_builtin(p)
                }
            if len(rels) != 1:
                raise ValueError(f"Could not determine the relation of the "
                                 f"reified edge '{edge}'")
            rel = rels.pop()
            rows.append((node_index[srcs[0]], rel, node_index[dsts[0]], edge))
            reified.add((srcs[0], rel, dsts[0]))

        for subj, pred, obj in self.graph:
            if (subj in node_index and obj in node_index
                    and not self.is_builtin(pred)
                    and (subj, pred, obj) not in reified):
                rows.append((node_index[subj], pred, node_index[obj], None))
        return rows

    def relations(self, rows: List[Tuple[int, Any, int, Any]]) -> List[Any]:
        r"""Returns the relations in :obj:`rows` plus all relations that are
        annotated with an index (which may have no assertions), in order.
        """
        used = {rel for _, rel, _, _ in rows}
        declared = {
            prop
            for prop in self.object_properties
            if (prop, self.pyg_index,
                None) in self.graph and not self.is_builtin(prop)
        }
        return self.ordered(used | declared)

    def get_labels(
        self,
        types: List[Set[Any]],
        targets: List[Any],
    ) -> Optional[Tensor]:
        if len(targets) == 0:
            return None
        types = [self.closure(t) for t in types]
        if len(targets) == 1:
            return torch.tensor([int(targets[0] in t) for t in types],
                                dtype=torch.long)

        y = torch.full((len(types), ), -1, dtype=torch.long)
        num_ambiguous = 0
        for i, t in enumerate(types):
            labels = [c for c, target in enumerate(targets) if target in t]
            if len(labels) > 0:
                y[i] = labels[0]
            num_ambiguous += int(len(labels) > 1)
        if num_ambiguous > 0:
            warnings.warn(
                f"Found {num_ambiguous} individual(s) that belong to more "
                f"than one target class. They are labeled with the first "
                f"matching class.", stacklevel=4)
        return y


def _sort_edges(edge: Tensor, num_nodes: int, num_relations: int) -> Tensor:
    r"""Returns the permutation that sorts edges, given as rows of (source,
    target, edge type), by source, target and edge type.
    """
    key = (edge[0] * num_nodes + edge[1]) * num_relations + edge[2]
    return key.argsort(stable=True)


def from_owl(
    source: Any,
    schema: SchemaType = None,
    target_classes: Optional[Sequence[str]] = None,
    ignore_classes: Optional[Sequence[str]] = None,
    features: Sequence[str] = ('types', 'data_properties'),
    add_inverse_edges: bool = True,
    node_types: Optional[Sequence[str]] = None,
    format: Optional[str] = None,
) -> Union['torch_geometric.data.OWLData',
           'torch_geometric.data.OWLHeteroData']:
    r"""Converts an OWL ontology to a
    :class:`torch_geometric.data.OWLData` instance (or a
    :class:`torch_geometric.data.OWLHeteroData` instance if
    :obj:`node_types` is given).

    * Every named individual becomes a node. The IRIs are stored in
      :obj:`data.node_iri`. Nodes are ordered by their :obj:`pyg:index`
      annotation (as written by :meth:`~torch_geometric.utils.to_owl`), and
      otherwise in natural order of their IRIs (*e.g.*, :obj:`node_2` before
      :obj:`node_10`), so the conversion is deterministic.
    * Every object property becomes an edge type in :obj:`data.edge_type`.
      If :obj:`add_inverse_edges=True`, edge type :obj:`2 * r + 1` holds the
      inverse edges of relation :obj:`r`, following the convention of
      :class:`~torch_geometric.datasets.Entities`. Edges are sorted by
      source node, target node and edge type.
    * Node features :obj:`data.x` contain a multi-hot encoding of the
      asserted classes of each individual (closed under asserted
      :obj:`rdfs:subClassOf` axioms) and one column per numeric or boolean
      data property.
    * Reified edges (individuals of class :obj:`pyg:Edge`, see
      :meth:`~torch_geometric.utils.to_owl`) are turned back into edges.
      Their classes and data properties become edge features
      :obj:`data.edge_attr`, and parallel edges are preserved.
    * Node labels :obj:`data.y` are derived from :obj:`target_classes`.
      Target classes are never used as node features.

    If :obj:`node_types` is given, the result is heterogeneous: every
    individual of one of the :obj:`node_types` classes becomes a node of
    that type (named after the local name of the class), and every object
    property assertion between them an edge of type :obj:`(src, rel, dst)`.
    Inverse edges are added as :obj:`(dst, "rev_" + rel, src)`, following
    :class:`~torch_geometric.transforms.ToUndirected`. Individuals of none of
    the classes are dropped.

    No reasoning beyond the :obj:`rdfs:subClassOf` closure is performed.
    The discovered IRIs are stored in :obj:`data.schema`, so the conversion
    can be reversed via :meth:`torch_geometric.utils.to_owl`.

    .. code-block:: python

        from torch_geometric.utils import from_owl

        data = from_owl('family.owl', target_classes=['Female'])
        # OWLData(x=[202, 17], edge_index=[2, 1296], edge_type=[1296],
        #         y=[202], node_iri=[202], num_nodes=202)

        data = from_owl('aifb.owl', node_types=['Person', 'Publication'])
        # OWLHeteroData(
        #   Person={ x=[...], node_iri=[...], num_nodes=... },
        #   Publication={ ... },
        #   (Publication, author, Person)={ edge_index=[2, ...] },
        #   (Person, rev_author, Publication)={ edge_index=[2, ...] },
        #   ...
        # )

    Args:
        source (str or rdflib.Graph): The path or URL of an OWL ontology, or
            an :class:`rdflib.Graph`.
        schema (OWLSchema or Dict[str, Any], optional): A schema whose
            :obj:`base_iri` is used to resolve local names, *e.g.*, in
            :obj:`target_classes`. (default: :obj:`None`)
        target_classes (List[str], optional): The OWL classes that define
            node labels. A single class results in binary labels. Multiple
            classes result in multi-class labels, where nodes that belong to
            none of the classes are labeled :obj:`-1`.
            (default: :obj:`None`)
        ignore_classes (List[str], optional): OWL classes that should not be
            used as node features (and not by concept learners).
            (default: :obj:`None`)
        features (List[str], optional): The node and edge features to
            compute. :obj:`"types"` adds the asserted classes,
            :obj:`"data_properties"` adds numeric and boolean data property
            values. If empty, :obj:`data.x` is set to :obj:`None`.
            (default: :obj:`("types", "data_properties")`)
        add_inverse_edges (bool, optional): If set to :obj:`True`, adds
            inverse edges with their own edge types. (default: :obj:`True`)
        node_types (List[str], optional): The OWL classes that define the
            node types of a heterogeneous graph. Individuals that belong to
            more than one of them get the first matching type.
            (default: :obj:`None`)
        format (str, optional): The :obj:`rdflib` parser format. If set to
            :obj:`None`, it is guessed from the file extension, with
            :obj:`".owl"` files parsed as RDF/XML. (default: :obj:`None`)
    """
    from torch_geometric.data import OWLData, OWLSchema

    reader = _OWLReader(source, schema, format, features)

    # Target classes and their sub classes entail the label, and are thus
    # never used as node features:
    targets = reader.resolve(target_classes)
    excluded = set(reader.resolve(ignore_classes))
    excluded |= {
        cls
        for cls in reader.classes if reader.closure({cls}) & set(targets)
    }

    if node_types is not None:
        return _hetero_from_owl(reader, node_types, targets, excluded,
                                add_inverse_edges)

    nodes = reader.ordered(reader.individuals)
    node_index = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # Edges ###################################################################
    rows = reader.get_rows(node_index)
    relations = reader.relations(rows)
    relation_index = {rel: r for r, rel in enumerate(relations)}
    num_relations = len(relations) * (2 if add_inverse_edges else 1)

    edge_list = []
    for src, rel, dst, _ in rows:
        r = relation_index[rel]
        if add_inverse_edges:
            edge_list.append([src, dst, 2 * r])
            edge_list.append([dst, src, 2 * r + 1])
        else:
            edge_list.append([src, dst, r])

    edge_attr, edge_feature_names, edge_feature_encoding = \
        reader.get_edge_features(rows, excluded)
    if len(edge_list) > 0:
        edge = torch.tensor(edge_list, dtype=torch.long).t()
        perm = _sort_edges(edge, num_nodes, num_relations)
        edge = edge[:, perm]
        edge_index, edge_type = edge[:2].contiguous(), edge[2].contiguous()
        if edge_attr is not None:
            if add_inverse_edges:
                edge_attr = edge_attr.repeat_interleave(2, dim=0)
            edge_attr = edge_attr[perm]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)

    # Node features and labels ################################################
    types = reader.asserted_types(nodes)
    x, feature_names, feature_encoding = reader.get_features(
        nodes, types, excluded)
    y = reader.get_labels(types, targets)

    out_schema = OWLSchema(
        base_iri=reader.schema.base_iri,
        node_class=None,
        relation_names=[str(rel) for rel in relations],
        inverse_edge_types=add_inverse_edges,
        feature_names=feature_names,
        feature_encoding=feature_encoding or 'auto',
        label_names=[str(t) for t in targets] if len(targets) > 1 else None,
        target_classes=[str(t) for t in targets] or None,
        ignore_classes=sorted(str(c) for c in excluded - set(targets)) or None,
        edge_feature_names=edge_feature_names or None,
        edge_feature_encoding=edge_feature_encoding or 'auto',
        reify_edges=True if any(row[3] is not None for row in rows) else None,
    )

    data = OWLData(x=x, edge_index=edge_index, edge_type=edge_type,
                   edge_attr=edge_attr, y=y, schema=out_schema,
                   source=reader.path)
    data.node_iri = [str(node) for node in nodes]
    data.num_nodes = num_nodes
    return data


def _unique_names(entities: List[Any]) -> Dict[Any, str]:
    r"""Names entities by their local name, or by their IRI if local names
    are ambiguous.
    """
    from torch_geometric.data.owl_data import local_name

    names = [local_name(str(entity)) for entity in entities]
    return {
        entity: name if names.count(name) == 1 else str(entity)
        for entity, name in zip(entities, names)
    }


def _hetero_from_owl(
    reader: _OWLReader,
    node_types: Sequence[str],
    targets: List[Any],
    excluded: Set[Any],
    add_inverse_edges: bool,
) -> 'torch_geometric.data.OWLHeteroData':
    from torch_geometric.data import OWLHeteroData, OWLSchema

    type_classes = reader.resolve(node_types)
    if len(set(type_classes)) != len(type_classes):
        raise ValueError("Found duplicated node types")
    type_names = _unique_names(type_classes)

    # Assign every individual to its first matching node type #################
    candidates = reader.ordered(reader.individuals)
    candidate_types = [
        reader.closure(t) for t in reader.asserted_types(candidates)
    ]
    members: Dict[Any, List[Any]] = {cls: [] for cls in type_classes}
    num_dropped = num_ambiguous = 0
    for node, node_classes in zip(candidates, candidate_types):
        matches = [cls for cls in type_classes if cls in node_classes]
        if len(matches) == 0:
            num_dropped += 1
            continue
        num_ambiguous += int(len(matches) > 1)
        members[matches[0]].append(node)
    if num_dropped > 0:
        warnings.warn(
            f"Found {num_dropped} individual(s) that belong to none of the "
            f"node types. They are dropped.", stacklevel=3)
    if num_ambiguous > 0:
        warnings.warn(
            f"Found {num_ambiguous} individual(s) that belong to more than "
            f"one node type. They are assigned to the first matching type.",
            stacklevel=3)

    # Order nodes within their type (`pyg:index` is unique per type):
    nodes = {cls: reader.ordered(members[cls]) for cls in type_classes}
    node_index: Dict[Any, int] = {}
    local_index: List[int] = []
    node_type_of: List[Any] = []
    for cls in type_classes:
        for i, node in enumerate(nodes[cls]):
            node_index[node] = len(local_index)
            local_index.append(i)
            node_type_of.append(cls)

    data = OWLHeteroData()
    feature_names: Dict[str, List[str]] = {}
    feature_encoding: Dict[str, str] = {}
    # Node type classes are constant within each node type:
    type_excluded = excluded | set(type_classes)
    for cls in type_classes:
        name = type_names[cls]
        store = data[name]
        store.node_iri = [str(node) for node in nodes[cls]]
        store.num_nodes = len(nodes[cls])
        types = reader.asserted_types(nodes[cls])
        x, names, encodings = reader.get_features(nodes[cls], types,
                                                  type_excluded)
        if x is not None:
            store.x = x
            feature_names[name] = names
            feature_encoding.update(zip(names, encodings))
        y = reader.get_labels(types, targets)
        if y is not None:
            store.y = y

    # Edges ###################################################################
    rows = reader.get_rows(node_index)
    relations = reader.relations(rows)
    relation_names = _unique_names(relations)
    relation_index = {rel: r for r, rel in enumerate(relations)}
    type_index = {cls: i for i, cls in enumerate(type_classes)}

    groups: Dict[Tuple[Any, Any, Any], List[int]] = defaultdict(list)
    for i, (src, rel, dst, _) in enumerate(rows):
        groups[(node_type_of[src], rel, node_type_of[dst])].append(i)

    def group_key(key: Tuple[Any, Any, Any]) -> Tuple[int, int, int]:
        return (relation_index[key[1]], type_index[key[0]], type_index[key[2]])

    edge_feature_names: Dict[str, List[str]] = {}
    edge_feature_encoding: Dict[str, str] = {}
    for key in sorted(groups.keys(), key=group_key):
        src_cls, rel, dst_cls = key
        group = [rows[i] for i in groups[key]]
        edge_index = torch.tensor(
            [[local_index[src] for src, _, _, _ in group],
             [local_index[dst] for _, _, dst, _ in group]], dtype=torch.long)
        rel_name = relation_names[rel]
        edge_attr, names, encodings = reader.get_edge_features(group, excluded)
        if edge_attr is not None:
            edge_feature_names[rel_name] = names
            edge_feature_encoding.update(zip(names, encodings))

        src_name, dst_name = type_names[src_cls], type_names[dst_cls]
        edge_types = [((src_name, rel_name, dst_name), edge_index)]
        if add_inverse_edges:
            edge_types.append(((dst_name, f'rev_{rel_name}', src_name),
                               edge_index.flip([0])))
        for edge_type, index in edge_types:
            num_dst = max(int(data[edge_type[2]].num_nodes or 0), 1)
            perm = _sort_edges(
                torch.cat([index, torch.zeros_like(index[:1])], dim=0),
                num_dst, 1)
            data[edge_type].edge_index = index[:, perm].contiguous()
            if edge_attr is not None:
                data[edge_type].edge_attr = edge_attr[perm]

    data.schema = OWLSchema(
        base_iri=reader.schema.base_iri,
        node_class=None,
        node_type_names={type_names[c]: str(c)
                         for c in type_classes},
        relation_names={relation_names[r]: str(r)
                        for r in relations},
        inverse_edge_types=add_inverse_edges,
        feature_names=feature_names or None,
        feature_encoding=feature_encoding or 'auto',
        label_names=[str(t) for t in targets] if len(targets) > 1 else None,
        target_classes=[str(t) for t in targets] or None,
        ignore_classes=sorted(str(c) for c in excluded - set(targets)) or None,
        edge_feature_names=edge_feature_names or None,
        edge_feature_encoding=edge_feature_encoding or 'auto',
        reify_edges=True if any(row[3] is not None for row in rows) else None,
    )
    data.source = reader.path
    return data
