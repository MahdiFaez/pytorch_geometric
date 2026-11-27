import os.path as osp
import re
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from urllib.parse import quote

import torch
from torch import Tensor

from torch_geometric.data.data import Data
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.typing import OptTensor
from torch_geometric.utils.mixin import CastMixin

FEATURE_ENCODINGS = {'auto', 'class', 'numeric', 'boolean', 'ignore'}

# The namespace of the structural vocabulary that PyG adds to an ontology:
# Annotations that preserve the order of nodes, relations and features, and
# the classes/properties of reified edges (see :meth:`utils.to_owl`).
PYG_NAMESPACE = 'https://pyg.org/owl#'
PYG_INDEX = PYG_NAMESPACE + 'index'
PYG_RELATION = PYG_NAMESPACE + 'relation'
PYG_EDGE = PYG_NAMESPACE + 'Edge'
PYG_HAS_OUTGOING_EDGE = PYG_NAMESPACE + 'hasOutgoingEdge'
PYG_HAS_INCOMING_EDGE = PYG_NAMESPACE + 'hasIncomingEdge'
PYG_GRAPH = PYG_NAMESPACE + 'Graph'
PYG_HAS_NODE = PYG_NAMESPACE + 'hasNode'

_ABSOLUTE_IRI = re.compile(r'^[A-Za-z][A-Za-z0-9+.-]*:[^\s]+$')


def is_absolute_iri(name: str) -> bool:
    r"""Returns :obj:`True` if :obj:`name` is an absolute IRI, *e.g.*,
    :obj:`"http://example.org/onto#Person"`.
    """
    return bool(_ABSOLUTE_IRI.match(name)) and ('/' in name or '#' in name
                                                or name.startswith('urn:'))


def local_name(iri: str) -> str:
    r"""Returns the local name of an IRI (the part after the last :obj:`#`
    or :obj:`/`).
    """
    return re.split(r'[#/]', iri.rstrip('#/'))[-1]


def natural_key(name: str) -> List[Any]:
    r"""Returns a sort key that orders numbers in :obj:`name` numerically,
    *e.g.*, :obj:`"node_2"` before :obj:`"node_10"`.
    """
    return [(0, int(part)) if part.isdigit() else (1, part)
            for part in re.split(r'(\d+)', name)]


Names = Union[List[str], Dict[str, List[str]]]


def _to_names(value: Any) -> Optional[Names]:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: list(names) for key, names in value.items()}
    return list(value)


def _select(names: Optional[Names], key: Optional[str]) -> Optional[List[str]]:
    r"""Returns the names of node/edge type :obj:`key` in case :obj:`names`
    is given per type.
    """
    if isinstance(names, dict):
        return names.get(key) if key is not None else None
    return names


def _resolve_encoding(
    spec: Union[str, List[str], Dict[str, str]],
    names: Optional[List[str]],
    index: int,
    column: Optional[Tensor],
    per_type: bool = False,
) -> str:
    encoding = 'auto'
    if isinstance(spec, str):
        encoding = spec
    elif isinstance(spec, list):
        if per_type:
            raise ValueError("Feature encodings of heterogeneous graphs need "
                             "to be given as a string or as a dictionary "
                             "that maps feature names to encodings")
        encoding = spec[index]
    elif names is not None:
        encoding = spec.get(names[index], 'auto')

    if encoding == 'auto':
        is_binary = column is not None and bool(
            ((column == 0) | (column == 1)).all())
        return 'class' if is_binary else 'numeric'
    return encoding


def _check_encoding(
    spec: Union[str, Sequence[str], Dict[str, str]],
) -> Union[str, List[str], Dict[str, str]]:
    if isinstance(spec, str):
        encodings: Iterable[str] = [spec]
    elif isinstance(spec, dict):
        spec = dict(spec)
        encodings = spec.values()
    else:
        spec = list(spec)
        encodings = spec
    for encoding in encodings:
        if encoding not in FEATURE_ENCODINGS:
            raise ValueError(f"Unknown feature encoding '{encoding}' "
                             f"(expected one of {sorted(FEATURE_ENCODINGS)})")
    return spec


@dataclass(init=False)
class OWLSchema(CastMixin):
    r"""The schema that describes how a graph maps to an OWL ontology and
    back.

    The schema serves two purposes:

    1. When converting a :class:`~torch_geometric.data.Data` object to OWL
       (see :meth:`torch_geometric.utils.to_owl`), it supplies the
       *background knowledge* that makes learned class expressions readable:
       names for individuals, relations and features, how each feature is
       encoded, and an optional class hierarchy.
    2. When reading an OWL ontology (see
       :meth:`torch_geometric.utils.from_owl`), it records the IRIs that
       were discovered, so that the conversion can be reversed and tensor
       indices can be mapped back to OWL entities.

    Names can either be local names (which are prefixed with
    :obj:`base_iri`) or absolute IRIs (which are used as they are).

    Edges are written as plain object property assertions. If the graph has
    edge features :obj:`edge_attr` or parallel edges, every edge is
    additionally *reified* as an individual of class :obj:`pyg:Edge`, which
    is linked to its source and target node via :obj:`pyg:hasOutgoingEdge`
    and :obj:`pyg:hasIncomingEdge` and carries the edge features. This way,
    concept learners can use edge features, *e.g.*,
    :obj:`∃ hasOutgoingEdge.DoubleBond`, and both edge features and
    parallel edges survive a round trip.

    For heterogeneous graphs (see
    :class:`~torch_geometric.data.OWLHeteroData`), every node type becomes an
    OWL class (a sub class of :obj:`node_class`), and every relation of an
    edge type :obj:`(src, rel, dst)` an object property. Then,
    :obj:`node_names` and :obj:`feature_names` are dictionaries that map
    node types to names, and :obj:`relation_names` and
    :obj:`edge_feature_names` are dictionaries that map relations to names.

    For datasets of many graphs (*e.g.*, molecules), all graphs are written
    to one ontology, and every graph becomes an individual of class
    :obj:`pyg:Graph` (and of :obj:`graph_class`, if given) that is linked to
    its nodes via :obj:`pyg:hasNode`. Domain names such as :obj:`hasAtom`
    are given via :obj:`has_node_name` and declared as a sub property of
    :obj:`pyg:hasNode`. Graph labels then become concepts over graphs, *e.g.*,
    :obj:`∃ hasAtom.(N ⊓ ∃ hasOutgoingEdge.DoubleBond)`.

    .. note::

        Ontolearn (via :obj:`owlapy`) only accepts IRIs that contain a
        :obj:`"/"`. Prefer :obj:`http(s)` IRIs over, *e.g.*, :obj:`urn:`
        IRIs for ontologies that are used for concept learning.

    .. code-block:: python

        schema = OWLSchema(
            base_iri='http://example.org/cora#',
            node_class='Paper',
            relation_names=['cites'],
            feature_names=['word_neural', 'word_network', ...],
            feature_encoding='class',
            class_hierarchy={'word_neural': 'MachineLearningTerm'},
        )

    Args:
        base_iri (str, optional): The namespace used for all local names.
            (default: :obj:`"http://example.org/pyg#"`)
        node_class (str, optional): The OWL class every node is asserted to
            be an instance of. If set to :obj:`None`, no such class is
            asserted. (default: :obj:`"Node"`)
        node_names (List[str], optional): The names of individuals, one per
            node. If set to :obj:`None`, nodes are named
            :obj:`"node_{i}"`. (default: :obj:`None`)
        relation_names (List[str], optional): The names of object properties,
            one per edge type. If set to :obj:`None`, a single relation is
            named :obj:`"related_to"`, multiple relations
            :obj:`"relation_{r}"`. (default: :obj:`None`)
        inverse_edge_types (bool, optional): If set to :obj:`True`, edge type
            :obj:`2 * r + 1` is the inverse of edge type :obj:`2 * r` (the
            convention of :class:`~torch_geometric.datasets.Entities`), and
            :obj:`relation_names` holds one name per forward relation.
            (default: :obj:`False`)
        feature_names (List[str], optional): The names of the columns of
            :obj:`data.x`. If set to :obj:`None`, columns are named
            :obj:`"feature_{j}"`. (default: :obj:`None`)
        feature_encoding (str or List[str] or Dict[str, str], optional): How
            each feature is encoded in OWL. Either a single encoding for all
            features, a list with one encoding per feature, or a dictionary
            mapping feature names to encodings (unlisted features use
            :obj:`"auto"`). The possible values are:

                - :obj:`"class"`: Nodes whose feature value is larger than
                  :obj:`feature_threshold` are asserted to be instances of an
                  OWL class named after the feature. Only true facts are
                  asserted, which keeps the ontology sparse.

                - :obj:`"numeric"`: The value is asserted via an
                  :obj:`xsd:double` data property.

                - :obj:`"boolean"`: The thresholded value is asserted via an
                  :obj:`xsd:boolean` data property.

                - :obj:`"ignore"`: The feature is not converted.

                - :obj:`"auto"`: :obj:`"class"` for columns that only hold
                  :obj:`0` and :obj:`1`, :obj:`"numeric"` otherwise.

            (default: :obj:`"auto"`)
        feature_threshold (float, optional): The threshold for the
            :obj:`"class"` and :obj:`"boolean"` encodings.
            (default: :obj:`0.0`)
        class_hierarchy (Dict[str, Union[str, List[str]]], optional): Maps a
            class name to the name(s) of its super class(es), which are
            asserted via :obj:`rdfs:subClassOf`. Grouping features (*e.g.*,
            :obj:`{"Red": "Color", "Blue": "Color"}`) allows concept learners
            to generalize. (default: :obj:`None`)
        label_names (List[str], optional): The names of the classes in
            :obj:`data.y`. (default: :obj:`None`)
        include_labels (bool, optional): If set to :obj:`True`, node labels
            are asserted as OWL classes. This leaks the label into the
            ontology, so a concept learner trivially rediscovers it. Only
            use it to provide labels as background knowledge for a different
            task. (default: :obj:`False`)
        target_classes (List[str], optional): The OWL classes that were turned
            into :obj:`data.y` by :meth:`~torch_geometric.utils.from_owl`.
            Concept learners must not use them.
            (default: :obj:`None`)
        ignore_classes (List[str], optional): Further OWL classes that are
            neither turned into node features nor used by concept learners,
            *e.g.*, classes that are trivially equivalent to the target.
            (default: :obj:`None`)
        edge_feature_names (List[str], optional): The names of the columns of
            :obj:`data.edge_attr`. If set to :obj:`None`, columns are named
            :obj:`"edge_feature_{j}"`. (default: :obj:`None`)
        edge_feature_encoding (str or List[str] or Dict[str, str], optional):
            How each edge feature is encoded in OWL, analogous to
            :obj:`feature_encoding`. (default: :obj:`"auto"`)
        reify_edges (bool, optional): If set to :obj:`True`, every edge is
            reified as an individual of class :obj:`pyg:Edge`. If set to
            :obj:`None`, edges are only reified if the graph holds edge
            features or parallel edges. (default: :obj:`None`)
        node_type_names (Dict[str, str], optional): Maps node types of
            heterogeneous graphs to the names of their OWL classes. Unlisted
            node types use the node type itself. (default: :obj:`None`)
        graph_class (str, optional): An OWL class every graph of a graph
            dataset is asserted to be an instance of, in addition to
            :obj:`pyg:Graph`, *e.g.*, :obj:`"Molecule"`.
            (default: :obj:`None`)
        graph_names (List[str], optional): The names of the graph
            individuals, one per graph. If set to :obj:`None`, graphs are
            named :obj:`"graph_{i}"`. (default: :obj:`None`)
        has_node_name (str, optional): The name of the object property that
            links graphs to their nodes, *e.g.*, :obj:`"hasAtom"`. It is
            declared as a sub property of :obj:`pyg:hasNode`. If set to
            :obj:`None`, :obj:`pyg:hasNode` is used. (default: :obj:`None`)
        graph_feature_names (List[str], optional): The names of the columns of
            the graph-level features :obj:`data.graph_attr`. If set to
            :obj:`None`, columns are named :obj:`"graph_feature_{j}"`.
            (default: :obj:`None`)
        graph_feature_encoding (str or List[str] or Dict[str, str], optional):
            How each graph feature is encoded in OWL, analogous to
            :obj:`feature_encoding`. (default: :obj:`"auto"`)
    """
    base_iri: str
    node_class: Optional[str]
    node_names: Optional[Names]
    relation_names: Optional[Union[List[str], Dict[str, str]]]
    inverse_edge_types: bool
    feature_names: Optional[Names]
    feature_encoding: Union[str, List[str], Dict[str, str]]
    feature_threshold: float
    class_hierarchy: Optional[Dict[str, Union[str, List[str]]]]
    label_names: Optional[List[str]]
    include_labels: bool
    target_classes: Optional[List[str]]
    ignore_classes: Optional[List[str]]
    edge_feature_names: Optional[Names]
    edge_feature_encoding: Union[str, List[str], Dict[str, str]]
    reify_edges: Optional[bool]
    node_type_names: Optional[Dict[str, str]]
    graph_class: Optional[str]
    graph_names: Optional[List[str]]
    has_node_name: Optional[str]
    graph_feature_names: Optional[List[str]]
    graph_feature_encoding: Union[str, List[str], Dict[str, str]]

    def __init__(
        self,
        base_iri: str = 'http://example.org/pyg#',
        node_class: Optional[str] = 'Node',
        node_names: Optional[Union[Sequence[str], Dict[str,
                                                       Sequence[str]]]] = None,
        relation_names: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        inverse_edge_types: bool = False,
        feature_names: Optional[Union[Sequence[str],
                                      Mapping[str, Sequence[str]]]] = None,
        feature_encoding: Union[str, Sequence[str], Dict[str, str]] = 'auto',
        feature_threshold: float = 0.0,
        class_hierarchy: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
        label_names: Optional[Sequence[str]] = None,
        include_labels: bool = False,
        target_classes: Optional[Sequence[str]] = None,
        ignore_classes: Optional[Sequence[str]] = None,
        edge_feature_names: Optional[Union[Sequence[str],
                                           Mapping[str,
                                                   Sequence[str]]]] = None,
        edge_feature_encoding: Union[str, Sequence[str], Dict[str,
                                                              str]] = 'auto',
        reify_edges: Optional[bool] = None,
        node_type_names: Optional[Dict[str, str]] = None,
        graph_class: Optional[str] = None,
        graph_names: Optional[Sequence[str]] = None,
        has_node_name: Optional[str] = None,
        graph_feature_names: Optional[Sequence[str]] = None,
        graph_feature_encoding: Union[str, Sequence[str], Dict[str,
                                                               str]] = 'auto',
    ):
        if not is_absolute_iri(base_iri):
            raise ValueError(f"'base_iri' needs to be an absolute IRI "
                             f"(got '{base_iri}')")
        if not base_iri.endswith(('#', '/')):
            base_iri = base_iri + '#'

        def _to_list(value: Optional[Sequence[str]]) -> Optional[List[str]]:
            return None if value is None else list(value)

        self.base_iri = base_iri
        self.node_class = node_class
        self.node_names = _to_names(node_names)
        self.relation_names = (dict(relation_names) if isinstance(
            relation_names, dict) else _to_list(relation_names))
        self.inverse_edge_types = inverse_edge_types
        self.feature_names = _to_names(feature_names)
        self.feature_encoding = _check_encoding(feature_encoding)
        self.feature_threshold = feature_threshold
        self.class_hierarchy = None
        if class_hierarchy is not None:
            self.class_hierarchy = {
                child: parents if isinstance(parents, str) else list(parents)
                for child, parents in class_hierarchy.items()
            }
        self.label_names = _to_list(label_names)
        self.include_labels = include_labels
        self.target_classes = _to_list(target_classes)
        self.ignore_classes = _to_list(ignore_classes)
        self.edge_feature_names = _to_names(edge_feature_names)
        self.edge_feature_encoding = _check_encoding(edge_feature_encoding)
        self.reify_edges = reify_edges
        self.node_type_names = (None if node_type_names is None else
                                dict(node_type_names))
        self.graph_class = graph_class
        self.graph_names = _to_list(graph_names)
        self.has_node_name = has_node_name
        self.graph_feature_names = _to_list(graph_feature_names)
        self.graph_feature_encoding = _check_encoding(graph_feature_encoding)

    def iri(self, name: str) -> str:
        r"""Returns the absolute IRI of :obj:`name`."""
        if is_absolute_iri(name):
            return name
        return self.base_iri + quote(name.strip().replace(' ', '_'), safe='')

    def node_iri(self, index: int, node_type: Optional[str] = None) -> str:
        r"""Returns the IRI of the individual of node :obj:`index` (of
        type :obj:`node_type` in heterogeneous graphs).
        """
        names = _select(self.node_names, node_type)
        if names is not None:
            return self.iri(names[index])
        if node_type is not None:
            return self.iri(f'{node_type}_{index}')
        return self.iri(f'node_{index}')

    def node_type_iri(self, node_type: str) -> str:
        r"""Returns the IRI of the OWL class of node type
        :obj:`node_type`.
        """
        return self.iri((self.node_type_names or {}).get(node_type, node_type))

    def relation_iri(
        self,
        relation: Union[int, str],
        num_relations: int = 1,
    ) -> str:
        r"""Returns the IRI of the object property of relation
        :obj:`relation`, given as an edge type index or, in heterogeneous
        graphs, as the relation of an edge type.
        """
        if isinstance(relation, str):
            names = self.relation_names
            if isinstance(names, dict):
                return self.iri(names.get(relation, relation))
            return self.iri(relation)
        if isinstance(self.relation_names, list):
            return self.iri(self.relation_names[relation])
        if num_relations == 1:
            return self.iri('related_to')
        return self.iri(f'relation_{relation}')

    def feature_iri(self, index: int, node_type: Optional[str] = None) -> str:
        r"""Returns the IRI of the class or data property of feature
        :obj:`index` (of node type :obj:`node_type` in heterogeneous graphs).
        """
        names = _select(self.feature_names, node_type)
        if names is not None:
            return self.iri(names[index])
        if node_type is not None:
            return self.iri(f'{node_type}_feature_{index}')
        return self.iri(f'feature_{index}')

    def edge_iri(self, index: int) -> str:
        r"""Returns the IRI of the individual of (reified) edge
        :obj:`index`.
        """
        return self.iri(f'edge_{index}')

    def edge_feature_iri(
        self,
        index: int,
        relation: Optional[str] = None,
    ) -> str:
        r"""Returns the IRI of the class or data property of edge feature
        :obj:`index` (of relation :obj:`relation` in heterogeneous graphs).
        """
        names = _select(self.edge_feature_names, relation)
        if names is not None:
            return self.iri(names[index])
        if relation is not None:
            return self.iri(f'{relation}_feature_{index}')
        return self.iri(f'edge_feature_{index}')

    def graph_iri(self, index: int) -> str:
        r"""Returns the IRI of the individual of graph :obj:`index`."""
        if self.graph_names is not None:
            return self.iri(self.graph_names[index])
        return self.iri(f'graph_{index}')

    def has_node_iri(self) -> str:
        r"""Returns the IRI of the object property that links graphs to their
        nodes.
        """
        if self.has_node_name is None:
            return PYG_HAS_NODE
        return self.iri(self.has_node_name)

    def graph_feature_iri(self, index: int) -> str:
        r"""Returns the IRI of the class or data property of graph feature
        :obj:`index`.
        """
        if self.graph_feature_names is not None:
            return self.iri(self.graph_feature_names[index])
        return self.iri(f'graph_feature_{index}')

    def label_iri(self, label: int) -> str:
        r"""Returns the IRI of the OWL class of label :obj:`label`."""
        if self.label_names is not None:
            return self.iri(self.label_names[label])
        return self.iri(f'label_{label}')

    def encoding(
        self,
        index: int,
        column: Optional[Tensor] = None,
        node_type: Optional[str] = None,
    ) -> str:
        r"""Returns the encoding of feature :obj:`index` (of node type
        :obj:`node_type` in heterogeneous graphs), resolving :obj:`"auto"`
        based on the feature values in :obj:`column`.
        """
        names = _select(self.feature_names, node_type)
        return _resolve_encoding(self.feature_encoding, names, index, column,
                                 per_type=node_type is not None)

    def edge_encoding(
        self,
        index: int,
        column: Optional[Tensor] = None,
        relation: Optional[str] = None,
    ) -> str:
        r"""Returns the encoding of edge feature :obj:`index` (of relation
        :obj:`relation` in heterogeneous graphs), resolving :obj:`"auto"`
        based on the feature values in :obj:`column`.
        """
        names = _select(self.edge_feature_names, relation)
        return _resolve_encoding(self.edge_feature_encoding, names, index,
                                 column, per_type=relation is not None)

    def graph_encoding(
        self,
        index: int,
        column: Optional[Tensor] = None,
    ) -> str:
        r"""Returns the encoding of graph feature :obj:`index`, resolving
        :obj:`"auto"` based on the feature values in :obj:`column`.
        """
        return _resolve_encoding(self.graph_feature_encoding,
                                 self.graph_feature_names, index, column)

    def excluded_classes(self) -> List[str]:
        r"""Returns the IRIs of all classes that concept learners must not
        use (:obj:`target_classes` and :obj:`ignore_classes`).
        """
        names = (self.target_classes or []) + (self.ignore_classes or [])
        return [self.iri(name) for name in names]


def to_owl_schema(
        schema: Optional[Union['OWLSchema', Dict[str, Any]]]) -> 'OWLSchema':
    r"""Returns :obj:`schema` as an :class:`OWLSchema` (a default schema if
    :obj:`schema` is :obj:`None`).
    """
    if schema is None:
        return OWLSchema()
    if isinstance(schema, OWLSchema):
        return schema
    return OWLSchema(**schema)


class _OWLMixin:
    r"""The OWL-related attributes and methods shared by :class:`OWLData`
    and :class:`OWLHeteroData`. The schema and source are stored outside of
    the attribute storage, so that they are neither collated nor moved
    across devices.
    """
    __dict__: Dict[str, Any]

    def _init_owl(
        self,
        schema: Optional[Union[OWLSchema, Dict[str, Any]]],
        source: Optional[str],
    ) -> None:
        self.__dict__['_schema'] = to_owl_schema(schema)
        self.__dict__['_source'] = source

    @property
    def schema(self) -> OWLSchema:
        r"""The :class:`OWLSchema` that maps the graph to OWL."""
        if self.__dict__.get('_schema') is None:
            self.__dict__['_schema'] = OWLSchema()
        return self.__dict__['_schema']

    @schema.setter
    def schema(self, schema: Union[OWLSchema, Dict[str, Any]]) -> None:
        self.__dict__['_schema'] = to_owl_schema(schema)

    @property
    def source(self) -> Optional[str]:
        r"""The path of the OWL file the data was read from, if any."""
        return self.__dict__.get('_source')

    @source.setter
    def source(self, source: Optional[str]) -> None:
        self.__dict__['_source'] = source

    def to_owl(self, **kwargs: Any) -> Any:
        r"""Converts the data object to an OWL ontology given as an
        :class:`rdflib.Graph`.
        See :meth:`torch_geometric.utils.to_owl` for the arguments.
        """
        from torch_geometric.utils import to_owl
        return to_owl(self, **kwargs)

    def save_owl(self, path: str, format: str = 'xml', **kwargs: Any) -> str:
        r"""Saves the data object as an OWL ontology and returns the path.

        Args:
            path (str): The file to write to.
            format (str, optional): The :obj:`rdflib` serialization format.
                :obj:`"xml"` (RDF/XML) can be read by every OWL tool.
                (default: :obj:`"xml"`)
            **kwargs (optional): Additional arguments of
                :meth:`torch_geometric.utils.to_owl`.
        """
        path = osp.expanduser(path)
        self.to_owl(**kwargs).serialize(destination=path, format=format)
        return path


class OWLData(_OWLMixin, Data):
    r"""A :class:`~torch_geometric.data.Data` object that is connected to an
    OWL ontology.

    :class:`OWLData` is a regular :class:`~torch_geometric.data.Data` object,
    so it works with every GNN and every mask-based explainer in
    :mod:`torch_geometric.explain`. In addition, it knows how its nodes,
    relations and features map to OWL individuals, object properties and
    classes (see :class:`OWLSchema`), which is what OWL concept learners such
    as :class:`~torch_geometric.explain.algorithm.OntolearnExplainer` need.

    It can be created in two directions:

    .. code-block:: python

        from torch_geometric.data import OWLData, OWLSchema

        # OWL -> PyG: individuals become nodes, object properties become
        # edge types, and asserted classes become node features:
        data = OWLData.from_owl('family.owl', target_classes=['Aunt'])

        # PyG -> OWL: a schema gives names to nodes, relations and features:
        data = OWLData.from_data(data, OWLSchema(
            base_iri='http://example.org/cora#',
            node_class='Paper',
            relation_names=['cites'],
        ))
        data.save_owl('cora.owl')

    The IRI of every node is stored in the node-level attribute
    :obj:`node_iri`, so the mapping survives
    :meth:`~torch_geometric.data.Data.subgraph` and batching.

    Args:
        x (torch.Tensor, optional): Node feature matrix.
            (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity.
            (default: :obj:`None`)
        edge_attr (torch.Tensor, optional): Edge feature matrix.
            (default: :obj:`None`)
        y (torch.Tensor, optional): Node labels. (default: :obj:`None`)
        pos (torch.Tensor, optional): Node position matrix.
            (default: :obj:`None`)
        time (torch.Tensor, optional): The timestamps.
            (default: :obj:`None`)
        schema (OWLSchema or Dict[str, Any], optional): The schema that maps
            the graph to OWL. (default: :obj:`None`)
        source (str, optional): The path of the OWL file the data was read
            from, if any. Concept learners use this file directly, so that
            the full ontology (including its TBox) is available.
            (default: :obj:`None`)
        **kwargs (optional): Additional attributes, *e.g.*,
            :obj:`edge_type` or :obj:`node_iri`.
    """
    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: Optional[Union[Tensor, int, float]] = None,
        pos: OptTensor = None,
        time: OptTensor = None,
        schema: Optional[Union[OWLSchema, Dict[str, Any]]] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                         pos=pos, time=time, **kwargs)
        self._init_owl(schema, source)

    @classmethod
    def from_owl(cls, source: Any, **kwargs: Any) -> 'OWLData':
        r"""Creates an :class:`OWLData` object from an OWL ontology.
        See :meth:`torch_geometric.utils.from_owl` for the arguments.
        """
        from torch_geometric.utils import from_owl
        if kwargs.get('node_types') is not None:
            raise ValueError("Use 'OWLHeteroData.from_owl' to read "
                             "heterogeneous graphs")
        if kwargs.get('graph_level'):
            raise ValueError("Use 'from_owl(..., graph_level=True)' to read "
                             "graph datasets (it returns a list of "
                             "'OWLData' objects)")
        out = from_owl(source, **kwargs)
        assert isinstance(out, OWLData)
        return out

    @classmethod
    def from_data(
        cls,
        data: Data,
        schema: Optional[Union[OWLSchema, Dict[str, Any]]] = None,
    ) -> 'OWLData':
        r"""Creates an :class:`OWLData` object from a
        :class:`~torch_geometric.data.Data` object.

        Args:
            data (torch_geometric.data.Data): The input graph.
            schema (OWLSchema or Dict[str, Any], optional): The schema that
                maps the graph to OWL. If set to :obj:`None` and :obj:`data`
                is an :class:`OWLData` object, its schema is reused.
                (default: :obj:`None`)
        """
        if schema is None and isinstance(data, OWLData):
            schema = data.schema
        source = data.source if isinstance(data, OWLData) else None

        out = cls(schema=schema, source=source)
        for key, value in data.items():
            out[key] = value
        if 'num_nodes' not in out and data.num_nodes is not None:
            out.num_nodes = data.num_nodes

        if 'node_iri' not in out and out.num_nodes is not None:
            out.node_iri = [
                out.schema.node_iri(i) for i in range(out.num_nodes)
            ]
        return out

    def index_of(self, iris: Union[str, Iterable[str]]) -> Tensor:
        r"""Returns the node indices of the individuals with the given IRIs.

        Args:
            iris (str or Iterable[str]): The IRI(s) of the individuals.
        """
        if isinstance(iris, str):
            iris = [iris]
        mapping = {iri: i for i, iri in enumerate(self.node_iri)}
        missing = [iri for iri in iris if iri not in mapping]
        if len(missing) > 0:
            raise KeyError(f"Found {len(missing)} unknown individual(s), "
                           f"e.g., '{missing[0]}'")
        return torch.tensor([mapping[iri] for iri in iris], dtype=torch.long)

    def set_labels_from_examples(
        self,
        positive: Iterable[str],
        negative: Iterable[str],
        mask_name: Optional[str] = 'train_mask',
    ) -> 'OWLData':
        r"""Sets binary node labels from positive and negative example
        individuals, *e.g.*, from an OWL learning problem.

        Nodes that are neither positive nor negative examples are labeled
        :obj:`-1`.

        Args:
            positive (Iterable[str]): The IRIs of the positive examples.
            negative (Iterable[str]): The IRIs of the negative examples.
            mask_name (str, optional): If set, additionally stores a boolean
                mask of all labeled nodes under this attribute name.
                (default: :obj:`"train_mask"`)
        """
        y = torch.full((self.num_nodes, ), -1, dtype=torch.long)
        y[self.index_of(list(negative))] = 0
        y[self.index_of(list(positive))] = 1
        self.y = y
        if mask_name is not None:
            self[mask_name] = y >= 0
        return self


class OWLHeteroData(_OWLMixin, HeteroData):
    r"""A :class:`~torch_geometric.data.HeteroData` object that is connected
    to an OWL ontology, the heterogeneous counterpart of :class:`OWLData`.

    Every node type corresponds to an OWL class, and every relation of an
    edge type :obj:`(src, rel, dst)` to an object property (see
    :class:`OWLSchema`). The IRIs of the nodes of every node type are stored
    in the node-level attribute :obj:`node_iri`.

    .. code-block:: python

        from torch_geometric.data import OWLHeteroData, OWLSchema

        # OWL -> PyG: The individuals of each class become one node type:
        data = OWLHeteroData.from_owl('aifb.owl',
                                      node_types=['Person', 'Publication'])

        # PyG -> OWL:
        data = OWLHeteroData.from_data(data, OWLSchema(
            base_iri='http://example.org/dblp#',
            feature_names={'paper': ['Theory', 'Systems']},
        ))
        data.save_owl('dblp.owl')

    Args:
        _mapping (Dict[str, Any], optional): A dictionary holding the node
            and edge type attributes, see
            :class:`~torch_geometric.data.HeteroData`. (default: :obj:`None`)
        schema (OWLSchema or Dict[str, Any], optional): The schema that maps
            the graph to OWL. (default: :obj:`None`)
        source (str, optional): The path of the OWL file the data was read
            from, if any. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(
        self,
        _mapping: Optional[Dict[str, Any]] = None,
        schema: Optional[Union[OWLSchema, Dict[str, Any]]] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(_mapping, **kwargs)
        self._init_owl(schema, source)

    def __setattr__(self, key: str, value: Any) -> None:
        prop = getattr(self.__class__, key, None)
        if isinstance(prop, property) and prop.fset is not None:
            prop.fset(self, value)
        else:
            super().__setattr__(key, value)

    @classmethod
    def from_owl(
        cls,
        source: Any,
        node_types: Sequence[str],
        **kwargs: Any,
    ) -> 'OWLHeteroData':
        r"""Creates an :class:`OWLHeteroData` object from an OWL ontology.
        See :meth:`torch_geometric.utils.from_owl` for the arguments.
        """
        from torch_geometric.utils import from_owl
        out = from_owl(source, node_types=node_types, **kwargs)
        assert isinstance(out, OWLHeteroData)
        return out

    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        schema: Optional[Union[OWLSchema, Dict[str, Any]]] = None,
    ) -> 'OWLHeteroData':
        r"""Creates an :class:`OWLHeteroData` object from a
        :class:`~torch_geometric.data.HeteroData` object.

        Args:
            data (torch_geometric.data.HeteroData): The input graph.
            schema (OWLSchema or Dict[str, Any], optional): The schema that
                maps the graph to OWL. If set to :obj:`None` and :obj:`data`
                is an :class:`OWLHeteroData` object, its schema is reused.
                (default: :obj:`None`)
        """
        if schema is None and isinstance(data, OWLHeteroData):
            schema = data.schema
        source = data.source if isinstance(data, OWLHeteroData) else None

        out = cls(schema=schema, source=source)
        for key, value in data._global_store.items():
            out[key] = value
        for node_type, store in data.node_items():
            for key, value in store.items():
                out[node_type][key] = value
            num_nodes = store.num_nodes
            if 'node_iri' not in out[node_type] and num_nodes is not None:
                out[node_type].num_nodes = num_nodes
                out[node_type].node_iri = [
                    out.schema.node_iri(i, node_type) for i in range(num_nodes)
                ]
        for edge_type, store in data.edge_items():
            for key, value in store.items():
                out[edge_type][key] = value
        return out

    def index_of(
        self,
        iris: Union[str, Iterable[str]],
        node_type: str,
    ) -> Tensor:
        r"""Returns the node indices of the individuals with the given IRIs.

        Args:
            iris (str or Iterable[str]): The IRI(s) of the individuals.
            node_type (str): The node type of the individuals.
        """
        if isinstance(iris, str):
            iris = [iris]
        mapping = {iri: i for i, iri in enumerate(self[node_type].node_iri)}
        missing = [iri for iri in iris if iri not in mapping]
        if len(missing) > 0:
            raise KeyError(f"Found {len(missing)} unknown individual(s) of "
                           f"node type '{node_type}', e.g., '{missing[0]}'")
        return torch.tensor([mapping[iri] for iri in iris], dtype=torch.long)
