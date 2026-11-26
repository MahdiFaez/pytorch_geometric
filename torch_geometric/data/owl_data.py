import os.path as osp
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from urllib.parse import quote

import torch
from torch import Tensor

from torch_geometric.data.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.utils.mixin import CastMixin

FEATURE_ENCODINGS = {'auto', 'class', 'numeric', 'boolean', 'ignore'}

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
    """
    base_iri: str
    node_class: Optional[str]
    node_names: Optional[List[str]]
    relation_names: Optional[List[str]]
    inverse_edge_types: bool
    feature_names: Optional[List[str]]
    feature_encoding: Union[str, List[str], Dict[str, str]]
    feature_threshold: float
    class_hierarchy: Optional[Dict[str, Union[str, List[str]]]]
    label_names: Optional[List[str]]
    include_labels: bool
    target_classes: Optional[List[str]]
    ignore_classes: Optional[List[str]]

    def __init__(
        self,
        base_iri: str = 'http://example.org/pyg#',
        node_class: Optional[str] = 'Node',
        node_names: Optional[Sequence[str]] = None,
        relation_names: Optional[Sequence[str]] = None,
        inverse_edge_types: bool = False,
        feature_names: Optional[Sequence[str]] = None,
        feature_encoding: Union[str, Sequence[str], Dict[str, str]] = 'auto',
        feature_threshold: float = 0.0,
        class_hierarchy: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
        label_names: Optional[Sequence[str]] = None,
        include_labels: bool = False,
        target_classes: Optional[Sequence[str]] = None,
        ignore_classes: Optional[Sequence[str]] = None,
    ):
        if not is_absolute_iri(base_iri):
            raise ValueError(f"'base_iri' needs to be an absolute IRI "
                             f"(got '{base_iri}')")
        if not base_iri.endswith(('#', '/')):
            base_iri = base_iri + '#'

        if isinstance(feature_encoding, str):
            encodings: Iterable[str] = [feature_encoding]
        elif isinstance(feature_encoding, dict):
            encodings = feature_encoding.values()
            feature_encoding = dict(feature_encoding)
        else:
            feature_encoding = list(feature_encoding)
            encodings = feature_encoding
        for encoding in encodings:
            if encoding not in FEATURE_ENCODINGS:
                raise ValueError(f"Unknown feature encoding '{encoding}' "
                                 f"(expected one of "
                                 f"{sorted(FEATURE_ENCODINGS)})")

        def _to_list(value: Optional[Sequence[str]]) -> Optional[List[str]]:
            return None if value is None else list(value)

        self.base_iri = base_iri
        self.node_class = node_class
        self.node_names = _to_list(node_names)
        self.relation_names = _to_list(relation_names)
        self.inverse_edge_types = inverse_edge_types
        self.feature_names = _to_list(feature_names)
        self.feature_encoding = feature_encoding  # type: ignore
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

    def iri(self, name: str) -> str:
        r"""Returns the absolute IRI of :obj:`name`."""
        if is_absolute_iri(name):
            return name
        return self.base_iri + quote(name.strip().replace(' ', '_'), safe='')

    def node_iri(self, index: int) -> str:
        r"""Returns the IRI of the individual of node :obj:`index`."""
        if self.node_names is not None:
            return self.iri(self.node_names[index])
        return self.iri(f'node_{index}')

    def relation_iri(self, relation: int, num_relations: int = 1) -> str:
        r"""Returns the IRI of the object property of relation
        :obj:`relation`.
        """
        if self.relation_names is not None:
            return self.iri(self.relation_names[relation])
        if num_relations == 1:
            return self.iri('related_to')
        return self.iri(f'relation_{relation}')

    def feature_iri(self, index: int) -> str:
        r"""Returns the IRI of the class or data property of feature
        :obj:`index`.
        """
        if self.feature_names is not None:
            return self.iri(self.feature_names[index])
        return self.iri(f'feature_{index}')

    def label_iri(self, label: int) -> str:
        r"""Returns the IRI of the OWL class of label :obj:`label`."""
        if self.label_names is not None:
            return self.iri(self.label_names[label])
        return self.iri(f'label_{label}')

    def encoding(self, index: int, column: Optional[Tensor] = None) -> str:
        r"""Returns the encoding of feature :obj:`index`, resolving
        :obj:`"auto"` based on the feature values in :obj:`column`.
        """
        encoding = 'auto'
        if isinstance(self.feature_encoding, str):
            encoding = self.feature_encoding
        elif isinstance(self.feature_encoding, list):
            encoding = self.feature_encoding[index]
        elif self.feature_names is not None:
            name = self.feature_names[index]
            encoding = self.feature_encoding.get(name, 'auto')

        if encoding == 'auto':
            is_binary = column is not None and bool(
                ((column == 0) | (column == 1)).all())
            return 'class' if is_binary else 'numeric'
        return encoding

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


class OWLData(Data):
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
        # The schema and source are stored outside of the attribute storage,
        # so that they are neither collated nor moved across devices:
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

    @classmethod
    def from_owl(cls, source: Any, **kwargs: Any) -> 'OWLData':
        r"""Creates an :class:`OWLData` object from an OWL ontology.
        See :meth:`torch_geometric.utils.from_owl` for the arguments.
        """
        from torch_geometric.utils import from_owl
        return from_owl(source, **kwargs)

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

    def to_owl(self) -> Any:
        r"""Converts the data object to an OWL ontology given as an
        :class:`rdflib.Graph`.
        See :meth:`torch_geometric.utils.to_owl` for details.
        """
        from torch_geometric.utils import to_owl
        return to_owl(self)

    def save_owl(self, path: str, format: str = 'xml') -> str:
        r"""Saves the data object as an OWL ontology and returns the path.

        Args:
            path (str): The file to write to.
            format (str, optional): The :obj:`rdflib` serialization format.
                :obj:`"xml"` (RDF/XML) can be read by every OWL tool.
                (default: :obj:`"xml"`)
        """
        path = osp.expanduser(path)
        self.to_owl().serialize(destination=path, format=format)
        return path

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
