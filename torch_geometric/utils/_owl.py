import os.path as osp
import warnings
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch import Tensor

import torch_geometric


def to_owl(
    data: 'torch_geometric.data.Data',
    schema: Optional[Union['torch_geometric.data.OWLSchema',
                           Dict[str, Any]]] = None,
    ontology_iri: Optional[str] = None,
) -> Any:
    r"""Converts a :class:`torch_geometric.data.Data` instance to an OWL
    ontology, given as an :class:`rdflib.Graph`.

    * Every node becomes an :obj:`owl:NamedIndividual` (named via
      :obj:`data.node_iri` or the schema).
    * Every edge type becomes an :obj:`owl:ObjectProperty`, and every edge
      an object property assertion (duplicate edges are removed).
    * Every column of :obj:`data.x` becomes an OWL class or a data property,
      depending on :obj:`schema.feature_encoding`.
    * :obj:`schema.class_hierarchy` is added via :obj:`rdfs:subClassOf`.
    * Node labels :obj:`data.y` are **not** added, unless
      :obj:`schema.include_labels=True`. Otherwise, a concept learner would
      trivially rediscover the label.

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
        data (torch_geometric.data.Data): The graph data object.
        schema (OWLSchema or Dict[str, Any], optional): The schema that maps
            the graph to OWL. If set to :obj:`None`, the schema of an
            :class:`~torch_geometric.data.OWLData` object is used, or a
            default schema otherwise. (default: :obj:`None`)
        ontology_iri (str, optional): The IRI of the ontology. If set to
            :obj:`None`, it is derived from :obj:`schema.base_iri`.
            (default: :obj:`None`)
    """
    import rdflib
    from rdflib import Literal, URIRef
    from rdflib.namespace import OWL, RDF, RDFS, XSD

    from torch_geometric.data import OWLData
    from torch_geometric.data.owl_data import to_owl_schema

    if schema is None and isinstance(data, OWLData):
        schema = data.schema
    owl_schema = to_owl_schema(schema)

    graph = rdflib.Graph()
    graph.bind('owl', OWL)
    graph.bind('', rdflib.Namespace(owl_schema.base_iri))

    onto = URIRef(ontology_iri or owl_schema.base_iri.rstrip('#/'))
    graph.add((onto, RDF.type, OWL.Ontology))

    def declare_class(iri: str) -> URIRef:
        node = URIRef(iri)
        graph.add((node, RDF.type, OWL.Class))
        return node

    # Individuals #############################################################
    num_nodes = data.num_nodes or 0
    if 'node_iri' in data:
        node_iri = list(data.node_iri)
        if len(node_iri) != num_nodes:
            raise ValueError(f"Expected 'node_iri' to hold {num_nodes} IRIs "
                             f"(got {len(node_iri)})")
    else:
        node_iri = [owl_schema.node_iri(i) for i in range(num_nodes)]
    if len(set(node_iri)) != len(node_iri):
        raise ValueError("Found duplicated node IRIs")

    individuals = [URIRef(iri) for iri in node_iri]
    node_class = None
    if owl_schema.node_class is not None:
        node_class = declare_class(owl_schema.iri(owl_schema.node_class))
    for individual in individuals:
        graph.add((individual, RDF.type, OWL.NamedIndividual))
        if node_class is not None:
            graph.add((individual, RDF.type, node_class))

    # Object properties #######################################################
    edge_index = data.edge_index
    if edge_index is not None and edge_index.numel() > 0:
        edge_index = edge_index.detach().cpu()
        if 'edge_type' in data and data.edge_type is not None:
            edge_type = data.edge_type.detach().cpu().view(-1)
        else:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

        if owl_schema.inverse_edge_types:
            # Inverse edges are implied by their forward assertion:
            mask = edge_type % 2 == 0
            edge_index, edge_type = edge_index[:, mask], edge_type[mask] // 2

        num_relations = int(edge_type.max()) + 1 if edge_type.numel() else 0
        if (owl_schema.relation_names is not None
                and num_relations > len(owl_schema.relation_names)):
            raise ValueError(
                f"Found {num_relations} relations but only "
                f"{len(owl_schema.relation_names)} relation names")
        num_relations = max(num_relations, len(owl_schema.relation_names
                                               or []))

        properties = []
        for r in range(num_relations):
            prop = URIRef(owl_schema.relation_iri(r, num_relations))
            graph.add((prop, RDF.type, OWL.ObjectProperty))
            properties.append(prop)

        edges = torch.stack([edge_index[0], edge_index[1], edge_type], dim=0)
        for src, dst, rel in torch.unique(edges, dim=1).t().tolist():
            graph.add((individuals[src], properties[rel], individuals[dst]))

    # Features ################################################################
    x = data.x if 'x' in data else None
    if isinstance(x, Tensor):
        x = x.detach().cpu()
        x = x.view(-1, 1) if x.dim() == 1 else x
        if x.dim() != 2:
            raise ValueError(f"Expected a two-dimensional feature matrix "
                             f"(got {x.dim()} dimensions)")
        if (owl_schema.feature_names is not None
                and len(owl_schema.feature_names) != x.size(1)):
            raise ValueError(f"Expected {x.size(1)} feature names (got "
                             f"{len(owl_schema.feature_names)})")

        for j in range(x.size(1)):
            column = x[:, j]
            encoding = owl_schema.encoding(j, column)
            if encoding == 'ignore':
                continue

            iri = URIRef(owl_schema.feature_iri(j))
            if encoding == 'class':
                declare_class(str(iri))
                index = (column
                         > owl_schema.feature_threshold).nonzero().view(-1)
                for i in index.tolist():
                    graph.add((individuals[i], RDF.type, iri))
                continue

            graph.add((iri, RDF.type, OWL.DatatypeProperty))
            if encoding == 'boolean':
                graph.add((iri, RDFS.range, XSD.boolean))
                values = (column > owl_schema.feature_threshold).tolist()
                for i, value in enumerate(values):
                    graph.add((individuals[i], iri,
                               Literal(bool(value), datatype=XSD.boolean)))
            else:  # encoding == 'numeric':
                graph.add((iri, RDFS.range, XSD.double))
                for i, value in enumerate(column.tolist()):
                    if value == value:  # Skip NaN values.
                        graph.add((individuals[i], iri,
                                   Literal(float(value), datatype=XSD.double)))

    # Labels ##################################################################
    y = data.y if 'y' in data else None
    if (owl_schema.include_labels and isinstance(y, Tensor) and y.dim() == 1
            and y.numel() == num_nodes and not y.is_floating_point()):
        labels = y.detach().cpu().tolist()
        label_classes = {
            label: declare_class(owl_schema.label_iri(label))
            for label in set(labels) if label >= 0
        }
        for i, label in enumerate(labels):
            if label >= 0:
                graph.add((individuals[i], RDF.type, label_classes[label]))

    # Class hierarchy #########################################################
    for child, parents in (owl_schema.class_hierarchy or {}).items():
        child_node = declare_class(owl_schema.iri(child))
        for parent in [parents] if isinstance(parents, str) else parents:
            parent_node = declare_class(owl_schema.iri(parent))
            graph.add((child_node, RDFS.subClassOf, parent_node))

    return graph


def from_owl(
    source: Any,
    schema: Optional[Union['torch_geometric.data.OWLSchema',
                           Dict[str, Any]]] = None,
    target_classes: Optional[Sequence[str]] = None,
    ignore_classes: Optional[Sequence[str]] = None,
    features: Sequence[str] = ('types', 'data_properties'),
    add_inverse_edges: bool = True,
    format: Optional[str] = None,
) -> 'torch_geometric.data.OWLData':
    r"""Converts an OWL ontology to a
    :class:`torch_geometric.data.OWLData` instance.

    * Every named individual becomes a node. Nodes are sorted by IRI, so the
      conversion is deterministic. The IRIs are stored in
      :obj:`data.node_iri`.
    * Every object property becomes an edge type in :obj:`data.edge_type`.
      If :obj:`add_inverse_edges=True`, edge type :obj:`2 * r + 1` holds the
      inverse edges of relation :obj:`r`, following the convention of
      :class:`~torch_geometric.datasets.Entities`.
    * Node features :obj:`data.x` contain a multi-hot encoding of the
      asserted classes of each individual (closed under asserted
      :obj:`rdfs:subClassOf` axioms), followed by one column per numeric or
      boolean data property.
    * Node labels :obj:`data.y` are derived from :obj:`target_classes`.
      Target classes are never used as node features.

    No reasoning beyond the :obj:`rdfs:subClassOf` closure is performed.
    The discovered IRIs are stored in :obj:`data.schema`, so the conversion
    can be reversed via :meth:`torch_geometric.utils.to_owl`.

    .. code-block:: python

        from torch_geometric.utils import from_owl

        data = from_owl('family.owl', target_classes=['Female'])
        # OWLData(x=[202, 17], edge_index=[2, 1296], edge_type=[1296],
        #         y=[202], node_iri=[202], num_nodes=202)

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
        features (List[str], optional): The node features to compute.
            :obj:`"types"` adds the asserted classes, :obj:`"data_properties"`
            adds numeric and boolean data property values. If empty,
            :obj:`data.x` is set to :obj:`None`.
            (default: :obj:`("types", "data_properties")`)
        add_inverse_edges (bool, optional): If set to :obj:`True`, adds
            inverse edges with their own edge types. (default: :obj:`True`)
        format (str, optional): The :obj:`rdflib` parser format. If set to
            :obj:`None`, it is guessed from the file extension, with
            :obj:`".owl"` files parsed as RDF/XML. (default: :obj:`None`)
    """
    import rdflib
    from rdflib import Literal, URIRef
    from rdflib.namespace import OWL, RDF, RDFS, XSD
    from rdflib.util import guess_format

    from torch_geometric.data import OWLData, OWLSchema
    from torch_geometric.data.owl_data import (
        is_absolute_iri,
        local_name,
        to_owl_schema,
    )

    for feature in features:
        if feature not in {'types', 'data_properties'}:
            raise ValueError(f"Unknown feature type '{feature}' (expected "
                             f"'types' or 'data_properties')")

    path: Optional[str] = None
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
        if osp.exists(path):
            path = osp.abspath(path)

    if schema is None:  # Derive the namespace from the ontology header:
        ontologies = sorted(
            str(onto) for onto in graph.subjects(RDF.type, OWL.Ontology)
            if isinstance(onto, URIRef) and is_absolute_iri(str(onto)))
        if len(ontologies) > 0:
            schema = OWLSchema(base_iri=ontologies[0])
    owl_schema = to_owl_schema(schema)

    # Collect declared entities ###############################################
    def uris(nodes: Any) -> Set[Any]:
        return {node for node in nodes if isinstance(node, URIRef)}

    builtin = {OWL.Thing, OWL.Nothing, OWL.NamedIndividual, OWL.Class}
    classes = uris(graph.subjects(RDF.type, OWL.Class))
    classes |= uris(graph.subjects(RDF.type, RDFS.Class))
    classes |= uris(graph.objects(None, RDFS.subClassOf))
    classes -= builtin
    object_properties = uris(graph.subjects(RDF.type, OWL.ObjectProperty))
    data_properties = uris(graph.subjects(RDF.type, OWL.DatatypeProperty))
    schema_entities = classes | object_properties | data_properties
    schema_entities |= uris(graph.subjects(RDF.type, OWL.AnnotationProperty))
    schema_entities |= uris(graph.subjects(RDF.type, OWL.Ontology))

    individuals = uris(graph.subjects(RDF.type, OWL.NamedIndividual))
    for subj, obj in graph.subject_objects(RDF.type):
        if isinstance(subj, URIRef) and obj in classes:
            individuals.add(subj)
    for prop in object_properties:
        for subj, obj in graph.subject_objects(prop):
            individuals |= uris([subj, obj])
    individuals -= schema_entities

    def is_builtin(pred: Any) -> bool:  # RDF/RDFS/OWL/XSD vocabulary.
        return str(pred).startswith((str(RDF), str(RDFS), str(OWL), str(XSD)))

    nodes = sorted(individuals, key=str)
    node_index = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    def resolve(names: Optional[Sequence[str]]) -> List[Any]:
        out = []
        for name in names or []:
            iri = URIRef(owl_schema.iri(name))
            if iri not in classes:
                matches = [c for c in classes if local_name(c) == name]
                if len(matches) != 1:
                    raise ValueError(f"Could not find the OWL class '{name}'")
                iri = matches[0]
            out.append(iri)
        return out

    # Class hierarchy (asserted, transitive) ##################################
    parents: Dict[Any, Set[Any]] = defaultdict(set)
    for child, parent in graph.subject_objects(RDFS.subClassOf):
        if isinstance(child, URIRef) and isinstance(parent, URIRef):
            parents[child].add(parent)

    def closure(types: Set[Any]) -> Set[Any]:
        out, stack = set(types), list(types)
        while len(stack) > 0:
            for parent in parents.get(stack.pop(), ()):
                if parent not in out and parent not in builtin:
                    out.add(parent)
                    stack.append(parent)
        return out

    # Target classes and their sub classes entail the label, and are thus
    # never used as node features:
    targets = resolve(target_classes)
    excluded = set(resolve(ignore_classes))
    excluded |= {cls for cls in classes if closure({cls}) & set(targets)}

    asserted: List[Set[Any]] = [set() for _ in range(num_nodes)]
    for subj, obj in graph.subject_objects(RDF.type):
        if subj in node_index and obj in classes:
            asserted[node_index[subj]].add(obj)

    # Edges ###################################################################
    triples: List[Tuple[int, Any, int]] = []
    for subj, pred, obj in graph:
        if subj in node_index and obj in node_index and not is_builtin(pred):
            triples.append((node_index[subj], pred, node_index[obj]))

    relations = sorted({pred for _, pred, _ in triples}, key=str)
    relation_index = {rel: r for r, rel in enumerate(relations)}
    num_relations = len(relations) * (2 if add_inverse_edges else 1)

    edge_list = []
    for src, pred, dst in triples:
        r = relation_index[pred]
        if add_inverse_edges:
            edge_list.append([src, dst, 2 * r])
            edge_list.append([dst, src, 2 * r + 1])
        else:
            edge_list.append([src, dst, r])

    if len(edge_list) > 0:
        edge = torch.tensor(edge_list, dtype=torch.long).t()
        key = (edge[0] * num_nodes + edge[1]) * num_relations + edge[2]
        edge = edge[:, key.argsort()]
        edge_index, edge_type = edge[:2].contiguous(), edge[2].contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)

    # Node features ###########################################################
    columns: List[Tensor] = []
    feature_names: List[str] = []
    feature_encoding: List[str] = []

    if 'types' in features:
        feature_types = [closure(t - excluded) - excluded for t in asserted]
        type_vocab = sorted(set().union(*feature_types), key=str)
        type_index = {cls: j for j, cls in enumerate(type_vocab)}
        type_x = torch.zeros(num_nodes, len(type_vocab))
        for i, node_feature_types in enumerate(feature_types):
            for cls in node_feature_types:
                type_x[i, type_index[cls]] = 1.0
        columns.append(type_x)
        feature_names += [str(cls) for cls in type_vocab]
        feature_encoding += ['class'] * len(type_vocab)

    if 'data_properties' in features:
        values: Dict[Any, Dict[int, Any]] = defaultdict(dict)
        for subj, pred, obj in graph:
            if subj in node_index and isinstance(obj, Literal):
                if pred in data_properties or not is_builtin(pred):
                    value = obj.toPython()
                    if isinstance(value, (bool, int, float, Decimal)):
                        values[pred].setdefault(node_index[subj], value)

        for prop in sorted(values.keys(), key=str):
            if prop in schema_entities and prop not in data_properties:
                continue  # Skip annotation properties.
            column = torch.zeros(num_nodes, 1)
            is_bool = all(isinstance(v, bool) for v in values[prop].values())
            for i, value in values[prop].items():
                column[i, 0] = float(value)
            columns.append(column)
            feature_names.append(str(prop))
            feature_encoding.append('boolean' if is_bool else 'numeric')

    x: Optional[Tensor] = None
    if len(columns) > 0 and sum(c.size(1) for c in columns) > 0:
        x = torch.cat(columns, dim=1)

    # Labels ##################################################################
    y: Optional[Tensor] = None
    if len(targets) > 0:
        node_types = [closure(t) for t in asserted]
        if len(targets) == 1:
            y = torch.tensor([int(targets[0] in t) for t in node_types],
                             dtype=torch.long)
        else:
            y = torch.full((num_nodes, ), -1, dtype=torch.long)
            num_ambiguous = 0
            for i, t in enumerate(node_types):
                labels = [c for c, target in enumerate(targets) if target in t]
                if len(labels) > 0:
                    y[i] = labels[0]
                num_ambiguous += int(len(labels) > 1)
            if num_ambiguous > 0:
                warnings.warn(
                    f"Found {num_ambiguous} individual(s) that belong to "
                    f"more than one target class. They are labeled with the "
                    f"first matching class.", stacklevel=2)

    out_schema = OWLSchema(
        base_iri=owl_schema.base_iri,
        node_class=None,
        relation_names=[str(rel) for rel in relations],
        inverse_edge_types=add_inverse_edges,
        feature_names=feature_names,
        feature_encoding=feature_encoding or 'auto',
        label_names=[str(t) for t in targets] if len(targets) > 1 else None,
        target_classes=[str(t) for t in targets] or None,
        ignore_classes=sorted(str(c) for c in excluded - set(targets)) or None,
    )

    data = OWLData(x=x, edge_index=edge_index, edge_type=edge_type, y=y,
                   schema=out_schema, source=path)
    data.node_iri = [str(node) for node in nodes]
    data.num_nodes = num_nodes
    return data
