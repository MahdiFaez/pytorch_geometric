import contextlib
import io
import logging
import os.path as osp
import tempfile
from collections import defaultdict
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor

from torch_geometric.data import (
    Batch,
    Data,
    HeteroData,
    OWLData,
    OWLHeteroData,
    OWLSchema,
)
from torch_geometric.data.owl_data import (
    PYG_HAS_INCOMING_EDGE,
    PYG_HAS_NODE,
    PYG_HAS_OUTGOING_EDGE,
    is_absolute_iri,
    local_name,
    to_owl_schema,
)
from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel


@contextlib.contextmanager
def _unique_primitive_names() -> Iterator[None]:
    r"""Makes the names of DEAP primitives and terminals unique while the
    context is active.

    :class:`ontolearn.concept_learner.EvoLearner` names its DEAP primitives
    and terminals after the local names of IRIs, and DEAP raises an error if
    two of them share a name, *e.g.*, if two object properties are called
    :obj:`next` in different namespaces. The OWL entities themselves are not
    renamed, so learned class expressions are not affected.
    """
    try:
        from deap import gp
    except ImportError:  # Learners that are not based on DEAP.
        yield
        return

    add_primitive = gp.PrimitiveSetTyped.addPrimitive
    add_terminal = gp.PrimitiveSetTyped.addTerminal

    def unique(pset: Any, name: str) -> str:
        candidate, suffix = name, 2
        while candidate in pset.context:
            candidate, suffix = f'{name}_{suffix}', suffix + 1
        return candidate

    def primitive(pset: Any, function: Any, in_types: Any, ret_type: Any,
                  name: Optional[str] = None) -> None:
        name = name.value if isinstance(name, Enum) else name
        if name in pset.context and pset.context[name] is not function:
            name = unique(pset, name)
        add_primitive(pset, function, in_types, ret_type, name=name)

    def terminal(pset: Any, value: Any, ret_type: Any,
                 name: Optional[str] = None) -> None:
        if name is not None and name in pset.context:
            if pset.context[name] == value:
                # Equal literals of data properties with equal local names
                # can share a name, which EvoLearner relies on when refitting:
                del pset.context[name]
            else:
                name = unique(pset, name)
        add_terminal(pset, value, ret_type, name=name)

    gp.PrimitiveSetTyped.addPrimitive = primitive  # type: ignore
    gp.PrimitiveSetTyped.addTerminal = terminal  # type: ignore
    try:
        yield
    finally:
        gp.PrimitiveSetTyped.addPrimitive = add_primitive  # type: ignore
        gp.PrimitiveSetTyped.addTerminal = add_terminal  # type: ignore


class OntolearnExplainer(ExplainerAlgorithm):
    r"""An explainer that explains the predictions of a GNN by an OWL class
    expression (a *concept*), learned by an
    `Ontolearn <https://github.com/dice-group/Ontolearn>`_ concept learner.

    In contrast to mask-based explainers, the explanation is a human-readable
    logical description of the nodes that are assigned to a class, *e.g.*,
    :obj:`Female ⊓ (∃ hasSibling.Parent)` for the class :obj:`"Aunt"`.
    Concept learners operate on an ontology, which is given by an
    :class:`~torch_geometric.data.OWLData` object (either read from an OWL
    file or converted from a :class:`~torch_geometric.data.Data` object via
    an :class:`~torch_geometric.data.OWLSchema`).

    The learning problem is derived from the :obj:`target` provided by the
    :class:`~torch_geometric.explain.Explainer`: nodes of the explained class
    are positive examples, all other nodes are negative examples.
    Thus, :obj:`explanation_type="model"` explains the predictions of the
    GNN, while :obj:`explanation_type="phenomenon"` explains the ground-truth
    labels.

    The returned :class:`~torch_geometric.explain.Explanation` holds

    * :obj:`concept`: The best class expression in description logic syntax.
    * :obj:`concepts`: The best :obj:`num_hypotheses` class expressions.
    * :obj:`concept_class`: The explained class.
    * :obj:`concept_accuracy` and :obj:`concept_f1`: The quality of
      :obj:`concept` on the examples of the learning problem.
    * :obj:`node_mask`: A :obj:`[num_nodes, 1]` mask that is :obj:`1` for all
      nodes that are instances of :obj:`concept`. This allows to evaluate
      the concept with the metrics in :mod:`torch_geometric.explain.metric`.
    * :obj:`concept_agreement`: A boolean :obj:`[num_nodes]` vector that is
      :obj:`True` where the decision of the concept (whether a node is an
      instance) agrees with :obj:`target`, *i.e.*, where the concept is a
      faithful explanation of the node's (predicted) class.
    * :obj:`witness_mask`: A :obj:`[num_nodes, 1]` mask of the *witnesses*
      of the explained nodes in :obj:`index` (of all instances if
      :obj:`index` is not given), *i.e.*, the nodes that make them instances
      of :obj:`concept`. For :obj:`Female ⊓ (∃ hasSibling.Parent)`, these are
      the siblings that are parents. Together with :obj:`concept`, this is
      the local explanation of the nodes in :obj:`index`. Negated parts
      (:obj:`¬`) and upper bounds (:obj:`≤ n`) have no witnesses.
    * :obj:`edge_mask` (if :obj:`edge_mask_type="object"`): A
      :obj:`[num_edges]` mask of the witness edges, *i.e.*, the edges that
      represent the object property assertions the concept follows (and
      their inverse edges), or whose (reified) edge individuals are
      witnesses.

    .. code-block:: python

        from torch_geometric.data import OWLData
        from torch_geometric.explain import Explainer
        from torch_geometric.explain.algorithm import OntolearnExplainer

        data = OWLData.from_owl('family.owl')

        explainer = Explainer(
            model=model,
            algorithm=OntolearnExplainer('EvoLearner', data=data,
                                         max_runtime=30),
            explanation_type='model',
            node_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw',
            ),
        )
        explanation = explainer(data.x, data.edge_index, index=0,
                                edge_type=data.edge_type)
        print(explanation.concept)

    For heterogeneous graphs, :obj:`data` is an
    :class:`~torch_geometric.data.OWLHeteroData` object, :obj:`node_type`
    is the node type that the model makes predictions for, and the returned
    :class:`~torch_geometric.explain.HeteroExplanation` holds one
    :obj:`node_mask` and :obj:`witness_mask` per node type (and one
    :obj:`edge_mask` per edge type), and :obj:`concept_agreement` is stored
    for :obj:`node_type`.

    For graph-level tasks (:obj:`task_level="graph"`), the inputs are a
    batch of graphs (passing :obj:`batch` to the explainer), and all graphs
    are converted to one ontology in which every graph is an individual of
    :obj:`pyg:Graph` (see :meth:`~torch_geometric.utils.to_owl`). The graphs
    of the explained class are positive examples, all other graphs negative
    examples, so the concept describes graphs, *e.g.*,
    :obj:`∃ hasAtom.(N ⊓ ∃ hasOutgoingEdge.Double)`. :obj:`index`,
    :obj:`example_index` and :obj:`target_class` then refer to graphs, and
    :obj:`data` can be a graph dataset (*e.g.*, the list returned by
    :meth:`~torch_geometric.utils.from_owl` with :obj:`graph_level=True`).
    The returned :class:`~torch_geometric.explain.Explanation` holds

    * :obj:`graph_mask`: A :obj:`[num_graphs, 1]` mask that is :obj:`1` for
      all graphs that are instances of :obj:`concept`, *i.e.*, the decision of
      the concept for every graph.
    * :obj:`concept_agreement`: A boolean :obj:`[num_graphs]` vector that is
      :obj:`True` where the decision of the concept agrees with
      :obj:`target`, *i.e.*, where the concept is a faithful explanation of
      the graph's (predicted) class.
    * :obj:`node_mask`: A :obj:`[num_nodes, 1]` mask of the *witnesses* of
      :obj:`concept` in all graphs that are instances, *i.e.*, the nodes that
      make a graph an instance of the concept. For
      :obj:`∃ hasAtom.(N ⊓ ∃ bond.O)`, these are the nitrogen atoms bonded to
      an oxygen atom, and those oxygen atoms.
    * :obj:`edge_mask` (if :obj:`edge_mask_type="object"`): The witness
      edges, as for node-level tasks.

    .. note::

        This explainer requires the :obj:`ontolearn` package.
        Concept learners explain a class globally, *i.e.*, the explanation is
        shared by all nodes (or graphs) of that class. If :obj:`index` is
        given, the class of the nodes (or graphs) in :obj:`index` is
        explained.

    Args:
        learner (str or type or Any, optional): The concept learner to use.
            Can be the name of a supported learner (:obj:`"EvoLearner"`,
            :obj:`"CELOE"`, :obj:`"OCEL"`), an Ontolearn learner class, or an
            already initialized learner instance (which then needs to be
            bound to a knowledge base of the same graph).
            (default: :obj:`"EvoLearner"`)
        data (OWLData or OWLHeteroData or List[Data], optional): The graph to
            explain, or the graph dataset to explain for graph-level tasks
            (in the order of the input batch). If it was read from an OWL
            file, the original ontology is used, including its full TBox.
            If set to :obj:`None`, an ontology is derived from the inputs of
            the explainer and :obj:`schema`. (default: :obj:`None`)
        schema (OWLSchema or Dict[str, Any], optional): The schema used to
            convert the inputs to OWL in case :obj:`data` is not given.
            (default: :obj:`None`)
        ignore_classes (List[str], optional): OWL classes the concept learner
            must not use, in addition to the target and ignored classes of
            the schema. (default: :obj:`None`)
        target_class (int, optional): The class to explain if :obj:`index` is
            not given. For binary classification, defaults to :obj:`1`.
            (default: :obj:`None`)
        num_hypotheses (int, optional): The number of class expressions to
            return. (default: :obj:`3`)
        example_index (torch.Tensor, optional): The nodes (or graphs, for
            graph-level tasks), given as indices or a boolean mask, used as
            examples of the learning problem. If set to :obj:`None`, all
            labeled nodes (or graphs) are used. (default: :obj:`None`)
        max_examples (int, optional): If set, samples at most this many
            positive and negative examples each. (default: :obj:`None`)
        balanced (bool, optional): If set to :obj:`True`, samples the same
            number of positive and negative examples. (default: :obj:`False`)
        seed (int, optional): The seed used for sampling examples.
            (default: :obj:`None`)
        node_type (str, optional): The node type to explain in heterogeneous
            graphs. Can be omitted for graphs with a single node type.
            (default: :obj:`None`)
        card_limit (int, optional): The largest number :obj:`n` the concept
            learner uses in cardinality restrictions such as
            :obj:`≥ n hasAtom.C`. For :obj:`"EvoLearner"`, it is passed as its
            :obj:`card_limit`. For :obj:`"CELOE"` and :obj:`"OCEL"`, it is
            passed to their refinement operator, which never uses a number
            larger than the maximum number of fillers in the knowledge base.
            If set to :obj:`None`, the learner's default is used (:obj:`10` in
            Ontolearn), which truncates concepts that need larger numbers,
            *e.g.*, molecule sizes. (default: :obj:`None`)
        **kwargs (optional): Additional arguments passed to the concept
            learner, *e.g.*, :obj:`max_runtime`.
    """
    SUPPORTED_LEARNERS = ['EvoLearner', 'CELOE', 'OCEL']

    def __init__(
        self,
        learner: Union[str, Any] = 'EvoLearner',
        data: Optional[Union[Data, HeteroData, Sequence[Data]]] = None,
        schema: Optional[Union[OWLSchema, Dict[str, Any]]] = None,
        ignore_classes: Optional[Sequence[str]] = None,
        target_class: Optional[int] = None,
        num_hypotheses: int = 3,
        example_index: Optional[Tensor] = None,
        max_examples: Optional[int] = None,
        balanced: bool = False,
        seed: Optional[int] = None,
        node_type: Optional[str] = None,
        card_limit: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()

        if isinstance(learner, str):
            names = {name.lower(): name for name in self.SUPPORTED_LEARNERS}
            if learner.lower() not in names:
                raise ValueError(f"{self.__class__.__name__} does not "
                                 f"support the learner '{learner}' (expected "
                                 f"one of {self.SUPPORTED_LEARNERS})")
            learner = names[learner.lower()]
        graph_data: Optional[List[Data]] = None
        if isinstance(data, Batch):
            graph_data = data.to_data_list()  # type: ignore
            data = None
        elif isinstance(data, HeteroData):
            if not isinstance(data, OWLHeteroData):
                data = OWLHeteroData.from_data(data, schema)
        elif isinstance(data, Data):
            if not isinstance(data, OWLData):
                data = OWLData.from_data(data, schema)
        elif data is not None:  # A graph dataset:
            graph_data = list(data)
            if len(graph_data) == 0 or any(not isinstance(item, Data)
                                           for item in graph_data):
                raise ValueError("Expected 'data' to be a non-empty list of "
                                 "'Data' objects for graph datasets")
            data = None
        if num_hypotheses < 1:
            raise ValueError(f"'num_hypotheses' needs to be positive (got "
                             f"{num_hypotheses})")
        if card_limit is not None:
            if card_limit < 1:
                raise ValueError(f"'card_limit' needs to be positive (got "
                                 f"{card_limit})")
            if not isinstance(learner, (str, type)):
                raise ValueError("'card_limit' cannot be applied to an "
                                 "already initialized learner")
            if 'refinement_operator' in kwargs:
                raise ValueError("Pass either 'card_limit' or a "
                                 "'refinement_operator'")

        self.learner = learner
        self.data: Optional[Union[OWLData, OWLHeteroData]] = data
        self.graph_data = graph_data
        self.schema = None if schema is None else to_owl_schema(schema)
        self.ignore_classes = list(ignore_classes or [])
        self.target_class = target_class
        self.num_hypotheses = num_hypotheses
        self.example_index = example_index
        self.max_examples = max_examples
        self.balanced = balanced
        self.seed = seed
        self.node_type = node_type
        self.card_limit = card_limit
        self.learner_kwargs = kwargs

        self._tmp_dir: Any = None
        self._kb_key: Optional[Tuple[Any, ...]] = None
        self._kb: Any = None
        self._concept_learner: Any = None
        # The IRIs of the explained nodes (or graphs), and the node type and
        # index of every individual (with node type `None` in homogeneous
        # graphs, and the graph index for graph-level tasks):
        self._example_iri: List[str] = []
        self._iri_index: Dict[str, Tuple[Optional[str], int]] = {}
        # The IRIs of all nodes (per node type, with node type `None` in
        # homogeneous graphs), and the node type and index of every node:
        self._node_iris: Dict[Optional[str], List[str]] = {}
        self._node_index: Dict[str, Tuple[Optional[str], int]] = {}
        self._node_iri: List[str] = []
        # The schema and edge types of the graph the knowledge base was built
        # from, which map edges to the facts they represent:
        self._kb_schema: OWLSchema = OWLSchema()
        self._kb_edge_type: Optional[Tensor] = None
        self._cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._instance_cache: Dict[Any, Set[str]] = {}
        self._filler_cache: Dict[Tuple[str, Any], List[str]] = {}
        self._edge_endpoints: Optional[Dict[str, Tuple[str, str]]] = None

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[str, Tensor]],
        edge_index: Union[Tensor, Dict[Any, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs: Any,
    ) -> Union[Explanation, HeteroExplanation]:
        if self.model_config.task_level == ModelTaskLevel.graph:
            return self._forward_graph(x, edge_index, target, index, kwargs)

        node_type = self._get_node_type(x)
        if isinstance(x, dict):
            num_nodes = x[node_type].size(0)
        else:
            num_nodes = x.size(0)

        if target.dim() > 1:
            target = target.view(-1) if target.size(-1) == 1 else target
        if target.dim() != 1 or target.numel() != num_nodes:
            raise ValueError(f"Expected one target per node (got a target "
                             f"of shape {tuple(target.size())})")
        target = target.detach().cpu().long()

        self._prepare_knowledge_base(x, edge_index, node_type, kwargs)
        cls = self._get_class(target, index)
        pos, neg = self._get_examples(target, cls)

        key = (cls, tuple(pos.tolist()), tuple(neg.tolist()))
        if key not in self._cache:
            self._cache[key] = self._learn(pos, neg)
        result = self._cache[key]

        def node_mask(size: int, members: Optional[Tensor],
                      device: torch.device) -> Tensor:
            mask = torch.zeros(size, 1, device=device)
            if members is not None:
                mask[members.to(device)] = 1.0
            return mask

        # The witnesses of the explained nodes that are instances (of all
        # instances if no `index` is given):
        members = result['members'].get(node_type)
        if members is None:
            members = torch.empty(0, dtype=torch.long)
        is_member = torch.zeros(num_nodes, dtype=torch.bool)
        is_member[members] = True
        if index is not None:
            roots = torch.as_tensor(index, dtype=torch.long).view(-1).cpu()
            roots = roots[is_member[roots]]
        else:
            roots = members
        individuals, facts = self._get_witnesses(
            result['class_expressions'][0],
            [self._example_iri[i] for i in roots.tolist()])
        witness_masks, edge_masks = self._get_witness_masks(
            individuals, facts, x, edge_index, kwargs.get('edge_type'))
        agreement = is_member == (target == cls)

        explanation: Union[Explanation, HeteroExplanation]
        if isinstance(x, dict):
            explanation = HeteroExplanation()
            explanation.set_value_dict(
                'node_mask', {
                    key:
                    node_mask(value.size(0), result['members'].get(key),
                              value.device)
                    for key, value in x.items()
                })
            explanation.set_value_dict(
                'witness_mask', {
                    key: witness_masks[key].to(value.device)
                    for key, value in x.items()
                })
            if edge_masks is not None:
                assert isinstance(edge_index, dict)
                explanation.set_value_dict(
                    'edge_mask', {
                        key: edge_masks[key].to(value.device)
                        for key, value in edge_index.items()
                    })
            explanation[node_type].concept_agreement = agreement.to(
                x[node_type].device)
        else:
            explanation = Explanation(node_mask=node_mask(
                num_nodes, result['members'].get(None), x.device))
            explanation.witness_mask = witness_masks[None].to(x.device)
            if edge_masks is not None:
                explanation.edge_mask = edge_masks[None].to(x.device)
            explanation.concept_agreement = agreement.to(x.device)

        explanation.concept = result['concepts'][0]
        explanation.concepts = result['concepts']
        explanation.concept_class = cls
        explanation.concept_accuracy = result['accuracy']
        explanation.concept_f1 = result['f1']
        explanation._class_expressions = result['class_expressions']
        return explanation

    def supports(self) -> bool:
        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level and graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False

        mode = self.model_config.mode
        if mode not in {
                ModelMode.binary_classification,
                ModelMode.multiclass_classification,
        }:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"classification tasks got (`mode={mode.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type != MaskType.object:
            value = None if node_mask_type is None else node_mask_type.value
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node masks of type 'object' "
                          f"got (`node_mask_type={value}`)")
            return False

        # Edge masks (which can only be of type 'object') hold the edges of
        # the witnesses of the concept.

        return True

    ###########################################################################

    def _get_node_type(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
    ) -> Optional[str]:
        r"""Returns the explained node type of heterogeneous graphs."""
        if not isinstance(x, dict):
            return None
        if self.node_type is not None:
            if self.node_type not in x:
                raise ValueError(f"The node type '{self.node_type}' is not "
                                 f"part of the input (got {list(x.keys())})")
            return self.node_type
        if len(x) == 1:
            return next(iter(x.keys()))
        raise ValueError(f"'{self.__class__.__name__}' requires 'node_type' "
                         f"to be set for heterogeneous graphs with more than "
                         f"one node type")

    def _prepare_knowledge_base(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
        edge_index: Union[Tensor, Dict[Any, Tensor]],
        node_type: Optional[str],
        kwargs: Dict[str, Any],
    ) -> None:
        r"""Builds (or re-uses) the knowledge base and concept learner."""
        if isinstance(x, dict):
            if not isinstance(edge_index, dict):
                raise ValueError("Expected 'edge_index' to be a dictionary "
                                 "for heterogeneous graphs")
            if self.data is not None and not isinstance(
                    self.data, OWLHeteroData):
                raise ValueError("Expected 'data' to be heterogeneous for "
                                 "heterogeneous inputs")
            if self.data is not None:
                for key, value in x.items():
                    num_nodes = self.data[key].num_nodes
                    if num_nodes != value.size(0):
                        raise ValueError(
                            f"The number of nodes of type '{key}' in 'data' "
                            f"({num_nodes}) does not match the number of "
                            f"nodes in the input ({value.size(0)})")
            tensors = list(x.values()) + list(edge_index.values())
        else:
            if isinstance(self.data, OWLHeteroData):
                raise ValueError("Expected 'data' to be homogeneous for "
                                 "homogeneous inputs")
            if self.data is not None and self.data.num_nodes != x.size(0):
                raise ValueError(f"The number of nodes in 'data' "
                                 f"({self.data.num_nodes}) does not match the "
                                 f"number of nodes in the input ({x.size(0)})")
            assert isinstance(edge_index, Tensor)
            tensors = [x, edge_index]

        if self.data is not None:
            key: Tuple[Any, ...] = (id(self.data), node_type)
        else:
            key = (node_type, ) + tuple(
                (t.data_ptr(), tuple(t.size())) for t in tensors)
        if key == self._kb_key:
            return

        data: Union[OWLData, OWLHeteroData, None] = self.data
        if data is None and isinstance(x, dict):
            assert isinstance(edge_index, dict)
            hetero = HeteroData()
            for key_, value in x.items():
                hetero[key_].x = value.detach().cpu()
            for key_, value in edge_index.items():
                hetero[key_].edge_index = value.detach().cpu()
            data = OWLHeteroData.from_data(hetero, self.schema)
        elif data is None:
            assert isinstance(x, Tensor) and isinstance(edge_index, Tensor)
            data = OWLData.from_data(
                Data(x=x.detach().cpu(), edge_index=edge_index.detach().cpu(),
                     edge_type=kwargs.get('edge_type'),
                     edge_attr=kwargs.get('edge_attr'), num_nodes=x.size(0)),
                self.schema,
            )
        assert data is not None

        node_iris: Dict[Optional[str], List[str]] = {}
        if isinstance(data, OWLHeteroData):
            for key_, store in data.node_items():
                node_iris[key_] = list(store.node_iri)
        else:
            node_iris[None] = list(data.node_iri)
        iri_index = {
            iri: (key_, i)
            for key_, iris in node_iris.items()
            for i, iri in enumerate(iris)
        }
        node_iri = node_iris[node_type]

        self._load_knowledge_base(data.source, data.save_owl, data.schema,
                                  list(iri_index.keys()))
        self._kb_edge_type = None
        if isinstance(data, OWLData) and data.edge_index is not None:
            self._kb_edge_type = self._edge_type_of(data)
        self._example_iri = node_iri
        self._iri_index = iri_index
        self._node_iris = node_iris
        self._node_index = iri_index
        self._node_iri = list(iri_index.keys())
        self._kb_key = key

    def _load_knowledge_base(
        self,
        source: Optional[str],
        save: Any,
        schema: OWLSchema,
        iris: List[str],
    ) -> None:
        r"""Loads the knowledge base from :obj:`source` (or from the file
        written by :obj:`save(path)` if :obj:`source` does not exist), and
        initializes the concept learner.
        """
        try:
            from ontolearn.knowledge_base import KnowledgeBase
        except ImportError as e:
            raise ImportError(f"'{self.__class__.__name__}' requires the "
                              f"'ontolearn' package (install it via "
                              f"`pip install ontolearn`)") from e

        # `owlapy` rejects IRIs without a "/", *e.g.*, "urn:" IRIs:
        invalid = [iri for iri in iris if '/' not in iri]
        if len(invalid) > 0:
            raise ValueError(f"Ontolearn only supports IRIs that contain a "
                             f"'/' (got '{invalid[0]}'). Use an 'http(s)' "
                             f"IRI as 'base_iri' of the schema instead.")

        path = source
        if path is None or not osp.exists(path):
            self._tmp_dir = tempfile.TemporaryDirectory()
            path = save(osp.join(self._tmp_dir.name, 'graph.owl'))

        with contextlib.redirect_stdout(io.StringIO()):
            kb = KnowledgeBase(path=osp.abspath(path))

        # Remove target classes and ignored classes from the knowledge base.
        # Names are either IRIs, or local names of classes in the ontology:
        concepts = {c.str: c for c in kb.get_concepts()}
        by_name: Dict[str, List[Any]] = defaultdict(list)
        for iri, concept in concepts.items():
            by_name[local_name(iri)].append(concept)

        ignored = set()
        names = schema.excluded_classes() + self.ignore_classes
        for name in names:
            if name in concepts:
                ignored.add(concepts[name])
            elif schema.iri(name) in concepts:
                ignored.add(concepts[schema.iri(name)])
            elif not is_absolute_iri(name) and len(by_name[name]) == 1:
                ignored.add(by_name[name][0])
            elif name in self.ignore_classes:
                raise ValueError(f"Could not find the OWL class '{name}' in "
                                 f"the knowledge base")
        # `pyg:hasNode` holds no assertions if graphs are linked to their
        # nodes via a sub property (*e.g.*, `hasAtom`), and the structural
        # reasoner of Ontolearn does not infer them. Then, it is removed so
        # that the concept learner does not search over it:
        from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLThing
        ignored_properties = [
            prop for prop in kb.get_object_properties()
            if prop.str == PYG_HAS_NODE and kb.individuals_count(
                OWLObjectSomeValuesFrom(prop, OWLThing)) == 0
        ]
        if len(ignored) > 0 or len(ignored_properties) > 0:
            kb = kb.ignore_and_copy(
                ignored_classes=ignored or None,
                ignored_object_properties=ignored_properties or None)

        if isinstance(self.learner, str):
            if self.learner == 'EvoLearner':
                from ontolearn.concept_learner import EvoLearner as cls
            else:
                import ontolearn.learners
                cls = getattr(ontolearn.learners, self.learner)
            with _unique_primitive_names():
                learner = cls(knowledge_base=kb,
                              **self._get_learner_kwargs(cls, kb))
        elif isinstance(self.learner, type):
            with _unique_primitive_names():
                learner = self.learner(
                    knowledge_base=kb,
                    **self._get_learner_kwargs(self.learner, kb))
        else:  # An already initialized learner:
            learner = self.learner
            kb = getattr(learner, 'kb', kb)

        self._kb, self._concept_learner = kb, learner
        self._kb_schema = schema
        self._cache = {}
        self._instance_cache = {}
        self._filler_cache = {}
        self._edge_endpoints = None

    def _get_learner_kwargs(self, cls: type, kb: Any) -> Dict[str, Any]:
        r"""Returns the arguments of learner :obj:`cls`, including
        :obj:`card_limit`.
        """
        kwargs = dict(self.learner_kwargs)
        if self.card_limit is None:
            return kwargs

        from ontolearn import base_concept_learner, refinement_operators
        from ontolearn.concept_learner import EvoLearner

        if issubclass(cls, EvoLearner):
            kwargs['card_limit'] = self.card_limit
        elif issubclass(cls,
                        base_concept_learner.RefinementBasedConceptLearner):
            # The default operator of CELOE and OCEL, with another limit:
            operator = refinement_operators.ModifiedCELOERefinement
            kwargs['refinement_operator'] = operator(
                kb, max_child_length=kwargs.get('max_child_length') or 10,
                card_limit=self.card_limit)
        else:
            raise ValueError(f"'card_limit' is not supported for the learner "
                             f"'{cls.__name__}'")
        return kwargs

    # Graph-level tasks #######################################################

    def _forward_graph(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
        edge_index: Union[Tensor, Dict[Any, Tensor]],
        target: Tensor,
        index: Optional[Union[int, Tensor]],
        kwargs: Dict[str, Any],
    ) -> Explanation:
        r"""Explains the graphs of a batch, see the class documentation."""
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"graph-level explanations of heterogeneous "
                             f"graphs yet")
        batch = kwargs.get('batch')
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        batch = batch.detach().cpu().view(-1)
        if batch.numel() != x.size(0):
            raise ValueError(f"Expected 'batch' to hold one entry per node "
                             f"(got {batch.numel()} entries for {x.size(0)} "
                             f"nodes)")

        if target.dim() > 1:
            target = target.view(-1) if target.size(-1) == 1 else target
        num_graphs = target.numel()
        if target.dim() != 1 or (batch.numel() > 0
                                 and int(batch.max()) >= num_graphs):
            raise ValueError(f"Expected one target per graph (got a target "
                             f"of shape {tuple(target.size())})")
        target = target.detach().cpu().long()

        self._prepare_graph_knowledge_base(x, edge_index, batch, num_graphs,
                                           kwargs)
        cls = self._get_class(target, index)
        pos, neg = self._get_examples(target, cls)

        key = (cls, tuple(pos.tolist()), tuple(neg.tolist()))
        if key not in self._cache:
            self._cache[key] = self._learn(pos, neg)
        result = self._cache[key]

        members = result['members'].get(None)
        if members is None:
            members = torch.empty(0, dtype=torch.long)
        graph_mask = torch.zeros(num_graphs, 1)
        graph_mask[members] = 1.0

        # Witnesses are computed once per concept and batch:
        if 'witnesses' not in result:
            result['witnesses'] = self._get_witnesses(
                result['class_expressions'][0],
                [self._example_iri[i] for i in members.tolist()])
        node_masks, edge_masks = self._get_witness_masks(
            *result['witnesses'], x, edge_index, kwargs.get('edge_type'))

        explanation = Explanation(node_mask=node_masks[None].to(x.device))
        if edge_masks is not None:
            explanation.edge_mask = edge_masks[None].to(x.device)
        explanation.graph_mask = graph_mask.to(x.device)
        agreement = graph_mask.view(-1).bool() == (target == cls)
        explanation.concept_agreement = agreement.to(x.device)

        explanation.concept = result['concepts'][0]
        explanation.concepts = result['concepts']
        explanation.concept_class = cls
        explanation.concept_accuracy = result['accuracy']
        explanation.concept_f1 = result['f1']
        explanation._class_expressions = result['class_expressions']
        return explanation

    def _prepare_graph_knowledge_base(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        num_graphs: int,
        kwargs: Dict[str, Any],
    ) -> None:
        r"""Builds (or re-uses) the knowledge base of a batch of graphs, in
        which every graph is an individual of :obj:`pyg:Graph`.
        """
        from torch_geometric.utils import to_owl

        if self.data is not None:
            raise ValueError("Expected 'data' to be a graph dataset (a list "
                             "of 'Data' objects) for graph-level tasks")
        graph_data = self.graph_data
        if graph_data is not None:
            if len(graph_data) != num_graphs:
                raise ValueError(f"The number of graphs in 'data' "
                                 f"({len(graph_data)}) does not match the "
                                 f"number of graphs in the input "
                                 f"({num_graphs})")
            num_nodes = sum(int(item.num_nodes or 0) for item in graph_data)
            if num_nodes != x.size(0):
                raise ValueError(f"The number of nodes in 'data' "
                                 f"({num_nodes}) does not match the number "
                                 f"of nodes in the input ({x.size(0)})")
            key: Tuple[Any, ...] = ('graph', id(graph_data))
        else:
            tensors = [x, edge_index, kwargs.get('batch')]
            key = ('graph', ) + tuple(
                None if t is None else (t.data_ptr(), tuple(t.size()))
                for t in tensors)
        if key == self._kb_key:
            return

        source: Optional[str] = None
        if graph_data is not None:
            data = Batch.from_data_list(graph_data)
            first = graph_data[0]
            schema = to_owl_schema(
                first.schema if isinstance(first, OWLData) else self.schema)
            sources = {
                item.source if isinstance(item, OWLData) else None
                for item in graph_data
            }
            if len(sources) == 1:
                source = sources.pop()
        else:
            attrs = {
                name: kwargs[name].detach().cpu()
                for name in ['edge_type', 'edge_attr', 'graph_attr']
                if isinstance(kwargs.get(name), Tensor)
            }
            data = Batch(x=x.detach().cpu(),
                         edge_index=edge_index.detach().cpu(), batch=batch,
                         **attrs)
            schema = to_owl_schema(self.schema)
        assert isinstance(data, Data)
        counts = torch.bincount(batch, minlength=num_graphs)
        data.ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

        node_iri = list(data.node_iri) if 'node_iri' in data else []
        if len(node_iri) > 0 and isinstance(node_iri[0], (list, tuple)):
            node_iri = [iri for iris in node_iri for iri in iris]
        if len(node_iri) != x.size(0):
            node_iri = [schema.node_iri(i) for i in range(x.size(0))]
        graph_iri = list(data.graph_iri) if 'graph_iri' in data else []
        if len(graph_iri) != num_graphs:
            graph_iri = [schema.graph_iri(i) for i in range(num_graphs)]
        data.node_iri, data.graph_iri = node_iri, graph_iri

        def save(path: str) -> str:
            to_owl(data, schema).serialize(destination=path, format='xml')
            return path

        self._load_knowledge_base(source, save, schema, node_iri + graph_iri)
        self._kb_edge_type = self._edge_type_of(data)
        self._example_iri = graph_iri
        self._iri_index = {iri: (None, i) for i, iri in enumerate(graph_iri)}
        self._node_iris = {None: node_iri}
        self._node_index = {iri: (None, i) for i, iri in enumerate(node_iri)}
        self._node_iri = node_iri
        self._kb_key = key

    def _get_witnesses(
        self,
        expression: Any,
        roots: List[str],
    ) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        r"""Returns the witnesses of :obj:`expression` for the instances
        :obj:`roots` (the individuals that make them instances of
        :obj:`expression`), and the object property assertions
        :obj:`(subject, property, object)` the expression follows.
        """
        individuals: Set[str] = set()
        facts: Set[Tuple[str, str, str]] = set()
        visited: Set[Tuple[Any, str]] = set()
        for root in roots:
            self._witness(expression, root, individuals, facts, visited)
        return individuals, facts

    def _witness(
        self,
        expression: Any,
        iri: str,
        individuals: Set[str],
        facts: Set[Tuple[str, str, str]],
        visited: Set[Tuple[Any, str]],
    ) -> None:
        r"""Collects the witnesses of the instance :obj:`iri` of
        :obj:`expression`.

        * :obj:`C ⊓ D`: The witnesses of :obj:`C` and :obj:`D`.
        * :obj:`C ⊔ D`: The witnesses of the operands :obj:`iri` is an
          instance of.
        * :obj:`∃ r.C`, :obj:`≥ n r.C`, :obj:`= n r.C`, :obj:`∀ r.C`: The
          :obj:`r`-fillers of :obj:`iri` that are instances of :obj:`C`, and
          their witnesses of :obj:`C`.
        * :obj:`∃ r.{a}`: The individual :obj:`a`.
        * Classes, data property restrictions, :obj:`¬ C` and
          :obj:`≤ n r.C`: No witnesses, since they hold for :obj:`iri` itself
          or describe the absence of fillers.
        """
        from owlapy.class_expression import (
            OWLObjectHasValue,
            OWLObjectIntersectionOf,
            OWLObjectMaxCardinality,
            OWLObjectUnionOf,
            OWLQuantifiedObjectRestriction,
        )
        from owlapy.owl_property import OWLObjectInverseOf

        if (expression, iri) in visited:
            return
        visited.add((expression, iri))

        def add(prop: Any, filler: str) -> None:
            individuals.add(filler)
            if isinstance(prop, OWLObjectInverseOf):
                facts.add((filler, prop.get_named_property().str, iri))
            else:
                facts.add((iri, prop.str, filler))

        if isinstance(expression, OWLObjectIntersectionOf):
            for operand in expression.operands():
                self._witness(operand, iri, individuals, facts, visited)
        elif isinstance(expression, OWLObjectUnionOf):
            for operand in expression.operands():
                if iri in self._instances(operand):
                    self._witness(operand, iri, individuals, facts, visited)
        elif isinstance(expression, OWLObjectHasValue):
            prop, value = expression.get_property(), expression.get_filler()
            if value.str in self._fillers(iri, prop):
                add(prop, value.str)
        elif (isinstance(expression, OWLQuantifiedObjectRestriction)
              and not isinstance(expression, OWLObjectMaxCardinality)):
            prop, filler = expression.get_property(), expression.get_filler()
            instances = self._instances(filler)
            for value in self._fillers(iri, prop):
                if value in instances:
                    add(prop, value)
                    self._witness(filler, value, individuals, facts, visited)

    def _instances(self, expression: Any) -> Set[str]:
        r"""Returns the IRIs of the instances of :obj:`expression`."""
        if expression not in self._instance_cache:
            self._instance_cache[expression] = {
                individual.str
                for individual in self._kb.individuals_set(expression)
            }
        return self._instance_cache[expression]

    def _fillers(self, iri: str, prop: Any) -> List[str]:
        r"""Returns the IRIs of the :obj:`prop`-fillers of :obj:`iri`."""
        from owlapy.iri import IRI
        from owlapy.owl_individual import OWLNamedIndividual

        key = (iri, prop)
        if key not in self._filler_cache:
            individual = OWLNamedIndividual(IRI.create(iri))
            self._filler_cache[key] = [
                value.str for value in self._kb.get_object_property_values(
                    individual, prop)
            ]
        return self._filler_cache[key]

    @staticmethod
    def _edge_type_of(data: Data) -> Optional[Tensor]:
        r"""Returns the edge types of :obj:`data` (all :obj:`0` if it has
        none, as assumed by :meth:`~torch_geometric.utils.to_owl`).
        """
        if not isinstance(data.edge_index, Tensor):
            return None
        edge_type = data.edge_type if 'edge_type' in data else None
        if isinstance(edge_type, Tensor):
            return edge_type.detach().cpu().view(-1)
        return torch.zeros(data.edge_index.size(1), dtype=torch.long)

    def _get_relations(
        self,
        edge_type: Optional[Tensor],
        num_edges: int,
    ) -> Optional[List[Tuple[str, bool]]]:
        r"""Returns the object property of every edge of a homogeneous graph,
        and whether the edge points from its subject to its object (or is an
        inverse edge), or :obj:`None` if the edge types are unknown.
        """
        if not isinstance(edge_type, Tensor) or edge_type.numel() != num_edges:
            edge_type = self._kb_edge_type
        if edge_type is None or edge_type.numel() != num_edges:
            return None
        schema = self._kb_schema
        types = edge_type.view(-1).tolist()
        inverse = schema.inverse_edge_types
        num_relations = 0
        if len(types) > 0:
            num_relations = (max(types) // 2 if inverse else max(types)) + 1
        if isinstance(schema.relation_names, list):
            num_relations = max(num_relations, len(schema.relation_names))
        iris = [
            schema.relation_iri(r, num_relations) for r in range(num_relations)
        ]
        if inverse:
            return [(iris[t // 2], t % 2 == 0) for t in types]
        return [(iris[t], True) for t in types]

    def _get_witness_masks(
        self,
        individuals: Set[str],
        facts: Set[Tuple[str, str, str]],
        x: Union[Tensor, Dict[str, Tensor]],
        edge_index: Union[Tensor, Dict[Any, Tensor]],
        edge_type: Optional[Tensor] = None,
    ) -> Tuple[Dict[Any, Tensor], Optional[Dict[Any, Tensor]]]:
        r"""Returns the :obj:`[num_nodes, 1]` masks of the witness nodes per
        node type and, if edge masks are requested, the masks of the witness
        edges per edge type (with type :obj:`None` in homogeneous graphs).

        Witness edges are the edges that represent the object property
        assertions the concept follows (including inverse edges of these
        assertions), and the edges whose reified individuals are witnesses.
        """
        if isinstance(x, Tensor):
            node_masks: Dict[Any, Tensor] = {None: torch.zeros(x.size(0), 1)}
        else:
            node_masks = {
                key: torch.zeros(value.size(0), 1)
                for key, value in x.items()
            }
        for iri in individuals:
            if iri in self._node_index:
                node_type, i = self._node_index[iri]
                if node_type in node_masks:
                    node_masks[node_type][i] = 1.0

        if self.explainer_config.edge_mask_type is None:
            return node_masks, None

        # Reified edges are matched by their source and target node, and
        # edges of unknown type by the nodes of the assertion:
        reified: Set[Tuple[str, str]] = set()
        if any(iri not in self._node_index for iri in individuals):
            endpoints = self._get_edge_endpoints()
            reified = {endpoints[i] for i in individuals if i in endpoints}
        pairs = {(a, b) for a, _, b in facts}

        def edge_mask(
            index: Tensor,
            src: Optional[str],
            dst: Optional[str],
            relations: Optional[List[Tuple[str, bool]]],
        ) -> Tensor:
            src_iri = self._node_iris.get(src, [])
            dst_iri = self._node_iris.get(dst, [])
            row, col = index.detach().cpu().tolist()
            if len(row) > 0 and (max(row) >= len(src_iri)
                                 or max(col) >= len(dst_iri)):
                return torch.zeros(len(row))
            mask = torch.zeros(len(row))
            for k, (i, j) in enumerate(zip(row, col)):
                a, b = src_iri[i], dst_iri[j]
                if (a, b) in reified:
                    mask[k] = 1.0
                elif relations is None:
                    mask[k] = float((a, b) in pairs)
                else:
                    rel, forward = relations[k]
                    fact = (a, rel, b) if forward else (b, rel, a)
                    mask[k] = float(fact in facts)
            return mask

        if isinstance(edge_index, Tensor):
            relations = self._get_relations(edge_type, edge_index.size(1))
            return node_masks, {
                None: edge_mask(edge_index, None, None, relations)
            }

        schema = self._kb_schema
        edge_masks = {}
        for key, value in edge_index.items():
            src, rel, dst = key
            forward = not (schema.inverse_edge_types and rel.startswith('rev_')
                           and (dst, rel[4:], src) in edge_index)
            iri = schema.relation_iri(rel if forward else rel[4:])
            edge_masks[key] = edge_mask(value, src, dst,
                                        [(iri, forward)] * value.size(1))
        return node_masks, edge_masks

    def _get_edge_endpoints(self) -> Dict[str, Tuple[str, str]]:
        r"""Returns the source and target node of every reified edge (an
        individual of :obj:`pyg:Edge`).
        """
        if self._edge_endpoints is None:
            from owlapy.iri import IRI
            from owlapy.owl_property import OWLObjectProperty

            outgoing = OWLObjectProperty(IRI.create(PYG_HAS_OUTGOING_EDGE))
            incoming = OWLObjectProperty(IRI.create(PYG_HAS_INCOMING_EDGE))
            src: Dict[str, str] = {}
            dst: Dict[str, str] = {}
            for iri in self._node_iri:
                for edge in self._fillers(iri, outgoing):
                    src[edge] = iri
                for edge in self._fillers(iri, incoming):
                    dst[edge] = iri
            self._edge_endpoints = {
                edge: (src[edge], dst[edge])
                for edge in src if edge in dst
            }
        return self._edge_endpoints

    ###########################################################################

    def _get_class(
        self,
        target: Tensor,
        index: Optional[Union[int, Tensor]],
    ) -> int:
        r"""Returns the class to explain."""
        if index is not None:
            index = torch.as_tensor(index, dtype=torch.long).view(-1).cpu()
            classes = target[index].unique()
            if classes.numel() != 1:
                raise ValueError(f"'{self.__class__.__name__}' explains one "
                                 f"class at a time, but the examples in "
                                 f"'index' belong to the classes "
                                 f"{classes.tolist()}")
            return int(classes[0])

        if self.target_class is not None:
            return self.target_class
        if self.model_config.mode == ModelMode.binary_classification:
            return 1
        raise ValueError(f"'{self.__class__.__name__}' requires either "
                         f"'index' or 'target_class' to be set for "
                         f"multi-class classification")

    def _get_examples(self, target: Tensor, cls: int) -> Tuple[Tensor, Tensor]:
        r"""Returns the positive and negative examples of the learning
        problem.
        """
        num_nodes = target.numel()
        if self.example_index is None:
            index = torch.arange(num_nodes)
        elif self.example_index.dtype == torch.bool:
            index = self.example_index.cpu().nonzero().view(-1)
        else:
            index = self.example_index.cpu().view(-1)
        index = index[target[index] >= 0]  # Skip unlabeled nodes.

        pos = index[target[index] == cls]
        neg = index[target[index] != cls]

        generator = None
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed)

        def sample(index: Tensor, num: Optional[int]) -> Tensor:
            if num is None or index.numel() <= num:
                return index
            perm = torch.randperm(index.numel(), generator=generator)
            return index[perm[:num]].sort()[0]

        num = self.max_examples
        if self.balanced:
            num = min(pos.numel(), neg.numel(), num or num_nodes)
        pos, neg = sample(pos, num), sample(neg, num)

        if pos.numel() == 0 or neg.numel() == 0:
            raise ValueError(f"The learning problem of class {cls} requires "
                             f"at least one positive and one negative example "
                             f"(got {pos.numel()} positive and {neg.numel()} "
                             f"negative examples)")
        return pos, neg

    def _learn(self, pos: Tensor, neg: Tensor) -> Dict[str, Any]:
        r"""Fits the concept learner and evaluates the learned concepts."""
        from ontolearn.learning_problem import PosNegLPStandard
        from owlapy.iri import IRI
        from owlapy.owl_individual import OWLNamedIndividual
        from owlapy.render import DLSyntaxObjectRenderer

        def individuals(index: Tensor) -> set:
            return {
                OWLNamedIndividual(IRI.create(self._example_iri[i]))
                for i in index.tolist()
            }

        problem = PosNegLPStandard(pos=individuals(pos), neg=individuals(neg))
        with contextlib.redirect_stdout(io.StringIO()), \
                _unique_primitive_names():
            self._concept_learner.fit(problem)
            hypotheses = self._concept_learner.best_hypotheses(
                n=self.num_hypotheses)
        if not isinstance(hypotheses, (list, tuple)):
            hypotheses = [hypotheses]
        expressions = [getattr(h, 'concept', h) for h in hypotheses]
        if len(expressions) == 0:
            raise RuntimeError("The concept learner did not return any class "
                               "expression")

        instances = self._kb.individuals_set(expressions[0])
        index: Dict[Optional[str], List[int]] = defaultdict(list)
        for individual in instances:
            if individual.str in self._iri_index:
                key, i = self._iri_index[individual.str]
                index[key].append(i)
        members = {
            key: torch.tensor(sorted(value), dtype=torch.long)
            for key, value in index.items()
        }

        mapping = {iri: i for i, iri in enumerate(self._example_iri)}
        covered = torch.zeros(len(self._example_iri), dtype=torch.bool)
        for individual in instances:
            if individual.str in mapping:
                covered[mapping[individual.str]] = True
        tp = int(covered[pos].sum())
        fp = int(covered[neg].sum())
        fn = pos.numel() - tp
        tn = neg.numel() - fp
        accuracy = (tp + tn) / (pos.numel() + neg.numel())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0

        renderer = DLSyntaxObjectRenderer()
        return dict(
            concepts=[renderer.render(e) for e in expressions],
            class_expressions=expressions,
            members=members,
            accuracy=accuracy,
            f1=f1,
        )

    def __repr__(self) -> str:
        learner = self.learner
        if not isinstance(learner, str):
            learner = getattr(learner, '__name__', type(learner).__name__)
        return f'{self.__class__.__name__}({learner})'
