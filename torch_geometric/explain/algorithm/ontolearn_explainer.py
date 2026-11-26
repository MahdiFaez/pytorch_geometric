import contextlib
import io
import logging
import os.path as osp
import tempfile
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, OWLData, OWLSchema
from torch_geometric.data.owl_data import (
    is_absolute_iri,
    local_name,
    to_owl_schema,
)
from torch_geometric.explain import Explanation
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

    .. note::

        This explainer requires the :obj:`ontolearn` package.
        Concept learners explain a class globally, *i.e.*, the explanation is
        shared by all nodes of that class. If :obj:`index` is given, the class
        of the nodes in :obj:`index` is explained.

    Args:
        learner (str or type or Any, optional): The concept learner to use.
            Can be the name of a supported learner (:obj:`"EvoLearner"`,
            :obj:`"CELOE"`, :obj:`"OCEL"`), an Ontolearn learner class, or an
            already initialized learner instance (which then needs to be
            bound to a knowledge base of the same graph).
            (default: :obj:`"EvoLearner"`)
        data (OWLData, optional): The graph to explain. If it was read from an
            OWL file, the original ontology is used, including its full TBox.
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
        example_index (torch.Tensor, optional): The nodes (indices or a boolean
            mask) used as examples of the learning problem. If set to
            :obj:`None`, all labeled nodes are used. (default: :obj:`None`)
        max_examples (int, optional): If set, samples at most this many
            positive and negative examples each. (default: :obj:`None`)
        balanced (bool, optional): If set to :obj:`True`, samples the same
            number of positive and negative examples. (default: :obj:`False`)
        seed (int, optional): The seed used for sampling examples.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments passed to the concept
            learner, *e.g.*, :obj:`max_runtime`.
    """
    SUPPORTED_LEARNERS = ['EvoLearner', 'CELOE', 'OCEL']

    def __init__(
        self,
        learner: Union[str, Any] = 'EvoLearner',
        data: Optional[OWLData] = None,
        schema: Optional[Union[OWLSchema, Dict[str, Any]]] = None,
        ignore_classes: Optional[Sequence[str]] = None,
        target_class: Optional[int] = None,
        num_hypotheses: int = 3,
        example_index: Optional[Tensor] = None,
        max_examples: Optional[int] = None,
        balanced: bool = False,
        seed: Optional[int] = None,
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
        if data is not None and not isinstance(data, OWLData):
            data = OWLData.from_data(data, schema)
        if num_hypotheses < 1:
            raise ValueError(f"'num_hypotheses' needs to be positive (got "
                             f"{num_hypotheses})")

        self.learner = learner
        self.data = data
        self.schema = None if schema is None else to_owl_schema(schema)
        self.ignore_classes = list(ignore_classes or [])
        self.target_class = target_class
        self.num_hypotheses = num_hypotheses
        self.example_index = example_index
        self.max_examples = max_examples
        self.balanced = balanced
        self.seed = seed
        self.learner_kwargs = kwargs

        self._tmp_dir: Any = None
        self._kb_key: Optional[Tuple[Any, ...]] = None
        self._kb: Any = None
        self._concept_learner: Any = None
        self._node_iri: List[str] = []
        self._cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs: Any,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise NotImplementedError(f"Heterogeneous graphs are not yet "
                                      f"supported in "
                                      f"'{self.__class__.__name__}'")

        num_nodes = x.size(0)
        if target.dim() > 1:
            target = target.view(-1) if target.size(-1) == 1 else target
        if target.dim() != 1 or target.numel() != num_nodes:
            raise ValueError(f"Expected one target per node (got a target "
                             f"of shape {tuple(target.size())})")
        target = target.detach().cpu().long()

        self._prepare_knowledge_base(x, edge_index, kwargs.get('edge_type'),
                                     kwargs.get('edge_attr'))
        cls = self._get_class(target, index)
        pos, neg = self._get_examples(target, cls)

        key = (cls, tuple(pos.tolist()), tuple(neg.tolist()))
        if key not in self._cache:
            self._cache[key] = self._learn(pos, neg)
        result = self._cache[key]

        node_mask = torch.zeros(num_nodes, 1, device=x.device)
        node_mask[result['members'].to(x.device)] = 1.0

        explanation = Explanation(
            node_mask=node_mask,
            concept=result['concepts'][0],
            concepts=result['concepts'],
            concept_class=cls,
            concept_accuracy=result['accuracy'],
            concept_f1=result['f1'],
        )
        explanation._class_expressions = result['class_expressions']
        return explanation

    def supports(self) -> bool:
        task_level = self.model_config.task_level
        if task_level != ModelTaskLevel.node:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level explanations "
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

        edge_mask_type = self.explainer_config.edge_mask_type
        if edge_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"edge masks got "
                          f"(`edge_mask_type={edge_mask_type.value}`)")
            return False

        return True

    ###########################################################################

    def _prepare_knowledge_base(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Optional[Tensor],
        edge_attr: Optional[Tensor] = None,
    ) -> None:
        r"""Builds (or re-uses) the knowledge base and concept learner."""
        if self.data is not None and self.data.num_nodes != x.size(0):
            raise ValueError(f"The number of nodes in 'data' "
                             f"({self.data.num_nodes}) does not match the "
                             f"number of nodes in the input ({x.size(0)})")

        if self.data is not None:
            key: Tuple[Any, ...] = (id(self.data), )
        else:
            key = (x.data_ptr(), tuple(x.size()), edge_index.data_ptr(),
                   edge_index.size(1))
        if key == self._kb_key:
            return

        try:
            from ontolearn.knowledge_base import KnowledgeBase
        except ImportError as e:
            raise ImportError(f"'{self.__class__.__name__}' requires the "
                              f"'ontolearn' package (install it via "
                              f"`pip install ontolearn`)") from e

        data = self.data
        if data is None:
            data = OWLData.from_data(
                Data(x=x.detach().cpu(), edge_index=edge_index.detach().cpu(),
                     edge_type=edge_type, edge_attr=edge_attr,
                     num_nodes=x.size(0)),
                self.schema,
            )

        # `owlapy` rejects IRIs without a "/", *e.g.*, "urn:" IRIs:
        invalid = [iri for iri in data.node_iri if '/' not in iri]
        if len(invalid) > 0:
            raise ValueError(f"Ontolearn only supports IRIs that contain a "
                             f"'/' (got '{invalid[0]}'). Use an 'http(s)' "
                             f"IRI as 'base_iri' of the schema instead.")

        path = data.source
        if path is None or not osp.exists(path):
            self._tmp_dir = tempfile.TemporaryDirectory()
            path = data.save_owl(osp.join(self._tmp_dir.name, 'graph.owl'))

        with contextlib.redirect_stdout(io.StringIO()):
            kb = KnowledgeBase(path=osp.abspath(path))

        # Remove target classes and ignored classes from the knowledge base.
        # Names are either IRIs, or local names of classes in the ontology:
        concepts = {c.str: c for c in kb.get_concepts()}
        by_name: Dict[str, List[Any]] = defaultdict(list)
        for iri, concept in concepts.items():
            by_name[local_name(iri)].append(concept)

        ignored = set()
        names = data.schema.excluded_classes() + self.ignore_classes
        for name in names:
            if name in concepts:
                ignored.add(concepts[name])
            elif data.schema.iri(name) in concepts:
                ignored.add(concepts[data.schema.iri(name)])
            elif not is_absolute_iri(name) and len(by_name[name]) == 1:
                ignored.add(by_name[name][0])
            elif name in self.ignore_classes:
                raise ValueError(f"Could not find the OWL class '{name}' in "
                                 f"the knowledge base")
        if len(ignored) > 0:
            kb = kb.ignore_and_copy(ignored_classes=ignored)

        if isinstance(self.learner, str):
            if self.learner == 'EvoLearner':
                from ontolearn.concept_learner import EvoLearner as cls
            else:
                import ontolearn.learners
                cls = getattr(ontolearn.learners, self.learner)
            with _unique_primitive_names():
                learner = cls(knowledge_base=kb, **self.learner_kwargs)
        elif isinstance(self.learner, type):
            with _unique_primitive_names():
                learner = self.learner(knowledge_base=kb,
                                       **self.learner_kwargs)
        else:  # An already initialized learner:
            learner = self.learner
            kb = getattr(learner, 'kb', kb)

        self._kb, self._concept_learner = kb, learner
        self._node_iri = list(data.node_iri)
        self._kb_key = key
        self._cache = {}

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
                                 f"class at a time, but the nodes in 'index' "
                                 f"belong to the classes {classes.tolist()}")
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
                OWLNamedIndividual(IRI.create(self._node_iri[i]))
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

        mapping = {iri: i for i, iri in enumerate(self._node_iri)}
        instances = self._kb.individuals_set(expressions[0])
        members = sorted(mapping[ind.str] for ind in instances
                         if ind.str in mapping)
        members_tensor = torch.tensor(members, dtype=torch.long)

        covered = torch.zeros(len(self._node_iri), dtype=torch.bool)
        covered[members_tensor] = True
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
            members=members_tensor,
            accuracy=accuracy,
            f1=f1,
        )

    def __repr__(self) -> str:
        learner = self.learner
        if not isinstance(learner, str):
            learner = getattr(learner, '__name__', type(learner).__name__)
        return f'{self.__class__.__name__}({learner})'
