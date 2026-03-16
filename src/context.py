"""
Context Space and Reward Functions.

Implements Definitions 1-3 from the paper in a fully generic way —
works for any domain without hard-coding domain-specific logic.

Definition 1 — Activation Context:
    c_t = (x_{t-k+1}, ..., x_{t-1}, v ; attrs)

Definition 2 — Valid Context Space:
    C = { c : phi(c) = True }

Definition 3 — Context Reward Function:
    f : C -> R+,  f(c) = 0 if c not in C
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# Context object
# ------------------------------------------------------------------

@dataclass
class Context:
    """
    Activation context c_t for a node v.

    Attributes:
        path:  Tuple of node IDs — last (k-1) predecessors + v.
        attrs: Optional attribute dict (topic, timestamp, credibility, ...).
    """
    path: Tuple[int, ...]
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def node(self) -> int:
        """The activated node v (last in path)."""
        return self.path[-1]

    @property
    def predecessors(self) -> Tuple[int, ...]:
        """The (k-1) predecessor nodes."""
        return self.path[:-1]

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        return isinstance(other, Context) and self.path == other.path


# ------------------------------------------------------------------
# Context space
# ------------------------------------------------------------------

class ContextSpace:
    """
    Generic valid context space  C = { c : phi(c) = True }.

    phi is a Boolean predicate supplied by the user — this class
    does NOT embed any domain-specific logic, making it reusable
    for Twitter, StackOverflow, Amazon, Wikipedia, Reddit, Enron,
    or any custom application.

    Args:
        memory:    Context memory order k.  k=1 recovers memoryless IC.
        predicate: phi(context) -> bool.  Default: accept all contexts.
        name:      Optional label for logging/reporting.
    """

    def __init__(
        self,
        memory: int = 3,
        predicate: Optional[Callable[['Context'], bool]] = None,
        name: str = 'generic',
    ):
        self.memory = memory
        self.name = name
        self._phi: Callable = predicate if predicate is not None else lambda c: True

    # ---- public API ------------------------------------------------

    def is_valid(self, context: 'Context') -> bool:
        """phi(c): True if context belongs to C."""
        return self._phi(context)

    def build_context(
        self,
        predecessors: List[int],
        node: int,
        attrs: Optional[Dict] = None,
    ) -> 'Context':
        """
        Construct a Context from a predecessor list and target node.

        Keeps only the last (memory-1) predecessors so path length = k.

        Args:
            predecessors: Ordered list of nodes before v.
            node:         The newly activated node v.
            attrs:        Optional attribute dictionary.

        Returns:
            Context object.
        """
        k_pred = predecessors[-(self.memory - 1):] if self.memory > 1 else []
        return Context(path=tuple(k_pred) + (node,), attrs=attrs or {})

    # ---- convenience callable for ic_simulation --------------------

    def __call__(self, predecessors: List[int], node: int) -> 'Context':
        """Shorthand: context_space(predecessors, node) -> Context."""
        return self.build_context(predecessors, node)


# ------------------------------------------------------------------
# Reward function
# ------------------------------------------------------------------

class RewardFunction:
    """
    Generic context reward function  f : C -> R+.

    Supports three modes from Section 3.2 of the paper:

    'binary'   — f(c) = 1 if c in C, else 0.
    'weighted' — f(c) = sum_i  w_i * attr_i(c).
                 Weights are a dict {attr_name: weight}.
    'custom'   — f(c) = user_fn(c), any callable.

    Args:
        context_space: The associated ContextSpace (for validity check).
        mode:          'binary', 'weighted', or 'custom'.
        weights:       Dict of {attribute_name: float} for 'weighted' mode.
        custom_fn:     Callable(Context) -> float for 'custom' mode.
    """

    def __init__(
        self,
        context_space: ContextSpace,
        mode: str = 'binary',
        weights: Optional[Dict[str, float]] = None,
        custom_fn: Optional[Callable[['Context'], float]] = None,
    ):
        assert mode in ('binary', 'weighted', 'custom'), \
            "mode must be 'binary', 'weighted', or 'custom'"
        self.context_space = context_space
        self.mode = mode
        self.weights = weights or {}
        self.custom_fn = custom_fn

    def __call__(self, context: 'Context') -> float:
        """
        Evaluate f(c).

        Args:
            context: Activation context.

        Returns:
            Non-negative reward; 0.0 if context is not valid.
        """
        if not self.context_space.is_valid(context):
            return 0.0

        if self.mode == 'binary':
            return 1.0

        elif self.mode == 'weighted':
            # f(c) = sum_i w_i * g_i(c)
            return sum(
                w * float(context.attrs.get(attr, 0.0))
                for attr, w in self.weights.items()
            )

        elif self.mode == 'custom':
            if self.custom_fn is None:
                raise ValueError("custom_fn must be provided for mode='custom'")
            return max(0.0, float(self.custom_fn(context)))

        return 0.0


# ------------------------------------------------------------------
# Ready-made context space builders (one per paper dataset)
# These are thin wrappers — all domain logic is in the predicate,
# so swapping context types never requires changing any other file.
# ------------------------------------------------------------------

def make_context_space(
    memory: int = 3,
    predicate: Optional[Callable] = None,
    name: str = 'custom',
) -> ContextSpace:
    """
    Generic factory: create a ContextSpace with any predicate.

    Args:
        memory:    k — context memory order.
        predicate: phi(context) -> bool.
        name:      Label for this context type.

    Returns:
        ContextSpace instance.

    Example — Twitter (verified source propagation):
        >>> def phi(c):
        ...     return c.attrs.get('verified', False)
        >>> cs = make_context_space(memory=3, predicate=phi, name='twitter')

    Example — StackOverflow (tag coherence):
        >>> def phi(c):
        ...     return c.attrs.get('tag_similarity', 0) > 0.5
        >>> cs = make_context_space(memory=3, predicate=phi, name='stackoverflow')

    Example — binary (accept everything, recovers classical IM):
        >>> cs = make_context_space(memory=1)
    """
    return ContextSpace(memory=memory, predicate=predicate, name=name)
