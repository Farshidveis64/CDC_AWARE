"""
Context Dominance Analysis.

Implements Algorithm 1 (EstimateContextDominance) and
Definition 5 (Pruned Candidate Set) from the paper.

Definition 4 — Context Dominance Score:
    D(v) = sum_{c in C: v in c}  f(c) * P[c is reachable]

Algorithm 1 — EstimateContextDominance:
    For m = 1..M:
        s  ~ Uniform(V)
        simulate IC from {s}, record trace
        for each valid activation with reward > 0:
            for each node v in the context path:
                D(v) += reward / M

Definition 5 — Pruned Candidate Set:
    V_theta = { v in V : D(v) >= theta }

Theorem 6 — Pruning Guarantee:
    sigma_C(S*) - sigma_C(S* ∩ V_theta)  <=  ell * theta * T
"""

import random
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from context import ContextSpace, RewardFunction
from ic_simulation import ic_simulation_with_context


def estimate_dominance(
    G: nx.DiGraph,
    context_space: ContextSpace,
    reward_fn: RewardFunction,
    context_fn: Optional[Callable] = None,
    M: int = 1000,
    p: float = 0.1,
    T: int = 20,
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[int, float]:
    """
    Algorithm 1: EstimateContextDominance.

    FIX vs previous version:
        - Uses node_rewards returned directly by ic_simulation_with_context
          instead of re-calling context_fn / reward_fn post-hoc.
          Eliminates redundant double-context-build.
        - Added `seed` parameter for reproducibility.

    Args:
        G:             Directed graph.
        context_space: Defines valid context space C and memory order k.
        reward_fn:     f(c) -> float reward.
        context_fn:    callable(predecessors, node) -> Context.
                       If None, uses context_space directly (no attrs).
                       REQUIRED for non-binary context types.
        M:             Number of Monte Carlo samples.
        p:             Default propagation probability.
        T:             Diffusion time horizon.
        seed:          Random seed for reproducibility.
        show_progress: Show tqdm progress bar.

    Returns:
        dominance: Dict {node_id -> D(v)}.
    """
    if seed is not None:
        random.seed(seed)

    _ctx_fn = context_fn if context_fn is not None else context_space

    nodes     = list(G.nodes())
    dominance: Dict[int, float] = {v: 0.0 for v in nodes}

    iterator = tqdm(range(M), desc="Estimating dominance") \
               if show_progress else range(M)

    for _ in iterator:
        s = random.choice(nodes)

        # Simulate IC — node_rewards already computed inside simulation.
        # No need to rebuild contexts post-hoc (FIX: was the bug before).
        _, paths, node_rewards = ic_simulation_with_context(
            G=G,
            seeds={s},
            context_fn=_ctx_fn,
            reward_fn=reward_fn,
            memory=context_space.memory,
            p=p,
            T=T,
        )

        for node, reward in node_rewards.items():
            if node == s or reward <= 0:
                continue

            # Give credit to every node that participated in this context
            path = paths[node]  # last-memory-nodes path ending at node
            for v in path:
                if v in dominance:
                    dominance[v] += reward / M

    return dominance


def prune_candidates(
    dominance: Dict[int, float],
    percentile: float = 90.0,
) -> Tuple[Set[int], float]:
    """
    Definition 5: Pruned Candidate Set.

    Retains the top (100 - percentile)% nodes by dominance score.
    At percentile=90, keeps the top 10% (paper default).

    NOTE: When all scores are 0 (e.g. disconnected graph or very low p),
    theta=0 and all nodes are retained — CDS degrades gracefully to
    full greedy over V.

    Args:
        dominance:   Dict {node -> D(v)}.
        percentile:  Threshold percentile alpha in [0, 100].

    Returns:
        candidates: Set V_theta.
        theta:      Threshold value used.
    """
    scores = np.array(list(dominance.values()))
    theta  = float(np.percentile(scores, percentile))
    candidates = {v for v, d in dominance.items() if d >= theta}
    return candidates, theta
