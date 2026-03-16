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
        for each valid context c_t containing node v:
            D(v) += f(c_t) / M

Definition 5 — Pruned Candidate Set:
    V_theta = { v in V : D(v) >= theta }

Theorem 6 — Pruning Guarantee:
    sigma_C(S*) - sigma_C(S* ∩ V_theta)  <=  ell * theta * T
"""

import random
from typing import Callable, Dict, Set, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from context import Context, ContextSpace, RewardFunction
from ic_simulation import ic_simulation_with_context


def estimate_dominance(
    G: nx.DiGraph,
    context_space: ContextSpace,
    reward_fn: RewardFunction,
    M: int = 1000,
    p: float = 0.1,
    T: int = 20,
    show_progress: bool = True,
) -> Dict[int, float]:
    """
    Algorithm 1: EstimateContextDominance.

    Args:
        G:             Directed graph.
        context_space: Defines C and memory order k.
        reward_fn:     f(c) -> float reward.
        M:             Number of Monte Carlo samples.
        p:             Default propagation probability.
        T:             Diffusion time horizon.
        show_progress: Show tqdm progress bar.

    Returns:
        dominance: Dict {node_id -> D(v)}.
    """
    nodes = list(G.nodes())
    dominance: Dict[int, float] = {v: 0.0 for v in nodes}

    iterator = tqdm(range(M), desc="Estimating dominance") if show_progress else range(M)

    for _ in iterator:
        # Step 1a: sample a random seed
        seed = random.choice(nodes)

        # Step 1b: simulate IC from {seed}, tracking paths
        _, paths = ic_simulation_with_context(
            G=G,
            seeds={seed},
            context_fn=context_space,
            reward_fn=reward_fn,
            memory=context_space.memory,
            p=p,
            T=T,
        )

        # Step 1c-d: extract contexts and accumulate
        for node, path in paths.items():
            if node == seed:
                continue  # seed has no incoming context

            ctx = context_space.build_context(path[:-1], node)

            if not context_space.is_valid(ctx):
                continue

            reward = reward_fn(ctx)
            if reward <= 0:
                continue

            # Every node in the context path gets credit
            for v in ctx.path:
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
    Default: percentile=90 keeps the top 10% (paper setting).

    Args:
        dominance:   Dict {node -> D(v)}.
        percentile:  Threshold percentile alpha.

    Returns:
        candidates: Set of retained nodes V_theta.
        theta:      The threshold value used.
    """
    scores = np.array(list(dominance.values()))
    theta = float(np.percentile(scores, percentile))
    candidates = {v for v, d in dominance.items() if d >= theta}
    return candidates, theta
