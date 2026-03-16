"""
Independent Cascade (IC) Model with Context Tracking.

Extends standard IC simulation to record full activation paths,
enabling context extraction per Definition 1 of the paper.
"""

import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm


# ------------------------------------------------------------------
# Core simulation
# ------------------------------------------------------------------

def ic_simulation(
    G: nx.DiGraph,
    seeds: Set[int],
    p: float = 0.1,
) -> int:
    """
    Standard Independent Cascade simulation (no context tracking).

    Args:
        G:     Directed graph. Uses edge attribute 'prob' if present, else p.
        seeds: Set of seed node IDs activated at t=0.
        p:     Default propagation probability.

    Returns:
        Number of activated nodes.
    """
    activated = set(seeds)
    frontier = set(seeds)

    while frontier:
        next_frontier = set()
        for u in frontier:
            for _, v, data in G.out_edges(u, data=True):
                if v not in activated:
                    prob = data.get('prob', p)
                    if random.random() < prob:
                        next_frontier.add(v)
        activated |= next_frontier
        frontier = next_frontier

    return len(activated)


def ic_simulation_with_context(
    G: nx.DiGraph,
    seeds: Set[int],
    context_fn: Callable,
    reward_fn: Callable,
    memory: int = 3,
    p: float = 0.1,
    T: int = 20,
) -> Tuple[float, Dict[int, List[int]]]:
    """
    IC simulation with context-aware reward accumulation.

    Tracks full activation paths so the context c_t(v) =
    (x_{t-k+1}, ..., x_{t-1}, v) can be evaluated at each step.

    Args:
        G:          Directed graph.
        seeds:      Seed set S activated at t=0.
        context_fn: callable(path, node) -> context object.
        reward_fn:  callable(context) -> float reward.
        memory:     Context memory order k (path length = k).
        p:          Default propagation probability.
        T:          Maximum diffusion time steps.

    Returns:
        total_reward: Sum of f(c_t(v)) over all valid activations.
        paths:        Dict mapping node -> activation path list.
    """
    activated = set(seeds)
    frontier = set(seeds)
    paths: Dict[int, List[int]] = {s: [s] for s in seeds}
    total_reward = 0.0

    for _ in range(T):
        if not frontier:
            break
        next_frontier = set()
        for u in frontier:
            path_u = paths[u]
            for _, v, data in G.out_edges(u, data=True):
                if v not in activated:
                    prob = data.get('prob', p)
                    if random.random() < prob:
                        # Build path to v, keeping last (memory-1) + v
                        path_v = (path_u + [v])[-(memory):]
                        paths[v] = path_v
                        next_frontier.add(v)

                        # Evaluate context reward
                        ctx = context_fn(path_v[:-1], v)
                        total_reward += reward_fn(ctx)

        activated |= next_frontier
        frontier = next_frontier

    return total_reward, paths


def monte_carlo_context_spread(
    G: nx.DiGraph,
    seeds: Set[int],
    context_fn: Callable,
    reward_fn: Callable,
    memory: int = 3,
    p: float = 0.1,
    T: int = 20,
    num_simulations: int = 1000,
) -> Tuple[float, float]:
    """
    Estimate sigma_C(S) via Monte Carlo:
    E[sum_t sum_v f(c_t(v))].

    Args:
        G:               Directed graph.
        seeds:           Seed set S.
        context_fn:      callable(path, node) -> context.
        reward_fn:       callable(context) -> float.
        memory:          Context memory order k.
        p:               Default propagation probability.
        T:               Time horizon.
        num_simulations: Number of MC rollouts R.

    Returns:
        mean_reward: Estimated sigma_C(S).
        std_reward:  Standard deviation across simulations.
    """
    rewards = []
    for _ in range(num_simulations):
        r, _ = ic_simulation_with_context(
            G, seeds, context_fn, reward_fn, memory, p, T
        )
        rewards.append(r)
    return float(np.mean(rewards)), float(np.std(rewards))


def evaluate_seeds(
    G: nx.DiGraph,
    seeds: List[int],
    context_fn: Callable,
    reward_fn: Callable,
    memory: int = 3,
    p: float = 0.1,
    T: int = 20,
    num_simulations: int = 1000,
    show_progress: bool = False,
) -> Dict:
    """
    Full evaluation of a seed set: context reward, efficiency, satisfaction.

    Args:
        G:               Directed graph.
        seeds:           Selected seed node IDs.
        context_fn:      callable(path, node) -> context.
        reward_fn:       callable(context) -> float.
        memory:          Context memory order k.
        p:               Default propagation probability.
        T:               Time horizon.
        num_simulations: MC simulations.
        show_progress:   Show tqdm bar.

    Returns:
        Dict with mean, std, reward_efficiency, context_satisfaction.
    """
    seed_set = set(seeds)
    rewards, activated_counts, valid_counts = [], [], []

    iterator = (
        tqdm(range(num_simulations), desc="Evaluating seeds")
        if show_progress else range(num_simulations)
    )

    for _ in iterator:
        reward, paths = ic_simulation_with_context(
            G, seed_set, context_fn, reward_fn, memory, p, T
        )
        n_activated = len(paths) - len(seed_set)
        n_valid = sum(
            1 for v, path in paths.items()
            if v not in seed_set and reward_fn(context_fn(path[:-1], v)) > 0
        )
        rewards.append(reward)
        activated_counts.append(n_activated)
        valid_counts.append(n_valid)

    mean_reward = float(np.mean(rewards))
    mean_activated = float(np.mean(activated_counts))
    mean_valid = float(np.mean(valid_counts))

    return {
        'mean':                mean_reward,
        'std':                 float(np.std(rewards)),
        'reward_efficiency':   mean_reward / mean_activated if mean_activated > 0 else 0.0,
        'context_satisfaction': (mean_valid / mean_activated * 100) if mean_activated > 0 else 0.0,
    }
