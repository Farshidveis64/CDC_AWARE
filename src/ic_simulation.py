"""
Independent Cascade (IC) Model with Context Tracking.

Extends standard IC simulation to record full activation paths,
enabling context extraction per Definition 1 of the paper.
"""

import random
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm


def ic_simulation(
    G: nx.DiGraph,
    seeds: Set[int],
    p: float = 0.1,
) -> int:
    """
    Standard IC simulation (no context tracking).
    Used for classical baselines (degree, pagerank comparisons).

    Args:
        G:     Directed graph with optional edge attribute 'prob'.
        seeds: Seed set activated at t=0.
        p:     Default propagation probability.

    Returns:
        Number of activated nodes (including seeds).
    """
    activated = set(seeds)
    frontier  = set(seeds)
    while frontier:
        next_frontier = set()
        for u in frontier:
            for _, v, data in G.out_edges(u, data=True):
                if v not in activated:
                    if random.random() < data.get('prob', p):
                        next_frontier.add(v)
        activated  |= next_frontier
        frontier    = next_frontier
    return len(activated)


def ic_simulation_with_context(
    G: nx.DiGraph,
    seeds: Set[int],
    context_fn: Callable,
    reward_fn: Callable,
    memory: int = 3,
    p: float = 0.1,
    T: int = 20,
) -> Tuple[float, Dict[int, List[int]], Dict[int, float]]:
    """
    IC simulation with context-aware reward accumulation.

    Tracks activation paths so context c_t(v) = (x_{t-k+1},...,v)
    is evaluated at every activation step.

    FIX vs previous version:
        - Returns node_rewards dict so callers never need to
          re-evaluate reward_fn(context_fn(...)) post-hoc.
        - Eliminates the double-context-build that existed in
          dominance.py and evaluate_seeds.

    Args:
        G:          Directed graph.
        seeds:      Seed set S activated at t=0.
        context_fn: callable(predecessors: List[int], node: int) -> Context.
                    Must inject node attributes for non-binary contexts.
        reward_fn:  callable(context) -> float.
        memory:     Context memory order k.
        p:          Default propagation probability.
        T:          Maximum diffusion time steps.

    Returns:
        total_reward:  Sum of f(c_t(v)) over all valid activations.
        paths:         Dict node -> last-memory-nodes activation path.
        node_rewards:  Dict node -> reward received at activation.
    """
    activated: Set[int]         = set(seeds)
    frontier:  Set[int]         = set(seeds)
    paths:     Dict[int, List[int]] = {s: [s] for s in seeds}
    node_rewards: Dict[int, float]  = {s: 0.0 for s in seeds}
    total_reward: float         = 0.0

    for _ in range(T):
        if not frontier:
            break
        next_frontier: Set[int] = set()
        for u in frontier:
            path_u = paths[u]
            for _, v, data in G.out_edges(u, data=True):
                if v not in activated:
                    if random.random() < data.get('prob', p):
                        # Build path: keep last (memory) nodes ending with v
                        path_v = (path_u + [v])[-memory:]
                        paths[v] = path_v

                        # Build context once — predecessors are all but last
                        ctx    = context_fn(path_v[:-1], v)
                        reward = reward_fn(ctx)

                        node_rewards[v] = reward
                        total_reward   += reward
                        next_frontier.add(v)

        activated |= next_frontier
        frontier   = next_frontier

    return total_reward, paths, node_rewards


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
    Estimate sigma_C(S) = E[sum_t sum_v f(c_t(v))] via Monte Carlo.

    Args:
        G:               Directed graph.
        seeds:           Seed set S.
        context_fn:      callable(predecessors, node) -> Context.
        reward_fn:       callable(context) -> float.
        memory:          Context memory order k.
        p:               Default propagation probability.
        T:               Time horizon.
        num_simulations: Number of MC rollouts R.

    Returns:
        mean_reward: Estimated sigma_C(S).
        std_reward:  Standard deviation.
    """
    if not seeds:
        return 0.0, 0.0
    rewards = [
        ic_simulation_with_context(G, seeds, context_fn, reward_fn, memory, p, T)[0]
        for _ in range(num_simulations)
    ]
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
    Evaluate a seed set on all four paper metrics.

    FIX vs previous version:
        - Uses node_rewards returned by ic_simulation_with_context
          instead of recomputing reward_fn post-hoc per node.
          This removes the redundant double-evaluation.

    Metrics (Section 5.1.3 of the paper):
        sigma_C(S)           = E[total reward]
        reward_efficiency    = sigma_C(S) / E[|activated non-seeds|]
        context_satisfaction = E[fraction of activations with reward > 0]

    Args:
        G:               Directed graph.
        seeds:           Selected seed node IDs.
        context_fn:      callable(predecessors, node) -> Context.
        reward_fn:       callable(context) -> float.
        memory:          Context memory order k.
        p:               Default propagation probability.
        T:               Time horizon.
        num_simulations: MC simulations.
        show_progress:   Show tqdm bar.

    Returns:
        Dict with keys: mean, std, reward_efficiency, context_satisfaction.
    """
    seed_set = set(seeds)
    rewards, n_activated, n_valid = [], [], []

    rng = tqdm(range(num_simulations), desc="Evaluating") if show_progress \
          else range(num_simulations)

    for _ in rng:
        total_r, paths, node_rewards = ic_simulation_with_context(
            G, seed_set, context_fn, reward_fn, memory, p, T
        )
        non_seeds     = [v for v in paths if v not in seed_set]
        valid_count   = sum(1 for v in non_seeds if node_rewards.get(v, 0) > 0)

        rewards.append(total_r)
        n_activated.append(len(non_seeds))
        n_valid.append(valid_count)

    mean_r   = float(np.mean(rewards))
    mean_act = float(np.mean(n_activated))
    mean_val = float(np.mean(n_valid))

    return {
        'mean':                 mean_r,
        'std':                  float(np.std(rewards)),
        'reward_efficiency':    mean_r / mean_act if mean_act > 0 else 0.0,
        'context_satisfaction': mean_val / mean_act * 100 if mean_act > 0 else 0.0,
    }
