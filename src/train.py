"""
Training / Experiment Script for CDS-IM.

Mirrors the structure of GraphRAG-IM/src/train.py:
  - run_cds()        runs the proposed CDS method
  - run_baseline()   runs one baseline method
  - evaluate()       evaluates a seed set (sigma_C, eta, rho)
  - train()          full pipeline: all methods + save results
  - main()           CLI entry point

Reproduces all paper results:
  Table 4  — Context-Aware Influence    (RQ1)
  Table 5  — Reward Efficiency          (RQ2)
  Figure 3 — Context Satisfaction       (RQ3)
  Table 6  — Runtime                    (RQ4)
  Table 7  — Ablation Study            (RQ5)
  Figure 4 — Parameter Sensitivity      (RQ6)

Usage:
    python src/train.py \
        --graph   data/enron/graph.edgelist \
        --attrs   data/enron/node_attrs.json \
        --context thread_structure \
        --budget  50
"""

import argparse
import json
import math
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from context import ContextSpace, RewardFunction, make_context_space
from data import CDSDataset
from ic_simulation import evaluate_seeds, monte_carlo_context_spread
from model import CDS, CDSResult


# ------------------------------------------------------------------
# Context presets — one per paper dataset (Section 5.1.1)
# ------------------------------------------------------------------

CONTEXT_PRESETS: Dict[str, Callable] = {
    'binary':           lambda mem: make_context_space(mem, None,                                          'binary'),
    'verified_source':  lambda mem: make_context_space(mem, lambda c: c.attrs.get('verified', False),      'verified_source'),
    'tag_coherence':    lambda mem: make_context_space(mem, lambda c: c.attrs.get('tag_similarity',0)>0.5, 'tag_coherence'),
    'category_path':    lambda mem: make_context_space(mem, lambda c: c.attrs.get('same_category',False),  'category_path'),
    'topic_coherence':  lambda mem: make_context_space(mem, lambda c: c.attrs.get('topic_similarity',0)>0.6,'topic_coherence'),
    'subreddit_chain':  lambda mem: make_context_space(mem, lambda c: c.attrs.get('same_subreddit',False), 'subreddit_chain'),
    'thread_structure': lambda mem: make_context_space(mem, lambda c: c.attrs.get('in_thread',False),      'thread_structure'),
}

REWARD_WEIGHTS: Dict[str, Dict[str, float]] = {
    'verified_source':  {'credibility': 0.4, 'topic_similarity': 0.4, 'recency': 0.2},
    'tag_coherence':    {'tag_similarity': 0.6, 'reputation': 0.4},
    'category_path':    {'review_quality': 0.5, 'same_category': 0.5},
    'topic_coherence':  {'topic_similarity': 1.0},
    'subreddit_chain':  {'karma': 0.3, 'same_subreddit': 0.7},
    'thread_structure': {'thread_coherence': 0.6, 'in_thread': 0.4},
}


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(
    G: nx.DiGraph,
    seeds: List[int],
    context_fn: Callable,
    reward_fn: RewardFunction,
    context_space: ContextSpace,
    p: float = 0.1,
    T: int = 20,
    num_simulations: int = 1000,
    show_progress: bool = False,
) -> Dict:
    """Evaluate seed set on sigma_C, reward efficiency, context satisfaction."""
    return evaluate_seeds(
        G=G, seeds=seeds,
        context_fn=context_fn, reward_fn=reward_fn,
        memory=context_space.memory, p=p, T=T,
        num_simulations=num_simulations, show_progress=show_progress,
    )


# ------------------------------------------------------------------
# CDS (proposed method)
# ------------------------------------------------------------------

def run_cds(
    G: nx.DiGraph,
    context_space: ContextSpace,
    reward_fn: RewardFunction,
    context_fn: Callable,
    budget: int = 50,
    R: int = 1000,
    M: int = 1000,
    prune_percentile: float = 90.0,
    p: float = 0.1,
    T: int = 20,
) -> Tuple[List[int], float]:
    """
    Run CDS algorithm (Algorithm 2 from paper).

    NOTE: context_fn is passed to select_seeds() so node attributes
    are injected into contexts during both Phase 1 (dominance
    estimation) and Phase 2 (greedy selection). Without this,
    non-binary context predicates receive empty attr dicts and
    always return 0 reward.
    """
    t0 = time.time()
    cds = CDS(
        context_space=context_space, reward_fn=reward_fn,
        budget=budget, R=R, M=M,
        prune_percentile=prune_percentile,
        p=p, T=T, verbose=True,
    )
    result: CDSResult = cds.select_seeds(G, context_fn=context_fn)
    return result.seeds, time.time() - t0


# ------------------------------------------------------------------
# Baselines (Table 4)
# ------------------------------------------------------------------

def run_baseline(
    G: nx.DiGraph,
    method: str,
    context_space: ContextSpace,
    reward_fn: RewardFunction,
    context_fn: Callable,
    budget: int = 50,
    R: int = 1000,
    M: int = 1000,
    p: float = 0.1,
    T: int = 20,
) -> Tuple[List[int], float]:
    """
    Run one baseline seed-selection method.

    Methods:
        degree         — top-B out-degree nodes
        pagerank       — top-B PageRank nodes
        greedy_ic      — greedy with context-aware MC estimation
        celf_context   — CELF lazy greedy with context-aware MC
        weighted_imm   — degree-weighted by context participation
        cds_no_prune   — ablation: CDS without dominance pruning
        cds_random     — ablation: CDS with random pruning
    """
    from dominance import estimate_dominance

    t0 = time.time()

    if method == 'degree':
        seeds = sorted(G.nodes(), key=lambda v: G.out_degree(v), reverse=True)[:budget]

    elif method == 'pagerank':
        pr = nx.pagerank(G, alpha=0.85)
        seeds = sorted(pr, key=pr.get, reverse=True)[:budget]

    elif method == 'greedy_ic':
        seeds, remaining, current_inf = [], set(G.nodes()), 0.0
        for _ in range(budget):
            best_v, best_gain = None, -1.0
            for v in remaining:
                new_inf, _ = monte_carlo_context_spread(
                    G, set(seeds) | {v}, context_fn, reward_fn,
                    context_space.memory, p, T, max(R // 10, 50)
                )
                gain = new_inf - current_inf
                if gain > best_gain:
                    best_gain, best_v = gain, v
            seeds.append(best_v)
            remaining.remove(best_v)
            current_inf += best_gain

    elif method == 'celf_context':
        seeds, remaining = [], set(G.nodes())
        delta_bar, current_inf = {v: math.inf for v in remaining}, 0.0
        for _ in range(budget):
            while True:
                v_star = max(remaining, key=lambda v: delta_bar[v])
                new_inf, _ = monte_carlo_context_spread(
                    G, set(seeds) | {v_star}, context_fn, reward_fn,
                    context_space.memory, p, T, R
                )
                delta_bar[v_star] = new_inf - current_inf
                if max(remaining, key=lambda v: delta_bar[v]) == v_star:
                    break
            seeds.append(v_star)
            current_inf += delta_bar[v_star]
            remaining.remove(v_star)

    elif method == 'weighted_imm':
        dominance = estimate_dominance(
            G, context_space, reward_fn, context_fn=context_fn,
            M=max(R // 5, 100), p=p, T=T, show_progress=False,
        )
        seeds = sorted(dominance, key=dominance.get, reverse=True)[:budget]

    elif method == 'cds_no_prune':
        cds = CDS(context_space, reward_fn, budget=budget,
                  R=R, M=M, prune_percentile=0.0, p=p, T=T, verbose=False)
        seeds = cds.select_seeds(G, context_fn=context_fn).seeds

    elif method == 'cds_random':
        nodes  = list(G.nodes())
        n_keep = max(budget, len(nodes) // 10)
        remaining = set(random.sample(nodes, min(n_keep, len(nodes))))
        seeds, delta_bar, current_inf = [], {v: math.inf for v in remaining}, 0.0
        for _ in range(budget):
            if not remaining:
                break
            while True:
                v_star = max(remaining, key=lambda v: delta_bar[v])
                new_inf, _ = monte_carlo_context_spread(
                    G, set(seeds) | {v_star}, context_fn, reward_fn,
                    context_space.memory, p, T, R
                )
                delta_bar[v_star] = new_inf - current_inf
                if max(remaining, key=lambda v: delta_bar[v]) == v_star:
                    break
            seeds.append(v_star)
            current_inf += delta_bar[v_star]
            remaining.remove(v_star)

    else:
        raise ValueError(f"Unknown method '{method}'.")

    return seeds, time.time() - t0


# ------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------

def train(
    graph_path: str,
    context_type: str = 'binary',
    node_attrs_path: Optional[str] = None,
    budget: int = 50,
    R: int = 1000,
    M: int = 1000,
    memory: int = 3,
    prune_percentile: float = 90.0,
    p: float = 0.1,
    T: int = 20,
    baselines: Optional[List[str]] = None,
    save_dir: str = './results',
) -> Dict:
    """
    Full experiment pipeline — mirrors GraphRAG-IM train().

    Runs CDS + specified baselines, evaluates all on four metrics,
    prints a summary table, and saves results to JSON.
    """
    os.makedirs(save_dir, exist_ok=True)

    if baselines is None:
        baselines = ['degree', 'pagerank', 'greedy_ic', 'celf_context', 'weighted_imm']

    # ---- Load data -------------------------------------------------
    dataset = CDSDataset(graph_path, node_attrs_path, default_prob=p)
    G = dataset.G

    print(f"\n{'='*60}")
    print(f"  Graph        : {os.path.basename(graph_path)}")
    print(f"  Nodes/Edges  : {G.number_of_nodes():,} / {G.number_of_edges():,}")
    print(f"  Context type : {context_type}")
    print(f"  Budget B     : {budget}")
    print(f"{'='*60}")

    # ---- Build context space, reward function, context_fn ----------
    context_space = CONTEXT_PRESETS[context_type](memory)
    weights       = REWARD_WEIGHTS.get(context_type, {})
    reward_fn     = RewardFunction(context_space,
                                   mode='weighted' if weights else 'binary',
                                   weights=weights)

    # context_fn wraps context_space + injects node attrs from store
    # This is essential: without it all non-binary predicates fail
    context_fn = dataset.get_context_fn(context_space)

    all_results: Dict[str, Dict] = {}

    # ---- CDS -------------------------------------------------------
    print(f"\n[1/{len(baselines)+1}] CDS (proposed)...")
    seeds_cds, t_cds = run_cds(
        G, context_space, reward_fn, context_fn,
        budget=budget, R=R, M=M,
        prune_percentile=prune_percentile, p=p, T=T,
    )
    m = evaluate(G, seeds_cds, context_fn, reward_fn,
                 context_space, p, T, R, show_progress=True)
    m['runtime'] = t_cds
    m['seeds']   = seeds_cds
    all_results['CDS'] = m

    # ---- Baselines -------------------------------------------------
    for i, method in enumerate(baselines, 2):
        print(f"\n[{i}/{len(baselines)+1}] Baseline: {method}...")
        seeds_bl, t_bl = run_baseline(
            G, method, context_space, reward_fn, context_fn,
            budget=budget, R=R, M=M, p=p, T=T,
        )
        m = evaluate(G, seeds_bl, context_fn, reward_fn,
                     context_space, p, T, R)
        m['runtime'] = t_bl
        m['seeds']   = seeds_bl
        all_results[method] = m

    # ---- Summary table (mirrors GraphRAG-IM output format) ---------
    print(f"\n{'='*70}")
    print(f"  Results — {context_type} context  |  B={budget}")
    print(f"{'='*70}")
    print(f"  {'Method':<18} {'sigma_C':>10} {'efficiency':>12} {'satisfaction':>14} {'time(s)':>9}")
    print(f"  {'-'*65}")
    for method, m in all_results.items():
        tag = '  <-- proposed' if method == 'CDS' else ''
        print(f"  {method:<18} {m['mean']:>10.3f} {m['reward_efficiency']:>12.4f} "
              f"{m['context_satisfaction']:>13.2f}% {m['runtime']:>9.1f}s{tag}")
    print(f"{'='*70}")

    # ---- Save JSON -------------------------------------------------
    def _s(v):
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, np.ndarray):     return v.tolist()
        return v

    out = {k: {kk: _s(vv) for kk, vv in mv.items()} for k, mv in all_results.items()}
    out['_config'] = dict(graph=graph_path, context_type=context_type,
                          budget=budget, R=R, M=M, memory=memory,
                          prune_percentile=prune_percentile, p=p, T=T)

    tag = os.path.splitext(os.path.basename(graph_path))[0]
    out_path = os.path.join(save_dir, f'{tag}_{context_type}_B{budget}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    return all_results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run CDS-IM experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--graph',     type=str, required=True)
    parser.add_argument('--context',   type=str, default='binary',
                        choices=list(CONTEXT_PRESETS.keys()))
    parser.add_argument('--attrs',     type=str, default=None)
    parser.add_argument('--budget',    type=int, default=50)
    parser.add_argument('--R',         type=int, default=1000)
    parser.add_argument('--M',         type=int, default=1000)
    parser.add_argument('--memory',    type=int, default=3)
    parser.add_argument('--prune',     type=float, default=90.0)
    parser.add_argument('--p',         type=float, default=0.1)
    parser.add_argument('--T',         type=int, default=20)
    parser.add_argument('--baselines', nargs='+',
                        default=['degree','pagerank','greedy_ic',
                                 'celf_context','weighted_imm'])
    parser.add_argument('--save_dir',  type=str, default='./results')
    args = parser.parse_args()

    train(
        graph_path=args.graph, context_type=args.context,
        node_attrs_path=args.attrs, budget=args.budget,
        R=args.R, M=args.M, memory=args.memory,
        prune_percentile=args.prune, p=args.p, T=args.T,
        baselines=args.baselines, save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
