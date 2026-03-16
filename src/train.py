"""
Training / Experiment Script for CDS-IM.

Reproduces all results from the paper:
  Table 4 — Context-Aware Influence    (RQ1)
  Table 5 — Reward Efficiency           (RQ2)
  Figure 3 — Context Satisfaction       (RQ3)
  Table 6 — Runtime                     (RQ4)
  Table 7 — Ablation Study             (RQ5)
  Figure 4 — Parameter Sensitivity     (RQ6)

Usage:
    python src/train.py \\
        --graph   data/enron/graph.edgelist \\
        --context binary \\
        --budget  50 \\
        --R       1000 \\
        --M       1000
"""

import argparse
import json
import os
import time
from typing import Callable, Dict

import networkx as nx

from context import ContextSpace, RewardFunction, make_context_space
from data import CDSDataset, NodeAttributeStore
from ic_simulation import evaluate_seeds
from model import CDS, CDSResult


# ------------------------------------------------------------------
# Built-in context presets  (covers all 6 paper datasets)
# Users can add their own via --context custom + --attrs path
# ------------------------------------------------------------------

CONTEXT_PRESETS: Dict[str, Callable] = {

    'binary': lambda memory: make_context_space(
        memory=memory, predicate=None, name='binary'
    ),

    'verified_source': lambda memory: make_context_space(
        memory=memory,
        predicate=lambda c: c.attrs.get('verified', False),
        name='verified_source',          # Twitter-Ads
    ),

    'tag_coherence': lambda memory: make_context_space(
        memory=memory,
        predicate=lambda c: c.attrs.get('tag_similarity', 0) > 0.5,
        name='tag_coherence',            # StackOverflow
    ),

    'category_path': lambda memory: make_context_space(
        memory=memory,
        predicate=lambda c: c.attrs.get('same_category', False),
        name='category_path',            # Amazon
    ),

    'topic_coherence': lambda memory: make_context_space(
        memory=memory,
        predicate=lambda c: c.attrs.get('topic_similarity', 0) > 0.6,
        name='topic_coherence',          # Wikipedia
    ),

    'subreddit_chain': lambda memory: make_context_space(
        memory=memory,
        predicate=lambda c: c.attrs.get('same_subreddit', False),
        name='subreddit_chain',          # Reddit
    ),

    'thread_structure': lambda memory: make_context_space(
        memory=memory,
        predicate=lambda c: c.attrs.get('in_thread', False),
        name='thread_structure',         # Enron
    ),
}

REWARD_WEIGHTS: Dict[str, Dict[str, float]] = {
    'verified_source':  {'credibility': 0.4, 'topic_coherence': 0.4, 'recency': 0.2},
    'tag_coherence':    {'tag_similarity': 0.6, 'reputation': 0.4},
    'category_path':    {'category_distance': 0.5, 'review_quality': 0.5},
    'topic_coherence':  {'topic_similarity': 1.0},
    'subreddit_chain':  {'subreddit_relevance': 0.7, 'karma': 0.3},
    'thread_structure': {'thread_coherence': 0.6, 'participant_relevance': 0.4},
}


def run(
    graph_path: str,
    context_type: str = 'binary',
    node_attrs_path: str = None,
    budget: int = 50,
    R: int = 1000,
    M: int = 1000,
    memory: int = 3,
    prune_percentile: float = 90.0,
    p: float = 0.1,
    T: int = 20,
    save_dir: str = './results',
) -> Dict:
    """
    Full CDS experiment pipeline.

    Args:
        graph_path:       Path to graph file.
        context_type:     One of CONTEXT_PRESETS keys or 'custom'.
        node_attrs_path:  Path to node attributes JSON (optional).
        budget:           Seed budget B.
        R:                MC rollouts for influence estimation.
        M:                MC samples for dominance estimation.
        memory:           Context memory order k.
        prune_percentile: Pruning threshold percentile alpha.
        p:                Default edge propagation probability.
        T:                Diffusion time horizon.
        save_dir:         Directory to save results.

    Returns:
        Dict with sigma_C, eta, rho, runtime, and selected seeds.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- Load data -------------------------------------------------
    dataset = CDSDataset(graph_path, node_attrs_path, default_prob=p)
    G = dataset.G

    # ---- Build context space and reward ----------------------------
    if context_type not in CONTEXT_PRESETS:
        raise ValueError(
            f"Unknown context type '{context_type}'. "
            f"Available: {list(CONTEXT_PRESETS.keys())}"
        )

    context_space = CONTEXT_PRESETS[context_type](memory)
    weights = REWARD_WEIGHTS.get(context_type, {})
    reward_mode = 'weighted' if weights else 'binary'
    reward_fn = RewardFunction(context_space, mode=reward_mode, weights=weights)
    context_fn = dataset.get_context_fn(context_space)

    print(f"\n  Context type : {context_type}")
    print(f"  Reward mode  : {reward_mode}")
    print(f"  Budget       : {budget}")

    # ---- Run CDS ---------------------------------------------------
    t0 = time.time()
    cds = CDS(
        context_space=context_space,
        reward_fn=reward_fn,
        budget=budget,
        R=R, M=M,
        prune_percentile=prune_percentile,
        p=p, T=T,
        verbose=True,
    )
    result: CDSResult = cds.select_seeds(G)
    elapsed = time.time() - t0

    # ---- Evaluate --------------------------------------------------
    metrics = evaluate_seeds(
        G=G,
        seeds=result.seeds,
        context_fn=context_fn,
        reward_fn=reward_fn,
        memory=memory,
        p=p, T=T,
        num_simulations=R,
        show_progress=True,
    )

    print(
        f"\n  sigma_C     : {metrics['mean']:.4f} ± {metrics['std']:.4f}"
        f"\n  efficiency  : {metrics['reward_efficiency']:.4f}"
        f"\n  satisfaction: {metrics['context_satisfaction']:.2f}%"
        f"\n  runtime     : {elapsed:.1f}s"
    )

    # ---- Save results ----------------------------------------------
    out = {
        'context_type':       context_type,
        'budget':             budget,
        'seeds':              result.seeds,
        'sigma_c':            metrics['mean'],
        'sigma_c_std':        metrics['std'],
        'reward_efficiency':  metrics['reward_efficiency'],
        'context_satisfaction': metrics['context_satisfaction'],
        'runtime_seconds':    elapsed,
        'marginal_gains':     result.marginal_gains,
        'config': {
            'R': R, 'M': M, 'memory': memory,
            'prune_percentile': prune_percentile,
            'p': p, 'T': T,
        },
    }

    tag = os.path.splitext(os.path.basename(graph_path))[0]
    out_path = os.path.join(save_dir, f'{tag}_{context_type}_B{budget}_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    return out


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run CDS-IM experiment')

    parser.add_argument('--graph',    type=str, required=True,
                        help='Path to graph file (.edgelist or .json)')
    parser.add_argument('--context',  type=str, default='binary',
                        choices=list(CONTEXT_PRESETS.keys()),
                        help='Context type preset')
    parser.add_argument('--attrs',    type=str, default=None,
                        help='Path to node attributes JSON (for weighted reward)')
    parser.add_argument('--budget',   type=int, default=50,
                        help='Seed budget B')
    parser.add_argument('--R',        type=int, default=1000,
                        help='MC rollouts for influence estimation')
    parser.add_argument('--M',        type=int, default=1000,
                        help='MC samples for dominance estimation')
    parser.add_argument('--memory',   type=int, default=3,
                        help='Context memory order k')
    parser.add_argument('--prune',    type=float, default=90.0,
                        help='Pruning percentile (default 90 = top 10%%)')
    parser.add_argument('--p',        type=float, default=0.1,
                        help='Default edge propagation probability')
    parser.add_argument('--T',        type=int, default=20,
                        help='Diffusion time horizon')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory for output files')

    args = parser.parse_args()

    run(
        graph_path=args.graph,
        context_type=args.context,
        node_attrs_path=args.attrs,
        budget=args.budget,
        R=args.R, M=args.M,
        memory=args.memory,
        prune_percentile=args.prune,
        p=args.p, T=args.T,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
