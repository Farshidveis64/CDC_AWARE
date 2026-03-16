"""
Example: Quick test of CDS-IM on synthetic data.

Tests all six context types from the paper on a small random graph
without needing any real dataset files.

Run:
    python example.py
"""

import sys
sys.path.append('src')

import random
import networkx as nx

from context import ContextSpace, RewardFunction, make_context_space
from ic_simulation import evaluate_seeds
from model import CDS


# ------------------------------------------------------------------
# Synthetic graph
# ------------------------------------------------------------------

def create_synthetic_graph(num_nodes: int = 200, num_edges: int = 600) -> nx.DiGraph:
    """
    Create a synthetic directed graph for testing.

    Args:
        num_nodes: Number of nodes.
        num_edges: Number of directed edges.

    Returns:
        nx.DiGraph with 'prob' attribute on every edge.
    """
    G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
    for u, v in G.edges():
        G[u][v]['prob'] = random.uniform(0.05, 0.3)
    return G


def create_synthetic_attrs(G: nx.DiGraph) -> dict:
    """
    Assign random context attributes to every node.
    Simulates the attribute dicts that real datasets provide.

    Args:
        G: Input graph.

    Returns:
        Dict mapping node_id -> attribute dict.
    """
    topics = ['tech', 'sports', 'politics', 'science']
    subreddits = ['r/python', 'r/science', 'r/news', 'r/tech']
    attrs = {}
    for node in G.nodes():
        attrs[node] = {
            # Twitter-Ads
            'verified':        random.random() > 0.7,
            # StackOverflow
            'tag_similarity':  random.uniform(0, 1),
            'reputation':      random.uniform(0, 1),
            # Amazon
            'same_category':   random.random() > 0.5,
            'review_quality':  random.uniform(0, 1),
            # Wikipedia
            'topic_similarity': random.uniform(0, 1),
            # Reddit
            'same_subreddit':  random.random() > 0.5,
            'karma':           random.uniform(0, 1),
            # Enron
            'in_thread':       random.random() > 0.4,
            'thread_coherence': random.uniform(0, 1),
        }
    return attrs


# ------------------------------------------------------------------
# Context presets  (same as train.py — six domains)
# ------------------------------------------------------------------

CONTEXTS = {
    'binary':           (lambda m: make_context_space(m, None,                                         'binary'),          {}),
    'verified_source':  (lambda m: make_context_space(m, lambda c: c.attrs.get('verified', False),     'verified_source'), {'credibility': 0.4, 'topic_similarity': 0.4, 'recency': 0.2}),
    'tag_coherence':    (lambda m: make_context_space(m, lambda c: c.attrs.get('tag_similarity',0)>0.5,'tag_coherence'),   {'tag_similarity': 0.6, 'reputation': 0.4}),
    'category_path':    (lambda m: make_context_space(m, lambda c: c.attrs.get('same_category',False), 'category_path'),   {'category_distance': 0.5, 'review_quality': 0.5}),
    'topic_coherence':  (lambda m: make_context_space(m, lambda c: c.attrs.get('topic_similarity',0)>0.6,'topic_coherence'),{'topic_similarity': 1.0}),
    'subreddit_chain':  (lambda m: make_context_space(m, lambda c: c.attrs.get('same_subreddit',False),'subreddit_chain'), {'subreddit_relevance': 0.7, 'karma': 0.3}),
    'thread_structure': (lambda m: make_context_space(m, lambda c: c.attrs.get('in_thread',False),     'thread_structure'),{'thread_coherence': 0.6, 'participant_relevance': 0.4}),
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 65)
    print("CDS-IM  Quick Test  —  6 Context Types")
    print("=" * 65)

    # 1. Synthetic data
    print("\n1. Creating synthetic graph (200 nodes, 600 edges)...")
    G = create_synthetic_graph(200, 600)
    attrs = create_synthetic_attrs(G)

    # Attach attrs to context function
    def make_context_fn(cs):
        def context_fn(predecessors, node):
            return cs.build_context(predecessors, node, attrs.get(node, {}))
        return context_fn

    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # 2. Run CDS for each context type
    print("\n2. Running CDS  (B=10, R=200, M=200 for speed)...\n")
    print(f"{'Context':<20} {'sigma_C':>10} {'efficiency':>12} {'satisfaction':>14} {'seeds'}")
    print("-" * 75)

    for ctx_name, (cs_factory, weights) in CONTEXTS.items():
        cs = cs_factory(memory=3)
        mode = 'weighted' if weights else 'binary'
        rf = RewardFunction(cs, mode=mode, weights=weights)
        context_fn = make_context_fn(cs)

        cds = CDS(
            context_space=cs,
            reward_fn=rf,
            budget=10,
            R=200, M=200,
            prune_percentile=90.0,
            p=0.1, T=10,
            verbose=False,
        )
        result = cds.select_seeds(G)

        metrics = evaluate_seeds(
            G=G,
            seeds=result.seeds,
            context_fn=context_fn,
            reward_fn=rf,
            memory=3, p=0.1, T=10,
            num_simulations=200,
        )

        print(
            f"{ctx_name:<20} "
            f"{metrics['mean']:>10.3f} "
            f"{metrics['reward_efficiency']:>12.4f} "
            f"{metrics['context_satisfaction']:>13.1f}% "
            f"  {result.seeds[:5]}..."
        )

    # 3. Quick sanity check — compare CDS vs random
    print("\n3. CDS vs Random baseline (binary context, B=10)...")
    cs = make_context_space(memory=3)
    rf = RewardFunction(cs, mode='binary')
    context_fn = make_context_fn(cs)

    cds = CDS(cs, rf, budget=10, R=500, M=500, verbose=False)
    cds_result = cds.select_seeds(G)
    cds_metrics = evaluate_seeds(G, cds_result.seeds, context_fn, rf,
                                  num_simulations=500)

    random_seeds = random.sample(list(G.nodes()), 10)
    rnd_metrics  = evaluate_seeds(G, random_seeds, context_fn, rf,
                                   num_simulations=500)

    improvement = (
        (cds_metrics['mean'] - rnd_metrics['mean']) / max(rnd_metrics['mean'], 1e-9) * 100
    )

    print(f"   CDS spread    : {cds_metrics['mean']:.2f} ± {cds_metrics['std']:.2f}")
    print(f"   Random spread : {rnd_metrics['mean']:.2f} ± {rnd_metrics['std']:.2f}")
    print(f"   Improvement   : {improvement:+.1f}%")

    print("\n" + "=" * 65)
    print("Quick test completed successfully!")
    print("=" * 65)


if __name__ == '__main__':
    main()
