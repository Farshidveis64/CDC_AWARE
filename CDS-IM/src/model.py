"""
Context-Dominant Selection (CDS) Algorithm.

Implements Algorithm 2 from the paper (Section 4.3).

Algorithm 2 — CDS:
    Phase 1  (Context Dominance Analysis):
        {D(v)} <- EstimateContextDominance(G, C, f, M)
        theta  <- Percentile({D(v)}, alpha)
        V'     <- {v in V : D(v) >= theta}

    Phase 2  (Greedy Selection with Lazy Evaluation):
        S <- {}
        delta_bar(v) <- inf  for all v in V'
        for i = 1..B:
            repeat:
                v* <- argmax_{v in V'S}  delta_bar(v)
                recompute delta_bar(v*) = sigma_C(S+{v*}) - sigma_C(S)
            until v* unchanged
            S <- S + {v*}

Theorem 5 (Algorithm Guarantee):
    Under (kappa, epsilon)-BCI, with prob >= 1-delta:
        sigma_C(S) >= (1 - exp(-(1 - kappa/B))) * sigma_C(S*)
                      - B*eps - ell*theta*T - O(sqrt(log(1/delta)/R))

Theorem 7 (Complexity):
    O( M*T*|E|  +  B*|V'|*R*T*|E| / L )
    where L >= 1 is the lazy evaluation speedup factor.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import networkx as nx
from tqdm import tqdm

from context import ContextSpace, RewardFunction
from dominance import estimate_dominance, prune_candidates
from ic_simulation import monte_carlo_context_spread


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class CDSResult:
    """
    Output of the CDS algorithm.

    Attributes:
        seeds:            Ordered list of selected seed node IDs.
        dominance_scores: D(v) for all nodes (full graph).
        candidate_set:    Pruned candidate set V'.
        theta:            Pruning threshold used.
        marginal_gains:   Marginal gain at each greedy step.
        influence:        Final estimated sigma_C(S).
    """
    seeds:            List[int]
    dominance_scores: Dict[int, float]
    candidate_set:    Set[int]
    theta:            float
    marginal_gains:   List[float] = field(default_factory=list)
    influence:        float = 0.0

    def __repr__(self):
        return (
            f"CDSResult(seeds={self.seeds}, "
            f"influence={self.influence:.4f}, "
            f"|V'|={len(self.candidate_set)})"
        )


# ------------------------------------------------------------------
# Main algorithm
# ------------------------------------------------------------------

class CDS:
    """
    Context-Dominant Selection algorithm.

    Combines:
      - Context dominance analysis for candidate pruning  (Phase 1)
      - Greedy seed selection with lazy evaluation        (Phase 2)

    Works with ANY ContextSpace / RewardFunction — domain logic
    lives entirely in those objects.

    Args:
        context_space:    Defines valid context space C and memory k.
        reward_fn:        f(c) -> float reward.
        budget:           Seed budget B.
        R:                MC rollouts per influence estimate.
        M:                MC samples for dominance estimation.
        prune_percentile: alpha — retain top (100-alpha)% by dominance.
        p:                Default edge propagation probability.
        T:                Diffusion time horizon.
        verbose:          Print progress to stdout.
    """

    def __init__(
        self,
        context_space: ContextSpace,
        reward_fn: RewardFunction,
        budget: int = 50,
        R: int = 1000,
        M: int = 1000,
        prune_percentile: float = 90.0,
        p: float = 0.1,
        T: int = 20,
        verbose: bool = True,
    ):
        self.context_space    = context_space
        self.reward_fn        = reward_fn
        self.budget           = budget
        self.R                = R
        self.M                = M
        self.prune_percentile = prune_percentile
        self.p                = p
        self.T                = T
        self.verbose          = verbose

    # ---- public entry point ----------------------------------------

    def select_seeds(self, G: nx.DiGraph) -> CDSResult:
        """
        Run CDS on graph G and return selected seed set.

        Args:
            G: Directed graph with optional edge attribute 'prob'.

        Returns:
            CDSResult containing seeds, dominance scores, and influence.
        """
        # ---- Phase 1: Context Dominance Analysis -------------------
        if self.verbose:
            print(f"\n[CDS / {self.context_space.name}]")
            print(f"  Phase 1 — Dominance estimation  (M={self.M})")

        dominance = estimate_dominance(
            G, self.context_space, self.reward_fn,
            M=self.M, p=self.p, T=self.T,
            show_progress=self.verbose,
        )

        candidates, theta = prune_candidates(dominance, self.prune_percentile)

        if self.verbose:
            print(
                f"  Pruned  {G.number_of_nodes()} -> {len(candidates)} candidates "
                f"(top {100 - self.prune_percentile:.0f}%,  theta={theta:.5f})"
            )

        # ---- Phase 2: Greedy Selection with Lazy Evaluation --------
        if self.verbose:
            print(f"  Phase 2 — Greedy selection  (B={self.budget}, R={self.R})")

        seeds: List[int] = []
        remaining: Set[int] = set(candidates)
        delta_bar: Dict[int, float] = {v: math.inf for v in remaining}
        marginal_gains: List[float] = []
        current_influence: Optional[float] = None

        pbar = tqdm(total=self.budget, desc="  Selecting seeds") if self.verbose else None

        for _ in range(self.budget):
            if not remaining:
                break

            # Lazy evaluation loop
            while True:
                v_star = max(remaining, key=lambda v: delta_bar[v])

                # Compute current influence (cached)
                if current_influence is None:
                    current_influence, _ = monte_carlo_context_spread(
                        G, set(seeds), self.context_space, self.reward_fn,
                        self.context_space.memory, self.p, self.T, self.R,
                    )

                # Compute true marginal gain for v_star
                new_inf, _ = monte_carlo_context_spread(
                    G, set(seeds) | {v_star}, self.context_space, self.reward_fn,
                    self.context_space.memory, self.p, self.T, self.R,
                )
                delta_bar[v_star] = new_inf - current_influence

                # Check if v_star is still the best
                if max(remaining, key=lambda v: delta_bar[v]) == v_star:
                    break

            seeds.append(v_star)
            remaining.remove(v_star)
            marginal_gains.append(delta_bar[v_star])
            current_influence += delta_bar[v_star]

            if pbar:
                pbar.set_postfix(node=v_star, gain=f"{delta_bar[v_star]:.3f}")
                pbar.update(1)

        if pbar:
            pbar.close()

        if self.verbose:
            print(f"  Done — sigma_C(S) ≈ {current_influence:.4f}")

        return CDSResult(
            seeds=seeds,
            dominance_scores=dominance,
            candidate_set=candidates,
            theta=theta,
            marginal_gains=marginal_gains,
            influence=current_influence or 0.0,
        )
