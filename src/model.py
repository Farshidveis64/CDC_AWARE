"""
Context-Dominant Selection (CDS) Algorithm.

Implements Algorithm 2 from the paper (Section 4.3).

Algorithm 2 — CDS:
    Phase 1 (Context Dominance Analysis):
        {D(v)} <- EstimateContextDominance(G, C, f, M)
        theta  <- Percentile({D(v)}, alpha)
        V'     <- {v in V : D(v) >= theta}

    Phase 2 (Greedy Selection with Lazy Evaluation):
        S <- empty, delta_bar(v) <- inf for all v in V'
        for i = 1..B:
            repeat:
                v* <- argmax_{v in V' minus S} delta_bar(v)
                delta_bar(v*) <- sigma_C(S + {v*}) - sigma_C(S)
            until v* unchanged
            S <- S + {v*}

Theorem 5 (Algorithm Guarantee):
    Under (kappa, epsilon)-BCI, with prob >= 1-delta:
        sigma_C(S) >= (1 - exp(-(1 - kappa/B))) * sigma_C(S*)
                      - B*eps - ell*theta*T - O(sqrt(log(1/delta)/R))

Theorem 7 (Complexity):
    O(M*T*|E| + B*|V'|*R*T*|E| / L)
    where L >= 1 is the lazy evaluation speedup factor.
"""

import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import networkx as nx
from tqdm import tqdm

from context import ContextSpace, RewardFunction
from dominance import estimate_dominance, prune_candidates
from ic_simulation import monte_carlo_context_spread


@dataclass
class CDSResult:
    """
    Output of the CDS algorithm.

    Attributes:
        seeds:            Ordered list of selected seed node IDs.
        dominance_scores: D(v) for all nodes.
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
    influence:        float       = 0.0

    def __repr__(self):
        return (
            f"CDSResult(seeds={self.seeds}, "
            f"influence={self.influence:.4f}, "
            f"|V'|={len(self.candidate_set)})"
        )


class CDS:
    """
    Context-Dominant Selection (CDS) algorithm.

    Args:
        context_space:    Defines valid context space C and memory k.
        reward_fn:        f(c) -> float reward.
        budget:           Seed budget B.
        R:                MC rollouts per influence estimate.
        M:                MC samples for dominance estimation.
        prune_percentile: alpha — retain top (100-alpha)% by dominance.
        p:                Default edge propagation probability.
        T:                Diffusion time horizon.
        seed:             Random seed for reproducibility (None = no fix).
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
        seed: Optional[int] = None,
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
        self.seed             = seed
        self.verbose          = verbose

    def select_seeds(
        self,
        G: nx.DiGraph,
        context_fn: Optional[Callable] = None,
    ) -> CDSResult:
        """
        Run CDS on graph G and return the selected seed set.

        Args:
            G:          Directed graph with optional edge attribute 'prob'.
            context_fn: callable(predecessors: List[int], node: int) -> Context.
                        Injects node attributes into contexts at evaluation time.
                        REQUIRED for weighted/predicate-based context types.
                        If None, context_space is used directly (binary only).

        Returns:
            CDSResult with seeds, dominance scores, and influence estimate.
        """
        if self.seed is not None:
            random.seed(self.seed)

        # Resolve context function
        _ctx_fn = context_fn if context_fn is not None else self.context_space

        # ---- Phase 1: Context Dominance Analysis -------------------
        if self.verbose:
            print(f"\n[CDS / {self.context_space.name}]")
            print(f"  Phase 1 — Dominance estimation  (M={self.M})")

        dominance = estimate_dominance(
            G, self.context_space, self.reward_fn,
            context_fn=_ctx_fn,
            M=self.M, p=self.p, T=self.T,
            show_progress=self.verbose,
        )

        candidates, theta = prune_candidates(dominance, self.prune_percentile)

        # Guard: if budget exceeds candidate set, warn and use all candidates
        if len(candidates) < self.budget:
            warnings.warn(
                f"Candidate set |V'|={len(candidates)} < budget B={self.budget}. "
                f"Returning {len(candidates)} seeds. "
                f"Consider lowering prune_percentile.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.verbose:
            print(
                f"  Pruned  {G.number_of_nodes()} -> {len(candidates)} candidates "
                f"(top {100 - self.prune_percentile:.0f}%,  theta={theta:.5f})"
            )

        # ---- Phase 2: Greedy Selection with Lazy Evaluation --------
        if self.verbose:
            print(f"  Phase 2 — Greedy selection  (B={self.budget}, R={self.R})")

        seeds:             List[int]      = []
        remaining:         Set[int]       = set(candidates)
        delta_bar:         Dict[int, float] = {v: math.inf for v in remaining}
        marginal_gains:    List[float]    = []
        # Cache: recomputed fresh after each seed is added (not incremental)
        # to avoid drift that accumulates with the += approximation.
        current_influence: Optional[float] = None

        pbar = tqdm(total=min(self.budget, len(candidates)),
                    desc="  Selecting seeds") if self.verbose else None

        for _ in range(min(self.budget, len(candidates))):
            if not remaining:
                break

            # Compute sigma_C(S) once per outer iteration (cached across lazy loop)
            if current_influence is None:
                current_influence, _ = monte_carlo_context_spread(
                    G, set(seeds), _ctx_fn, self.reward_fn,
                    self.context_space.memory, self.p, self.T, self.R,
                )

            # Lazy evaluation (CELF) loop
            while True:
                v_star = max(remaining, key=lambda v: delta_bar[v])

                new_inf, _ = monte_carlo_context_spread(
                    G, set(seeds) | {v_star}, _ctx_fn, self.reward_fn,
                    self.context_space.memory, self.p, self.T, self.R,
                )
                delta_bar[v_star] = new_inf - current_influence

                if max(remaining, key=lambda v: delta_bar[v]) == v_star:
                    break

            seeds.append(v_star)
            remaining.remove(v_star)
            marginal_gains.append(delta_bar[v_star])

            # Reset cache: recompute fresh at the start of next outer iteration
            # This is more expensive than += but avoids cumulative drift.
            current_influence = None

            if pbar:
                pbar.set_postfix(node=v_star, gain=f"{delta_bar[v_star]:.3f}")
                pbar.update(1)

        if pbar:
            pbar.close()

        # Final influence estimate on the complete seed set
        final_influence, _ = monte_carlo_context_spread(
            G, set(seeds), _ctx_fn, self.reward_fn,
            self.context_space.memory, self.p, self.T, self.R,
        )

        if self.verbose:
            print(f"  Done — sigma_C(S) = {final_influence:.4f}")

        return CDSResult(
            seeds=seeds,
            dominance_scores=dominance,
            candidate_set=candidates,
            theta=theta,
            marginal_gains=marginal_gains,
            influence=final_influence,
        )
