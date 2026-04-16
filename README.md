# Context-Constrained Influence Maximization: A
Path-Dependent Reward Framework

[![Paper](https://img.shields.io/badge/Paper-ESWA%202026-blue)](link)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Anonymous submission — double-blind review at **Expert Systems with Applications**.

Official implementation of **CDS-IM**, a path-dependent reward framework
for quality-aware influence maximization. Unlike classical methods that
maximize *how many* nodes are activated, CDS-IM maximizes *how valuable*
the activations are by accounting for the propagation context.



## 🏗️ Architecture

Mirrors **Figure 1** from the paper — three phases, two side annotations.

```
     ┌──────────────────────────────────────────────────────┐
     │  Input: Graph G, Context Space C, Reward f, Budget B │
     └──────────────────────────┬───────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────┐   - - -  ┌──────────────────┐
     │  Phase 1: Context Dominance Analysis                 │◀ - - - - │  Monte Carlo     │
     │  D(v) = Σ f(c) · P[c reachable],  c ∈ C, v ∈ c     │           │  M diffusions    │
     │  Algorithm 1 · Definition 4                          │           └──────────────────┘
     └──────────────────────────┬───────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────┐
     │  Phase 2: Candidate Pruning                          │
     │  V' = { v ∈ V : D(v) ≥ θ }  ·  top 10% retained    │
     │  Definition 5 · Theorem 6                            │
     └──────────────────────────┬───────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────┐   - - -  ┌──────────────────┐
     │  Phase 3: Greedy Selection + Lazy Evaluation         │◀ - - - - │  Context-aware   │
     │  v* = arg max Δ̄(v|S),  v ∈ V'\S                    │           │  R rollouts      │
     │  repeat until v* stable, then S ← S ∪ {v*}          │           └──────────────────┘
     │  Algorithm 2 · Theorem 5                             │
     └──────────────────────────┬───────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────┐
     │  Output: Seed Set S,  |S| = B,  maximises σ_C(S)     │
     └──────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
CDS-IM/
├── src/
│   ├── context.py          # Context space + reward function (Defs 1–3)
│   ├── ic_simulation.py    # IC model with context tracking
│   ├── dominance.py        # Algorithm 1: EstimateContextDominance
│   ├── model.py            # Algorithm 2: CDS (main algorithm)
│   ├── data.py             # Graph loading + node attribute store
│   └── train.py            # Experiment runner (reproduces all tables)
├── data/
│   ├── enron/
│   │   ├── graph.edgelist
│   │   └── node_attrs.json
│   └── README.md           # Dataset download instructions
├── results/                # Saved JSON results + seed nodes
├── example.py              # Quick test on synthetic data
├── requirements.txt
└── README.md
```

---

## 🔄 Context Types

The framework supports **any** context predicate. Six presets match
the paper's datasets:

| `--context` flag | Domain | Predicate |
|-----------------|--------|-----------|
| `binary` | Any | Accept all (classical IM) |
| `verified_source` | Twitter-Ads | Source node is verified |
| `tag_coherence` | StackOverflow | Tag similarity > 0.5 |
| `category_path` | Amazon | Same product category path |
| `topic_coherence` | Wikipedia | Topic similarity > 0.6 |
| `subreddit_chain` | Reddit | Same subreddit chain |
| `thread_structure` | Enron | Within email thread |

To add your own context type, pass a custom predicate:

```python
my_cs = make_context_space(
    memory=3,
    predicate=lambda c: c.attrs.get('my_attribute') == 'some_value',
    name='my_context'
)
```

---

## 📋 Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--budget` / `B` | 50 | Seed budget |
| `--memory` / `k` | 3 | Context memory order |
| `--R` | 1000 | MC rollouts per influence estimate |
| `--M` | 1000 | MC samples for dominance estimation |
| `--prune` | 90.0 | Pruning percentile (keeps top 10%) |
| `--p` | 0.1 | Default edge propagation probability |
| `--T` | 20 | Diffusion time horizon |

---



---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
