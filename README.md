# CDS-IM: Context-Dominant Selection for Influence Maximization

[![Paper](https://img.shields.io/badge/Paper-ESWA%202026-blue)](link)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Anonymous submission — double-blind review at **Expert Systems with Applications**.

Official implementation of **CDS-IM**, a path-dependent reward framework
for quality-aware influence maximization. Unlike classical methods that
maximize *how many* nodes are activated, CDS-IM maximizes *how valuable*
the activations are by accounting for the propagation context.

---

## 📊 Results

| Method | Twitter | StackOF | Amazon | Wiki | Reddit | Enron |
|--------|---------|---------|--------|------|--------|-------|
| IMM | 1,623 | 1,267 | 1,893 | 2,389 | 1,498 | 1,134 |
| CELF-Context | 2,034 | 1,589 | 2,389 | 3,012 | 1,889 | 1,423 |
| **CDS (Ours)** | **2,512** | **1,978** | **2,978** | **3,789** | **2,367** | **1,789** |
| *Improvement* | *+23.5%* | *+24.5%* | *+24.7%* | *+25.8%* | *+25.3%* | *+25.7%* |

---

## 🏗️ Architecture

```
Phase 1: Dominance Analysis        Phase 2: Greedy Selection
┌───────────────────────┐     ┌────────────────────────────┐
│  M random diffusions  │     │  Lazy evaluation loop      │
│  Extract contexts     │────▶│  sigma_C(S ∪ {v*}) - σ(S) │
│  D(v) = Σ f(c)*P[c]  │     │  Add v* with max gain      │
│  Prune top 10%  → V' │     │  Repeat for B seeds        │
└───────────────────────┘     └────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/anonymous-review/CDS-IM.git
cd CDS-IM
pip install -r requirements.txt
```

### Quick test (no data needed)

```bash
python example.py
```

This runs all **six context types** on a synthetic graph and prints
sigma_C, reward efficiency, and context satisfaction for each.

### Run on your graph

```bash
# Binary context (recovers classical IM)
python src/train.py \
    --graph  data/enron/graph.edgelist \
    --context binary \
    --budget  50

# Thread-structure context (Enron email)
python src/train.py \
    --graph  data/enron/graph.edgelist \
    --attrs  data/enron/node_attrs.json \
    --context thread_structure \
    --budget  50 \
    --R 1000 --M 1000
```

### Use in your own code

```python
import sys
sys.path.append('src')

import networkx as nx
from context import make_context_space, RewardFunction
from model import CDS

# Load your graph
G = nx.read_edgelist('graph.edgelist', create_using=nx.DiGraph(),
                     data=[('prob', float)])

# Define context: valid when 'verified' attribute is True
cs = make_context_space(
    memory=3,
    predicate=lambda c: c.attrs.get('verified', False),
    name='verified_source'
)
rf = RewardFunction(cs, mode='binary')

# Run CDS
cds = CDS(cs, rf, budget=50, R=1000, M=1000)
result = cds.select_seeds(G)

print("Seeds:", result.seeds)
print("Influence:", result.influence)
```

---

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

## 📖 Citation

```bibtex
@article{anonymous2026cds,
  title   = {Context-Constrained Influence Maximization:
             A Path-Dependent Reward Framework},
  author  = {Anonymous Authors},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under review}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
