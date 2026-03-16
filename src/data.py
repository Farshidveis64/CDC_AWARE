"""
Data loading and graph utilities for CDS-IM.
"""

import json
import os
import pickle
from typing import Any, Callable, Dict, Optional

import networkx as nx
import numpy as np


class GraphLoader:
    """
    Load directed graphs from various file formats.

    Formats supported:
        .edgelist / .txt  — space-separated: src dst [prob]
        .json             — {"edges": [[u,v], ...], "probs": {"u,v": prob}}
        .pkl              — standard Python pickle of nx.DiGraph

    Args:
        default_prob: Edge probability when not specified in the file.
    """

    def __init__(self, default_prob: float = 0.1):
        self.default_prob = default_prob

    def load(self, path: str) -> nx.DiGraph:
        if path.endswith('.edgelist') or path.endswith('.txt'):
            return self._load_edgelist(path)
        elif path.endswith('.json'):
            return self._load_json(path)
        elif path.endswith('.pkl') or path.endswith('.gpickle'):
            # FIX: nx.read_gpickle removed in NetworkX 3.x — use stdlib pickle
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported graph format: '{path}'. "
                f"Use .edgelist, .json, or .pkl"
            )

    def _load_edgelist(self, path: str) -> nx.DiGraph:
        G = nx.DiGraph()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                u, v  = int(parts[0]), int(parts[1])
                prob  = float(parts[2]) if len(parts) > 2 else self.default_prob
                G.add_edge(u, v, prob=prob)
        return G

    def _load_json(self, path: str) -> nx.DiGraph:
        with open(path) as f:
            data = json.load(f)
        G     = nx.DiGraph()
        probs = data.get('probs', {})
        for u, v in data['edges']:
            prob = probs.get(f"{u},{v}", self.default_prob)
            G.add_edge(u, v, prob=prob)
        return G


class NodeAttributeStore:
    """
    Per-node attributes attached to contexts at runtime.

    These attributes are what make non-binary context predicates work.
    Without them, predicates like phi(c) = c.attrs.get('verified')
    always receive an empty dict and always return False.

    JSON format expected:
        { "0": {"verified": true, "karma": 0.8, ...}, "1": {...}, ... }

    Args:
        attr_dict: Dict mapping int node_id -> Dict[str, Any].
    """

    def __init__(self, attr_dict: Optional[Dict[int, Dict]] = None):
        self._attrs: Dict[int, Dict] = attr_dict or {}

    @classmethod
    def from_json(cls, path: str) -> 'NodeAttributeStore':
        with open(path) as f:
            raw = json.load(f)
        return cls({int(k): v for k, v in raw.items()})

    @classmethod
    def empty(cls) -> 'NodeAttributeStore':
        """No attributes — use only with binary context."""
        return cls({})

    def get(self, node: int) -> Dict[str, Any]:
        """Return attribute dict for node; {} if not found."""
        return self._attrs.get(node, {})

    def make_context_fn(self, context_space) -> Callable:
        """
        Build context_fn that auto-attaches node attributes.

        This is the key bridge between NodeAttributeStore and the IC
        simulation. Pass the returned function as context_fn to CDS,
        estimate_dominance, and evaluate_seeds.

        Args:
            context_space: ContextSpace to use for building contexts.

        Returns:
            callable(predecessors: List[int], node: int) -> Context
        """
        def context_fn(predecessors, node):
            return context_space.build_context(
                predecessors, node, self.get(node)
            )
        return context_fn


class CDSDataset:
    """
    Dataset wrapper: loads graph + node attributes from disk.

    Args:
        graph_path:      Path to graph file.
        node_attrs_path: Path to node attributes JSON (optional).
        default_prob:    Default edge propagation probability.
    """

    def __init__(
        self,
        graph_path: str,
        node_attrs_path: Optional[str] = None,
        default_prob: float = 0.1,
    ):
        self.G = GraphLoader(default_prob).load(graph_path)

        if node_attrs_path and os.path.exists(node_attrs_path):
            self.attr_store = NodeAttributeStore.from_json(node_attrs_path)
        else:
            self.attr_store = NodeAttributeStore.empty()

        print(
            f"Loaded: {self.G.number_of_nodes():,} nodes, "
            f"{self.G.number_of_edges():,} edges  "
            f"(node attrs: {'yes' if self.attr_store._attrs else 'no'})"
        )

    def get_context_fn(self, context_space) -> Callable:
        """
        Build context_fn with auto-attached node attributes.
        Pass this to CDS.select_seeds() and evaluate_seeds().
        """
        return self.attr_store.make_context_fn(context_space)

    def summary(self) -> Dict[str, Any]:
        return {
            'nodes':          self.G.number_of_nodes(),
            'edges':          self.G.number_of_edges(),
            'avg_out_degree': np.mean([d for _, d in self.G.out_degree()]),
            'density':        nx.density(self.G),
            'has_node_attrs': bool(self.attr_store._attrs),
        }
