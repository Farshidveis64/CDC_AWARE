"""
Data loading and graph utilities for CDS-IM.

Handles:
  - Loading graphs from edge list / JSON / pickle
  - Assigning edge propagation probabilities
  - Computing node attribute dicts for context enrichment
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


# ------------------------------------------------------------------
# Graph loading
# ------------------------------------------------------------------

class GraphLoader:
    """
    Load directed graphs from various file formats.

    Supported formats:
        .edgelist  — space-separated  src  dst  [prob]
        .json      — {"edges": [[u,v], ...], "probs": {edge: prob}}
        .pkl       — networkx pickle

    Args:
        default_prob: Edge probability to use when not in the file.
    """

    def __init__(self, default_prob: float = 0.1):
        self.default_prob = default_prob

    def load(self, path: str) -> nx.DiGraph:
        """
        Load graph from file.

        Args:
            path: Path to graph file.

        Returns:
            nx.DiGraph with 'prob' attribute on every edge.
        """
        if path.endswith('.edgelist') or path.endswith('.txt'):
            return self._load_edgelist(path)
        elif path.endswith('.json'):
            return self._load_json(path)
        elif path.endswith('.pkl') or path.endswith('.gpickle'):
            return nx.read_gpickle(path)
        else:
            raise ValueError(f"Unsupported format: {path}")

    def _load_edgelist(self, path: str) -> nx.DiGraph:
        G = nx.DiGraph()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                u, v = int(parts[0]), int(parts[1])
                prob = float(parts[2]) if len(parts) > 2 else self.default_prob
                G.add_edge(u, v, prob=prob)
        return G

    def _load_json(self, path: str) -> nx.DiGraph:
        with open(path) as f:
            data = json.load(f)
        G = nx.DiGraph()
        probs = data.get('probs', {})
        for u, v in data['edges']:
            prob = probs.get(f"{u},{v}", self.default_prob)
            G.add_edge(u, v, prob=prob)
        return G


# ------------------------------------------------------------------
# Node attribute store
# ------------------------------------------------------------------

class NodeAttributeStore:
    """
    Stores per-node attributes that are attached to contexts at runtime.

    The attribute dict for each node is used by the reward function
    to evaluate quality criteria such as:
        verified (bool), topic (str), karma (float), category (str), etc.

    Usage:
        store = NodeAttributeStore.from_json('data/twitter/node_attrs.json')
        attrs = store.get(node_id)

    Args:
        attr_dict: Dict mapping node_id (int) -> Dict[str, Any].
    """

    def __init__(self, attr_dict: Optional[Dict[int, Dict]] = None):
        self._attrs: Dict[int, Dict] = attr_dict or {}

    @classmethod
    def from_json(cls, path: str) -> 'NodeAttributeStore':
        """
        Load node attributes from a JSON file.

        Expected format:
            { "0": {"verified": true, "topic": "tech", ...}, ... }

        Args:
            path: Path to JSON file.

        Returns:
            NodeAttributeStore instance.
        """
        with open(path) as f:
            raw = json.load(f)
        attr_dict = {int(k): v for k, v in raw.items()}
        return cls(attr_dict)

    @classmethod
    def empty(cls) -> 'NodeAttributeStore':
        """Create a store with no attributes (binary reward mode)."""
        return cls({})

    def get(self, node: int) -> Dict[str, Any]:
        """
        Get attributes for a node.

        Args:
            node: Node ID.

        Returns:
            Attribute dict; empty dict if node has no attributes.
        """
        return self._attrs.get(node, {})

    def make_context_fn(self, context_space) -> Callable:
        """
        Build a context_fn compatible with ic_simulation that
        auto-attaches node attributes from this store.

        Args:
            context_space: ContextSpace to use for building contexts.

        Returns:
            callable(predecessors, node) -> Context with attrs.
        """
        def context_fn(predecessors, node):
            attrs = self.get(node)
            return context_space.build_context(predecessors, node, attrs)
        return context_fn


# ------------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------------

class CDSDataset:
    """
    Dataset class for a single graph with context specification.

    Bundles graph, node attributes, context space, and reward
    function into one object — mirrors GraphRAGIMDataset pattern.

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
        loader = GraphLoader(default_prob)
        self.G = loader.load(graph_path)

        if node_attrs_path and os.path.exists(node_attrs_path):
            self.attr_store = NodeAttributeStore.from_json(node_attrs_path)
        else:
            self.attr_store = NodeAttributeStore.empty()

        print(
            f"Loaded graph: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        )

    def get_context_fn(self, context_space) -> Callable:
        """
        Build context function with auto-attached node attributes.

        Args:
            context_space: The ContextSpace for this experiment.

        Returns:
            callable(predecessors, node) -> Context.
        """
        return self.attr_store.make_context_fn(context_space)

    def summary(self) -> Dict[str, Any]:
        """Return basic graph statistics."""
        return {
            'nodes':            self.G.number_of_nodes(),
            'edges':            self.G.number_of_edges(),
            'avg_degree':       np.mean([d for _, d in self.G.degree()]),
            'density':          nx.density(self.G),
            'has_node_attrs':   len(self.attr_store._attrs) > 0,
        }
