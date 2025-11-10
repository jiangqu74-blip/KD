"""Synthetic hypergraph data generation utilities.

This module provides helpers for constructing the synthetic hypergraph dataset
used throughout the project.  The dataset mimics multi-participation group
interactions with long-tailed hyperedge sizes.  It is designed so that the
teacher network can exploit hypergraph structure while the student remains a
0-hop model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class HypergraphData:
    """Container for synthetic hypergraph outputs.

    Parameters
    ----------
    X:
        Node features with shape ``(N, D)`` where ``N`` is the number of nodes
        and ``D`` is the feature dimension.
    y:
        Integer node labels with shape ``(N,)``.
    H:
        Binary incidence matrix with shape ``(N, E)`` describing membership of
        nodes (rows) in hyperedges (columns).
    train_mask, val_mask, test_mask:
        Boolean masks covering mutually exclusive splits.
    """

    X: np.ndarray
    y: np.ndarray
    H: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Return the data as a plain dictionary (便于存档/序列化)."""

        return {
            "X": self.X,
            "y": self.y,
            "H": self.H,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask,
        }


def _power_law_sizes(
    rng: np.random.Generator,
    n_edges: int,
    min_edge: int,
    max_edge: int,
    alpha: float = 2.5,
) -> np.ndarray:
    """Sample hyperedge sizes from a truncated Pareto distribution."""

    raw = min_edge * (1.0 + rng.pareto(alpha, size=n_edges))
    sizes = np.floor(raw).astype(int)
    upper = min(max_edge, max_edge if max_edge > 0 else sizes.max())
    sizes = np.clip(sizes, min_edge, upper)
    sizes = np.clip(sizes, min_edge, max_edge)
    return sizes


def make_synth_hypergraph(
    n_nodes: int = 4000,
    n_classes: int = 6,
    feat_dim: int = 32,
    n_edges: int = 3000,
    p_mixed: float = 0.35,
    min_edge: int = 3,
    max_edge: int = 60,
    seed: int = 1,
) -> Dict[str, np.ndarray]:
    r"""Construct a synthetic hypergraph with long-tailed hyperedge sizes.

    This generator purposely creates *hypergraphs* (not 2-section graphs) by
    sampling an incidence matrix ``H \in {0,1}^{|V|\times|E|}`` directly.  Node
    features are sampled from Gaussian clusters corresponding to latent
    categories, and hyperedges either connect nodes mostly within a single
    category or mix classes according to ``p_mixed``.

    Parameters
    ----------
    n_nodes:
        Number of nodes (|V|).
    n_classes:
        Number of latent classes used to generate feature clusters.
    feat_dim:
        Dimensionality of node features.
    n_edges:
        Number of hyperedges (|E|).
    p_mixed:
        Probability of sampling a mixed hyperedge that pulls members from
        multiple classes.
    min_edge / max_edge:
        Minimum and maximum hyperedge cardinality.  The maximum is clipped to
        ``n_nodes`` internally to avoid invalid sampling.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing features, labels, incidence matrix and dataset
        masks.  The arrays are ``float32``/``int64``/``bool`` for downstream
        compatibility.
    """

    rng = np.random.default_rng(seed)
    n_nodes = int(n_nodes)
    n_edges = int(n_edges)
    max_edge = min(int(max_edge), n_nodes)
    min_edge = max(int(min_edge), 2)

    # Sample Gaussian clusters for node features.
    nodes_per_class = np.full(n_classes, n_nodes // n_classes, dtype=int)
    nodes_per_class[: n_nodes % n_classes] += 1
    means = rng.normal(0.0, 5.0, size=(n_classes, feat_dim)).astype(np.float32)
    cov_scale = rng.uniform(0.4, 1.5, size=n_classes).astype(np.float32)

    features = []
    labels = []
    start = 0
    for cls, count in enumerate(nodes_per_class):
        if count == 0:
            continue
        eps = rng.normal(0.0, cov_scale[cls], size=(count, feat_dim)).astype(np.float32)
        features.append(means[cls] + eps)
        labels.extend([cls] * count)
        start += count

    X = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    # Pre-compute class-biased sampling weights for homogeneous edges.
    class_indices = [np.flatnonzero(y == cls) for cls in range(n_classes)]
    class_weights: list[np.ndarray] = []
    for cls_indices in class_indices:
        weights = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float64)
        if len(cls_indices) > 0:
            bias = 0.85
            other = np.setdiff1d(np.arange(n_nodes), cls_indices, assume_unique=True)
            if len(other) == 0:
                weights = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float64)
            else:
                weights[cls_indices] = bias / len(cls_indices)
                weights[other] = (1.0 - bias) / len(other)
        weights /= weights.sum()
        class_weights.append(weights)

    sizes = _power_law_sizes(rng, n_edges, min_edge, max_edge)
    H = np.zeros((n_nodes, n_edges), dtype=np.float32)

    for e_idx, size in enumerate(sizes):
        size = int(min(size, n_nodes))
        if size == 0:
            continue
        if rng.random() < p_mixed:
            members = rng.choice(n_nodes, size=size, replace=False)
        else:
            cls = int(rng.integers(0, n_classes))
            members = rng.choice(
                n_nodes, size=size, replace=False, p=class_weights[cls]
            )
        H[members, e_idx] = 1.0

    # Dataset splits (train/val/test) with fixed ratios.
    idx = rng.permutation(n_nodes)
    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train : n_train + n_val]] = True
    test_mask[idx[n_train + n_val :]] = True

    data = HypergraphData(
        X=X,
        y=y,
        H=H,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data.as_dict()


def degree_mats(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute node degrees ``Dv`` and hyperedge sizes ``De`` (含 epsilon).

    Parameters
    ----------
    H:
        Incidence matrix ``(N, E)`` stored as ``float32``/``float64``/``int``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``Dv`` and ``De`` are one-dimensional arrays (length ``N`` and ``E``)
        containing node degrees and edge cardinalities with a small epsilon to
        avoid division-by-zero issues in downstream normalisation.
    """

    H = np.asarray(H, dtype=np.float32)
    dv = H.sum(axis=1, dtype=np.float32) + 1e-6
    de = H.sum(axis=0, dtype=np.float32) + 1e-6
    return dv, de
