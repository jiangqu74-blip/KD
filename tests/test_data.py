"""Tests for synthetic hypergraph data generation utilities."""

from __future__ import annotations

import numpy as np

from src import data


def test_make_synth_hypergraph_shapes() -> None:
    """Synthetic hypergraph generator should return consistent shapes."""

    out = data.make_synth_hypergraph(
        n_nodes=120,
        n_classes=4,
        feat_dim=8,
        n_edges=80,
        seed=42,
    )
    assert out["X"].shape == (120, 8)
    assert out["y"].shape == (120,)
    assert out["H"].shape == (120, 80)
    assert out["X"].dtype == np.float32
    assert out["y"].dtype == np.int64
    assert out["H"].dtype == np.float32

    for key in ("train_mask", "val_mask", "test_mask"):
        mask = out[key]
        assert mask.dtype == bool
        assert mask.shape == (120,)
    # Masks should form a partition of the node set.
    combined = out["train_mask"].astype(int) + out["val_mask"].astype(int) + out[
        "test_mask"
    ].astype(int)
    assert np.all(combined == 1)


def test_degree_mats_shapes() -> None:
    """Degree helpers return node and edge degrees with epsilon."""

    H = np.zeros((5, 3), dtype=np.float32)
    H[[0, 1, 2], 0] = 1.0
    H[[1, 3], 1] = 1.0
    dv, de = data.degree_mats(H)

    assert dv.shape == (5,)
    assert de.shape == (3,)
    assert np.all(dv >= 1e-6)
    assert np.all(de >= 1e-6)
    # Node 4 is isolated but should still have epsilon degree.
    assert np.isclose(dv[4], 1e-6, atol=1e-7)
