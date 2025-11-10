"""Tests for distillation package export logic."""

from __future__ import annotations

import numpy as np
import torch

from src.distill import export_distill_package


def test_export_distill_package_shapes(tmp_path) -> None:
    torch.manual_seed(0)
    N, E = 6, 3
    logits = torch.randn(N, 4)
    roles_node = [[torch.softmax(torch.randn(N, 5), dim=-1)]]
    roles_edge = [[torch.softmax(torch.randn(E, 7), dim=-1)]]
    H = torch.zeros(N, E)
    H[:3, 0] = 1
    H[2:, 1] = 1
    H[[0, 4], 2] = 1

    out_path = tmp_path / "pack.npz"
    data = export_distill_package(
        logits, roles_node, roles_edge, H, topk=3, out_path=out_path
    )

    assert data["node_idx"].shape == (1, N, 3)
    assert data["edge_val"].shape == (1, N, 3)
    soft = data["soft_labels"]
    assert soft.shape == (N, 4)
    assert np.allclose(soft.sum(axis=-1), np.ones(N))
    assert np.allclose(data["node_val"].sum(axis=-1), np.ones((1, N)))
    assert out_path.exists()
