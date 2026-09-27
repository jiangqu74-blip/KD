"""Tests for hypergraph models and RGVQ-lite components."""

from __future__ import annotations

import pytest
import torch

from src.models import StudentMLP, n2e_aggregate
from src.rgvq import RGVQConfig, RGVQLite


def test_n2e_permutation_invariance() -> None:
    torch.manual_seed(0)
    X = torch.randn(5, 4)
    H = torch.tensor(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    De = H.sum(dim=0)
    baseline = n2e_aggregate(X, H, De)
    perm = torch.randperm(5)
    H_permuted = H[perm]
    X_permuted = X[perm]
    result = n2e_aggregate(X_permuted, H_permuted, De)
    assert torch.allclose(baseline, result, atol=1e-6)


def test_rgq_outputs_and_temperature_clamp() -> None:
    config = RGVQConfig(dim=3, codebook_size=5, n_levels=2)
    rgq = RGVQLite(config)
    x = torch.randn(7, 3)
    roles, recons = rgq(x, temperature=0.01)
    assert len(roles) == 2
    assert len(recons) == 2
    for prob in roles:
        assert prob.shape == (7, 5)
        assert torch.allclose(prob.sum(dim=-1), torch.ones(7), atol=1e-5)
    for recon in recons:
        assert recon.shape == (7, 3)
    clamped = rgq.clamp_temperature(torch.tensor(10.0))
    assert float(clamped) <= config.max_temperature
    clamped_low = rgq.clamp_temperature(torch.tensor(0.01))
    assert float(clamped_low) >= config.min_temperature


def test_student_is_zero_hop() -> None:
    student = StudentMLP(
        input_dim=4,
        hidden_dim=8,
        num_classes=3,
        n_stages=1,
        n_levels=1,
        codebook_size_edge=6,
        codebook_size_node=6,
    )
    X = torch.randn(2, 4)
    out = student(X)
    assert out["logits"].shape == (2, 3)
    assert out["node_role_logits"].shape == (1, 2, 6)
    with pytest.raises(TypeError):
        student(X, H=torch.randn(2, 1))
