"""Tests for student training utilities."""

from __future__ import annotations

import torch

from src.models import StudentMLP
from src.train_student import masked_kl


def test_masked_kl_matches_manual() -> None:
    student_logits = torch.log(torch.tensor([[0.2, 0.5, 0.3]]))
    target_idx = torch.tensor([[1, 2]])
    target_val = torch.tensor([[0.7, 0.3]])

    loss = masked_kl(student_logits, target_idx, target_val)
    log_probs = torch.log_softmax(student_logits, dim=-1)
    gathered = log_probs.gather(-1, target_idx)
    target = target_val / target_val.sum(dim=-1, keepdim=True)
    manual = (target * (torch.log(target) - gathered)).sum()
    assert torch.allclose(loss, manual)


def test_student_forward_shapes() -> None:
    model = StudentMLP(
        input_dim=6,
        hidden_dim=8,
        num_classes=4,
        n_stages=2,
        n_levels=2,
        codebook_size_node=5,
        codebook_size_edge=7,
    )
    X = torch.randn(10, 6)
    out = model(X)
    assert out["logits"].shape == (10, 4)
    assert out["node_role_logits"].shape == (4, 10, 5)
    assert out["edge_role_logits"].shape == (4, 10, 7)
