"""End-to-end smoke test for the full hypergraph distillation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data import degree_mats
from src.distill import export_distill_package
from src.eval import evaluate
from src.models import TeacherHGNN
from src.train_student import train_student
from src.train_teacher import train_teacher


def test_full_pipeline(tmp_path: Path) -> None:
    cfg = {
        "n_nodes": 120,
        "n_classes": 4,
        "feat_dim": 8,
        "n_edges": 90,
        "p_mixed": 0.3,
        "min_edge": 3,
        "max_edge": 20,
        "seed": 2,
    }
    teacher_path = train_teacher(
        tmp_path,
        device="cpu",
        L=1,
        hidden_dim=32,
        K_edge=16,
        K_node=16,
        M_levels=1,
        epochs=2,
        lr=5e-3,
        data_kwargs=cfg,
    )

    state = torch.load(teacher_path, map_location="cpu")
    data = np.load(tmp_path / "data_cached.npz")
    X = torch.as_tensor(data["X"], dtype=torch.float32)
    H = torch.as_tensor(data["H"], dtype=torch.float32)
    dv_np, de_np = degree_mats(data["H"])
    Dv = torch.as_tensor(dv_np)
    De = torch.as_tensor(de_np)

    teacher = TeacherHGNN(
        input_dim=int(state["meta"]["input_dim"]),
        hidden_dim=int(state["meta"]["hidden_dim"]),
        num_classes=int(state["meta"]["num_classes"]),
        n_stages=int(state["meta"]["n_stages"]),
        codebook_size_edge=int(state["meta"]["codebook_edge"]),
        codebook_size_node=int(state["meta"]["codebook_node"]),
        n_levels=int(state["meta"]["n_levels"]),
    )
    teacher.load_state_dict(state["model_state"])
    teacher.eval()
    with torch.no_grad():
        outputs = teacher(X, H, Dv=Dv, De=De, return_roles=True)

    distill_path = tmp_path / "distill_pack.npz"
    export_distill_package(
        outputs["logits"],
        outputs["roles_node"],
        outputs["roles_edge"],
        H,
        out_path=distill_path,
    )

    train_student(
        tmp_path,
        tmp_path / "data_cached.npz",
        distill_path,
        ablation="kd",
        epochs=2,
        lr=5e-3,
        hidden_dim=32,
    )
    train_student(
        tmp_path,
        tmp_path / "data_cached.npz",
        distill_path,
        ablation="kd_topk",
        epochs=2,
        lr=5e-3,
        hidden_dim=32,
    )

    results = evaluate(tmp_path, device="cpu", data_npz=tmp_path / "data_cached.npz")
    assert "kd" in results and "kd_topk" in results
