"""Evaluation utilities for teacher and student models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .models import StudentMLP


def _load_npz(path: Path | str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _load_student(path: Path | str, device: torch.device) -> StudentMLP:
    ckpt = torch.load(path, map_location=device)
    meta = ckpt["meta"]
    model = StudentMLP(
        input_dim=meta["input_dim"],
        hidden_dim=meta["hidden_dim"],
        num_classes=meta["num_classes"],
        n_layers=meta.get("n_layers", 2),
        n_stages=meta.get("n_stages", 2),
        n_levels=meta.get("n_levels", 2),
        codebook_size_node=meta.get("codebook_node", 128),
        codebook_size_edge=meta.get("codebook_edge", 128),
        use_role_heads=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate(
    out_dir: Path | str,
    *,
    device: str = "cpu",
    data_npz: Path | str | None = None,
    student_kd: Path | str | None = None,
    student_kd_topk: Path | str | None = None,
) -> Dict[str, float]:
    """Compute test accuracy for KD vs KD+TopK students."""

    out_path = Path(out_dir)
    device_t = torch.device(device)
    data_path = Path(data_npz) if data_npz is not None else out_path / "data_cached.npz"
    data = _load_npz(data_path)

    X = torch.as_tensor(data["X"], device=device_t)
    y = torch.as_tensor(data["y"], device=device_t).long()
    test_mask = torch.as_tensor(data["test_mask"], device=device_t).bool()

    results: Dict[str, float] = {}

    ckpt_paths = {
        "kd": (
            Path(student_kd) if student_kd is not None else out_path / "student_kd.pt"
        ),
        "kd_topk": (
            Path(student_kd_topk)
            if student_kd_topk is not None
            else out_path / "student_kd_topk.pt"
        ),
    }

    for name, ckpt_path in ckpt_paths.items():
        if not ckpt_path.exists():
            continue
        model = _load_student(ckpt_path, device_t)
        with torch.no_grad():
            logits = model(X)["logits"]
            preds = logits.argmax(dim=-1)
            acc = (preds[test_mask] == y[test_mask]).float().mean().item()
        results[name] = acc

    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "eval_results.json").open("w", encoding="utf8") as f:
        json.dump(results, f, indent=2)

    return results
