"""Training loop for the hypergraph teacher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import degree_mats, make_synth_hypergraph
from .models import TeacherHGNN


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, device=device)


def train_teacher(
    out_dir: Path | str,
    *,
    device: str = "cpu",
    L: int = 2,
    hidden_dim: int = 64,
    K_edge: int = 128,
    K_node: int = 128,
    M_levels: int = 2,
    epochs: int = 20,
    lr: float = 2e-3,
    data_kwargs: Optional[Dict[str, int | float]] = None,
) -> Path:
    """Train the teacher HGNN and cache the dataset."""

    device_t = torch.device(device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_cfg = data_kwargs or {}
    data = make_synth_hypergraph(**data_cfg)
    data_npz = out_path / "data_cached.npz"
    np.savez(data_npz, **data)

    X = _to_tensor(data["X"], device_t)
    y = _to_tensor(data["y"], device_t).long()
    H = _to_tensor(data["H"], device_t)
    train_mask = _to_tensor(data["train_mask"], device_t).bool()
    val_mask = _to_tensor(data["val_mask"], device_t).bool()

    dv_np, de_np = degree_mats(data["H"])
    Dv = _to_tensor(dv_np, device_t)
    De = _to_tensor(de_np, device_t)

    num_classes = int(y.max().item() + 1)

    model = TeacherHGNN(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        n_stages=L,
        codebook_size_edge=K_edge,
        codebook_size_node=K_node,
        n_levels=M_levels,
    ).to(device_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_state: Dict[str, torch.Tensor] | None = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X, H, Dv=Dv, De=De, return_roles=False)
        logits = out["logits"]
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_logits = model(X, H, Dv=Dv, De=De, return_roles=False)["logits"]
            preds = val_logits.argmax(dim=-1)
            val_acc = (preds[val_mask] == y[val_mask]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_state = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    teacher_path = out_path / "teacher.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "meta": {
                "input_dim": int(X.shape[1]),
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
                "n_stages": L,
                "n_levels": M_levels,
                "codebook_edge": K_edge,
                "codebook_node": K_node,
            },
            "data_path": str(data_npz),
        },
        teacher_path,
    )

    metrics = {
        "best_val_acc": best_val,
        "epochs": epochs,
        "train_nodes": int(train_mask.sum().item()),
        "val_nodes": int(val_mask.sum().item()),
    }
    with (out_path / "teacher_metrics.json").open("w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)

    return teacher_path
