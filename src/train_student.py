"""Student-side training utilities for Top-K 掩码蒸馏."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .models import StudentMLP


def masked_kl(
    student_logits: torch.Tensor, target_idx: torch.Tensor, target_val: torch.Tensor
) -> torch.Tensor:
    """Compute Top-K masked KL divergence for a batch."""

    log_probs = F.log_softmax(student_logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=target_idx)
    target = target_val / target_val.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    loss = target * (torch.log(target.clamp_min(1e-12)) - gathered)
    return loss.sum(dim=-1).mean()


def _load_npz(path: Path | str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def train_student(
    out_dir: Path | str,
    data_npz: Path | str,
    distill_npz: Path | str,
    *,
    ablation: str = "kd",
    device: str = "cpu",
    epochs: int = 20,
    lr: float = 2e-3,
    topk_weight: float = 1.0,
    hidden_dim: int = 64,
    n_layers: int = 2,
    use_infonce: bool = False,
    use_ib: bool = False,
) -> Path:
    """Train the zero-hop student under the specified ablation."""

    if use_infonce or use_ib:
        raise NotImplementedError("InfoNCE/IB hooks are reserved for future work.")

    device_t = torch.device(device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = _load_npz(data_npz)
    distill = _load_npz(distill_npz)

    X = torch.as_tensor(data["X"], device=device_t)
    y = torch.as_tensor(data["y"], device=device_t).long()
    train_mask = torch.as_tensor(data["train_mask"], device=device_t).bool()
    val_mask = torch.as_tensor(data["val_mask"], device=device_t).bool()

    soft_labels = torch.as_tensor(distill["soft_labels"], device=device_t)
    node_idx = torch.as_tensor(distill["node_idx"], device=device_t).long()
    node_val = torch.as_tensor(distill["node_val"], device=device_t)
    edge_idx = torch.as_tensor(distill["edge_idx"], device=device_t).long()
    edge_val = torch.as_tensor(distill["edge_val"], device=device_t)

    n_stages = int(distill["n_stages"]) if "n_stages" in distill else 2
    n_levels = int(distill["n_levels"]) if "n_levels" in distill else 2
    codebook_node = (
        int(distill["codebook_node"])
        if "codebook_node" in distill
        else node_val.shape[-1]
    )
    codebook_edge = (
        int(distill["codebook_edge"])
        if "codebook_edge" in distill
        else edge_val.shape[-1]
    )

    num_classes = soft_labels.shape[1]

    student = StudentMLP(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        n_layers=n_layers,
        n_stages=n_stages,
        n_levels=n_levels,
        codebook_size_edge=codebook_edge,
        codebook_size_node=codebook_node,
        use_role_heads=True,
    ).to(device_t)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    best_val = -1.0
    best_state = None

    for _ in range(epochs):
        student.train()
        optimizer.zero_grad()
        out = student(X)
        logits = out["logits"]
        log_probs = F.log_softmax(logits[train_mask], dim=-1)
        kd_loss = F.kl_div(log_probs, soft_labels[train_mask], reduction="batchmean")

        total_loss = kd_loss
        if ablation == "kd_topk":
            node_logits = out["node_role_logits"][:, train_mask]
            edge_logits = out["edge_role_logits"][:, train_mask]
            node_targets_idx = node_idx[:, train_mask]
            node_targets_val = node_val[:, train_mask]
            edge_targets_idx = edge_idx[:, train_mask]
            edge_targets_val = edge_val[:, train_mask]

            n_roles = node_logits.size(0)
            node_loss = torch.stack(
                [
                    masked_kl(node_logits[i], node_targets_idx[i], node_targets_val[i])
                    for i in range(n_roles)
                ]
            ).mean()
            edge_loss = torch.stack(
                [
                    masked_kl(edge_logits[i], edge_targets_idx[i], edge_targets_val[i])
                    for i in range(n_roles)
                ]
            ).mean()
            topk_loss = 0.5 * (node_loss + edge_loss)
            total_loss = total_loss + topk_weight * topk_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            student.eval()
            val_logits = student(X)["logits"]
            preds = val_logits.argmax(dim=-1)
            val_acc = (preds[val_mask] == y[val_mask]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_state = {
                    k: v.detach().cpu() for k, v in student.state_dict().items()
                }

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    ckpt_path = out_path / f"student_{ablation}.pt"
    torch.save(
        {
            "model_state": student.state_dict(),
            "meta": {
                "input_dim": int(X.shape[1]),
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
                "n_layers": n_layers,
                "n_stages": n_stages,
                "n_levels": n_levels,
                "codebook_node": codebook_node,
                "codebook_edge": codebook_edge,
                "ablation": ablation,
            },
            "distill_npz": str(distill_npz),
        },
        ckpt_path,
    )

    metrics = {
        "best_val_acc": best_val,
        "epochs": epochs,
        "ablation": ablation,
        "train_nodes": int(train_mask.sum().item()),
        "val_nodes": int(val_mask.sum().item()),
    }
    with (out_path / f"student_{ablation}_metrics.json").open(
        "w", encoding="utf8"
    ) as f:
        json.dump(metrics, f, indent=2)

    return ckpt_path
