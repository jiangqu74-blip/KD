"""Utilities for distillation package export."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


def _as_tensor(
    x: Tensor | np.ndarray, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=dtype, device=device)
    return x.to(device=device, dtype=dtype)


def pool_edge_roles_to_node(
    roles_edge_levels: List[List[Tensor]],
    H: Tensor | np.ndarray,
) -> List[List[Tensor]]:
    """Pool hyperedge role probabilities onto nodes (edge@node 角色)."""

    if not roles_edge_levels:
        return []
    device = roles_edge_levels[0][0].device
    dtype = roles_edge_levels[0][0].dtype
    H_tensor = _as_tensor(H, device=device, dtype=dtype)
    degrees = H_tensor.sum(dim=1, keepdim=True).clamp_min(1e-6)

    pooled: List[List[Tensor]] = []
    for stage_roles in roles_edge_levels:
        stage_pooled: List[Tensor] = []
        for probs in stage_roles:
            node_roles = H_tensor @ probs
            node_roles = node_roles / degrees
            stage_pooled.append(node_roles)
        pooled.append(stage_pooled)
    return pooled


def _topk_indices_values(prob: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
    prob = prob.clamp_min(0.0)
    prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    k = min(topk, prob.size(-1))
    values, indices = torch.topk(prob, k=k, dim=-1)
    if k < topk:
        pad = topk - k
        pad_idx = torch.zeros(prob.size(0), pad, dtype=torch.long, device=prob.device)
        pad_val = torch.zeros(prob.size(0), pad, dtype=prob.dtype, device=prob.device)
        indices = torch.cat([indices, pad_idx], dim=-1)
        values = torch.cat([values, pad_val], dim=-1)
    norm = values.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(values, 1.0 / max(topk, 1))
    values = torch.where(norm > 0, values / norm.clamp_min(1e-12), uniform)
    return indices, values


def export_distill_package(
    logits: Tensor,
    roles_node: List[List[Tensor]],
    roles_edge: List[List[Tensor]],
    H: Tensor | np.ndarray,
    topk: int = 3,
    out_path: str | Path = "artifacts/distill_pack.npz",
) -> Dict[str, np.ndarray]:
    """Export the distilled supervision package for the student model."""

    logits = logits.detach()
    soft_labels = torch.softmax(logits, dim=-1)

    pooled_edge = pool_edge_roles_to_node(roles_edge, H)

    n_stages = len(roles_node)
    n_levels = len(roles_node[0]) if n_stages > 0 else 0
    n_nodes = logits.size(0)

    node_idx_list: List[Tensor] = []
    node_val_list: List[Tensor] = []
    edge_idx_list: List[Tensor] = []
    edge_val_list: List[Tensor] = []

    for stage in range(n_stages):
        for level in range(n_levels):
            node_prob = roles_node[stage][level].detach()
            edge_prob = pooled_edge[stage][level].detach()
            node_indices, node_values = _topk_indices_values(node_prob, topk)
            edge_indices, edge_values = _topk_indices_values(edge_prob, topk)
            node_idx_list.append(node_indices)
            node_val_list.append(node_values)
            edge_idx_list.append(edge_indices)
            edge_val_list.append(edge_values)

    node_idx = torch.stack(node_idx_list, dim=0)
    node_val = torch.stack(node_val_list, dim=0)
    edge_idx = torch.stack(edge_idx_list, dim=0)
    edge_val = torch.stack(edge_val_list, dim=0)

    data = {
        "soft_labels": soft_labels.cpu().numpy(),
        "node_idx": node_idx.cpu().numpy(),
        "node_val": node_val.cpu().numpy(),
        "edge_idx": edge_idx.cpu().numpy(),
        "edge_val": edge_val.cpu().numpy(),
        "n_stages": np.array(n_stages, dtype=np.int64),
        "n_levels": np.array(n_levels, dtype=np.int64),
        "topk": np.array(topk, dtype=np.int64),
        "n_nodes": np.array(n_nodes, dtype=np.int64),
        "codebook_node": np.array(roles_node[0][0].size(-1), dtype=np.int64),
        "codebook_edge": np.array(roles_edge[0][0].size(-1), dtype=np.int64),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data)
    return data
