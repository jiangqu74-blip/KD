"""Model definitions for teacher and student networks."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .rgvq import RGVQConfig, RGVQLite


def _ensure_tensor(x: Tensor | torch.Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        return torch.as_tensor(x)
    return x


def n2e_aggregate(X: Tensor, H: Tensor, De: Optional[Tensor] = None) -> Tensor:
    """Aggregate node representations onto hyperedges (N2E 聚合，置换不敏感).

    Parameters
    ----------
    X:
        Node embeddings of shape ``(N, D)``.
    H:
        Incidence matrix ``(N, E)`` with binary memberships.
    De:
        Optional pre-computed hyperedge cardinalities ``(|E|,)``.  If omitted
        the function recomputes ``De`` via ``H.sum(dim=0)``.

    Returns
    -------
    Tensor
        Hyperedge embeddings ``(E, D)`` obtained by mean pooling.
    """

    X = _ensure_tensor(X)
    H = _ensure_tensor(H).to(X.dtype)
    if De is None:
        De = H.sum(dim=0)
    De = _ensure_tensor(De).to(X.dtype)
    denom = De.clamp_min(1e-6).unsqueeze(-1)
    edge_sum = H.transpose(0, 1) @ X
    return edge_sum / denom


def e2n_aggregate(E: Tensor, H: Tensor, Dv: Optional[Tensor] = None) -> Tensor:
    """Aggregate hyperedge representations back to nodes (E2N 聚合).

    Parameters
    ----------
    E:
        Hyperedge embeddings ``(E, D)``.
    H:
        Incidence matrix ``(N, E)``.
    Dv:
        Optional node degrees ``(|V|,)``.  Recomputed from ``H`` if missing.

    Returns
    -------
    Tensor
        Node embeddings ``(N, D)`` after averaging incident hyperedge features.
    """

    E = _ensure_tensor(E)
    H = _ensure_tensor(H).to(E.dtype)
    if Dv is None:
        Dv = H.sum(dim=1)
    Dv = _ensure_tensor(Dv).to(E.dtype)
    denom = Dv.clamp_min(1e-6).unsqueeze(-1)
    node_sum = H @ E
    return node_sum / denom


class HypergraphStage(nn.Module):
    """Single Stage = N2E → Tap-E → E2N → Tap-N pipeline."""

    def __init__(
        self,
        dim: int,
        codebook_edge: int,
        codebook_node: int,
        n_levels: int,
        *,
        dropout: float = 0.1,
        use_edge_gate: bool = False,
    ) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_rgq = RGVQLite(
            RGVQConfig(
                dim=dim,
                codebook_size=codebook_edge,
                n_levels=n_levels,
                use_gate=use_edge_gate,
            )
        )
        self.node_rgq = RGVQLite(
            RGVQConfig(
                dim=dim,
                codebook_size=codebook_node,
                n_levels=n_levels,
            )
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        X: Tensor,
        H: Tensor,
        *,
        Dv: Tensor,
        De: Tensor,
        temperature_edge: Optional[Tensor | float] = None,
        temperature_node: Optional[Tensor | float] = None,
        edge_gate: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """Execute the stage and return updated node states and role lists."""

        edges = n2e_aggregate(X, H, De)
        edges = self.edge_mlp(edges)
        edge_roles, edge_recons = self.edge_rgq(
            edges, temperature=temperature_edge, gate=edge_gate
        )
        edge_quant = torch.stack(edge_recons, dim=0).sum(dim=0)
        edges = edges + edge_quant

        nodes = e2n_aggregate(edges, H, Dv)
        nodes = self.node_mlp(nodes)
        node_roles, node_recons = self.node_rgq(nodes, temperature=temperature_node)
        node_quant = torch.stack(node_recons, dim=0).sum(dim=0)
        X = self.norm(X + nodes + node_quant)
        return X, edge_roles, node_roles


class TeacherHGNN(nn.Module):
    """Hypergraph teacher with multi-stage RGVQ-lite taps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        n_stages: int = 2,
        codebook_size_edge: int = 128,
        codebook_size_node: int = 128,
        n_levels: int = 2,
        dropout: float = 0.1,
        use_edge_gate: bool = False,
    ) -> None:
        super().__init__()
        self.n_stages = n_stages
        self.n_levels = n_levels
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.stages = nn.ModuleList(
            [
                HypergraphStage(
                    dim=hidden_dim,
                    codebook_edge=codebook_size_edge,
                    codebook_node=codebook_size_node,
                    n_levels=n_levels,
                    dropout=dropout,
                    use_edge_gate=use_edge_gate,
                )
                for _ in range(n_stages)
            ]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        X: Tensor,
        H: Tensor,
        *,
        Dv: Optional[Tensor] = None,
        De: Optional[Tensor] = None,
        temperature_edge: Optional[Tensor | float] = None,
        temperature_node: Optional[Tensor | float] = None,
        return_roles: bool = True,
    ) -> Dict[str, torch.Tensor | List[List[Tensor]]]:
        """Forward pass with optional role collection."""

        X = _ensure_tensor(X).float()
        H = _ensure_tensor(H).to(X.device).float()
        if Dv is None:
            Dv = H.sum(dim=1)
        if De is None:
            De = H.sum(dim=0)
        Dv = Dv.to(X.device)
        De = De.to(X.device)

        X = self.input_proj(X)
        roles_edge: List[List[Tensor]] = []
        roles_node: List[List[Tensor]] = []

        edge_gate = torch.log(De.clamp_min(1.0)).unsqueeze(-1)

        for stage in self.stages:
            X, edge_roles, node_roles = stage(
                X,
                H,
                Dv=Dv,
                De=De,
                temperature_edge=temperature_edge,
                temperature_node=temperature_node,
                edge_gate=edge_gate,
            )
            roles_edge.append(edge_roles)
            roles_node.append(node_roles)

        logits = self.classifier(X)
        if not return_roles:
            return {"logits": logits, "embeddings": X}
        return {
            "logits": logits,
            "embeddings": X,
            "roles_edge": roles_edge,
            "roles_node": roles_node,
        }


class StudentMLP(nn.Module):
    """Zero-hop student network with auxiliary role heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_stages: int = 2,
        n_levels: int = 2,
        codebook_size_edge: int = 128,
        codebook_size_node: int = 128,
        use_role_heads: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * n_layers
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.use_role_heads = use_role_heads
        self.n_roles = n_stages * n_levels
        if use_role_heads:
            self.node_role_heads = nn.ModuleList(
                [nn.Linear(hidden_dim, codebook_size_node) for _ in range(self.n_roles)]
            )
            self.edge_role_heads = nn.ModuleList(
                [nn.Linear(hidden_dim, codebook_size_edge) for _ in range(self.n_roles)]
            )
        else:
            self.node_role_heads = nn.ModuleList()
            self.edge_role_heads = nn.ModuleList()

    def forward(self, X: Tensor) -> Dict[str, Tensor | List[Tensor]]:
        """Compute logits and (optionally) role predictions without using ``H``."""

        X = _ensure_tensor(X).float()
        h = self.backbone(X)
        h = self.norm(h)
        logits = self.classifier(h)
        output: Dict[str, Tensor | List[Tensor]] = {"logits": logits, "embedding": h}
        if self.use_role_heads:
            node_logits = torch.stack([head(h) for head in self.node_role_heads], dim=0)
            edge_logits = torch.stack([head(h) for head in self.edge_role_heads], dim=0)
            output["node_role_logits"] = node_logits
            output["edge_role_logits"] = edge_logits
        return output
