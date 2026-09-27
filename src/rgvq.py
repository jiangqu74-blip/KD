"""Residual Gumbel Vector Quantisation (lite) components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class RGVQConfig:
    """Configuration container for :class:`RGVQLite` (参数配置容器)."""

    dim: int
    codebook_size: int
    n_levels: int
    temperature: float = 1.0
    min_temperature: float = 0.1
    max_temperature: float = 5.0
    use_gate: bool = False


class RGVQLite(nn.Module):
    """Multi-level residual soft quantiser used in Tap-E/Tap-N stages.

    The module performs a sequence of differentiable vector quantisation steps
    without relying on hard assignments.  Each level computes similarity-based
    logits against a learnable codebook, applies a temperature-controlled
    softmax to obtain role distributions ("soft assignments"), reconstructs an
    approximation of the residual vector, and then subtracts the reconstruction
    from the residual before proceeding to the next level.  This yields
    multi-scale role probabilities that can be distilled via Top-K掩码 KL.
    """

    def __init__(self, config: RGVQConfig) -> None:
        super().__init__()
        self.config = config
        self.temperature = nn.Parameter(torch.tensor(float(config.temperature)))

        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(config.codebook_size, config.dim) * 0.02)
                for _ in range(config.n_levels)
            ]
        )
        if config.use_gate:
            # Linear gate that maps scalar signals (e.g. log |e|) to logits.
            self.gate = nn.Linear(1, config.codebook_size, bias=False)
        else:
            self.register_parameter("gate", None)

    def clamp_temperature(self, value: Tensor | float) -> Tensor:
        """Clamp the temperature into the valid range ([min, max])."""

        temp = torch.as_tensor(
            value, dtype=self.temperature.dtype, device=self.temperature.device
        )
        cfg = self.config
        return temp.clamp(cfg.min_temperature, cfg.max_temperature)

    def forward(
        self,
        h: Tensor,
        *,
        temperature: Optional[Tensor | float] = None,
        gate: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Run residual soft quantisation on ``h``.

        Parameters
        ----------
        h:
            Input activations with shape ``(B, D)`` where ``D`` equals the
            configured dimension.
        temperature:
            Optional override for the sampling temperature ``τ``.  Values are
            clamped to the configured ``[min_temperature, max_temperature]``
            range to avoid degeneracy (温度裁剪，避免退火失控).
        gate:
            Optional scalar signal per sample (``(B, 1)``) that modulates the
            logits through a small linear transformation.  This exposes the
            ``|e|`` gating interface required by Tap-E.  When the gate is not
            enabled during instantiation the argument is ignored.

        Returns
        -------
        roles, recons:
            ``roles`` is a list of length ``n_levels`` containing probability
            tensors with shape ``(B, K)``.  ``recons`` contains reconstructed
            residual contributions with shape ``(B, D)`` per level.
        """

        if temperature is None:
            tau = self.clamp_temperature(self.temperature)
        else:
            tau = self.clamp_temperature(temperature)

        roles: List[Tensor] = []
        recons: List[Tensor] = []
        residual = h

        gate_term: Optional[Tensor] = None
        if self.gate is not None and gate is not None:
            gate_term = self.gate(gate)

        for codebook in self.codebooks:
            # Compute similarity logits using the 2 r·c - ||c||^2 formulation.
            logits = (
                2.0 * residual @ codebook.t()
                - codebook.pow(2).sum(dim=1, keepdim=True).t()
            )
            if gate_term is not None:
                logits = logits + gate_term
            probs = F.softmax(logits / tau, dim=-1)
            recon = probs @ codebook
            roles.append(probs)
            recons.append(recon)
            residual = residual - recon

        return roles, recons
