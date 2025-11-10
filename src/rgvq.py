"""Residual Gumbel Vector Quantisation (lite) components."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor


class RGVQLite(nn.Module):
    """Placeholder RGVQ-lite module, to be implemented in later milestones."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, h: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        raise NotImplementedError("Implemented in later milestones")
