"""Model definitions for teacher and student networks."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch import Tensor


def n2e_aggregate(*args: Any, **kwargs: Any) -> Tensor:
    """Placeholder node-to-edge aggregation."""

    raise NotImplementedError("Implemented in later milestones")


def e2n_aggregate(*args: Any, **kwargs: Any) -> Tensor:
    """Placeholder edge-to-node aggregation."""

    raise NotImplementedError("Implemented in later milestones")


class TeacherHGNN(nn.Module):
    """Placeholder teacher hypergraph neural network."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("Implemented in later milestones")


class StudentMLP(nn.Module):
    """Placeholder student MLP that will be implemented later."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError("Implemented in later milestones")
