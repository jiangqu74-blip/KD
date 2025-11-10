"""Plotting utilities for experiment visualisations (绘图工具)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_student_comparison(results: Dict[str, float], out_path: str | Path) -> Path:
    """Create a bar chart comparing KD vs KD+TopK student accuracy."""

    names = list(results.keys())
    values = [results[name] for name in names]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, values, color=["#4c72b0", "#55a868"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Student Ablation Comparison")
    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center"
        )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_codebook_utilisation(
    probabilities: Iterable[torch.Tensor], out_path: str | Path
) -> Path:
    """Plot optional histogram of codebook utilisation given role probabilities."""

    data = []
    for probs in probabilities:
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        data.append(probs.reshape(-1))
    if not data:
        raise ValueError("No probability tensors provided for utilisation plot.")

    flat = np.concatenate(data)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(flat, bins=20, color="#8172b2", alpha=0.8)
    plt.xlabel("Assignment Probability")
    plt.ylabel("Frequency")
    plt.title("Codebook Utilisation Histogram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
