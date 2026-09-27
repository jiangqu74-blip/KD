#!/usr/bin/env python
"""Generate docs/REPORT.md from latest experiment artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


def describe_dataset(data_path: Path) -> Dict[str, int]:
    if not data_path.exists():
        return {}
    with np.load(data_path) as data:
        n_nodes, feat_dim = data["X"].shape
        n_edges = data["H"].shape[1]
        n_classes = int(data["y"].max() + 1)
        splits = {
            "train": int(data["train_mask"].sum()),
            "val": int(data["val_mask"].sum()),
            "test": int(data["test_mask"].sum()),
        }
    return {
        "n_nodes": n_nodes,
        "feat_dim": feat_dim,
        "n_edges": n_edges,
        "n_classes": n_classes,
        **{f"{k}_nodes": v for k, v in splits.items()},
    }


def describe_distill(pack_path: Path) -> Dict[str, int]:
    if not pack_path.exists():
        return {}
    with np.load(pack_path) as pack:
        return {
            "topk": int(pack["topk"]),
            "n_stages": int(pack["n_stages"]),
            "n_levels": int(pack["n_levels"]),
            "codebook_node": int(pack["codebook_node"]),
            "codebook_edge": int(pack["codebook_edge"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--report", default="docs/REPORT.md")
    parser.add_argument("--plot_target", default="docs/student_ablation.png")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    data_info = describe_dataset(out_dir / "data_cached.npz")
    distill_info = describe_distill(out_dir / "distill_pack.npz")
    results = load_json(out_dir / "eval_results.json")

    teacher_metrics = load_json(out_dir / "teacher_metrics.json")
    kd_metrics = load_json(out_dir / "student_kd_metrics.json")
    topk_metrics = load_json(out_dir / "student_kd_topk_metrics.json")

    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# 实验报告 | MSRD4 Hypergraph RGVQ Distillation",
        "",
        f"_自动生成于 {now}_",
        "",
        "## 数据集概要",
    ]
    if data_info:
        lines.extend(
            [
                "- 节点数 (|V|): {n_nodes}".format(**data_info),
                "- 特征维度: {feat_dim}".format(**data_info),
                "- 超边数 (|E|): {n_edges}".format(**data_info),
                "- 类别数: {n_classes}".format(**data_info),
                (
                    "- 划分: 训练 {train_nodes} / 验证 {val_nodes} / "
                    "测试 {test_nodes}"
                ).format(**data_info),
            ]
        )
    else:
        lines.append("- 尚未生成数据缓存。")

    lines.extend(["", "## 蒸馏包配置"])
    if distill_info:
        lines.extend(
            [
                "- Top-K: {topk}".format(**distill_info),
                "- Stage × Level: {n_stages} × {n_levels}".format(**distill_info),
                "- Node/Edge 码本规模: {codebook_node} / {codebook_edge}".format(
                    **distill_info
                ),
            ]
        )
    else:
        lines.append("- 尚未导出蒸馏包。")

    lines.extend(["", "## 训练摘要"])
    if teacher_metrics:
        lines.append(
            "- 教师验证准确率: {:.3f}".format(teacher_metrics.get("best_val_acc", 0.0))
        )
    if kd_metrics:
        lines.append(
            "- 学生 (KD) 验证准确率: {:.3f}".format(kd_metrics.get("best_val_acc", 0.0))
        )
    if topk_metrics:
        lines.append(
            "- 学生 (KD+TopK) 验证准确率: {:.3f}".format(
                topk_metrics.get("best_val_acc", 0.0)
            )
        )

    lines.extend(
        ["", "## 测试集结果", "", "| Student | Test Accuracy |", "| --- | --- |"]
    )
    if results:
        for name in sorted(results):
            lines.append(f"| {name} | {results[name]:.4f} |")
    else:
        lines.append("| (no results) | - |")

    if (out_dir / "plots" / "student_ablation.png").exists():
        lines.extend(["", "![Student comparison](student_ablation.png)"])
        target = Path(args.plot_target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((out_dir / "plots" / "student_ablation.png").read_bytes())

    report_path.write_text("\n".join(lines), encoding="utf8")


if __name__ == "__main__":
    main()
