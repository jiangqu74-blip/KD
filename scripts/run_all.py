"""Command-line orchestrator for the full hypergraph distillation pipeline.

This Python helper mirrors ``scripts/run_all.sh`` but runs entirely through the
Python interpreter so that Windows users who do not have a Bash environment can
still execute the end-to-end workflow with a single command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_step(label: str, args: List[str]) -> None:
    """Execute one pipeline step with a friendly prefix."""

    print(label, flush=True)
    subprocess.run([sys.executable, "-m", "src.main", *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run teacher training, distillation export, student ablations, "
            "evaluation, and report update in sequence."
        )
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts",
        type=Path,
        help="Directory used for cached datasets, checkpoints, and reports.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device passed through to the CLI (default: cpu).",
    )

    args = parser.parse_args()

    common = ["--out_dir", str(args.out_dir), "--device", args.device]

    run_step("[1/6] Training teacher", ["--mode", "train_teacher", *common])
    run_step("[2/6] Exporting distillation package", ["--mode", "export_distill", *common])
    run_step("[3/6] Training student (kd)", ["--mode", "train_student", "--ablation", "kd", *common])
    run_step(
        "[4/6] Training student (kd_topk)",
        ["--mode", "train_student", "--ablation", "kd_topk", *common],
    )
    run_step("[5/6] Evaluating", ["--mode", "eval", *common])

    print("[6/6] Updating report", flush=True)
    subprocess.run(
        [sys.executable, "scripts/update_report.py", "--out_dir", str(args.out_dir)],
        check=True,
    )


if __name__ == "__main__":
    main()
