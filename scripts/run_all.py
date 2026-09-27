"""Run the whole experiment pipeline from one YAML config (一键跑完整实验).

Run from the repository root (在仓库根目录运行)::

    python scripts/run_all.py --config configs/quick.yaml      # smoke run
    python scripts/run_all.py --config configs/default.yaml
    python scripts/run_all.py --config configs/default.yaml --device cuda

Steps: train teacher -> export distillation pack -> train student (kd)
-> train student (kd_topk) -> evaluate -> write report.

Large outputs (checkpoints, cached data) go to ``artifacts/<run>/`` and are not
committed. A small summary (config, metrics, report, plot, run info) goes to
``results/<run>/`` so it can be committed and compared later.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Sequence, Set

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Config keys allowed in each section. Every key is passed to
# ``python -m src.main`` as ``--<key> <value>``.
SECTION_KEYS: Dict[str, Set[str]] = {
    "data": {
        "n_nodes",
        "n_edges",
        "n_classes",
        "feat_dim",
        "p_mixed",
        "min_edge",
        "max_edge",
    },
    "teacher": {"L", "hidden_dim", "K_edge", "K_node", "M_levels", "epochs", "lr"},
    "distill": {"topk", "temperature_edge", "temperature_node"},
    "student": {"hidden_dim_student", "n_layers", "epochs", "lr", "topk_weight"},
}
TOP_LEVEL_KEYS = {"name", "device", "seed", *SECTION_KEYS}

# Which config sections feed which pipeline step.
STEP_SECTIONS: Dict[str, Sequence[str]] = {
    "train_teacher": ("data", "teacher"),
    "export_distill": ("distill",),
    "train_student": ("student",),
    "eval": (),
}

SUMMARY_FILES = (
    "teacher_metrics.json",
    "student_kd_metrics.json",
    "student_kd_topk_metrics.json",
    "eval_results.json",
)


def load_config(path: Path) -> Dict:
    """Read a YAML config and reject unknown keys before anything runs."""

    if not path.exists() and (REPO_ROOT / path).exists():
        path = REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    unknown = sorted(set(cfg) - TOP_LEVEL_KEYS)
    if unknown:
        raise SystemExit(f"{path}: unknown top-level key(s): {unknown}")
    for section, allowed in SECTION_KEYS.items():
        unknown = sorted(set(cfg.get(section) or {}) - allowed)
        if unknown:
            raise SystemExit(
                f"{path}: unknown key(s) {unknown} in section '{section}'. "
                f"Allowed: {sorted(allowed)}"
            )

    cfg.setdefault("name", path.stem)
    cfg.setdefault("device", "cpu")
    cfg.setdefault("seed", 1)
    return cfg


def to_flags(cfg: Dict, sections: Sequence[str]) -> List[str]:
    flags: List[str] = []
    for section in sections:
        for key, value in (cfg.get(section) or {}).items():
            if value is not None:  # a blank value means "use the CLI default"
                flags += [f"--{key}", str(value)]
    return flags


def run_step(label: str, mode: str, cfg: Dict, out_dir: Path, *extra: str) -> None:
    print(label, flush=True)
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--mode",
        mode,
        "--out_dir",
        str(out_dir),
        "--device",
        str(cfg["device"]),
        "--seed",
        str(cfg["seed"]),
        *to_flags(cfg, STEP_SECTIONS[mode]),
        *extra,
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def git_info() -> Dict[str, object]:
    """Record which code version produced a result (for reproducibility)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown (not a git checkout)"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline from a YAML config."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--name", help="Override the run name from the config.")
    parser.add_argument("--device", help="Override the device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, help="Override the random seed.")
    parser.add_argument(
        "--artifacts_root",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Where large outputs go (not committed).",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=REPO_ROOT / "results",
        help="Where the small, committable summary goes.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key in ("name", "device", "seed"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    started = dt.datetime.now().astimezone()
    run_name = f"{started:%Y%m%d-%H%M%S}_{cfg['name']}"
    out_dir = args.artifacts_root.resolve() / run_name
    results_dir = args.results_root.resolve() / run_name
    info: Dict[str, object] = {
        "run_name": run_name,
        "config_file": str(args.config),
        "started": started.isoformat(timespec="seconds"),
        "git": git_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    t0 = time.perf_counter()

    run_step("[1/6] Training teacher", "train_teacher", cfg, out_dir)
    run_step("[2/6] Exporting distillation package", "export_distill", cfg, out_dir)
    run_step(
        "[3/6] Training student (kd)",
        "train_student",
        cfg,
        out_dir,
        "--ablation",
        "kd",
    )
    run_step(
        "[4/6] Training student (kd_topk)",
        "train_student",
        cfg,
        out_dir,
        "--ablation",
        "kd_topk",
    )
    run_step("[5/6] Evaluating", "eval", cfg, out_dir)

    print("[6/6] Writing report", flush=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/update_report.py",
            "--out_dir",
            str(out_dir),
            "--report",
            str(results_dir / "REPORT.md"),
            "--plot_target",
            str(results_dir / "student_ablation.png"),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    for name in SUMMARY_FILES:
        if (out_dir / name).exists():
            shutil.copy2(out_dir / name, results_dir / name)
    with (results_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    info["torch"] = version("torch")
    info["device"] = cfg["device"]
    info["duration_seconds"] = round(time.perf_counter() - t0, 1)
    with (results_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nDone in {info['duration_seconds']} s.")
    print(f"Summary (commit this): {results_dir}")
    print(f"Checkpoints (not committed): {out_dir}")
    results_file = results_dir / "eval_results.json"
    if results_file.exists():
        print("Test accuracy:", results_file.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
