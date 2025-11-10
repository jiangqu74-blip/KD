"""Command line interface for msrd4-hypergraph-rgvq-distill 项目."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .data import degree_mats
from .distill import export_distill_package
from .eval import evaluate
from .models import TeacherHGNN
from .plots import plot_student_comparison
from .train_student import train_student
from .train_teacher import train_teacher


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "train_teacher",
            "export_distill",
            "train_student",
            "eval",
        ],
    )
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--K_edge", type=int, default=128)
    parser.add_argument("--K_node", type=int, default=128)
    parser.add_argument("--M_levels", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--ablation", choices=["kd", "kd_topk"], default="kd")
    parser.add_argument("--topk_weight", type=float, default=1.0)
    parser.add_argument("--data_npz")
    parser.add_argument("--distill_npz")
    parser.add_argument("--teacher_ckpt")
    parser.add_argument("--student_kd")
    parser.add_argument("--student_kd_topk")
    parser.add_argument("--temperature_edge", type=float, default=None)
    parser.add_argument("--temperature_node", type=float, default=None)
    parser.add_argument("--n_nodes", type=int, default=1200)
    parser.add_argument("--n_edges", type=int, default=800)
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--feat_dim", type=int, default=16)
    parser.add_argument("--p_mixed", type=float, default=0.35)
    parser.add_argument("--min_edge", type=int, default=3)
    parser.add_argument("--max_edge", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_infonce", action="store_true")
    parser.add_argument("--use_ib", action="store_true")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim_student", type=int, default=64)
    return parser


def _load_data_for_teacher(path: Path | None) -> Dict[str, np.ndarray]:
    if path is not None and path.exists():
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    raise FileNotFoundError(f"Data file {path} not found")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train_teacher":
        data_kwargs = dict(
            n_nodes=args.n_nodes,
            n_classes=args.n_classes,
            feat_dim=args.feat_dim,
            n_edges=args.n_edges,
            p_mixed=args.p_mixed,
            min_edge=args.min_edge,
            max_edge=args.max_edge,
            seed=args.seed,
        )
        ckpt = train_teacher(
            out_dir,
            device=args.device,
            L=args.L,
            hidden_dim=args.hidden_dim,
            K_edge=args.K_edge,
            K_node=args.K_node,
            M_levels=args.M_levels,
            epochs=args.epochs,
            lr=args.lr,
            data_kwargs=data_kwargs,
        )
        print(f"Teacher saved to {ckpt}")
        return

    if args.mode == "export_distill":
        teacher_ckpt = (
            Path(args.teacher_ckpt) if args.teacher_ckpt else out_dir / "teacher.pt"
        )
        device_t = torch.device(args.device)
        state = torch.load(teacher_ckpt, map_location=device_t)
        meta = state["meta"]
        data_path = (
            Path(args.data_npz)
            if args.data_npz
            else Path(state.get("data_path", out_dir / "data_cached.npz"))
        )
        data = _load_data_for_teacher(data_path)
        X = torch.as_tensor(data["X"], device=device_t)
        H = torch.as_tensor(data["H"], device=device_t)
        dv_np, de_np = degree_mats(data["H"])
        Dv = torch.as_tensor(dv_np, device=device_t)
        De = torch.as_tensor(de_np, device=device_t)
        model = TeacherHGNN(
            input_dim=meta["input_dim"],
            hidden_dim=meta["hidden_dim"],
            num_classes=meta["num_classes"],
            n_stages=meta["n_stages"],
            codebook_size_edge=meta["codebook_edge"],
            codebook_size_node=meta["codebook_node"],
            n_levels=meta["n_levels"],
        )
        model.load_state_dict(state["model_state"])
        model = model.to(device_t)
        model.eval()
        with torch.no_grad():
            outputs = model(
                X,
                H,
                Dv=Dv,
                De=De,
                temperature_edge=args.temperature_edge,
                temperature_node=args.temperature_node,
                return_roles=True,
            )
        distill_path = (
            Path(args.distill_npz) if args.distill_npz else out_dir / "distill_pack.npz"
        )
        export_distill_package(
            outputs["logits"],
            outputs["roles_node"],
            outputs["roles_edge"],
            H,
            topk=args.topk,
            out_path=distill_path,
        )
        print(f"Distillation package saved to {distill_path}")
        return

    if args.mode == "train_student":
        data_npz = Path(args.data_npz) if args.data_npz else out_dir / "data_cached.npz"
        distill_npz = (
            Path(args.distill_npz) if args.distill_npz else out_dir / "distill_pack.npz"
        )
        ckpt = train_student(
            out_dir,
            data_npz,
            distill_npz,
            ablation=args.ablation,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            topk_weight=args.topk_weight,
            hidden_dim=args.hidden_dim_student,
            n_layers=args.n_layers,
            use_infonce=args.use_infonce,
            use_ib=args.use_ib,
        )
        print(f"Student ({args.ablation}) saved to {ckpt}")
        return

    if args.mode == "eval":
        data_npz = Path(args.data_npz) if args.data_npz else out_dir / "data_cached.npz"
        results = evaluate(
            out_dir,
            device=args.device,
            data_npz=data_npz,
            student_kd=args.student_kd,
            student_kd_topk=args.student_kd_topk,
        )
        if results:
            plot_path = plot_student_comparison(
                results, out_dir / "plots" / "student_ablation.png"
            )
            print(f"Evaluation results: {results}\nPlot saved to {plot_path}")
        else:
            print("No student checkpoints found for evaluation.")
        return


if __name__ == "__main__":
    main()
