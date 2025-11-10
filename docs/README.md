# MSRD4 Hypergraph RGVQ Distillation

> 超图结构 + RGVQ-lite Top-K 角色蒸馏基线。This repository implements the
> "msrd4-hypergraph-rgvq-distill" research scaffold end-to-end, from synthetic
> hypergraph generation to teacher/student training, Top-K role slicing, and
> reporting.

## Why Hypergraphs?

Unlike ordinary graphs, a hypergraph models higher-order interactions where a
single hyperedge can connect an arbitrary number of nodes.  The teacher network
operates directly on the incidence matrix `H ∈ {0,1}^{|V|×|E|}` using the stage
pattern described in the project brief:

1. **N2E** – permutation-invariant mean pooling from nodes to hyperedges.
2. **Tap-E (RGVQ-lite)** – multi-level residual soft quantisation on hyperedge
   embeddings, yielding *hyperedge role distributions*.
3. **E2N** – propagate quantised hyperedge features back to nodes.
4. **Tap-N (RGVQ-lite)** – obtain *node participation roles*.

The resulting *multi-scale node signatures* are cached as Top-K role slices and
fed to a strictly 0-hop MLP student via masked KL objectives.  No 2-section
conversion is performed at any point – the model always reasons over the true
hypergraph incidence matrix.

## Repository Layout

```
msrd4-hypergraph-rgvq-distill/
├─ src/
│  ├─ data.py              # synthetic hypergraph generator & degree helpers
│  ├─ models.py            # TeacherHGNN, StudentMLP, permutation-invariant ops
│  ├─ rgvq.py              # RGVQ-lite implementation with temperature gating
│  ├─ distill.py           # edge→node role pooling & Top-K export utilities
│  ├─ train_teacher.py     # teacher training loop (saves data + checkpoint)
│  ├─ train_student.py     # student training with masked KL objectives
│  ├─ eval.py              # accuracy evaluation & JSON summary export
│  ├─ plots.py             # matplotlib helpers (ablation comparison, etc.)
│  └─ main.py              # CLI entry point wrapping the entire pipeline
├─ tests/                  # pytest suite (data, models, distill, smoke)
├─ scripts/
│  ├─ run_all.sh           # teacher → distill → students → eval → report
│  ├─ plot_results.sh      # refresh evaluation & comparison plot
│  └─ update_report.py     # regenerate docs/REPORT.md from artifacts
├─ docs/
│  ├─ README.md            # this document
│  ├─ REPORT.md            # auto-generated experiment summary
│  └─ PR.md                # ready-to-use PR template
└─ artifacts/              # generated checkpoints, npz files, and plots
```

## Getting Started

1. **Install dependencies**

   ```bash
   make setup
   ```

2. **End-to-end demo (default CPU configuration)**

   ```bash
   scripts/run_all.sh
   ```

   This runs the teacher, exports Top-K slices, trains both KD variants of the
   student, evaluates them, and refreshes the report + plots in `docs/`.

3. **Manual steps** (all driven by the CLI `python -m src.main`):

   ```bash
   make train_teacher      # train teacher & cache dataset (artifacts/data_cached.npz)
   make export_distill     # export artifacts/distill_pack.npz
   make train_student_kd   # train KD-only student → artifacts/student_kd.pt
   make train_student_topk # train KD + Top-K student → artifacts/student_kd_topk.pt
   make eval               # compute test accuracy & update artifacts/plots
   make report             # sync docs/REPORT.md + copy latest plot
   ```

The CLI exposes dataset and model hyperparameters (`--n_nodes`, `--n_edges`,
`--L`, `--K_edge`, …).  See `src/main.py` for the full argument list.

## Reproducibility Notes

- **Synthetic data** is generated with fixed seeds via `numpy.random.Generator`.
  The default CI-friendly configuration uses `N=1200`, `E=800`, `feat_dim=16`,
  and a long-tailed edge cardinality sampled from a truncated Pareto law.
- **Teacher training** caches both the model checkpoint and the dataset split in
  `artifacts/`.  Subsequent steps reuse these files, ensuring consistent
  supervision across runs.
- **Distillation package** stores only node-level Top-K slices (`O(|V|)` size)
  for both node roles and pooled edge@node roles, plus the soft class labels.
- **Student training** never touches the incidence matrix at inference time; the
  forward pass consumes only node attributes, keeping deployment strictly 0-hop.

### Minimal Reproduction (3–4 minute CPU run)

```bash
PYTHON=python DEVICE=cpu OUT_DIR=artifacts scripts/run_all.sh
```

This configuration executes with `epochs=10` for both teacher and students,
matching the Makefile defaults.  For even faster smoke tests (as used in the
pytest suite), set `TEACHER_ARGS="--epochs 2 --L 1 --hidden_dim 32 --K_edge 16 --K_node 16 --M_levels 1"`
and `STUDENT_ARGS="--epochs 2 --lr 5e-3 --hidden_dim_student 32"` before calling
`make` targets.

## Quality Gates

- `pytest -q` covers data generation, permutation invariance, RGVQ temperature
  clamping, Top-K masked KL, and a full end-to-end smoke test.
- `ruff` + `black` ensure code style; both are wired into `make lint`/`make format`.
- GitHub Actions (see `.github/workflows/ci.yml`) installs dependencies, runs
  the smoke tests, and verifies the mini pipeline with reduced epochs.

## Citation & Attribution

If you reference this baseline, please cite it as:

```
MSRD4 Hypergraph RGVQ Distillation Baseline, 2024. Repository:
https://github.com/msrd/msrd4-hypergraph-rgvq-distill
```

Happy experimenting! 欢迎基于本仓库拓展 InfoNCE / 信息瓶颈等后续增强模块。
