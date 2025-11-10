# feat: MSRD4.0 (RGVQ) hypergraph distillation baseline

## Summary
- implement synthetic hypergraph generation, teacher HGNN with RGVQ-lite taps,
  and strict 0-hop MLP student with Top-K masked KL supervision
- add distillation export, CLI orchestration, plotting/reporting scripts, and
  Makefile automation for the entire pipeline
- provide comprehensive pytest coverage including an end-to-end smoke test and
  auto-generated experiment report hooks

## Testing
- `pytest -q`
- `python -m src.main --mode train_teacher --epochs 2 --L 1 --hidden_dim 32 --K_edge 16 --K_node 16 --M_levels 1 --out_dir artifacts_ci`
- `python -m src.main --mode export_distill --out_dir artifacts_ci`
- `python -m src.main --mode train_student --ablation kd --epochs 2 --hidden_dim_student 32 --out_dir artifacts_ci`
- `python -m src.main --mode train_student --ablation kd_topk --epochs 2 --hidden_dim_student 32 --out_dir artifacts_ci`
- `python -m src.main --mode eval --out_dir artifacts_ci`
