#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-artifacts}
DEVICE=${DEVICE:-cpu}

echo "[1/6] Training teacher"
$PYTHON -m src.main --mode train_teacher --out_dir "$OUT_DIR" --device "$DEVICE"

echo "[2/6] Exporting distillation package"
$PYTHON -m src.main --mode export_distill --out_dir "$OUT_DIR" --device "$DEVICE"

echo "[3/6] Training student (kd)"
$PYTHON -m src.main --mode train_student --ablation kd --out_dir "$OUT_DIR" --device "$DEVICE"

echo "[4/6] Training student (kd_topk)"
$PYTHON -m src.main --mode train_student --ablation kd_topk --out_dir "$OUT_DIR" --device "$DEVICE"

echo "[5/6] Evaluating"
$PYTHON -m src.main --mode eval --out_dir "$OUT_DIR" --device "$DEVICE"

echo "[6/6] Updating report"
$PYTHON scripts/update_report.py --out_dir "$OUT_DIR"
