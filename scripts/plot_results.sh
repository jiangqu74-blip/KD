#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-artifacts}
DEVICE=${DEVICE:-cpu}

$PYTHON -m src.main --mode eval --out_dir "$OUT_DIR" --device "$DEVICE"
