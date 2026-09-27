.PHONY: help setup train_teacher export_distill train_student_kd train_student_topk eval report lint format test

PYTHON ?= python
OUT_DIR ?= artifacts
DEVICE ?= cpu
DATA_ARGS ?= --n_nodes 1200 --n_edges 800 --n_classes 6 --feat_dim 16 --p_mixed 0.35 --min_edge 3 --max_edge 40 --seed 1
TEACHER_ARGS ?= --L 2 --hidden_dim 64 --K_edge 64 --K_node 64 --M_levels 2 --epochs 10 --lr 2e-3
STUDENT_ARGS ?= --epochs 10 --lr 2e-3 --hidden_dim_student 64 --n_layers 2

help:
@echo "Available targets:"
@echo "  setup                Install Python requirements and pre-commit hooks"
@echo "  train_teacher        Train the hypergraph teacher and cache the dataset"
@echo "  export_distill       Export Top-K distillation slices from the teacher"
@echo "  train_student_kd     Train the 0-hop student with vanilla KD"
@echo "  train_student_topk   Train the 0-hop student with KD + Top-K role KL"
@echo "  eval                 Evaluate both students and generate comparison plot"
@echo "  report               Refresh docs/REPORT.md using latest results"
@echo "  lint                 Run ruff for static analysis"
@echo "  format               Run black on the source tree"
@echo "  test                 Run pytest for the entire suite"

setup:
$(PYTHON) -m pip install -r requirements.txt
pre-commit install || true

train_teacher:
$(PYTHON) -m src.main --mode train_teacher --out_dir $(OUT_DIR) --device $(DEVICE) $(TEACHER_ARGS) $(DATA_ARGS)

export_distill:
$(PYTHON) -m src.main --mode export_distill --out_dir $(OUT_DIR) --device $(DEVICE)

train_student_kd:
$(PYTHON) -m src.main --mode train_student --out_dir $(OUT_DIR) --device $(DEVICE) --ablation kd $(STUDENT_ARGS)

train_student_topk:
$(PYTHON) -m src.main --mode train_student --out_dir $(OUT_DIR) --device $(DEVICE) --ablation kd_topk $(STUDENT_ARGS)

eval:
$(PYTHON) -m src.main --mode eval --out_dir $(OUT_DIR) --device $(DEVICE)

report:
$(PYTHON) scripts/update_report.py --out_dir $(OUT_DIR)

lint:
ruff check src tests

format:
black src tests

test:
pytest -q
