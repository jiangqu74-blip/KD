.PHONY: help setup train_teacher export_distill train_student_kd train_student_topk eval report

help:
	@echo "Available targets:"
	@echo "  setup                Install Python requirements and pre-commit hooks"
	@echo "  train_teacher        Placeholder target for teacher training"
	@echo "  export_distill       Placeholder target for distillation export"
	@echo "  train_student_kd     Placeholder target for KD student training"
	@echo "  train_student_topk   Placeholder target for KD+TopK student training"
	@echo "  eval                 Placeholder evaluation target"
	@echo "  report               Placeholder report generation target"

setup:
	pip install -r requirements.txt
	pre-commit install || true

train_teacher:
	@echo "Teacher training will be implemented in later milestones"

export_distill:
	@echo "Distillation export will be implemented in later milestones"

train_student_kd:
	@echo "Student KD training will be implemented in later milestones"

train_student_topk:
	@echo "Student KD+TopK training will be implemented in later milestones"

eval:
	@echo "Evaluation will be implemented in later milestones"

report:
	@echo "Report generation will be implemented in later milestones"
