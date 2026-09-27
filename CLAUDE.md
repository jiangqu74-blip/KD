# CLAUDE.md

Research code for hypergraph knowledge distillation: a teacher HGNN with
RGVQ-lite role taps is distilled into a 0-hop MLP student, comparing plain KD
against KD + Top-K role supervision. Method notes: `docs/method.md`.

## Commands (run from the repository root)

- Install: `pip install -r requirements.txt` (CPU is fine; cloud sessions have no GPU)
- Tests: `python -m pytest -q`
- Lint: `ruff check src tests scripts && black --check src tests scripts`
- Smoke run of the whole pipeline: `python scripts/run_all.py --config configs/quick.yaml`
- Full run: `python scripts/run_all.py --config configs/default.yaml`

CI (`.github/workflows/ci.yml`) runs lint, tests and the quick smoke run; keep it green.

## Conventions

- Code imports the project as the `src` package (`from src.data import ...`), so
  always run from the repository root.
- Experiment settings live in YAML files under `configs/`; each key maps to the
  `python -m src.main` flag of the same name, and `scripts/run_all.py` rejects
  unknown keys. Add a new config instead of hardcoding values.
- `scripts/run_all.py` writes large outputs to `artifacts/<run>/` (gitignored)
  and a small summary to `results/<run>/`, which is meant to be committed.
- Run experiments you intend to keep from committed code, so `run_info.json`
  records a clean commit.
- The current synthetic data is linearly separable from node features alone
  (see `docs/method.md`), so results on it cannot distinguish methods yet.

## Working with the repository owner

- Explain changes in Chinese; put a Chinese translation in brackets after
  English technical terms.
- Work on a branch and let the owner merge through a pull request; never push
  to `main` directly.
