"""Pytest configuration for msrd4-hypergraph-rgvq-distill.

Ensures that the project `src/` directory is importable when running tests
without installing the package. This mirrors the layout expected by the
scripts and CLI entry points during local development.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
