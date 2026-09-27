"""Site customization to make the local `src` directory importable.

Pytest imports test modules before loading per-directory ``conftest`` files,
so we provide a `sitecustomize` hook that ensures the project source tree is
always available on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
