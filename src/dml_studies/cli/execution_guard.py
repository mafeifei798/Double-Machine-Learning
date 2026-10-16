"""Require an explicitly selected Linux checkout before study execution."""
from __future__ import annotations

import os
from pathlib import Path
import platform


def require_execution_root(root: Path) -> None:
    configured = os.environ.get("DML_EXECUTION_ROOT")
    if platform.system() != "Linux" or not configured:
        raise RuntimeError(
            "Execution requires Linux and an explicitly configured DML_EXECUTION_ROOT."
        )
    selected = Path(configured)
    if not selected.is_absolute() or selected.resolve() != root.resolve():
        raise RuntimeError(
            "DML_EXECUTION_ROOT must be the absolute path of this checkout."
        )
