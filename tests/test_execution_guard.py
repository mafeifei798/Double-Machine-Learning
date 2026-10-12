"""Test execution authorization without importing learners or launching studies."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
require_execution_root = runpy.run_path(
    str(ROOT / "scripts" / "execution_guard.py")
)["require_execution_root"]


class ExecutionGuardTests(unittest.TestCase):
    def check(self, system: str, configured: str) -> None:
        with patch("platform.system", return_value=system), patch.dict(
            os.environ, {"DML_EXECUTION_ROOT": configured}
        ):
            require_execution_root(ROOT)

    def test_missing_root_is_rejected(self):
        with patch("platform.system", return_value="Linux"), patch.dict(os.environ):
            os.environ.pop("DML_EXECUTION_ROOT", None)
            with self.assertRaises(RuntimeError):
                require_execution_root(ROOT)

    def test_empty_root_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check("Linux", "")

    def test_non_linux_is_rejected(self):
        for system in ("Darwin", "Windows"):
            with self.subTest(system=system), self.assertRaises(RuntimeError):
                self.check(system, str(ROOT))

    def test_relative_root_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check("Linux", ".")

    def test_different_root_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check("Linux", str(ROOT.parent))

    def test_explicit_linux_root_is_accepted(self):
        self.check("Linux", str(ROOT))

    def test_equivalent_absolute_root_is_accepted(self):
        self.check("Linux", str(ROOT / ".." / ROOT.name))


if __name__ == "__main__":
    unittest.main()
