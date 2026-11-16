"""Fixed locations of this checkout, so no command derives them from its own file's depth."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPORT_DIR = REPO / "exported"
FIXTURES_DIR = REPO / "tests" / "fixtures"
