"""Repo-root paths for Isaac Lab wrapper (run scripts from repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_ROOT = Path(__file__).resolve().parents[1]


def setup_paths() -> Path:
    """Insert repo root and wrapper root on sys.path. Returns repo root."""
    for p in (REPO_ROOT, WRAPPER_ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return REPO_ROOT


def data_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)
