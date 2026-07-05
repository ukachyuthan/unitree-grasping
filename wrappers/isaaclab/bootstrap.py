"""Call once after AppLauncher, before env/model imports."""

from __future__ import annotations

import sys
from pathlib import Path

_WRAPPER = Path(__file__).resolve().parent


def bootstrap() -> Path:
    if str(_WRAPPER) not in sys.path:
        sys.path.insert(0, str(_WRAPPER))
    from envs._paths import setup_paths

    return setup_paths()
