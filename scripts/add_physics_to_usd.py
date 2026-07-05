#!/usr/bin/env python3
"""Moved to wrappers/isaaclab/scripts/add_physics_to_usd.py"""
import runpy
import sys
from pathlib import Path

_target = Path(__file__).resolve().parents[1] / "wrappers/isaaclab/scripts/add_physics_to_usd.py"
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")
