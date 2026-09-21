"""
Single source of truth for the set of training/eval object shape names.

Both grasp envs (grasp_pose_env / g1_grasp_env) and their cfg files used to
each hardcode the same 12 procedural shape names in three separate places
(env `_SHAPE_NAMES`, cfg `_OBJECT_PRIM_NAMES`, one `RigidObjectCfg` field per
shape). Real (YCB-derived) object families are read dynamically from the
manifest produced by scripts/generate_ycb_meshes.py, so adding/removing real
objects never requires touching env/cfg code — only regenerating the dataset.
"""

from __future__ import annotations

import json
import re

from envs._paths import data_path

PROCEDURAL_SHAPE_NAMES: list[str] = [
    "torus", "l_shape", "t_shape", "c_shape", "dumbbell",
    "wedge", "star_prism", "bracket", "stepped_cyl",
    "twisted_bar", "irregular_ext", "convex_hull",
]


def ycb_shape_names(split: str) -> list[str]:
    """Sorted, de-duplicated `ycb_*` family names present in a split's manifest.

    Only families with a spawn-ready `000.usd` are returned — some YCB families
    may have deformed instances (001+) but no graspable vanilla `000` after the
    antipodal filter in generate_ycb_meshes.py.

    Returns an empty list if the manifest doesn't exist yet (e.g. before
    fetch_ycb.py / generate_ycb_meshes.py have been run) rather than raising,
    so importing this module never breaks a procedural-only checkout.
    """
    manifest_path = data_path("data/objects", split, "manifest.json")
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        manifest = json.load(f)
    names = {entry["family"] for entry in manifest if entry["family"].startswith("ycb_")}
    ready = []
    for name in sorted(names):
        usd = data_path("data/objects", split, name, "000.usd")
        if usd.exists():
            ready.append(name)
    skipped = sorted(names - set(ready))
    if skipped:
        print(f"[object_registry] Skipping {len(skipped)} YCB families in {split} "
              f"with no 000.usd (no graspable vanilla instance): {skipped[:5]}"
              f"{'...' if len(skipped) > 5 else ''}")
    return ready


def shape_split(shape_name: str) -> str:
    """Which of data/objects/{train,eval} a shape's assets live under."""
    if shape_name in PROCEDURAL_SHAPE_NAMES:
        return "train"
    return "eval" if shape_name in ycb_shape_names("eval") else "train"


def prim_name(shape_name: str) -> str:
    """USD prim name for a shape (PascalCase, alphanumeric only) — must match
    the naming used by `_usd_obj()` in grasp_pose_env_cfg.py / g1_grasp_env_cfg.py.

    Strips underscores AND hyphens: some official YCB names contain hyphens
    (e.g. "072-a_toy_airplane", "065-a_cups"), which are invalid in USD prim
    names.
    """
    return re.sub(r"[^0-9a-zA-Z]", "", shape_name.title())
