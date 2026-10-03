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

import functools
import json
import re

from envs._paths import data_path

PROCEDURAL_SHAPE_NAMES: list[str] = [
    "torus", "l_shape", "t_shape", "c_shape", "dumbbell",
    "wedge", "star_prism", "bracket", "stepped_cyl",
    "twisted_bar", "irregular_ext", "convex_hull",
]


def ycb_shape_names(split: str) -> list[str]:
    return list(_ycb_shape_names_cached(split))


@functools.lru_cache(maxsize=None)
def _ycb_shape_names_cached(split: str) -> tuple[str, ...]:
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
        return ()
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
    return tuple(ready)


# ── Mutated variants ──────────────────────────────────────────────────────────
# generate_meshes.py / generate_ycb_meshes.py write N instances per family:
# <family>/000.* (for YCB: the vanilla real mesh) and 001..N-1 (mutated
# variants). Each instance is its own env "shape": instance 0 keeps the bare
# family name (so existing checkpoints, replay queues and VR cases still match),
# instance k>0 is "<family>.v<kkk>" — only [A-Za-z0-9_.-], which the VR app's
# dataset server requires of case ids and file names.
VARIANT_SEP = ".v"


def variant_name(family: str, inst: int) -> str:
    return family if inst == 0 else f"{family}{VARIANT_SEP}{inst:03d}"


def split_variant(shape_name: str) -> tuple[str, int]:
    """'ycb_006_mustard_bottle.v003' → ('ycb_006_mustard_bottle', 3); bare family → (family, 0)."""
    family, sep, inst = shape_name.rpartition(VARIANT_SEP)
    if sep and inst.isdigit():
        return family, int(inst)
    return shape_name, 0


def shape_split(shape_name: str) -> str:
    """Which of data/objects/{train,eval} a shape's assets live under."""
    family, _ = split_variant(shape_name)
    if family in PROCEDURAL_SHAPE_NAMES:
        return "train"
    return "eval" if family in ycb_shape_names("eval") else "train"


def shape_asset(shape_name: str, suffix: str):
    """Path of one instance file, e.g. shape_asset('torus.v002', '_pc.npy') → .../torus/002_pc.npy."""
    family, inst = split_variant(shape_name)
    return data_path("data/objects", shape_split(shape_name), family, f"{inst:03d}{suffix}")


def family_variants(family: str, max_variants: int | None = None) -> list[str]:
    """Shape names for a family's spawn-ready instances: the family itself
    (instance 000) first, then every mutated instance with a .usd on disk.

    max_variants caps the total per family (None or <= 0 → all available).
    Instances the generator's graspability gate discarded leave gaps in the
    numbering; those are simply skipped.
    """
    split = shape_split(family)
    family_dir = data_path("data/objects", split, family)
    insts = sorted(
        int(p.stem) for p in family_dir.glob("[0-9][0-9][0-9].usd") if int(p.stem) > 0
    ) if family_dir.is_dir() else []
    names = [family] + [variant_name(family, i) for i in insts]
    if max_variants is not None and max_variants > 0:
        names = names[:max_variants]
    return names


def expand_variants(families: list[str], max_variants: int | None = None) -> list[str]:
    return [name for fam in families for name in family_variants(fam, max_variants)]


def prim_name(shape_name: str) -> str:
    """USD prim name for a shape (PascalCase, alphanumeric only) — must match
    the naming used by `_usd_obj()` in grasp_pose_env_cfg.py / g1_grasp_env_cfg.py.

    Strips underscores AND hyphens: some official YCB names contain hyphens
    (e.g. "072-a_toy_airplane", "065-a_cups"), which are invalid in USD prim
    names.
    """
    return re.sub(r"[^0-9a-zA-Z]", "", shape_name.title())
