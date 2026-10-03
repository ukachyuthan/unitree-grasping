#!/usr/bin/env python3
"""
Collect human VR demonstrations of failed grasp cases into BC training pairs.

The VR app (vr-grasping-project) writes one JSON per successful demo under
    data/vr_failures/<run>/demos/<case_id>/<timestamp>.json

Each demo records the thumb and opposing-finger contacts at the moment of grasp,
in the scaled object-local frame — the same frame as GraspPoseEnv's
_contact_left_local / _contact_right_local, so a demo is directly a (c1, c2)
target for the two-point action head (see scripts/pretrain_grasp_2pt.py).

Usage:
    python scripts/load_vr_demos.py                       # summary
    python scripts/load_vr_demos.py --out data/vr_failures/demos.npz
    python scripts/load_vr_demos.py --modes opposed       # drop pinch-only grasps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEMO_SCHEMA = "unitree-grasp-demo/v1"


def load_vr_demos(dataset_dir: str | Path, modes: tuple[str, ...] = ("opposed", "pinch")) -> list[dict]:
    """Successful demos across every run in ``dataset_dir``, oldest first."""
    pairs = []
    for path in sorted(Path(dataset_dir).glob("*/demos/*/*.json")):
        with open(path, encoding="utf-8") as f:
            demo = json.load(f)
        if demo.get("schema") != DEMO_SCHEMA or not demo.get("success"):
            continue
        grasp = demo.get("grasp") or {}
        if grasp.get("mode") not in modes:
            continue
        c1 = np.asarray(grasp["c1_local"], dtype=np.float32)
        c2 = np.asarray(grasp["c2_local"], dtype=np.float32)
        pairs.append({
            "run": demo["run"],
            "case_id": demo["case_id"],
            "shape": demo["shape"],
            "mode": grasp["mode"],
            "c1": c1,
            "c2": c2,
            "width": float(np.linalg.norm(c2 - c1)),
            "path": str(path),
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/vr_failures")
    parser.add_argument("--modes", nargs="+", default=["opposed", "pinch"])
    parser.add_argument("--out", default=None, help="Optional .npz with shapes, c1, c2")
    args = parser.parse_args()

    pairs = load_vr_demos(args.dataset, tuple(args.modes))
    print(f"[vr-demos] {len(pairs)} successful demos in {args.dataset}")
    by_shape: dict[str, int] = {}
    for p in pairs:
        by_shape[p["shape"]] = by_shape.get(p["shape"], 0) + 1
    for shape, n in sorted(by_shape.items()):
        print(f"  {shape:24s} {n}")

    if args.out and pairs:
        np.savez(
            args.out,
            shapes=np.array([p["shape"] for p in pairs]),
            case_ids=np.array([p["case_id"] for p in pairs]),
            c1=np.stack([p["c1"] for p in pairs]),
            c2=np.stack([p["c2"] for p in pairs]),
        )
        print(f"[vr-demos] wrote {args.out}")


if __name__ == "__main__":
    main()
