#!/usr/bin/env python3
"""
Converts all OBJ files in a directory to USD format for Isaac Lab.

Must be run inside Isaac Lab's process (uses Omniverse USD stack).

Usage:
    ./rl_unitree/bin/python scripts/convert_to_usd.py \
        --input data/objects/train \
        --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input",  type=str, default="data/objects/train")
parser.add_argument("--collision", type=str, default="convexDecomposition",
                    choices=["convexDecomposition", "convexHull", "meshSimplification"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import glob, json, os, time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

_COLLISION_CFG = {
    "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
    "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
    "meshSimplification": schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
}


def convert_one(obj_path: str, collision_approx: str) -> str | None:
    """Convert a single OBJ → USD, return the USD path."""
    mesh_collision_cls = _COLLISION_CFG[collision_approx]
    cfg = MeshConverterCfg(
        asset_path=obj_path,
        usd_dir=os.path.dirname(obj_path),
        usd_file_name=os.path.splitext(os.path.basename(obj_path))[0] + ".usd",
        force_usd_conversion=False,
        make_instanceable=False,
        mass_props=schemas_cfg.MassPropertiesCfg(mass=0.15),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        mesh_collision_props=mesh_collision_cls(),
    )
    try:
        converter = MeshConverter(cfg)
        return converter.usd_path
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    obj_files = sorted(glob.glob(os.path.join(args.input, "**/*.obj"), recursive=True))
    print(f"Found {len(obj_files)} OBJ files in {args.input}")

    manifest_path = os.path.join(args.input, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest_list = json.load(f)
        manifest = {m["obj_path"]: m for m in manifest_list}

    t0 = time.time()
    usd_paths = {}

    for i, obj_path in enumerate(obj_files):
        rel = os.path.relpath(obj_path, args.input)
        usd_path = os.path.splitext(obj_path)[0] + ".usd"

        if os.path.exists(usd_path):
            print(f"  [{i+1:3d}/{len(obj_files)}]  {rel}  (cached)")
            usd_paths[rel] = usd_path
            continue

        print(f"  [{i+1:3d}/{len(obj_files)}]  {rel}  converting...", end="", flush=True)
        result = convert_one(obj_path, args.collision)
        elapsed = time.time() - t0
        if result:
            usd_paths[rel] = result
            print(f"  ok  ({elapsed:.0f}s)")
        else:
            print(f"  FAILED")

    # Write USD paths back to manifest
    if manifest:
        for m in manifest_list:
            obj_rel = m["obj_path"]
            if obj_rel in usd_paths:
                m["usd_path"] = os.path.relpath(usd_paths[obj_rel], args.input)
        with open(manifest_path, "w") as f:
            json.dump(manifest_list, f, indent=2)
        print(f"\nManifest updated: {manifest_path}")

    print(f"\nConverted {len(usd_paths)}/{len(obj_files)} files in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
    simulation_app.close()
