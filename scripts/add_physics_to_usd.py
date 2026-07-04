#!/usr/bin/env python3
"""
Adds rigid body physics properties to existing USD mesh files.

The MeshConverter creates the mesh geometry USD but fails to add
physics (mass, collision, rigid body) when prim names start with digits.
This script patches the existing USDs in-place using pxr directly.

Usage:
    ./rl_unitree/bin/python scripts/add_physics_to_usd.py --input data/objects/train --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input",  type=str, default="data/objects/train")
parser.add_argument("--mass",   type=float, default=0.15, help="Object mass in kg")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import glob, os
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf


def patch_usd(usd_path: str, mass_kg: float) -> bool:
    """Open a USD, find the mesh prim, add physics API schemas, save."""
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as e:
        print(f"    open failed: {e}")
        return False

    # Find root xform prim (the default prim)
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        # Fall back: find first Xform
        for p in stage.Traverse():
            if p.GetTypeName() == "Xform":
                root = p
                break
    if not root or not root.IsValid():
        print("    no root prim found")
        return False

    # Check if physics already applied
    if UsdPhysics.RigidBodyAPI(root):
        existing = UsdPhysics.MassAPI(root).GetMassAttr()
        if existing and existing.IsValid():
            return True   # already patched

    # Apply RigidBodyAPI
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr(True)

    # Apply MassAPI
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(mass_kg)

    # Find mesh child prims and apply CollisionAPI
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            col = UsdPhysics.CollisionAPI.Apply(prim)
            col.CreateCollisionEnabledAttr(True)
            # Use convex hull approximation (fast + stable for grasping)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col.CreateApproximationAttr("convexHull")

    stage.Save()
    return True


def main():
    usd_files = sorted(glob.glob(os.path.join(args.input, "**/*.usd"), recursive=True))
    print(f"Found {len(usd_files)} USD files in {args.input}")

    ok = fail = skip = 0
    for i, usd_path in enumerate(usd_files):
        rel = os.path.relpath(usd_path, args.input)
        result = patch_usd(usd_path, args.mass)
        if result:
            ok += 1
            status = "ok"
        else:
            fail += 1
            status = "FAILED"
        print(f"  [{i+1:3d}/{len(usd_files)}]  {rel}  {status}")

    print(f"\nPatched {ok}/{len(usd_files)} USD files  ({fail} failed)")


if __name__ == "__main__":
    main()
    simulation_app.close()
