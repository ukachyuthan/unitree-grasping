#!/usr/bin/env python3
"""
Standalone script — no Isaac Lab needed.

Generates N procedural mesh instances across all 13 shape families,
exports as OBJ files, and computes + saves antipodal grasp labels.

Usage:
    ./rl_unitree/bin/python scripts/generate_meshes.py --n_per_family 8 --seed 0 --out data/objects/train
    ./rl_unitree/bin/python scripts/generate_meshes.py --n_per_family 2 --seed 99 --out data/objects/eval
"""

import argparse, json, os, random, sys, time
import numpy as np
import trimesh
import trimesh.creation as creation
import trimesh.sample

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.generate_objects import (
    make_superquadric, make_torus, make_l_shape, make_t_shape,
    make_c_shape, make_dumbbell, make_wedge, make_star_prism,
    make_bracket, make_stepped_cylinder, make_twisted_bar,
    make_irregular_extrusion, make_convex_hull,
)

FAMILIES = [
    ("superquadric",   make_superquadric),
    ("torus",          make_torus),
    ("l_shape",        make_l_shape),
    ("t_shape",        make_t_shape),
    ("c_shape",        make_c_shape),
    ("dumbbell",       make_dumbbell),
    ("wedge",          make_wedge),
    ("star_prism",     make_star_prism),
    ("bracket",        make_bracket),
    ("stepped_cyl",    make_stepped_cylinder),
    ("twisted_bar",    make_twisted_bar),
    ("irregular_ext",  make_irregular_extrusion),
    ("convex_hull",    make_convex_hull),
]

GRIPPER_MAX_W = 0.08   # G1 gripper max span (m)
PC_N          = 512    # surface points to pre-sample for labels


def compute_antipodal_grasps(mesh, n_candidates=600, gripper_w=GRIPPER_MAX_W):
    pts, fidx = trimesh.sample.sample_surface(mesh, n_candidates)
    normals   = mesh.face_normals[fidx]
    grasps = []
    for p1, n1 in zip(pts, normals):
        origin = p1 - n1 * 1e-4
        locs, _, _ = mesh.ray.intersects_location([origin], [-n1])
        if len(locs) == 0:
            continue
        dists = np.linalg.norm(locs - p1, axis=1)
        p2 = locs[np.argmin(dists)]
        w  = np.linalg.norm(p2 - p1)
        if w > gripper_w or w < 0.005:
            continue
        n1n = n1 / (np.linalg.norm(n1) + 1e-9)
        quality = abs(np.dot(n1n, (p1 - p2) / (w + 1e-9)))
        up = np.array([0., 0., 1.])
        bn = np.cross(n1n, up)
        if np.linalg.norm(bn) < 1e-6:
            bn = np.array([1., 0., 0.])
        bn /= np.linalg.norm(bn)
        # Grasp rotation matrix: [approach, binormal, up]
        approach = n1n
        tangent  = np.cross(bn, approach)
        R = np.stack([approach, bn, tangent], axis=-1)   # (3,3) rotation
        grasps.append(dict(
            center=((p1 + p2) / 2).tolist(),
            approach=approach.tolist(),
            binormal=bn.tolist(),
            rotation=R.tolist(),
            width=float(w),
            quality=float(quality),
        ))
    grasps.sort(key=lambda g: -g["quality"])
    return grasps


def make_mesh(family_fn, rng, max_tries=30):
    for _ in range(max_tries):
        try:
            mesh = family_fn(rng)
        except Exception:
            mesh = None
        if mesh is None:
            continue
        if not mesh.is_volume:
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
            trimesh.repair.fill_holes(mesh)
        if len(mesh.vertices) >= 4 and mesh.is_watertight:
            return mesh
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_family", type=int, default=8,
                   help="Number of instances per shape family")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--out",   type=str, default="data/objects/train")
    p.add_argument("--grasp_candidates", type=int, default=600)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    manifest = []   # list of {obj_path, label_path, family, ...}
    total = len(FAMILIES) * args.n_per_family
    count = 0
    t0 = time.time()

    for family_name, family_fn in FAMILIES:
        family_dir = os.path.join(args.out, family_name)
        os.makedirs(family_dir, exist_ok=True)
        generated = 0

        for inst in range(args.n_per_family):
            mesh = make_mesh(family_fn, rng)
            if mesh is None:
                print(f"  [{family_name}] instance {inst}: failed to generate, skipping")
                continue

            # Paths
            obj_path   = os.path.join(family_dir, f"{inst:03d}.obj")
            label_path = os.path.join(family_dir, f"{inst:03d}_grasps.json")
            pc_path    = os.path.join(family_dir, f"{inst:03d}_pc.npy")

            # Export mesh
            trimesh.exchange.export.export_mesh(mesh, obj_path)

            # Pre-sample PC for PointNet input (stored for fast loading during training)
            pc, _ = trimesh.sample.sample_surface(mesh, PC_N)
            np.save(pc_path, pc.astype(np.float32))

            # Compute grasp labels
            grasps = compute_antipodal_grasps(mesh, n_candidates=args.grasp_candidates)
            label = dict(
                family=family_name,
                instance=inst,
                n_vertices=len(mesh.vertices),
                extents=mesh.extents.tolist(),
                n_grasps=len(grasps),
                grasps=grasps[:20],    # keep top-20
            )
            with open(label_path, "w") as f:
                json.dump(label, f, indent=2)

            manifest.append(dict(
                family=family_name,
                instance=inst,
                obj_path=os.path.relpath(obj_path, args.out),
                label_path=os.path.relpath(label_path, args.out),
                pc_path=os.path.relpath(pc_path, args.out),
                n_grasps=len(grasps),
                best_quality=grasps[0]["quality"] if grasps else 0.0,
            ))

            count += 1
            generated += 1
            elapsed = time.time() - t0
            print(f"  [{count:3d}/{total}]  {family_name}/{inst:03d}"
                  f"  verts={len(mesh.vertices):5d}"
                  f"  grasps={len(grasps):3d}"
                  f"  q={grasps[0]['quality']:.3f}" if grasps else "  no grasps"
                  f"  ({elapsed:.0f}s elapsed)")

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} meshes in {args.out}/")
    print(f"Manifest: {manifest_path}")
    by_family = {}
    for m in manifest:
        by_family.setdefault(m["family"], []).append(m)
    for fam, items in by_family.items():
        avg_q = np.mean([i["best_quality"] for i in items])
        avg_g = np.mean([i["n_grasps"]     for i in items])
        print(f"  {fam:20s}  n={len(items):3d}  avg_grasps={avg_g:.0f}  avg_best_q={avg_q:.3f}")


if __name__ == "__main__":
    main()
