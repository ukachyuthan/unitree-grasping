#!/usr/bin/env python3
"""
Write a small synthetic failure-case set, for testing the VR app without Isaac.

Runs the real FailureCurator (training/vr_failures.py) on procedurally generated
meshes with scripted grasps that fail on every replay, so the output has exactly
the layout of a training run folder. Each scenario is a different way a
two-finger grasp goes wrong:

    torus    pinched across the hole — both fingers close on air
    block    jaws set along the long axis, wider than the gripper opens
    cylinder both contacts on the curved top — nothing to squeeze against
    bracket  aimed at the empty inside corner of the L
    wedge    pinched near the apex — the slopes squeeze it out

    python3 scripts/make_sample_vr_dataset.py --out ../vr-grasping-project/sample-events --run example
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.vr_failures import FailureCurator, FailureCuratorConfig  # noqa: E402

OBJECT_SCALE = 1.5


# ── Meshes (unscaled, object-local, z-up) ───────────────────────────────────

def torus(R=0.025, r=0.01, nu=32, nv=16):
    verts, faces = [], []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            verts.append(((R + r * math.cos(v)) * math.cos(u), (R + r * math.cos(v)) * math.sin(u), r * math.sin(v)))
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            faces += [(a, b, c), (a, c, d)]
    return np.array(verts), np.array(faces)


def box(size, center=(0.0, 0.0, 0.0)):
    x, y, z = (s / 2 for s in size)
    verts = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                      [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]]) + np.asarray(center)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return verts, faces


def merge(*meshes):
    verts, faces, offset = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def cylinder_on_side(r=0.015, length=0.07, n=32):
    """Axis along x, so it lies on the table."""
    ring = [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    verts = [(-length / 2, y, z) for y, z in ring] + [(length / 2, y, z) for y, z in ring]
    verts += [(-length / 2, 0, 0), (length / 2, 0, 0)]
    c0, c1 = 2 * n, 2 * n + 1
    faces = []
    for k in range(n):
        a, b = k, (k + 1) % n
        faces += [(a, b, n + b), (a, n + b, n + a), (c0, b, a), (c1, n + a, n + b)]
    return np.array(verts), np.array(faces)


def l_bracket(arm=0.06, thick=0.015, height=0.02):
    """An L in the xy plane: one leg along x, one along y, meeting at (-x, -y)."""
    h = arm / 2
    leg_x = box((arm, thick, height), (0.0, -h + thick / 2, 0.0))
    leg_y = box((thick, arm - thick, height), (-h + thick / 2, thick / 2, 0.0))
    return merge(leg_x, leg_y)


def wedge(base=0.05, height=0.035, depth=0.04):
    """Triangular prism: ridge along y at the top."""
    z0, z1, x, y = -height / 2, height / 2, base / 2, depth / 2
    verts = np.array([[-x, -y, z0], [x, -y, z0], [0, -y, z1], [-x, y, z0], [x, y, z0], [0, y, z1]])
    faces = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]])
    return verts, faces


# (mesh, xy on the table, yaw, failed contact pair in the SCALED object frame)
SCENARIOS = {
    "torus": (torus(), (0.50, 0.03), 0.3, ((0.0, -0.012, 0.0), (0.0, 0.012, 0.0))),
    "block": (box((0.06, 0.025, 0.035)), (0.47, -0.08), 1.2, ((-0.045, 0.0, 0.0), (0.045, 0.0, 0.0))),
    "cylinder": (cylinder_on_side(), (0.55, 0.10), -0.6, ((-0.01, 0.0, 0.0225), (0.01, 0.0, 0.0225))),
    "bracket": (l_bracket(), (0.45, 0.10), 2.4, ((0.0, 0.01, 0.0), (0.02, 0.02, 0.0))),
    "wedge": (wedge(), (0.56, -0.06), -1.9, ((-0.005, 0.0, 0.022), (0.005, 0.0, 0.022))),
}


def write_obj(path: Path, verts, faces):
    lines = [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")


def sample_surface(verts, faces, n, rng):
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(faces), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/vr_failures")
    parser.add_argument("--run", default="example")
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    mesh_dir = Path(args.out) / f"_{args.run}_src_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    names = list(SCENARIOS)
    pcs, poses = [], []
    for name in names:
        (v, f), (x, y), yaw, _ = SCENARIOS[name]
        write_obj(mesh_dir / f"{name}.obj", v, f)
        pcs.append(sample_surface(v, f, 512, rng) * OBJECT_SCALE)
        # Resting on the table: yaw only, so the lowest vertex sets the height.
        rest_z = -v[:, 2].min() * OBJECT_SCALE
        poses.append((np.array([x, y, rest_z]), [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))

    scene = {
        "robot": {"name": "franka_panda", "base_pos": [0.0, 0.0, 0.0]},
        "table": {"center": [0.5, 0.0, -0.01], "size": [0.8, 0.6, 0.02], "surface_z": 0.0},
        "object_scale": OBJECT_SCALE,
        "lift_target_m": 0.05,
        "gripper_width_bounds": [0.01, 0.08],
    }
    n = len(names)
    # Capacity = one slot per scenario: the second round of fresh failures is
    # turned away, so each scenario is queued (and exported) exactly once.
    curator = FailureCurator(
        FailureCuratorConfig(out_dir=args.out, run_name=args.run, warmup_iters=0,
                             replay_prob=1.0, queue_capacity=n),
        num_envs=n,
        shape_names=names,
        object_pcs=np.stack(pcs),
        mesh_path_fn=lambda s: mesh_dir / f"{s}.obj",
        scene=scene,
    )

    running = list(range(n))   # scenario each env is running; replays may swap them
    for it in range(1, 200):
        if curator.num_cases >= n:
            break
        replays = curator.plan_replays(100 + it)
        shape_ids = np.array(running)
        c1 = np.array([SCENARIOS[names[s]][3][0] for s in running]) + rng.normal(scale=0.002, size=(n, 3))
        c2 = np.array([SCENARIOS[names[s]][3][1] for s in running]) + rng.normal(scale=0.002, size=(n, 3))
        lift = rng.uniform(0.0, 0.2, size=n)
        new = curator.observe(
            iteration=100 + it,
            shape_ids=shape_ids,
            obj_pos=np.stack([poses[s][0] for s in running]),
            obj_quat_wxyz=np.array([poses[s][1] for s in running]),
            lift_frac=lift,
            reward=0.75 * lift + rng.uniform(0.0, 0.1, size=n),
            reward_terms={"lift": lift, "contact": rng.uniform(0.0, 0.4, size=n)},
            contact_left_local=c1,
            contact_right_local=c2,
            grasp_center_local=(c1 + c2) / 2,
            grasp_width=np.linalg.norm(c2 - c1, axis=1),
        )
        for b, s in zip(replays["env_ids"], replays["shape_ids"]):
            running[int(b)] = int(s)
        for cid in new:
            print(f"exported {cid}")

    for p in mesh_dir.iterdir():
        p.unlink()
    mesh_dir.rmdir()
    print(f"{curator.num_cases} scenarios → {curator.run_dir}")


if __name__ == "__main__":
    main()
