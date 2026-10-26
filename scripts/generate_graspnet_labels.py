#!/usr/bin/env python3
"""
Offline 6-DoF grasp labels from a pretrained Contact-GraspNet (Panda gripper).

For every object instance in a split's manifest (procedural and YCB alike):
  1. place the mesh, scaled by OBJECT_SCALE like the env, resting on a table plane;
  2. ray-cast a virtual depth camera from several viewpoints (partial views of the
     object on the table — what Contact-GraspNet was trained on);
  3. run Contact-GraspNet on each view, keeping grasps on the object segment;
  4. express the grasps in the object's mesh frame and keep the best ones.

Output: <instance>_graspnet.json next to the mesh. Positions are in the UNSCALED
mesh frame (the env multiplies by object_scale), directions are unit vectors:

    {"family", "instance", "frame": "mesh_unscaled", "object_scale",
     "grasps": [{"c1", "c2", "center", "approach", "width", "quality"}, ...]}

    c1, c2   finger contact points (c1 → c2 is the closing direction)
    approach direction toward the fingertips (hand +Z)
    center   midpoint of c1, c2 (what the env's graspnet reward term uses)

Consumed by scripts/pretrain_grasp_6dof.py (warm start) and the env's
graspnet reward term (use_graspnet_reward).

Needs the separate `graspnet` conda env (torch cu128 + contact_graspnet_pytorch,
https://github.com/elchun/contact_graspnet_pytorch, cloned to ~/contact_graspnet_pytorch):

    conda activate graspnet
    python scripts/generate_graspnet_labels.py --data data/objects/train
    python scripts/generate_graspnet_labels.py --data data/objects/eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh

OBJECT_SCALE = 1.5          # must match grasp_pose_env_cfg.OBJECT_SCALE
GRIPPER_MAX_W = 0.08        # Panda opening


def look_at_cv(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """4x4 world←camera transform, OpenCV camera (x right, y down, z forward)."""
    z = target - eye
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    y = -up + (up @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def render_view(obj: trimesh.Trimesh, table: trimesh.Trimesh, T_wc: np.ndarray,
                width=320, height=240, f=900.0):
    """Ray-cast one depth view. Returns (points in camera frame, is_object mask)."""
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    d_cam = np.stack([(u.ravel() - width / 2) / f, (v.ravel() - height / 2) / f,
                      np.ones(u.size)], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    d_w = d_cam @ T_wc[:3, :3].T
    o_w = np.repeat(T_wc[:3, 3][None], len(d_w), axis=0)
    scene = trimesh.util.concatenate([obj, table])
    n_obj_faces = len(obj.faces)
    loc, ray_idx, tri_idx = scene.ray.intersects_location(o_w, d_w, multiple_hits=False)
    if len(loc) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=bool)
    pts_c = (loc - T_wc[:3, 3]) @ T_wc[:3, :3]
    return pts_c, tri_idx < n_obj_faces


def label_instance(estimator, mesh_path: str, n_views_az: int, elevations, cam_dist: float,
                   top_k: int, rng: np.random.Generator) -> list[dict]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    mesh.apply_scale(OBJECT_SCALE)                    # env object-local frame
    z_table = float(mesh.vertices[:, 2].min())        # rests upright on its lowest point
    center = mesh.bounds.mean(axis=0)
    table = trimesh.creation.box(extents=(0.8, 0.8, 0.02))
    table.apply_translation([center[0], center[1], z_table - 0.01])

    grasps = []
    az0 = rng.uniform(0, 2 * np.pi)
    for elev in elevations:
        for k in range(n_views_az):
            az = az0 + 2 * np.pi * k / n_views_az
            eye = center + cam_dist * np.array([np.cos(elev) * np.cos(az),
                                                np.cos(elev) * np.sin(az), np.sin(elev)])
            T_wc = look_at_cv(eye, center)
            pts, is_obj = render_view(mesh, table, T_wc)
            if is_obj.sum() < 50:
                continue
            pred, scores, contacts, openings = estimator.predict_scene_grasps(
                pts, pc_segments={1: pts[is_obj]}, local_regions=True, filter_grasps=True,
                forward_passes=1)
            if 1 not in pred or len(pred[1]) == 0:
                continue
            G = np.asarray(pred[1]).reshape(-1, 4, 4)
            S = np.asarray(scores[1]).reshape(-1)
            C = np.asarray(contacts[1]).reshape(-1, 3)
            W = np.asarray(openings[1]).reshape(-1)
            R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]
            for g, s, c, w in zip(G, S, C, W):
                x_w = R_wc @ g[:3, 0]                  # closing direction (c1 → c2)
                z_w = R_wc @ g[:3, 2]                  # approach (toward the fingertips)
                c1 = R_wc @ c + t_wc
                width = float(np.clip(w - estimator._contact_grasp_cfg["TEST"]["extra_opening"],
                                      0.005, GRIPPER_MAX_W))
                c2 = c1 + width * x_w
                grasps.append(dict(c1=c1, c2=c2, approach=z_w / np.linalg.norm(z_w),
                                   width=width, quality=float(s)))

    # Best first, de-duplicated (centres within 4 mm and approaches within ~15°).
    grasps.sort(key=lambda g: -g["quality"])
    kept = []
    for g in grasps:
        ctr = 0.5 * (g["c1"] + g["c2"])
        if any(np.linalg.norm(ctr - 0.5 * (h["c1"] + h["c2"])) < 0.004
               and g["approach"] @ h["approach"] > 0.966 for h in kept):
            continue
        kept.append(g)
        if len(kept) >= top_k:
            break
    out = []
    for g in kept:   # → unscaled mesh frame (positions); directions unchanged
        c1, c2 = g["c1"] / OBJECT_SCALE, g["c2"] / OBJECT_SCALE
        out.append(dict(c1=c1.round(6).tolist(), c2=c2.round(6).tolist(),
                        center=(0.5 * (c1 + c2)).round(6).tolist(),
                        approach=g["approach"].round(6).tolist(),
                        width=round(g["width"] / OBJECT_SCALE, 6), quality=round(g["quality"], 6)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="split dir with manifest.json (data/objects/train|eval)")
    p.add_argument("--cgn_root", default=os.path.expanduser("~/contact_graspnet_pytorch"))
    p.add_argument("--views", type=int, default=2, help="camera azimuths per elevation")
    # Low views give side approaches (Contact-GraspNet approaches tend to face the camera).
    p.add_argument("--elevations_deg", type=float, nargs="+", default=[20.0, 55.0])
    p.add_argument("--cam_dist", type=float, default=0.65)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--limit", type=int, default=0, help="label only the first N instances (debug)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only_empty", action="store_true",
                   help="relabel only instances whose existing file has no grasps (e.g. flat "
                        "boxes), typically with a lower --first_thres and more --views")
    p.add_argument("--first_thres", type=float, default=None,
                   help="Contact-GraspNet confidence threshold (checkpoint default 0.15)")
    args = p.parse_args()

    sys.path.insert(0, args.cgn_root)
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

    ckpt_dir = os.path.join(args.cgn_root, "checkpoints", "contact_graspnet")
    arg_configs = []
    if args.first_thres is not None:
        arg_configs = [f"TEST.first_thres:{args.first_thres}", f"TEST.second_thres:{args.first_thres}"]
    cfg = config_utils.load_config(ckpt_dir, batch_size=1, arg_configs=arg_configs)
    estimator = GraspEstimator(cfg)
    # The checkpoint stores numpy scalars (e.g. the step count); allow just those
    # instead of disabling torch.load's weights-only safety check.
    import torch
    from numpy.core.multiarray import scalar as np_scalar
    # (object, "name") pairs: numpy 2 moved these to numpy._core, but the
    # checkpoint was pickled with the numpy 1 names.
    np_globals = [(np_scalar, "numpy.core.multiarray.scalar"), np.dtype] + [
        type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32")]
    with torch.serialization.safe_globals(np_globals):
        CheckpointIO(checkpoint_dir=os.path.join(ckpt_dir, "checkpoints"),
                     model=estimator.model).load("model.pt")
    estimator.model.eval()

    manifest_path = os.path.join(args.data, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    rng = np.random.default_rng(args.seed)
    elevations = [np.radians(e) for e in args.elevations_deg]

    t0, n_done = time.time(), 0
    entries = manifest[: args.limit] if args.limit else manifest
    for entry in entries:
        base = os.path.join(args.data, entry["pc_path"][: -len("_pc.npy")])
        out_path = base + "_graspnet.json"
        if os.path.exists(out_path):
            if args.only_empty:
                with open(out_path) as f:
                    if json.load(f)["grasps"]:
                        continue
            elif not args.overwrite:
                continue
        elif args.only_empty:
            continue
        import torch
        with torch.no_grad():
            grasps = label_instance(estimator, os.path.join(args.data, entry["obj_path"]),
                                    args.views, elevations, args.cam_dist, args.top_k, rng)
        with open(out_path, "w") as f:
            json.dump(dict(family=entry["family"], instance=entry["instance"], frame="mesh_unscaled",
                           object_scale=OBJECT_SCALE, grasps=grasps), f)
        entry["graspnet_path"] = os.path.relpath(out_path, args.data)
        entry["graspnet_n"] = len(grasps)
        entry["graspnet_best_quality"] = grasps[0]["quality"] if grasps else 0.0
        n_done += 1
        print(f"  [{n_done}/{len(entries)}] {entry['family']}/{entry['instance']:03d}  "
              f"{len(grasps):3d} grasps  best={entry['graspnet_best_quality']:.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done: {n_done} instances labeled → *_graspnet.json; manifest updated.")


if __name__ == "__main__":
    main()
