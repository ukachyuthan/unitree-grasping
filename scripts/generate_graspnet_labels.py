#!/usr/bin/env python3
"""
Offline GraspNet-quality label generation.

Runs a pretrained Contact-GraspNet model once over every object's pre-sampled
point cloud (procedural AND real — same manifest, same treatment) to produce
learned grasp-quality labels, cached to disk exactly like the geometric
antipodal-grasp labels (scripts/generate_meshes.py:compute_antipodal_grasps).
These are consumed by:
  - scripts/pretrain_grasp.py     (--label_source graspnet / both)
  - the RL reward-shaping term    (use_graspnet_reward cfg flag, on by default)

No live model inference happens during RL rollout — this script runs once,
offline, and results are memory-mapped at env init the same way `_obj_pcs`
already is (see grasp_pose_env.py / g1_grasp_env.py).

NOTE: the exact import path / call signature for contact-graspnet-pytorch
depends on which build you install (see requirements.txt) — the model
load/predict calls are isolated in _load_model()/_predict_grasps() below;
adjust only those two functions to match your installed package's actual API.

Usage:
    ./rl_unitree/bin/python scripts/generate_graspnet_labels.py \
        --data data/objects/train --checkpoint /path/to/contact_graspnet_ckpt.pt
    ./rl_unitree/bin/python scripts/generate_graspnet_labels.py \
        --data data/objects/eval  --checkpoint /path/to/contact_graspnet_ckpt.pt
"""

import argparse
import json
import os
import sys

import numpy as np


def _load_model(checkpoint: str, device: str):
    """Load a pretrained Contact-GraspNet model.

    Adjust the import / construction below to match your installed
    contact-graspnet-pytorch build — package layout varies across forks.
    """
    import torch
    try:
        from contact_graspnet_pytorch.model import ContactGraspNet
    except ImportError as e:
        sys.exit(
            "contact-graspnet-pytorch not importable. Install it (see requirements.txt) "
            f"or adjust _load_model() in this script to match your build's module path.\n{e}"
        )
    model = ContactGraspNet()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device).eval()
    return model


def _predict_grasps(model, points: np.ndarray, device: str, top_k: int = 20) -> list[dict]:
    """Run inference on one object's point cloud.

    Returns grasps in the same schema as generate_meshes.py's
    compute_antipodal_grasps(): {center, approach, width, quality}, so
    downstream consumers (pretrain_grasp.py, the RL reward lookup) don't need
    to special-case the label source.

    Adjust to match your model's actual output format (grasp pose
    representation, confidence scores) — this assumes a `model.predict(pc)`
    call returning (grasp_poses, scores, widths).
    """
    import torch
    with torch.no_grad():
        pc = torch.from_numpy(points).float().unsqueeze(0).to(device)
        pred_grasps, pred_scores, pred_widths, _ = model.predict(pc)

    grasps = []
    for g, s, w in zip(pred_grasps[0], pred_scores[0], pred_widths[0]):
        g = np.asarray(g)
        center = g[:3, 3] if g.shape == (4, 4) else g[:3]
        approach = g[:3, 2] if g.shape == (4, 4) else np.array([0.0, 0.0, 1.0])
        grasps.append(dict(
            center=center.tolist(),
            approach=approach.tolist(),
            width=float(w),
            quality=float(s),
        ))
    grasps.sort(key=lambda x: -x["quality"])
    return grasps[:top_k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/objects/train")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = os.path.join(args.data, "manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit(f"No manifest.json in {args.data}/ — run generate_meshes.py / "
                  f"generate_ycb_meshes.py first.")
    with open(manifest_path) as f:
        manifest = json.load(f)

    model = _load_model(args.checkpoint, device)

    n_done = 0
    for entry in manifest:
        pc_path = os.path.join(args.data, entry["pc_path"])
        base = pc_path[: -len("_pc.npy")]
        graspnet_path = base + "_graspnet.json"

        if os.path.exists(graspnet_path) and not args.overwrite:
            entry["graspnet_path"] = os.path.relpath(graspnet_path, args.data)
            continue
        if not os.path.exists(pc_path):
            continue

        points = np.load(pc_path).astype(np.float32)
        grasps = _predict_grasps(model, points, device)

        with open(graspnet_path, "w") as f:
            json.dump(dict(family=entry["family"], instance=entry["instance"], grasps=grasps), f, indent=2)

        entry["graspnet_path"] = os.path.relpath(graspnet_path, args.data)
        entry["graspnet_best_quality"] = grasps[0]["quality"] if grasps else 0.0
        n_done += 1
        print(f"  [{n_done}] {entry['family']}/{entry['instance']:03d}  -> {len(grasps)} grasps")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {n_done} objects labeled. Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
