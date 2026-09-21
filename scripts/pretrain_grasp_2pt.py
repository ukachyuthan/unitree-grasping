#!/usr/bin/env python3
"""
Behavioral cloning warm-start for the 6D two-contact grasp head.

Reads antipodal labels (generate_meshes.py) and reconstructs contact pairs:
    c1, c2 = center ± 0.5 * width * approach
then normalizes into the same tanh action space used by GraspPoseEnv
(two_point_grasp=True).

Filters out near-vertical closing axes (incompatible with top-down palm IK)
and widths outside the env clamp. Loss is min-distance to any of the top-K
labeled contact pairs (with left/right swap invariance).

Usage:
    python scripts/pretrain_grasp_2pt.py
    python scripts/pretrain_grasp_2pt.py --epochs 200 --out data/grasp_weights

Then RL:
    python wrappers/isaaclab/scripts/train_grasp_pose.py --headless \\
        --pretrain data/grasp_weights/grasp_2pt_pretrain_best.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.pointnet_encoder import PointNetEncoder

# Must match GraspPoseEnvCfg (scaled object frame).
OBJECT_SCALE = 1.5
X_BOUNDS = (-0.05, 0.05)
Y_BOUNDS = (-0.05, 0.05)
Z_BOUNDS = (-0.04, 0.04)
W_BOUNDS = (0.01, 0.08)
# Reject grasps whose closing axis is too aligned with world +Z (top-down IK
# needs a roughly horizontal jaw).
MAX_ABS_APPROACH_Z = 0.65


def _norm_axis(x: float, lo: float, hi: float) -> float:
    return float(np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0))


def _contacts_to_action(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Canonicalize left/right by +y, then map xyz→tanh for both contacts."""
    if c1[1] > c2[1]:
        c1, c2 = c2, c1
    vals = []
    for pt in (c1, c2):
        vals.append(_norm_axis(pt[0], *X_BOUNDS))
        vals.append(_norm_axis(pt[1], *Y_BOUNDS))
        vals.append(_norm_axis(pt[2], *Z_BOUNDS))
    return np.asarray(vals, dtype=np.float32)


def _label_to_contacts(g: dict) -> np.ndarray | None:
    """Reconstruct scaled (c1,c2) from an antipodal / GraspNet label dict."""
    center = np.asarray(g["center"], dtype=np.float64) * OBJECT_SCALE
    approach = np.asarray(g["approach"], dtype=np.float64)
    n = np.linalg.norm(approach) + 1e-9
    approach = approach / n
    if abs(approach[2]) > MAX_ABS_APPROACH_Z:
        return None
    width = float(g["width"]) * OBJECT_SCALE
    if width < W_BOUNDS[0] or width > W_BOUNDS[1]:
        return None
    # Ray was along -approach from p1 → p1 = center + 0.5 w approach.
    c1 = center + 0.5 * width * approach
    c2 = center - 0.5 * width * approach
    return np.stack([c1, c2], axis=0).astype(np.float32)


class TwoPointGraspDataset(Dataset):
    """(point_cloud, K×6 tanh contact actions) from antipodal labels."""

    def __init__(
        self,
        data_dir: str,
        n_pts: int = 128,
        n_grasp: int = 10,
        augment: bool = True,
        label_source: str = "antipodal",
    ):
        self.n_pts = n_pts
        self.n_grasp = n_grasp
        self.augment = augment
        self.samples = []

        for pc_path in sorted(
            glob.glob(os.path.join(data_dir, "**/*_pc.npy"), recursive=True)
        ):
            base = pc_path.replace("_pc.npy", "")
            g_path = base + "_grasps.json"
            gn_path = base + "_graspnet.json"
            has_g = os.path.exists(g_path)
            has_gn = os.path.exists(gn_path)
            if label_source == "antipodal" and has_g:
                self.samples.append((pc_path, g_path, None))
            elif label_source == "graspnet" and has_gn:
                self.samples.append((pc_path, None, gn_path))
            elif label_source == "both" and (has_g or has_gn):
                self.samples.append(
                    (pc_path, g_path if has_g else None, gn_path if has_gn else None)
                )

        if not self.samples:
            raise RuntimeError(
                f"No (pc, grasp) pairs under {data_dir} for label_source={label_source}"
            )
        print(
            f"[TwoPointGraspDataset] {len(self.samples)} samples in {data_dir} "
            f"(label_source={label_source})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc_path, g_path, gn_path = self.samples[idx]
        pc_full = np.load(pc_path).astype(np.float32) * OBJECT_SCALE
        chosen = np.random.choice(len(pc_full), self.n_pts, replace=False)
        pc = pc_full[chosen]

        grasp_list = []
        for path in (g_path, gn_path):
            if path is None:
                continue
            with open(path) as f:
                data = json.load(f)
            grasp_list += data["grasps"] if isinstance(data, dict) else data
        grasp_list = sorted(grasp_list, key=lambda g: g["quality"], reverse=True)

        actions = []
        for g in grasp_list:
            contacts = _label_to_contacts(g)
            if contacts is None:
                continue
            actions.append(_contacts_to_action(contacts[0], contacts[1]))
            if len(actions) >= self.n_grasp:
                break

        if not actions:
            # Fallback: small horizontal pinch around PC centroid.
            mid = pc.mean(axis=0)
            half = 0.5 * W_BOUNDS[0]
            c1 = mid + np.array([0.0, -half, 0.0], dtype=np.float32)
            c2 = mid + np.array([0.0, +half, 0.0], dtype=np.float32)
            actions = [_contacts_to_action(c1, c2)]

        actions = np.stack(actions, axis=0)
        if len(actions) < self.n_grasp:
            pad = np.repeat(actions[-1:], self.n_grasp - len(actions), axis=0)
            actions = np.vstack([actions, pad])
        else:
            actions = actions[: self.n_grasp]

        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            pc = (R @ pc.T).T
            # Rotate denormalized contacts, then re-normalize (yaw-only).
            new_actions = []
            for a in actions:
                pts = []
                for i in range(2):
                    ax, ay, az = a[3 * i : 3 * i + 3]
                    x = (ax + 1) / 2 * (X_BOUNDS[1] - X_BOUNDS[0]) + X_BOUNDS[0]
                    y = (ay + 1) / 2 * (Y_BOUNDS[1] - Y_BOUNDS[0]) + Y_BOUNDS[0]
                    z = (az + 1) / 2 * (Z_BOUNDS[1] - Z_BOUNDS[0]) + Z_BOUNDS[0]
                    pts.append(R @ np.array([x, y, z], dtype=np.float32))
                new_actions.append(_contacts_to_action(pts[0], pts[1]))
            actions = np.stack(new_actions, axis=0)

        return (
            torch.from_numpy(np.ascontiguousarray(pc)),
            torch.from_numpy(np.ascontiguousarray(actions)),
        )


def min_action_loss(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    pred    : (B, 6)
    targets : (B, K, 6)  — canonical left/right by +y
    Also scores the swapped ordering so left/right flips are not punished.
    """
    B, K, _ = targets.shape
    swap = targets[:, :, [3, 4, 5, 0, 1, 2]]
    cand = torch.stack([targets, swap], dim=2)  # (B, K, 2, 6)
    diff = cand - pred.view(B, 1, 1, 6)
    dists = diff.norm(dim=-1)  # (B, K, 2)
    return dists.amin(dim=(1, 2)).mean()


def _make_head(embed_dim: int, num_actions: int = 6) -> nn.Sequential:
    """Match GraspPoseActorCritic.actor_head so weights load 1:1."""
    return nn.Sequential(
        nn.Linear(embed_dim, 128), nn.ELU(),
        nn.Linear(128, 64), nn.ELU(),
        nn.Linear(64, num_actions),
        nn.Tanh(),
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pretrain-2pt] device={device}")

    dataset = TwoPointGraspDataset(
        args.data,
        n_pts=args.n_pts,
        n_grasp=args.n_grasp,
        augment=not args.no_augment,
        label_source=args.label_source,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    encoder = PointNetEncoder(embed_dim=args.embed_dim).to(device)
    head = _make_head(args.embed_dim, num_actions=6).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    os.makedirs(args.out, exist_ok=True)
    best_loss = float("inf")
    avg = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for pc, actions in loader:
            pc = pc.to(device)
            actions = actions.to(device)
            pred = head(encoder(pc))
            loss = min_action_loss(pred, actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / max(len(loader), 1)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if avg < best_loss:
            best_loss = avg
            save_path = os.path.join(args.out, "grasp_2pt_pretrain_best.pt")
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "head": head.state_dict(),
                    "epoch": epoch,
                    "loss": avg,
                    "embed_dim": args.embed_dim,
                    "n_pts": args.n_pts,
                    "num_actions": 6,
                    "mode": "two_point",
                    "object_scale": OBJECT_SCALE,
                    "bounds": {
                        "x": X_BOUNDS,
                        "y": Y_BOUNDS,
                        "z": Z_BOUNDS,
                        "w": W_BOUNDS,
                    },
                },
                save_path,
            )

    final_path = os.path.join(args.out, "grasp_2pt_pretrain_final.pt")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "epoch": args.epochs,
            "loss": avg,
            "embed_dim": args.embed_dim,
            "n_pts": args.n_pts,
            "num_actions": 6,
            "mode": "two_point",
            "object_scale": OBJECT_SCALE,
        },
        final_path,
    )
    print(f"\n[pretrain-2pt] done.  best loss={best_loss:.4f}")
    print(f"  best  → {os.path.join(args.out, 'grasp_2pt_pretrain_best.pt')}")
    print(f"  final → {final_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/objects/train")
    p.add_argument("--out", default="data/grasp_weights")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_pts", type=int, default=128)
    p.add_argument("--n_grasp", type=int, default=10)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument(
        "--label_source",
        type=str,
        default="antipodal",
        choices=["antipodal", "graspnet", "both"],
    )
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
