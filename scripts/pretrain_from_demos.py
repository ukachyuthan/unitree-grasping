#!/usr/bin/env python3
"""
Offline imitation of verified demo grasp points, before RL.

Imitation during RL (train_grasp_pose.py --demos_dir) moves slowly: PPO's own
updates keep pulling the policy around. This trains the RL policy's actor
directly on the demos first, like the Contact-GraspNet warm start
(scripts/pretrain_grasp_6dof.py), whose batch construction it shares:

  demos       pose-transfer demos (teach_failure_cases.py --transfer): a human's
              two grasp points on an object in a 3-D orientation where they were
              verified to lift. Observation: the object's point cloud in that
              orientation (plus a random turn about the vertical, which keeps the
              same support), with the env's sensor noise, centred on the noisy
              centroid. Target: the two contact points only — the approach is
              left to RL.
  anchor      ordinary training objects, flat and in random 3-D orientations,
              mixed into every batch (--anchor_frac) with the starting policy's
              own action as the target, so it keeps what RL taught it. (Anchoring
              to the Contact-GraspNet labels instead undid RL's gains over them:
              training success fell from ~50% to 27%.)

Starts from an RL checkpoint and writes one (actor trained; critic copied), so
RL resumes straight from it:

    python scripts/pretrain_from_demos.py \\
        --init wrappers/isaaclab/data/grasp_logs/<run>/grasp_pose_125.pt \\
        --demos data/pose_transfer data/pose_transfer_variants \\
        --out data/grasp_weights/demo_bc.pt
    cd wrappers/isaaclab && python scripts/train_grasp_pose.py --headless \\
        --resume ../../data/grasp_weights/demo_bc.pt --demos_dir ...
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "wrappers", "isaaclab"))
from grasping.pointcloud_utils import add_sensor_noise                     # noqa: E402
from models.grasp_pose_actor_critic import GraspPoseActorCritic            # noqa: E402
from training.demo_replay import SOURCE_TRANSFER, load_demos               # noqa: E402
from envs._object_registry import shape_asset, split_variant               # noqa: E402
from pretrain_grasp_6dof import (                                          # noqa: E402
    BOUNDS, NUM_PC_POINTS, OBJECT_SCALE, PC_DROPOUT, PC_NOISE_M, PC_OUTLIER,
    load_split,
)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) wxyz → (N, 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def load_demo_set(roots):
    """Pose-transfer demos → per-demo shape index, rotation, contacts; and the shape PCs."""
    demos = load_demos(roots, sources=(SOURCE_TRANSFER,))
    shapes = sorted({d.shape for d in demos})
    pcs = []
    for s in shapes:
        pcs.append(torch.from_numpy(np.load(shape_asset(s, "_pc.npy")).astype(np.float32)) * OBJECT_SCALE)
    index = {s: i for i, s in enumerate(shapes)}
    shape_id = torch.tensor([index[d.shape] for d in demos])
    R = quat_to_matrix(torch.tensor(np.stack([d.quat_wxyz for d in demos]), dtype=torch.float32))
    # Every human grasp on a shape is a correct answer for that shape (grasp points
    # are object-local), so each sample is scored against the nearest of them, as
    # the Contact-GraspNet warm start is; one target per sample would average
    # several good grasps into a point on none of them.
    pairs = defaultdict(set)
    for d in demos:
        pairs[index[d.shape]].add(tuple(np.round(np.concatenate([d.c1_local, d.c2_local]), 4)))
    K = max(len(v) for v in pairs.values())
    c = torch.zeros(len(shapes), K, 6)
    cm = torch.zeros(len(shapes), K, dtype=torch.bool)
    for sid, v in pairs.items():
        c[sid, : len(v)] = torch.tensor(sorted(v), dtype=torch.float32)
        cm[sid, : len(v)] = True
    print(f"[demo-bc] up to {K} distinct human grasps per shape "
          f"(mean {np.mean([len(v) for v in pairs.values()]):.1f})")
    # Held-out demos for monitoring: every demo transferred from the same human
    # demo (same origin case) stays on one side, so this measures new cases.
    groups = defaultdict(list)
    for i, d in enumerate(demos):
        m = re.match(r"(.*?__it\d+__q\d+)", d.case_id)       # the failure case it came from
        groups[m.group(1) if m else d.case_id].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(keys)
    ev = {i for k in keys[: max(1, len(keys) // 10)] for i in groups[k]}
    is_eval = torch.tensor([i in ev for i in range(len(demos))])
    print(f"[demo-bc] {len(demos)} pose-transfer demos on {len(shapes)} shapes "
          f"({int(is_eval.sum())} held out for monitoring)")
    return torch.stack(pcs), shape_id, R, (c, cm), is_eval


def load_demo_shapes(roots) -> set[str]:
    return {d.shape for d in load_demos(roots, sources=(SOURCE_TRANSFER,))}


def demo_batch(pcs, shape_id, R, c, idx, device):
    """Env-identical observation + every human grasp of the shape as a candidate
    target (normalized), for demos idx. c = (pairs (S, K, 6), mask (S, K))."""
    c, cm = c
    B = len(idx)
    pc = pcs[shape_id[idx]].to(device)
    yaw = torch.empty(B, device=device).uniform_(-math.pi, math.pi)
    cy, sy = yaw.cos(), yaw.sin()
    Rz = torch.zeros(B, 3, 3, device=device)
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = cy, -sy, sy, cy, 1.0
    Rt = Rz @ R[idx].to(device)                         # turn about the vertical keeps the support
    pick = torch.randint(0, pc.shape[1], (B, NUM_PC_POINTS), device=device)
    pts = pc.gather(1, pick.unsqueeze(-1).expand(-1, -1, 3)) @ Rt.transpose(1, 2)

    def rng(r):
        return torch.empty(B, device=device).uniform_(*r)
    pts = add_sensor_noise(pts, gaussian_std=rng(PC_NOISE_M), dropout_frac=rng(PC_DROPOUT),
                           outlier_frac=rng(PC_OUTLIER))
    centroid = pts.mean(dim=1, keepdim=True)
    sid = shape_id[idx]
    K = c.shape[1]
    cc = c[sid].to(device).view(B, K * 2, 3) @ Rt.transpose(1, 2) - centroid
    lo = torch.tensor([b[0] for b in BOUNDS], device=device)
    hi = torch.tensor([b[1] for b in BOUNDS], device=device)
    n = (2 * (cc - lo) / (hi - lo) - 1).view(B, K, 6)
    ok = cm[sid].to(device) & (n.abs() <= 1).all(-1)
    return (pts - centroid).reshape(B, -1), n, ok


def anchor_obs(pcs, idx, device):
    """Observations of ordinary training objects as the env makes them: half flat
    at a random heading, half in a random 3-D orientation (tilted spawns)."""
    B = len(idx)
    pc = pcs[idx].to(device)
    q = torch.randn(B, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True)                  # uniform over SO(3)
    flat = torch.rand(B, device=device) < 0.5
    yaw = torch.empty(B, device=device).uniform_(-math.pi, math.pi)
    q[flat] = torch.stack([(yaw / 2).cos(), 0 * yaw, 0 * yaw, (yaw / 2).sin()], -1)[flat]
    Rt = quat_to_matrix(q)
    pick = torch.randint(0, pc.shape[1], (B, NUM_PC_POINTS), device=device)
    pts = pc.gather(1, pick.unsqueeze(-1).expand(-1, -1, 3)) @ Rt.transpose(1, 2)

    def rng(r):
        return torch.empty(B, device=device).uniform_(*r)
    pts = add_sensor_noise(pts, gaussian_std=rng(PC_NOISE_M), dropout_frac=rng(PC_DROPOUT),
                           outlier_frac=rng(PC_OUTLIER))
    return (pts - pts.mean(dim=1, keepdim=True)).reshape(B, -1)


def contact_loss(pred, tgt, ok):
    """Distance on the two contacts only to the nearest candidate grasp, either
    finger order. tgt (B, K, 6), ok (B, K)."""
    p = pred[:, None, :6]
    swap = torch.cat([tgt[..., 3:6], tgt[..., 0:3]], -1)
    d = torch.minimum((p - tgt).norm(dim=-1), (p - swap).norm(dim=-1))
    d = torch.where(ok, d, torch.full_like(d, 1e6)).min(dim=1).values
    has = ok.any(dim=1)
    return (d[has].mean() if has.any() else d.sum() * 0), d[has]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True, help="RL checkpoint to start from (grasp_pose_*.pt)")
    p.add_argument("--demos", nargs="+", required=True, help="pose-transfer demo roots")
    p.add_argument("--graspnet_train", default=os.path.join(ROOT, "data/objects/train"))
    p.add_argument("--out", default=os.path.join(ROOT, "data/grasp_weights/demo_bc.pt"))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--anchor_frac", type=float, default=0.5,
                   help="fraction of each batch that keeps the starting policy's own actions "
                        "on ordinary objects")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcs, shape_id, R, c, is_eval = load_demo_set(args.demos)
    train_ids = torch.nonzero(~is_eval).squeeze(-1)
    eval_ids = torch.nonzero(is_eval).squeeze(-1)
    gn = load_split(args.graspnet_train, 100)
    # Anchor only on shapes without demos: on demonstrated shapes the demos are
    # meant to change the policy, and keeping its old action there fights them.
    demo_shapes = {split_variant(s) for s in load_demo_shapes(args.demos)}
    keep = torch.tensor([(n.split("/")[0], int(n.split("/")[1])) not in demo_shapes for n in gn[3]])
    anchor_pcs = gn[0][keep]
    print(f"[demo-bc] anchor: {len(anchor_pcs)} of {len(gn[0])} training shapes (those without demos)")

    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    ac = GraspPoseActorCritic(num_actor_obs=NUM_PC_POINTS * 3, num_critic_obs=NUM_PC_POINTS * 3,
                              num_actions=7, num_pc_points=NUM_PC_POINTS).to(device)
    ac.load_state_dict(ckpt.get("model", ckpt))
    start = copy.deepcopy(ac).eval()
    for q in start.parameters():
        q.requires_grad_(False)
    params = list(ac.actor_encoder.parameters()) + list(ac.actor_head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    def evaluate():
        ac.eval()
        with torch.no_grad():
            ds = []
            for _ in range(4):
                for i in range(0, len(eval_ids), 512):
                    obs, tgt, ok = demo_batch(pcs, shape_id, R, c, eval_ids[i:i + 512], device)
                    ds.append(contact_loss(ac.act_inference(obs), tgt, ok)[1])
            d_demo = torch.cat(ds).mean().item()
            obs = anchor_obs(anchor_pcs, torch.randint(0, len(anchor_pcs), (1024,)), device)
            d_anchor = (ac.act_inference(obs) - start.act_inference(obs)).norm(dim=-1).mean().item()
        ac.train()
        return d_demo, d_anchor

    d0 = evaluate()
    print(f"[demo-bc] start: held-out demo contact error {d0[0]:.3f}")
    n_gn = int(args.batch * args.anchor_frac)
    for step in range(1, args.steps + 1):
        idx = train_ids[torch.randint(0, len(train_ids), (args.batch - n_gn,))]
        obs, tgt, ok = demo_batch(pcs, shape_id, R, c, idx, device)
        loss_demo, _ = contact_loss(ac.act_inference(obs), tgt, ok)
        loss = loss_demo
        if n_gn:
            aobs = anchor_obs(anchor_pcs, torch.randint(0, len(anchor_pcs), (n_gn,)), device)
            with torch.no_grad():
                keep = start.act_inference(aobs)
            loss = loss + (ac.act_inference(aobs) - keep).norm(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            d = evaluate()
            print(f"  step {step:5d}  train demo {loss_demo.item():.3f}  held-out demo {d[0]:.3f}  "
                  f"drift from start policy {d[1]:.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": ac.state_dict(), "iteration": 0, "init": args.init,
                "demo_roots": args.demos, "steps": args.steps}, args.out)
    d = evaluate()
    print(f"[demo-bc] held-out demo contact error {d0[0]:.3f} → {d[0]:.3f}; "
          f"drift from start policy on ordinary objects {d[1]:.3f}  → {args.out}")


if __name__ == "__main__":
    main()
