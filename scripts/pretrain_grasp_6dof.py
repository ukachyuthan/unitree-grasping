#!/usr/bin/env python3
"""
Behaviour-cloning warm start for the 6-DoF grasp policy, from Contact-GraspNet.

Distils the offline Contact-GraspNet grasps (scripts/generate_graspnet_labels.py,
<instance>_graspnet.json) into the RL actor (GraspPoseActorCritic actor_encoder +
actor_head), in exactly the observation / action space of GraspPoseEnv with
grasp_mode="six_dof", obs_frame="centroid":

  observation  128 object points, random spawn yaw, the env's sensor noise
               (jitter / dropout / outliers), centred on the noisy centroid,
               world-aligned axes — no object pose
  action (7)   c1, c2 in that frame, normalized by the grasp bounds, and
               ψ / six_dof_max_tilt_rad, where ψ is the approach angle about the
               jaw axis (envs/six_dof_geometry.approach_from_jaw)

Each object has many good grasps, so the loss is the distance to the NEAREST
labelled grasp; the finger swap (c1↔c2, ψ→−ψ) is the same grasp.

Sim-free (no Isaac). Then RL:

    python scripts/pretrain_grasp_6dof.py
    cd wrappers/isaaclab && python scripts/train_grasp_pose.py --headless \\
        --pretrain ../../data/grasp_weights/grasp_6dof_pretrain_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wrappers", "isaaclab"))
from grasping.pointcloud_utils import add_sensor_noise            # noqa: E402
from models.grasp_pose_actor_critic import GraspPoseActorCritic   # noqa: E402
from envs.six_dof_geometry import approach_from_jaw               # noqa: E402

# Must match GraspPoseEnvCfg (grasp_pose_env_cfg.py).
OBJECT_SCALE = 1.5
NUM_PC_POINTS = 128
BOUNDS = ((-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07))   # grasp_{x,y,z}_bounds
MAX_TILT = math.pi / 2                                   # six_dof_max_tilt_rad
PC_NOISE_M = (0.0, 0.005)
PC_DROPOUT = (0.0, 0.10)
PC_OUTLIER = (0.0, 0.02)


def psi_of(c1: torch.Tensor, c2: torch.Tensor, approach: torch.Tensor) -> torch.Tensor:
    """Approach angle ψ about the jaw c1→c2 such that approach_from_jaw(u, ψ) = approach."""
    u = c2 - c1
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    down = torch.zeros_like(u)
    down[:, 2] = -1.0
    a0 = down - (down * u).sum(-1, keepdim=True) * u
    a0 = a0 / a0.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    b = torch.cross(u, a0, dim=-1)
    return torch.atan2((approach * b).sum(-1), (approach * a0).sum(-1))


def load_split(split_dir: str, max_labels: int):
    """→ pcs (I, P, 3), labels (I, K, 7) [c1, c2, ψ] in the scaled object frame, mask (I, K)."""
    manifest = json.load(open(os.path.join(split_dir, "manifest.json")))
    pcs, labels, masks, names = [], [], [], []
    n_skip_jaw = n_skip_below = 0
    for e in manifest:
        gpath = os.path.join(split_dir, e["pc_path"][: -len("_pc.npy")] + "_graspnet.json")
        if not os.path.exists(gpath):
            continue
        grasps = json.load(open(gpath))["grasps"][:max_labels]
        if not grasps:
            continue
        c1 = torch.tensor([g["c1"] for g in grasps], dtype=torch.float32) * OBJECT_SCALE
        c2 = torch.tensor([g["c2"] for g in grasps], dtype=torch.float32) * OBJECT_SCALE
        a = torch.tensor([g["approach"] for g in grasps], dtype=torch.float32)
        u = (c2 - c1) / (c2 - c1).norm(dim=-1, keepdim=True).clamp(min=1e-9)
        ok = u[:, 2].abs() < 0.98            # vertical jaw: ψ reference is the env's radial direction
        n_skip_jaw += int((~ok).sum())
        psi = psi_of(c1, c2, a)
        below = psi.abs() > MAX_TILT + 1e-3  # would approach from under the table
        n_skip_below += int((below & ok).sum())
        ok &= ~below
        if not ok.any():
            continue
        lab = torch.cat([c1, c2, psi.clamp(-MAX_TILT, MAX_TILT).unsqueeze(-1)], dim=-1)[ok]
        pc = torch.from_numpy(np.load(os.path.join(split_dir, e["pc_path"])).astype(np.float32)) * OBJECT_SCALE
        pcs.append(pc)
        labels.append(lab)
        names.append(f"{e['family']}/{e['instance']:03d}")
    K = max(len(l) for l in labels)
    L = torch.zeros(len(labels), K, 7)
    M = torch.zeros(len(labels), K, dtype=torch.bool)
    for i, l in enumerate(labels):
        L[i, : len(l)] = l
        M[i, : len(l)] = True
    print(f"[pretrain-6dof] {split_dir}: {len(pcs)} instances, {int(M.sum())} grasps "
          f"(skipped {n_skip_jaw} vertical-jaw, {n_skip_below} from-below)")
    return torch.stack(pcs), L, M, names


def make_batch(pcs, labels, mask, idx, device, noise=True):
    """Env-identical observation + targets for instances idx (random yaw, noise, centroid)."""
    B = len(idx)
    pc, lab, m = pcs[idx].to(device), labels[idx].to(device), mask[idx].to(device)
    yaw = torch.empty(B, device=device).uniform_(-math.pi, math.pi)
    c, s = yaw.cos(), yaw.sin()
    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
    pick = torch.rand(B, pc.shape[1], device=device).argsort(dim=1)[:, :NUM_PC_POINTS]
    pts = pc.gather(1, pick.unsqueeze(-1).expand(-1, -1, 3)) @ R.transpose(1, 2)
    if noise:
        def rng(r):
            return torch.empty(B, device=device).uniform_(*r)
        pts = add_sensor_noise(pts, gaussian_std=rng(PC_NOISE_M), dropout_frac=rng(PC_DROPOUT),
                               outlier_frac=rng(PC_OUTLIER))
    centroid = pts.mean(dim=1, keepdim=True)
    obs = pts - centroid
    c1 = lab[..., 0:3] @ R.transpose(1, 2) - centroid      # yaw leaves ψ unchanged
    c2 = lab[..., 3:6] @ R.transpose(1, 2) - centroid
    lo = torch.tensor([b[0] for b in BOUNDS], device=device)
    hi = torch.tensor([b[1] for b in BOUNDS], device=device)
    n1 = 2 * (c1 - lo) / (hi - lo) - 1
    n2 = 2 * (c2 - lo) / (hi - lo) - 1
    in_bounds = (n1.abs() <= 1).all(-1) & (n2.abs() <= 1).all(-1)
    tgt = torch.cat([n1, n2, lab[..., 6:7] / MAX_TILT], dim=-1)
    return obs.reshape(B, -1), tgt, m & in_bounds


def nearest_grasp_loss(pred, tgt, mask):
    """Mean over the batch of the distance to the nearest valid label (either finger order)."""
    swap = torch.cat([tgt[..., 3:6], tgt[..., 0:3], -tgt[..., 6:7]], dim=-1)
    d = torch.stack([(tgt - pred[:, None]).norm(dim=-1), (swap - pred[:, None]).norm(dim=-1)], dim=-1)
    d = torch.where(mask.unsqueeze(-1), d, torch.full_like(d, 1e6))
    best = d.flatten(1).min(dim=1).values
    has = mask.any(dim=1)
    return best[has].mean(), best[has]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=os.path.join(ROOT, "data/objects/train"))
    p.add_argument("--eval", default=os.path.join(ROOT, "data/objects/eval"))
    p.add_argument("--out", default=os.path.join(ROOT, "data/grasp_weights"))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_labels", type=int, default=100, help="top-K GraspNet grasps per object")
    p.add_argument("--pc_embed_dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = load_split(args.train, args.max_labels)
    ev = load_split(args.eval, args.max_labels) if os.path.exists(os.path.join(args.eval, "manifest.json")) else None

    ac = GraspPoseActorCritic(num_actor_obs=NUM_PC_POINTS * 3, num_critic_obs=NUM_PC_POINTS * 3,
                              num_actions=7, num_pc_points=NUM_PC_POINTS,
                              pc_embed_dim=args.pc_embed_dim).to(device)
    params = list(ac.actor_encoder.parameters()) + list(ac.actor_head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    os.makedirs(args.out, exist_ok=True)

    def evaluate(split, n_rounds=4):
        pcs, L, M, _ = split
        ac.eval()
        ds = []
        with torch.no_grad():
            for _ in range(n_rounds):
                for i in range(0, len(pcs), 512):
                    idx = torch.arange(i, min(i + 512, len(pcs)))
                    obs, tgt, m = make_batch(pcs, L, M, idx, device)
                    _, d = nearest_grasp_loss(ac.act_inference(obs), tgt, m)
                    ds.append(d)
        ac.train()
        d = torch.cat(ds)
        return d.mean().item(), d.median().item()

    best = float("inf")
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(tr[0]), (args.batch,))
        obs, tgt, m = make_batch(*tr[:3], idx, device)
        loss, _ = nearest_grasp_loss(ac.act_inference(obs), tgt, m)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 1:
            msg = f"  step {step:6d}  train_loss={loss.item():.4f}"
            score = loss.item()
            if ev is not None:
                ev_mean, ev_med = evaluate(ev)
                msg += f"  eval nearest-grasp dist mean={ev_mean:.4f} median={ev_med:.4f}"
                score = ev_mean
            print(msg + f"  lr={sched.get_last_lr()[0]:.1e}", flush=True)
            ckpt = {
                "encoder": ac.actor_encoder.state_dict(), "head": ac.actor_head.state_dict(),
                "mode": "six_dof", "obs_frame": "centroid", "num_actions": 7,
                "embed_dim": args.pc_embed_dim, "n_pts": NUM_PC_POINTS, "object_scale": OBJECT_SCALE,
                "bounds": BOUNDS, "max_tilt_rad": MAX_TILT, "label_source": "contact_graspnet",
                "step": step, "score": score,
            }
            if score < best:
                best = score
                torch.save(ckpt, os.path.join(args.out, "grasp_6dof_pretrain_best.pt"))
    torch.save(ckpt, os.path.join(args.out, "grasp_6dof_pretrain_final.pt"))
    print(f"[pretrain-6dof] best score {best:.4f} → {args.out}/grasp_6dof_pretrain_best.pt")


if __name__ == "__main__":
    main()
