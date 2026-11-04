#!/usr/bin/env python3
"""
Human grasp prior: where on a point cloud would a human grasp?

Imitating the demos directly (train_grasp_pose.py --demos_dir) teaches the policy
individual grasps on the ~200 demonstrated shapes, and stalls about 1.5–2 cm short
of them. This instead learns the pattern behind the demos, so it applies to shapes
nobody demonstrated:

  data      pose-transfer demos of the human VR grasps (teach_failure_cases.py
            --transfer): contact points in the object frame, verified to lift
            with the Franka hand. Pose-independent, so every demo of a shape
            labels the same point cloud.
  label     per point of the shape's 512-point cloud, exp(-d² / 2σ²) to the
            nearest human contact (σ = --sigma_m).
  model     a small EdgeConv (DGCNN-style) segmentation net: per-point logit,
            trained with soft BCE under random 3-D rotations.
  check     --folds cross-validation over whole families (no family is seen in
            both train and test): per-point AUC and the hit rate of human
            contacts in the model's top 10% of points, against chance.
  output    the final model (trained on every family) scores every shape in
            data/objects/{train,eval}; scores are normalised to max 1 per shape
            → <out>/prior.npz, read by the env's human-prior reward term
            (--human_prior in train_grasp_pose.py).

    python scripts/train_human_prior.py \\
        --demos data/pose_transfer data/pose_transfer_variants data/pt_human_flat \\
        --out data/human_prior
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wrappers", "isaaclab"))
from training.demo_replay import SOURCE_TRANSFER, load_demos               # noqa: E402
from envs._object_registry import shape_asset, split_variant, variant_name  # noqa: E402

OBJECT_SCALE = 1.5   # must match grasp_pose_env_cfg.OBJECT_SCALE


def random_rotations(n: int, gen: torch.Generator) -> torch.Tensor:
    """(n, 3, 3) uniform over SO(3) (QR of a Gaussian, sign-fixed)."""
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, generator=gen))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    det = torch.det(q)
    q[det < 0, :, 0] *= -1
    return q


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """x (B, N, C) → (B, N, k) indices of the nearest neighbours (self included)."""
    return torch.cdist(x, x).topk(k, dim=-1, largest=False).indices


def edge_features(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(B, N, C), (B, N, k) → (B, N, k, 2C): [x_i, x_j − x_i]."""
    B, N, k = idx.shape
    offset = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
    nb = x.reshape(B * N, -1)[(idx + offset).reshape(-1)].view(B, N, k, -1)
    xi = x.unsqueeze(2).expand_as(nb)
    return torch.cat([xi, nb - xi], dim=-1)


class GraspPriorNet(nn.Module):
    """Two EdgeConv layers + a global feature → one logit per point."""

    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k
        self.e1 = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.e2 = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.glob = nn.Sequential(nn.Linear(128, 256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(128 + 256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        x = pc - pc.mean(dim=1, keepdim=True)
        f1 = self.e1(edge_features(x, knn(x, self.k))).max(dim=2).values
        f2 = self.e2(edge_features(f1, knn(f1, self.k))).max(dim=2).values
        f = torch.cat([f1, f2], dim=-1)
        g = self.glob(f).max(dim=1, keepdim=True).values.expand(-1, f.shape[1], -1)
        return self.head(torch.cat([f, g], dim=-1)).squeeze(-1)


def load_shape_pc(shape: str) -> np.ndarray:
    return np.load(shape_asset(shape, "_pc.npy")).astype(np.float32) * OBJECT_SCALE


def build_labels(roots, sigma_m: float):
    """→ shapes, pcs (S, P, 3), heat (S, P), contacts per shape."""
    contacts = defaultdict(list)
    for d in load_demos(roots, sources=(SOURCE_TRANSFER,)):
        contacts[d.shape] += [d.c1_local, d.c2_local]
    shapes = sorted(contacts)
    pcs = np.stack([load_shape_pc(s) for s in shapes])
    heat = np.zeros(pcs.shape[:2], dtype=np.float32)
    for i, s in enumerate(shapes):
        c = np.stack(contacts[s])                                      # (M, 3)
        d2 = ((pcs[i][:, None, :] - c[None]) ** 2).sum(-1).min(-1)     # (P,)
        heat[i] = np.exp(-d2 / (2 * sigma_m ** 2))
    return shapes, pcs, heat, {s: np.stack(v) for s, v in contacts.items()}


def train(pcs, heat, steps, batch, lr, device, seed):
    gen = torch.Generator().manual_seed(seed)
    net = GraspPriorNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    P = torch.from_numpy(pcs)
    H = torch.from_numpy(heat)
    pos_weight = torch.tensor((1 - heat.mean()) / max(heat.mean(), 1e-6)).clamp(max=20.0).to(device)
    for step in range(steps):
        i = torch.randint(len(P), (batch,), generator=gen)
        R = random_rotations(batch, gen)
        x = (P[i] @ R.transpose(1, 2) + 0.001 * torch.randn(batch, P.shape[1], 3, generator=gen)).to(device)
        y = H[i].to(device)
        logit = net(x)
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step + 1}/{steps}  loss={loss.item():.4f}", flush=True)
    return net.eval()


@torch.no_grad()
def score(net, pcs: np.ndarray, device, batch: int = 32) -> np.ndarray:
    out = []
    for s in range(0, len(pcs), batch):
        out.append(torch.sigmoid(net(torch.from_numpy(pcs[s:s + batch]).to(device))).cpu().numpy())
    return np.concatenate(out)


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank AUC: P(score of a positive point > score of a negative one)."""
    if positive.all() or not positive.any():
        return float("nan")
    ranks = scores.argsort().argsort().astype(np.float64)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def top_hit_rate(scores: np.ndarray, pc: np.ndarray, contacts: np.ndarray,
                 frac: float = 0.1, radius_m: float = 0.0075) -> float:
    """Fraction of human contacts within radius of one of the top-`frac` points."""
    top = pc[scores >= np.quantile(scores, 1 - frac)]
    d = np.linalg.norm(contacts[:, None, :] - top[None], axis=-1).min(-1)
    return float((d < radius_m).mean())


def evaluate(scores, pcs, heat, contacts_list, rng):
    a, hit, chance = [], [], []
    for s, pc, h, c in zip(scores, pcs, heat, contacts_list):
        a.append(auc(s, h > 0.5))
        hit.append(top_hit_rate(s, pc, c))
        chance.append(top_hit_rate(rng.random(len(s)), pc, c))
    return float(np.nanmean(a)), float(np.mean(hit)), float(np.mean(chance))


def all_shapes() -> list[tuple[str, Path]]:
    """(shape name, point-cloud path) of every instance on disk, both splits."""
    found = {}
    for split in ("train", "eval"):
        for p in sorted(Path(ROOT, "data", "objects", split).glob("*/[0-9][0-9][0-9]_pc.npy")):
            found.setdefault(variant_name(p.parent.name, int(p.name[:3])), p)
    return sorted(found.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", nargs="+", required=True, help="Pose-transfer demo roots (human).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma_m", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5, help="Family-level CV folds (0 = skip).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shapes, pcs, heat, contacts = build_labels(args.demos, args.sigma_m)
    families = np.array([split_variant(s)[0] for s in shapes])
    fam_list = sorted(set(families))
    print(f"[prior] {len(shapes)} shapes, {len(fam_list)} families, "
          f"{sum(len(c) for c in contacts.values()) // 2} human grasps; "
          f"{(heat > 0.5).mean() * 100:.1f}% of points labelled positive")

    report = {"shapes": len(shapes), "families": len(fam_list), "folds": []}
    if args.folds:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(fam_list)
        for f in range(args.folds):
            test_fam = set(order[f::args.folds])
            te = np.isin(families, list(test_fam))
            print(f"[prior] fold {f + 1}/{args.folds}: {te.sum()} test shapes from {len(test_fam)} unseen families")
            net = train(pcs[~te], heat[~te], args.steps, args.batch, args.lr, device, args.seed + f)
            s = score(net, pcs[te], device)
            a, hit, chance = evaluate(s, pcs[te], heat[te], [contacts[shapes[i]] for i in np.where(te)[0]], rng)
            print(f"[prior]   unseen families: AUC {a:.3f}   human contacts in top 10%: "
                  f"{hit * 100:.1f}% (chance {chance * 100:.1f}%)")
            report["folds"].append({"families": sorted(test_fam), "auc": a, "top10_hit": hit, "chance": chance})
        for key in ("auc", "top10_hit", "chance"):
            report[key] = float(np.mean([r[key] for r in report["folds"]]))
        print(f"[prior] CV over unseen families: AUC {report['auc']:.3f}   top-10% hit "
              f"{report['top10_hit'] * 100:.1f}% vs chance {report['chance'] * 100:.1f}%")

    print("[prior] final model on every family")
    net = train(pcs, heat, args.steps, args.batch, args.lr, device, args.seed)
    torch.save(net.state_dict(), out / "prior_net.pt")
    found = all_shapes()
    names = [n for n, _ in found]
    all_pcs = np.stack([np.load(p).astype(np.float32) * OBJECT_SCALE for _, p in found])
    s = score(net, all_pcs, device)
    s = s / s.max(axis=1, keepdims=True).clip(min=1e-6)
    np.savez_compressed(out / "prior.npz", names=np.array(names), scores=s.astype(np.float16))
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(f"[prior] scored {len(names)} shapes → {out / 'prior.npz'}")


if __name__ == "__main__":
    main()
