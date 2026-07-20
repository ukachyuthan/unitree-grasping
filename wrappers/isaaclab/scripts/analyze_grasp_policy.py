#!/usr/bin/env python3
"""
Visualize where a trained policy grasps each object shape.

Runs N_PER_SHAPE episodes per shape (headless), records where the policy
chose to grasp in object-local coordinates, then saves:
  1. Grasp-distribution figure: PC (blue) + grasp targets colored by success/fail.
  2. Orientation figure: (tilt, roll) scatter colored by reward per shape.
  3. Optional .npz with raw data for further offline analysis.

Usage:
    python wrappers/isaaclab/scripts/analyze_grasp_policy.py --headless \\
        --checkpoint data/grasp_logs/.../grasp_pose_final.pt \\
        --n_per_shape 30 --out data/viz/grasp_analysis.png

    # Sanity check without a checkpoint:
    python wrappers/isaaclab/scripts/analyze_grasp_policy.py --headless \\
        --debug_action random --out data/viz/grasp_random.png
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grasp-policy distribution visualization")
parser.add_argument("--checkpoint",     type=str,   default=None,
                    help="Trained .pt checkpoint (omit with --debug_action)")
parser.add_argument("--debug_action",   type=str,   choices=["zero", "random"], default=None)
parser.add_argument("--n_per_shape",    type=int,   default=30,
                    help="Episodes per shape family")
parser.add_argument("--out",            type=str,   default="data/viz/grasp_analysis.png")
parser.add_argument("--save_npz",       type=str,   default=None,
                    help="Also save raw episode data as .npz")
parser.add_argument("--num_pc_points",  type=int,   default=128)
parser.add_argument("--pc_embed_dim",   type=int,   default=128)
parser.add_argument("--success_thresh", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap
bootstrap()

from envs.grasp_pose_env_cfg import GraspPoseEnvCfg, OBS_DIM, NUM_ACTIONS
from envs.grasp_pose_env import GraspPoseEnv, _SHAPE_NAMES, NUM_SHAPES
from models.grasp_pose_actor_critic import GraspPoseActorCritic


# ── Policy loader ──────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: str) -> GraspPoseActorCritic:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model = GraspPoseActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=NUM_ACTIONS,
        num_pc_points=args.num_pc_points,
        pc_embed_dim=args.pc_embed_dim,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _obs_tensor(obs_td, device):
    return obs_td["policy"].to(device)


# ── Episode data collection ────────────────────────────────────────────────────

def collect_grasp_data(env: GraspPoseEnv, policy, device: str, n_per_shape: int):
    """Run n_per_shape × NUM_SHAPES episodes; return per-shape stats dicts."""
    u = env.unwrapped
    u.cfg.eval_cycle_shapes = True
    u.cfg.eval_num_shapes = NUM_SHAPES
    u.cfg.task_mode_prob_place = 0.0  # lift-only so reward = pure lift quality

    per_shape = {
        name: {"targets": [], "tilts": [], "rolls": [], "rewards": []}
        for name in _SHAPE_NAMES
    }

    total = n_per_shape * NUM_SHAPES
    obs_dict, _ = env.reset()
    collected = 0

    while collected < total:
        # Shape for THIS episode is already in _env_shape (set during last reset).
        current_shapes = u._env_shape.clone()

        if args.debug_action == "zero":
            action = torch.zeros(1, NUM_ACTIONS, device=device)
        elif args.debug_action == "random":
            action = torch.empty(1, NUM_ACTIONS, device=device).uniform_(-1, 1)
        else:
            obs = _obs_tensor(obs_dict, device)
            with torch.no_grad():
                action = policy.act_inference(obs)

        obs_dict, rew, _, _, _ = env.step(action)

        # _pre_physics_step already set _grasp_target from this action;
        # _reset_idx does NOT clear it so it still holds the episode's value.
        shape_idx = current_shapes[0].item()
        name = _SHAPE_NAMES[shape_idx]
        per_shape[name]["targets"].append(u._grasp_target[0].detach().cpu().numpy())
        per_shape[name]["tilts"].append(u._tilt_target[0].item())
        per_shape[name]["rolls"].append(u._roll_target[0].item())
        per_shape[name]["rewards"].append(rew[0].item())

        collected += 1
        if collected % max(n_per_shape, 10) == 0:
            print(f"  {collected}/{total} episodes collected")

    # Trim each shape to exactly n_per_shape (may overshoot by rounding).
    for name in _SHAPE_NAMES:
        d = per_shape[name]
        for key in ("targets", "tilts", "rolls", "rewards"):
            arr = np.array(d[key])
            d[key] = arr[:n_per_shape]

    return per_shape


# ── Grasp-distribution figure ──────────────────────────────────────────────────

def plot_grasp_distribution(env: GraspPoseEnv, per_shape: dict, out_path: str,
                            success_thresh: float):
    u = env.unwrapped
    pcs_all = u._obj_pcs.detach().cpu().numpy()   # (NUM_SHAPES, 512, 3)

    COLS, ROWS = 4, 3
    fig = plt.figure(figsize=(COLS * 4, ROWS * 3.6))
    fig.suptitle(
        "Grasp-point distribution per shape  (top view XY)\n"
        "Blue = object PC  |  Green = success  |  Red ✕ = fail",
        fontsize=12, y=0.98,
    )
    gs = gridspec.GridSpec(ROWS, COLS, figure=fig, hspace=0.44, wspace=0.32)

    success_color = "#2ecc71"
    fail_color    = "#e74c3c"
    pc_color      = "#2980b9"
    overall: list[float] = []

    for idx, name in enumerate(_SHAPE_NAMES):
        row, col = divmod(idx, COLS)
        ax = fig.add_subplot(gs[row, col])

        pc = pcs_all[idx]          # (512, 3)
        d  = per_shape[name]
        tgt = d["targets"]         # (N, 3)
        rew = d["rewards"]         # (N,)
        ok  = rew >= success_thresh
        rate = 100.0 * ok.sum() / max(len(rew), 1)
        overall.extend(rew.tolist())

        ax.scatter(pc[:, 0], pc[:, 1],
                   s=3, c=pc_color, alpha=0.25, zorder=1, linewidths=0)

        if ok.any():
            ax.scatter(tgt[ok, 0], tgt[ok, 1],
                       s=45, c=success_color, alpha=0.8, zorder=3,
                       edgecolors="white", linewidths=0.4)
        if (~ok).any():
            ax.scatter(tgt[~ok, 0], tgt[~ok, 1],
                       s=45, c=fail_color, alpha=0.7, zorder=2,
                       marker="x", linewidths=1.5)

        ax.set_title(f"{name.replace('_',' ')}  ({rate:.0f}%  {ok.sum()}/{len(rew)})",
                     fontsize=8.5)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        lim = 0.10
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=pc_color,
               markersize=6, label="object PC"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=success_color,
               markersize=7, label=f"success (reward ≥ {success_thresh:.1f})"),
        Line2D([0], [0], marker="x", color=fail_color, markersize=7, lw=0,
               label="fail"),
    ]
    fig.legend(handles=legend_elems, loc="lower right", fontsize=8,
               framealpha=0.85, ncol=3)

    overall_rate = 100.0 * sum(r >= success_thresh for r in overall) / max(len(overall), 1)
    fig.text(0.01, 0.01,
             f"Overall success rate: {overall_rate:.1f}%  ({len(overall)} episodes)",
             fontsize=9, ha="left", va="bottom",
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[analyze] saved → {out_path}")
    plt.close(fig)
    return overall_rate


# ── Orientation distribution figure ───────────────────────────────────────────

def plot_orientation_dist(per_shape: dict, out_path: str, success_thresh: float):
    """Scatter of (tilt, roll) per shape colored by reward — shows learned orientations."""
    COLS, ROWS = 4, 3
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3.5, ROWS * 3), squeeze=False)
    fig.suptitle("EE orientation choices  (tilt = panda_joint5, roll = panda_joint7)\n"
                 "Dot color = reward  (green = high)", fontsize=11, y=0.99)

    for idx, name in enumerate(_SHAPE_NAMES):
        row, col = divmod(idx, COLS)
        ax = axes[row][col]
        d  = per_shape[name]
        ok = np.array(d["rewards"]) >= success_thresh
        ax.scatter(d["tilts"], d["rolls"], c=d["rewards"],
                   cmap="RdYlGn", vmin=0.0, vmax=1.0,
                   s=30, alpha=0.75, edgecolors="grey", linewidths=0.3)
        ax.set_title(f"{name.replace('_',' ')}  ({ok.sum()}/{len(d['rewards'])})",
                     fontsize=8)
        ax.set_xlabel("tilt j5 (rad)", fontsize=7)
        ax.set_ylabel("roll j7 (rad)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="reward")

    orient_path = out_path.replace(".png", "_orientation.png")
    fig.savefig(orient_path, dpi=120, bbox_inches="tight")
    print(f"[analyze] saved → {orient_path}")
    plt.close(fig)


# ── Summary print ──────────────────────────────────────────────────────────────

def print_summary(per_shape: dict, success_thresh: float):
    print("\n── Per-shape success rates ──────────────────────────────────")
    for name in _SHAPE_NAMES:
        d   = per_shape[name]
        rew = d["rewards"]
        ok  = (rew >= success_thresh).sum()
        print(f"  {name:18s}  {ok:2d}/{len(rew)}"
              f"  ({100*ok/max(len(rew),1):5.1f}%)"
              f"  tilt={np.mean(d['tilts']):.2f}±{np.std(d['tilts']):.2f}"
              f"  roll={np.mean(d['rolls']):.2f}±{np.std(d['rolls']):.2f}")
    print("────────────────────────────────────────────────────────────\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not args.debug_action and not args.checkpoint:
        print("[analyze] provide --checkpoint or --debug_action zero|random")
        sys.exit(1)

    device = args.device

    env_cfg = GraspPoseEnvCfg()
    env_cfg.scene.num_envs = 1   # single env to iterate shapes sequentially
    env_cfg.sim.device = device
    env_cfg.visualize_grasp_point = False
    env_cfg.eval_cycle_shapes = True
    env_cfg.eval_num_shapes = NUM_SHAPES
    env_cfg.task_mode_prob_place = 0.0   # lift-only for clean reward signal

    env = GraspPoseEnv(cfg=env_cfg, render_mode=None)

    policy = None
    if args.debug_action:
        print(f"[analyze] debug mode: {args.debug_action}")
    else:
        ckpt = os.path.abspath(args.checkpoint)
        if not os.path.isfile(ckpt):
            print(f"[analyze] checkpoint not found: {ckpt}")
            sys.exit(1)
        policy = load_policy(ckpt, device)
        print(f"[analyze] loaded {ckpt}")

    total = args.n_per_shape * NUM_SHAPES
    print(f"[analyze] collecting {total} episodes ({args.n_per_shape} per shape × {NUM_SHAPES} shapes)")

    per_shape = collect_grasp_data(env, policy, device, args.n_per_shape)

    print_summary(per_shape, args.success_thresh)
    overall_rate = plot_grasp_distribution(env, per_shape, args.out, args.success_thresh)
    plot_orientation_dist(per_shape, args.out, args.success_thresh)

    if args.save_npz:
        npz_data = {}
        for name in _SHAPE_NAMES:
            d = per_shape[name]
            npz_data[f"{name}_targets"] = d["targets"]
            npz_data[f"{name}_rewards"] = d["rewards"]
            npz_data[f"{name}_tilts"]   = d["tilts"]
            npz_data[f"{name}_rolls"]   = d["rolls"]
        save_dir = os.path.dirname(os.path.abspath(args.save_npz)) or "."
        os.makedirs(save_dir, exist_ok=True)
        np.savez(args.save_npz, **npz_data)
        print(f"[analyze] raw data → {args.save_npz}")

    print(f"[analyze] overall success rate: {overall_rate:.1f}%")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
