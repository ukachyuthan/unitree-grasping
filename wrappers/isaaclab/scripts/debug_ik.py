#!/usr/bin/env python3
"""
IK diagnostic: run one grasp episode headless, record EE + target + object
positions at every physics step, save a PNG showing the trajectories.

Usage:
    python wrappers/isaaclab/scripts/debug_ik.py --headless
    python wrappers/isaaclab/scripts/debug_ik.py --headless --num_envs 1 --out data/viz/debug_ik.png
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out",      type=str, default="data/viz/debug_ik.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.grasp_pose_env_cfg import GraspPoseEnvCfg, EXEC_STEPS, N_APPROACH, N_CLOSE, N_LIFT, NUM_ACTIONS
from envs.grasp_pose_env import GraspPoseEnv


def run_episode(env: GraspPoseEnv, action: torch.Tensor):
    """
    Manually step through one complete episode, recording positions.
    Returns arrays of shape (EXEC_STEPS, 3).
    """
    ee_traj   = []  # EE world position
    tgt_traj  = []  # grasp target (robot-base frame) → converted to world
    obj_traj  = []  # active object world position

    obs, _ = env.reset()
    env._pre_physics_step(action)

    for step in range(EXEC_STEPS):
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        ee_w   = env._robot.data.body_pos_w[:, env._ee_body_idx, :3].detach().cpu()
        obj_w  = env._get_active_obj_pos().detach().cpu()

        tgt_world   = env._obj_anchor[:, :3].detach().cpu() + env._grasp_target.detach().cpu()

        ee_traj.append(ee_w[0].numpy())
        tgt_traj.append(tgt_world[0].numpy())
        obj_traj.append(obj_w[0].numpy())

    return (
        np.array(ee_traj),   # (T, 3)
        np.array(tgt_traj),  # (T, 3)
        np.array(obj_traj),  # (T, 3)
    )


def plot_results(ee, tgt, obj, action_local, out_path):
    T = len(ee)
    t = np.arange(T)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("IK Debug: one grasp episode", fontsize=14)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    phase_ends = [N_APPROACH, N_APPROACH + N_CLOSE, N_APPROACH + N_CLOSE + N_LIFT]
    phase_labels = ["approach", "close", "lift", "hold"]
    phase_colors = ["#4c8eda", "#e07b2a", "#3baf5e", "#999999"]

    def add_phase_lines(ax):
        prev = 0
        for i, end in enumerate(phase_ends):
            ax.axvspan(prev, end, alpha=0.07, color=phase_colors[i],
                       label=phase_labels[i])
            prev = end
        ax.axvspan(prev, T, alpha=0.07, color=phase_colors[3],
                   label=phase_labels[3])

    axes_labels = ["X (m)", "Y (m)", "Z (m)"]
    for dim in range(3):
        ax = fig.add_subplot(gs[0, dim])
        ax.plot(t, ee[:, dim],  label="EE",     color="steelblue",  lw=1.5)
        ax.plot(t, tgt[:, dim], label="target", color="darkorange", lw=1.5, ls="--")
        ax.plot(t, obj[:, dim], label="object", color="green",      lw=1.5, ls=":")
        add_phase_lines(ax)
        ax.set_xlabel("physics step")
        ax.set_ylabel(axes_labels[dim])
        ax.set_title(f"World-frame {axes_labels[dim]}")
        if dim == 2:
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    dist = np.linalg.norm(ee - tgt, axis=1)
    ax.plot(t, dist * 100, color="crimson", lw=1.5)
    add_phase_lines(ax)
    for end in phase_ends:
        ax.axvline(end, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("physics step")
    ax.set_ylabel("distance (cm)")
    ax.set_title("EE ↔ target distance")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    delta_z = (obj[:, 2] - obj[0, 2]) * 100
    ax.plot(t, delta_z, color="darkgreen", lw=1.5)
    add_phase_lines(ax)
    ax.axhline(3.0, color="red", lw=0.8, ls="--", label="reward threshold (3cm)")
    ax.set_xlabel("physics step")
    ax.set_ylabel("Δz from spawn (cm)")
    ax.set_title("Object lift above spawn")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(ee[:, 0],  ee[:, 1],  "b-",  label="EE",     lw=1.5)
    ax.plot(tgt[:, 0], tgt[:, 1], "o--", label="target",  color="darkorange", ms=3)
    ax.plot(obj[0, 0], obj[0, 1], "g*",  label="object",  ms=12)
    ax.plot(ee[0, 0],  ee[0, 1],  "bs",  ms=8, label="EE start")
    ax.plot(ee[-1, 0], ee[-1, 1], "b^",  ms=8, label="EE end")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Bird's-eye view (XY plane)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    final_lift = (obj[-1, 2] - obj[0, 2]) * 100
    final_dist = np.linalg.norm(ee[N_APPROACH - 1] - tgt[N_APPROACH - 1]) * 100
    summary = (
        f"Grasp target (local): {action_local.numpy().round(3)}\n"
        f"Object spawn (world): {obj[0].round(3)}\n"
        f"EE at end of approach: {ee[N_APPROACH-1].round(3)}\n"
        f"Target world:          {tgt[N_APPROACH-1].round(3)}\n"
        f"Approach error (final): {final_dist:.1f} cm\n"
        f"Object lift at end:     {final_lift:.1f} cm\n"
        f"SUCCESS: {final_lift > 3.0}"
    )
    fig.text(0.01, 0.01, summary, fontsize=8, family="monospace",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[debug_ik] saved → {out_path}")
    print(f"\n{summary}")
    plt.close(fig)


def print_jacobian_diagnostics(env: GraspPoseEnv):
    """Print Jacobian values to verify IK setup."""
    J_full = env._robot.root_physx_view.get_jacobians()
    print(f"\n── Jacobian diagnostics ──")
    print(f"  J_full shape:    {tuple(J_full.shape)}")
    print(f"  robot.num_bodies:{env._robot.num_bodies}")
    print(f"  _ee_body_idx:    {env._ee_body_idx}")
    print(f"  _arm_dof_idx:    {env._arm_dof_idx}")
    print(f"  _grip_dof_idx:   {env._grip_dof_idx}")

    for offset in (0, 1):
        row = env._ee_body_idx - offset
        if 0 <= row < J_full.shape[1]:
            J_ee = J_full[0, row, :3, :]
            J_arm = J_ee[:, env._arm_dof_idx]
            col_norms = J_arm.norm(dim=0)
            print(f"  J[ee_idx-{offset}] arm-col norms: {col_norms.cpu().numpy().round(4)}")
            print(f"    row total norm: {J_arm.norm().item():.4f}")

    ee_w   = env._robot.data.body_pos_w[0, env._ee_body_idx, :3].cpu()
    root_w = env._robot.data.root_pos_w[0, :3].cpu()
    print(f"  EE world pos:  {ee_w.numpy().round(3)}")
    print(f"  Root world pos:{root_w.numpy().round(3)}")
    print(f"  EE base  pos:  {(ee_w - root_w).numpy().round(3)}")
    print(f"  Joint pos (arm): {env._robot.data.joint_pos[0, env._arm_dof_idx].cpu().numpy().round(3)}")
    print(f"────────────────────────\n")


def main():
    env_cfg = GraspPoseEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = GraspPoseEnv(cfg=env_cfg, render_mode=None)

    obs, _ = env.reset()
    print_jacobian_diagnostics(env)

    action = torch.zeros(args.num_envs, NUM_ACTIONS, device=env.device)
    print(f"[debug_ik] Testing action = {action[0].cpu().numpy()}  (object centre)")
    print(f"[debug_ik] Env device: {env.device}")
    print(f"[debug_ik] Robot root world z: {env._robot.data.root_pos_w[0, 2].item():.3f}")

    ee, tgt, obj = run_episode(env, action)

    print(f"[debug_ik] Object spawn z (world): {obj[0, 2]:.3f}")
    print(f"[debug_ik] EE start z (world):     {ee[0, 2]:.3f}")
    print(f"[debug_ik] Target z at step 0:     {tgt[0, 2]:.3f}")
    print(f"[debug_ik] EE z at end of approach (step {N_APPROACH-1}): {ee[N_APPROACH-1, 2]:.3f}")
    print(f"[debug_ik] Target z at end of approach:                  {tgt[N_APPROACH-1, 2]:.3f}")
    print(f"[debug_ik] Object lift:  {(obj[-1, 2] - obj[0, 2])*100:.1f} cm")

    plot_results(ee, tgt, obj, action[0].cpu(), args.out)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
