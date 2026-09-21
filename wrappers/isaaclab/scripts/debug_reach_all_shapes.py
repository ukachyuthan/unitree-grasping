#!/usr/bin/env python3
"""
Fixed-contact reach smoke test across procedural shapes (no policy).

Reports finger-mid and per-finger errors at end of approach (gripper still open).
With an open parallel jaw (~80 mm) vs ~40 mm contacts, L/R body→contact cannot
reach <5 mm even with perfect mid — so PASS is mid < 15 mm and jaw_align > 0.7.

Usage (repo root, conda unitree_isaaclab):
    python -u scripts/debug_reach_all_shapes.py --headless
    python -u scripts/debug_reach_all_shapes.py --headless --ik_write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed-contact reach test all shapes")
parser.add_argument("--ik_write", action="store_true",
                    help="Kinematic write joints during approach (prove IK vs PD)")
parser.add_argument("--num_shapes", type=int, default=10)
parser.add_argument("--pinch_width", type=float, default=0.04,
                    help="Total pinch width for fixed contacts along object +Y (m)")
parser.add_argument("--mid_thresh_m", type=float, default=0.015,
                    help="PASS if finger-mid error below this (open-gripper)")
parser.add_argument("--jaw_thresh", type=float, default=0.70,
                    help="PASS if |finger·jaw| alignment above this")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.grasp_pose_env import GraspPoseEnv
from envs.grasp_pose_env_cfg import GraspPoseEnvCfg
from envs._object_registry import PROCEDURAL_SHAPE_NAMES


def rollout(env: GraspPoseEnv, action: torch.Tensor):
    u = env.unwrapped
    u._pre_physics_step(action)
    for _ in range(u.cfg.decimation):
        u._apply_action()
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
    return u


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    shapes = PROCEDURAL_SHAPE_NAMES[: args.num_shapes]
    half = 0.5 * args.pinch_width

    cfg = GraspPoseEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.two_point_grasp = True
    cfg.use_fixed_grasp_contacts = True
    cfg.fixed_contact_1 = (-half, 0.0, 0.02)
    cfg.fixed_contact_2 = (+half, 0.0, 0.02)
    cfg.project_grasp_to_pc = True          # ray-snap fixed contacts onto surface
    cfg.project_grasp_ray_down = True
    cfg.use_diff_ik = True
    cfg.pin_wrist_during_ik = False
    cfg.ik_write_joint_state = bool(args.ik_write)
    cfg.ik_write_approach_only = True
    cfg.randomize_object_spawn = False
    cfg.eval_cycle_shapes = True
    cfg.eval_num_shapes = len(PROCEDURAL_SHAPE_NAMES)
    cfg.use_real_objects = False
    cfg.use_camera_pc = False
    cfg.visualize_grasp_point = False
    cfg.two_point_approach_clearance_m = 0.0

    print(f"[reach-all] use_diff_ik={cfg.use_diff_ik}  pin_wrist={cfg.pin_wrist_during_ik}  "
          f"ik_write={cfg.ik_write_joint_state}  half_width={half:.3f}m  "
          f"PASS=mid<{args.mid_thresh_m*1000:.0f}mm & jaw>{args.jaw_thresh}")

    env = GraspPoseEnv(cfg=cfg, render_mode=None)
    env.seed(42)
    dummy = torch.zeros(1, 6, device=device)

    rows = []
    for i, name in enumerate(shapes):
        env.unwrapped._eval_shape_step = PROCEDURAL_SHAPE_NAMES.index(name)
        env.reset()
        # Re-apply fixed contacts each episode (reset clears them via pre_physics).
        env.unwrapped.cfg.fixed_contact_1 = (-half, 0.0, 0.02)
        env.unwrapped.cfg.fixed_contact_2 = (+half, 0.0, 0.02)
        u = rollout(env, dummy)
        le = float(u._ik_left_contact_err[0].item())
        re_ = float(u._ik_right_contact_err[0].item())
        fe = float(u._ik_finger_err[0].item())
        ja = float(u._ik_jaw_align[0].item())
        ok = (fe < args.mid_thresh_m) and (ja > args.jaw_thresh)
        rows.append((name, le, re_, fe, ja, ok))
        print(
            f"  {name:14s}  L={le*1000:5.1f}mm  R={re_*1000:5.1f}mm  "
            f"mid={fe*1000:5.1f}mm  jaw={ja:.2f}  {'OK' if ok else 'FAIL'}"
        )

    n_ok = sum(1 for *_, ok in rows if ok)
    mean_mid = sum(r[3] for r in rows) / len(rows)
    print(f"[reach-all] {n_ok}/{len(rows)} shapes PASS  mean_mid={mean_mid*1000:.1f}mm")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
