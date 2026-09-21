#!/usr/bin/env python3
"""
Minimal control test: two fixed contact points in object-local space → reach → pinch → lift.

No policy, no PC projection, no action decoding. Orange/cyan markers = the two targets.

Usage (from repo root, conda env unitree_isaaclab):
    python scripts/debug_reach_pinch.py --headless
    python scripts/debug_reach_pinch.py --headless --enable_cameras --video \\
        --out data/viz/debug_reach_pinch.mp4
    python scripts/debug_reach_pinch.py --shape star_prism --c1 -0.04 0 0 --c2 0.04 0 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed two-point reach + pinch test")
parser.add_argument("--shape", type=str, default="star_prism",
                    help="Procedural object name (default: star_prism)")
parser.add_argument("--c1", type=float, nargs=3, default=[-0.040, 0.0, 0.020],
                    help="Contact 1 in object-local metres (x y z)")
parser.add_argument("--c2", type=float, nargs=3, default=[+0.040, 0.0, 0.020],
                    help="Contact 2 in object-local metres (x y z)")
parser.add_argument("--video", action="store_true")
parser.add_argument("--out", type=str, default="data/viz/debug_reach_pinch.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.grasp_pose_env import GraspPoseEnv
from envs.grasp_pose_env_cfg import GraspPoseEnvCfg, N_APPROACH, N_CLOSE, N_DESCEND, N_LIFT
from envs._object_registry import PROCEDURAL_SHAPE_NAMES


def rollout_episode(env: GraspPoseEnv, action: torch.Tensor):
    """Run one scripted episode; return reward tensor."""
    u = env.unwrapped
    u._pre_physics_step(action)
    for _ in range(u.cfg.decimation):
        u._apply_action()
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
    return u._get_rewards()


def rollout_dense_frames(env: GraspPoseEnv, action: torch.Tensor):
    u = env.unwrapped
    u._pre_physics_step(action)
    capture = u.render_mode == "rgb_array"
    is_rendering = capture or u.sim.has_gui() or u.sim.has_rtx_sensors()
    frames = []
    for _ in range(u.cfg.decimation):
        u._apply_action()
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        if is_rendering:
            u.sim.render()
            if capture:
                frame = u.render(recompute=True)
                if frame is not None and frame.size > 0 and frame.any():
                    frames.append(frame)
        u.scene.update(dt=u.physics_dt)
    return frames, u._get_rewards()


def write_mp4(frames: list, path: str, fps: int = 30):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[reach-pinch] saved video → {path}  ({len(frames)} frames)")


def main():
    if args.shape not in PROCEDURAL_SHAPE_NAMES:
        print(f"[reach-pinch] unknown shape {args.shape!r}; "
              f"pick one of: {PROCEDURAL_SHAPE_NAMES[:10]}")
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = GraspPoseEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.two_point_grasp = True
    cfg.use_fixed_grasp_contacts = True
    cfg.fixed_contact_1 = tuple(args.c1)
    cfg.fixed_contact_2 = tuple(args.c2)
    cfg.project_grasp_to_pc = False
    cfg.ik_write_joint_state = False
    cfg.randomize_object_spawn = False
    cfg.eval_cycle_shapes = True
    cfg.eval_num_shapes = len(PROCEDURAL_SHAPE_NAMES)
    cfg.visualize_grasp_point = True
    cfg.visualize_object_pc = False
    cfg.use_real_objects = False  # procedural only — one known object
    cfg.use_camera_pc = False

    cfg.two_point_approach_clearance_m = 0.0

    shape_idx = PROCEDURAL_SHAPE_NAMES.index(args.shape)
    render_mode = "rgb_array" if args.video else None
    env = GraspPoseEnv(cfg=cfg, render_mode=render_mode)
    env.seed(42)
    env.unwrapped._eval_shape_step = shape_idx

    c1 = torch.tensor(args.c1, device=device)
    c2 = torch.tensor(args.c2, device=device)
    width = (c2 - c1).norm().item()
    print("[reach-pinch] FIXED contacts (object-local metres):")
    print(f"  shape = {args.shape}")
    print(f"  c1    = ({args.c1[0]:+.3f}, {args.c1[1]:+.3f}, {args.c1[2]:+.3f})")
    print(f"  c2    = ({args.c2[0]:+.3f}, {args.c2[1]:+.3f}, {args.c2[2]:+.3f})")
    print(f"  width = {width:.3f} m")
    print("[reach-pinch] phases: approach → pinch (arm frozen) → lift")
    print(f"  steps: approach={N_APPROACH}  close={N_CLOSE}  descend={N_DESCEND}  lift={N_LIFT}")
    print("[reach-pinch] tip: set c1/c2 on the object surface (object-local metres).")
    print("  e.g. star_prism: --c1 -0.04 0 0.02 --c2 0.04 0 0.02")

    dummy_action = torch.zeros(1, 6, device=device)
    env.reset()
    if args.video:
        env.unwrapped.sim.set_camera_view(eye=[0.85, -0.35, 0.45], target=[0.50, 0.0, 0.12])
        frames, reward = rollout_dense_frames(env, dummy_action)
        write_mp4(frames, os.path.abspath(args.out))
    else:
        reward = rollout_episode(env, dummy_action)

    u = env.unwrapped
    c1w, c2w = u._predicted_contacts_w(for_viz=False)
    print(f"[reach-pinch] world targets: c1={c1w[0].cpu().numpy().round(3)}  "
          f"c2={c2w[0].cpu().numpy().round(3)}")
    fe = float(u._ik_finger_err[0].item())
    le = float(u._ik_left_contact_err[0].item())
    re_ = float(u._ik_right_contact_err[0].item())
    lec = float(u._ik_left_contact_err_closed[0].item())
    rec = float(u._ik_right_contact_err_closed[0].item())
    obj_z0 = float(u._spawn_z[0].item())
    obj_z1 = float(u._get_active_obj_pos()[0, 2].item())
    lift_dz = obj_z1 - obj_z0
    contact = bool(u._lift_had_contact[0].item())
    lift_ok = float(reward[0].item()) >= 0.5 if args.video else False
    if not args.video:
        lift_ok = lift_dz >= cfg.lift_threshold_m and (
            not cfg.require_contact_for_lift_reward or contact
        )

    print("[reach-pinch] results:")
    print(f"  finger_mid err @ pinch start : {fe*100:.1f} cm")
    print(f"  L/R finger → contact @ pinch : {le*100:.1f} / {re_*100:.1f} cm")
    print(f"  L/R finger → contact @ closed : {lec*100:.1f} / {rec*100:.1f} cm")
    print(f"  had_contact={contact}  obj_lift={lift_dz*100:.1f} cm  lift_ok={lift_ok}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
