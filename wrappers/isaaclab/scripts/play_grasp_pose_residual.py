#!/usr/bin/env python3
"""Evaluate / record video for residual grasp policy."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play residual grasp-pose policy")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_episodes", type=int, default=3)
parser.add_argument("--out", type=str, default="data/viz/grasp_residual_rollout.mp4")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_pc_points", type=int, default=128)
parser.add_argument("--pc_embed_dim", type=int, default=128)
parser.add_argument("--cam_eye", type=float, nargs=3, default=[0.85, -0.35, 0.45])
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.50, 0.0, 0.12])
parser.add_argument("--visualize_grasp", action="store_true")
parser.add_argument("--cycle_shapes", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.grasp_pose_residual_env_cfg import OBS_DIM, NUM_ACTIONS, POLICY_STEPS_PER_EPISODE
from envs.grasp_pose_residual_env import GraspPoseResidualEnv, GraspPoseResidualEnvCfg
from envs.grasp_pose_env import _SHAPE_NAMES
from models.grasp_pose_residual_actor_critic import GraspPoseResidualActorCritic


def load_policy(ckpt_path: str, device: str) -> GraspPoseResidualActorCritic:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model = GraspPoseResidualActorCritic(
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


def rollout_episode(env, policy, device, capture_video: bool) -> tuple[float, list]:
    """Run one full episode (~POLICY_STEPS_PER_EPISODE policy steps)."""
    u = env.unwrapped
    obs_dict, _ = env.reset()
    total_rew = 0.0
    frames: list = []
    capture = capture_video and u.render_mode == "rgb_array"

    for _ in range(POLICY_STEPS_PER_EPISODE + 2):
        obs = _obs_tensor(obs_dict, device)
        with torch.no_grad():
            action = policy.act_inference(obs)

        if capture:
            u._pre_physics_step(action)
            is_rendering = u.sim.has_gui() or u.sim.has_rtx_sensors()
            for _ in range(u.cfg.decimation):
                u._apply_action()
                u.scene.write_data_to_sim()
                u.sim.step(render=False)
                if is_rendering:
                    u.sim.render()
                    frame = u.render(recompute=True)
                    if frame is not None and frame.size > 0 and frame.any():
                        frames.append(frame)
                u.scene.update(dt=u.physics_dt)
            rew = u._get_rewards()
            term, _ = u._get_dones()
            if term.any():
                u._reset_idx(torch.arange(u.num_envs, device=u.device))
            obs_dict = u._get_observations()
        else:
            obs_dict, rew, terminated, _, _ = env.step(action)
            term = terminated

        total_rew += rew.mean().item()
        if term.all():
            break

    # Terminal sparse reward is the meaningful metric.
    terminal = u.terminal_lift_reward.mean().item()
    return terminal, frames


def write_mp4(frames: list, path: str, fps: int = 15):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    writer = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[play-residual] saved video → {path}  ({len(frames)} frames)")


def main():
    device = args.device
    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        print(f"[play-residual] checkpoint not found: {ckpt}")
        sys.exit(1)

    render_mode = "rgb_array" if args.video else None
    env_cfg = GraspPoseResidualEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    env_cfg.visualize_grasp_point = args.visualize_grasp or args.video or not args.headless
    if args.cycle_shapes:
        env_cfg.eval_cycle_shapes = True
        env_cfg.eval_num_shapes = 10
    if args.video:
        env_cfg.sim.render_interval = 1

    env = GraspPoseResidualEnv(cfg=env_cfg, render_mode=render_mode)
    if args.seed is not None:
        env.seed(args.seed)

    if args.video:
        env.unwrapped.sim.set_camera_view(eye=args.cam_eye, target=args.cam_target)

    policy = load_policy(ckpt, device)
    print(f"[play-residual] loaded {ckpt}")
    print(f"[play-residual] device={device}  envs={args.num_envs}  episodes={args.num_episodes}")

    successes, total_lift = 0, 0.0
    all_frames: list = []

    for ep in range(1, args.num_episodes + 1):
        capture = args.video and ep <= args.video_episodes
        lift_r, frames = rollout_episode(env, policy, device, capture_video=capture)
        total_lift += lift_r
        if lift_r >= 0.5:
            successes += 1
        if capture:
            all_frames.extend(frames)
        shape = _SHAPE_NAMES[env.unwrapped._env_shape[0].item()]
        print(f"  ep {ep:3d}/{args.num_episodes}  shape={shape:14s}  lift_r={lift_r:.3f}  ok={lift_r >= 0.5}")

    n = args.num_episodes
    print(f"\n[play-residual] mean_lift={total_lift/n:.3f}  success_rate={100*successes/n:.1f}%")

    if args.video and all_frames:
        write_mp4(all_frames, os.path.abspath(args.out))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
