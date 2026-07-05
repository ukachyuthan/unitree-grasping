#!/usr/bin/env python3
"""
Evaluate a trained grasp-pose policy and optionally record MP4 rollouts.

Usage:
    # Live viewport (recommended on 8 GB GPU — use 1 env)
    python wrappers/isaaclab/scripts/play_grasp_pose.py \\
        --checkpoint data/grasp_logs/grasp_pose_envs32_*/grasp_pose_final.pt \\
        --num_envs 1

    # Headless MP4 (approach → close → lift, ~80 frames per episode)
    python wrappers/isaaclab/scripts/play_grasp_pose.py --headless --enable_cameras \\
        --checkpoint data/grasp_logs/.../grasp_pose_final.pt \\
        --video --video_episodes 5 --num_envs 1 \\
        --out data/viz/grasp_pose_rollout.mp4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play grasp-pose RL policy")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=20,
                    help="Episodes to evaluate (stats printed)")
parser.add_argument("--video", action="store_true",
                    help="Record dense MP4 of grasp execution")
parser.add_argument("--video_episodes", type=int, default=3,
                    help="Episodes included in the MP4 when --video is set")
parser.add_argument("--out", type=str, default="data/viz/grasp_pose_rollout.mp4")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_pc_points", type=int, default=128)
parser.add_argument("--pc_embed_dim", type=int, default=128)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.grasp_pose_env_cfg import GraspPoseEnvCfg, OBS_DIM, NUM_ACTIONS, EXEC_STEPS
from envs.grasp_pose_env import GraspPoseEnv
from models.grasp_pose_actor_critic import GraspPoseActorCritic


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


def rollout_dense_frames(env: GraspPoseEnv, action: torch.Tensor) -> list:
    """Step physics manually and capture an rgb frame each sub-step."""
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

    rew = u._get_rewards()
    done_ids = torch.arange(u.num_envs, device=u.device)
    u._reset_idx(done_ids)
    return frames, rew


def write_mp4(frames: list, path: str, fps: int = 30):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    writer = imageio.get_writer(
        path, fps=fps, codec="libx264",
        pixelformat="yuv420p", quality=8,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[play] saved video → {path}  ({len(frames)} frames)")


def main():
    device = args.device
    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        print(f"[play] checkpoint not found: {ckpt}")
        sys.exit(1)

    render_mode = "rgb_array" if args.video else None
    env_cfg = GraspPoseEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    if args.video:
        env_cfg.sim.render_interval = 1

    env = GraspPoseEnv(cfg=env_cfg, render_mode=render_mode)
    if args.seed is not None:
        env.seed(args.seed)

    policy = load_policy(ckpt, device)
    print(f"[play] loaded {ckpt}")
    print(f"[play] device={device}  envs={args.num_envs}  episodes={args.num_episodes}")

    obs_dict, _ = env.reset()
    successes, total_reward = 0, 0.0
    video_frames: list = []
    threshold = env_cfg.lift_threshold_m / env_cfg.lift_target_m

    for ep in range(1, args.num_episodes + 1):
        obs = _obs_tensor(obs_dict, device)
        with torch.no_grad():
            action = policy.act_inference(obs)

        if args.video and ep <= args.video_episodes:
            frames, rew = rollout_dense_frames(env, action)
            video_frames.extend(frames)
            obs_dict = env._get_observations()
        else:
            obs_dict, rew, _, _, _ = env.step(action)

        r = rew.mean().item()
        total_reward += r
        if r >= threshold:
            successes += 1
        print(f"  ep {ep:3d}/{args.num_episodes}  reward={r:.3f}  "
              f"lift_ok={r >= threshold}")

    n = args.num_episodes
    print(f"\n[play] mean_reward={total_reward/n:.3f}  "
          f"success_rate={100*successes/n:.1f}%  (reward ≥ {threshold:.2f})")

    if args.video and video_frames:
        write_mp4(video_frames, os.path.abspath(args.out))
    elif args.video:
        print("[play] WARNING: no frames captured — try without --headless or check --enable_cameras")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
