#!/usr/bin/env python3
"""
Evaluate / visualise the trained grasping policy.

Three modes:

  sim_primitives   — replay on the training-time box/sphere/cylinder (sanity check).

  sim_novel        — swap in held-out REAL objects (scripts/fetch_ycb.py +
                     generate_ycb_meshes.py's eval split) to prove zero-shot
                     generalisation on never-seen real geometry. Falls back to
                     sim_primitives if no eval real objects have been generated yet.

  real_camera      — read depth frames from an Intel RealSense D4xx camera,
                     run the PointNet policy, print predicted EE targets.
                     Does NOT command the robot; bridge to G1 SDK separately.

Usage:
    # 1. Verify the env works after training
    ./rl_unitree/bin/python scripts/play_g1_grasp.py \\
        --checkpoint data/grasp_logs/<run>/g1_grasp_final.pt --num_envs 4

    # 2. Zero-shot on held-out real objects (generate first):
    #    python scripts/fetch_ycb.py && python scripts/generate_ycb_meshes.py
    ./rl_unitree/bin/python scripts/play_g1_grasp.py \\
        --checkpoint ... --mode sim_novel

    # 3. Real-camera inference (requires pyrealsense2 + real camera):
    ./rl_unitree/bin/python scripts/play_g1_grasp.py \\
        --checkpoint ... --mode real_camera

Sim-to-real deployment notes:
    • Provide real camera intrinsics K (from `rs.video_stream_profile.get_intrinsics()`)
    • Provide extrinsic T_cam_robot from hand-eye calibration
    • Bridge action → G1 joint commands via Unitree G1 SDK
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play G1 grasping policy")
parser.add_argument("--checkpoint",    type=str, required=True)
parser.add_argument("--mode",          type=str,
                    choices=["sim_primitives", "sim_novel", "real_camera"],
                    default="sim_primitives")
parser.add_argument("--num_envs",      type=int, default=4)
parser.add_argument("--num_episodes",  type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.g1_grasp_env_cfg import G1GraspEnvCfg, OBS_DIM, NUM_ACTIONS
from envs.g1_grasp_env import G1GraspEnv
from envs._object_registry import ycb_shape_names
from models.grasp_actor_critic import PointNetActorCritic


def load_policy(ckpt_path: str, device: str) -> PointNetActorCritic:
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model = PointNetActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=NUM_ACTIONS,
        num_pc_points=128,
        pc_embed_dim=128,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_sim(device: str, mode: str):
    env_cfg = G1GraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    if mode == "sim_novel":
        # Held-out REAL objects (scripts/fetch_ycb.py + generate_ycb_meshes.py),
        # never seen during training if --use_real_objects was on for that run —
        # eval_object_mode swaps the env's entire object set to just these.
        eval_names = ycb_shape_names("eval")
        if eval_names:
            env_cfg.eval_object_mode = True
            preview = ", ".join(eval_names[:5]) + ("..." if len(eval_names) > 5 else "")
            print(f"[play] sim_novel: {len(eval_names)} held-out real objects ({preview})")
        else:
            print(f"[play] sim_novel: no held-out real objects found under data/objects/eval/.")
            print("       Generate first: python scripts/fetch_ycb.py && "
                  "python scripts/generate_ycb_meshes.py")
            print("       Falling back to procedural shapes (same as sim_primitives).")

    env    = G1GraspEnv(cfg=env_cfg)
    policy = load_policy(args.checkpoint, device)

    successes, total, episode = 0, 0, 0
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"].to(device)

    while episode < args.num_episodes:
        with torch.no_grad():
            action = policy.act_inference(obs)
        obs_dict, rew, terminated, truncated, _ = env.step(action)
        obs = obs_dict["policy"].to(device)
        done = terminated | truncated
        if done.any():
            n_done    = done.sum().item()
            n_success = terminated.sum().item()
            successes += n_success
            total     += n_done
            episode   += n_done
            pct = 100 * successes / max(total, 1)
            print(f"  ep {episode:3d}/{args.num_episodes}  "
                  f"success {successes}/{total} = {pct:.1f}%  "
                  f"reward {rew.mean().item():.2f}")

    print(f"\nFinal success rate: {100*successes/max(total,1):.1f}%")
    env.close()


def run_real_camera(device: str):
    """
    Real-hardware inference loop.
    Reads depth from RealSense D400-series, extracts point cloud,
    runs PointNet policy, prints EE action.
    Requires: pyrealsense2  (pip install pyrealsense2)
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("Install pyrealsense2:  pip install pyrealsense2")
        return

    from grasping.pointcloud_utils import depth_to_pointcloud_world

    policy = load_policy(args.checkpoint, device)

    pipeline = rs.pipeline()
    cfg_rs   = rs.config()
    cfg_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(cfg_rs)

    profile = pipeline.get_active_profile()
    intr    = rs.video_stream_profile(
        profile.get_stream(rs.stream.depth)
    ).get_intrinsics()

    K = torch.tensor([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0,  0,       1.0   ],
    ], dtype=torch.float32, device=device).unsqueeze(0)  # (1,3,3)

    # ── Replace these with your hand-eye calibration results ──────────────
    # cam_pos_w : world position of the camera optical centre
    # cam_quat_w: quaternion (w,x,y,z) rotating camera frame to world frame
    cam_pos_w  = torch.zeros(1, 3, device=device)
    cam_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # Dummy EE / goal (replace with real G1 SDK state queries)
    prev_action    = torch.zeros(1, NUM_ACTIONS, device=device)
    dummy_ee_pos   = torch.zeros(1, 3, device=device)
    dummy_ee_quat  = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=device)
    dummy_goal_pos = torch.tensor([[0.5, 0.3, 0.8]], device=device)
    dummy_goal_quat = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=device)
    dummy_grip     = torch.zeros(1, 1, device=device)

    print("RealSense connected. Ctrl-C to stop.")
    try:
        while True:
            frames    = pipeline.wait_for_frames()
            depth_np  = np.asanyarray(
                frames.get_depth_frame().get_data()
            ).astype(np.float32) * 0.001   # mm → m

            depth_t = torch.from_numpy(depth_np).unsqueeze(0).to(device)  # (1,H,W)
            pts_w   = depth_to_pointcloud_world(depth_t, K, cam_pos_w, cam_quat_w)

            # Filter & sample 128 pts
            above  = pts_w[0, :, 2] > 0.05
            valid  = pts_w[0][above]
            M = valid.shape[0]
            if M < 10:
                print("  [warn] too few valid points")
                continue
            idx = (torch.randperm(M)[:128] if M >= 128
                   else torch.randint(M, (128,)))
            pc_rob = valid[idx].reshape(1, -1).to(device)   # (1, 384)

            proprio = torch.cat([
                dummy_ee_pos, dummy_ee_quat,
                dummy_goal_pos, dummy_goal_quat,
                dummy_grip, prev_action,
            ], dim=-1)

            obs = torch.cat([pc_rob, proprio], dim=-1)
            with torch.no_grad():
                action = policy.act_inference(obs)

            grip = "CLOSE" if action[0, -1] > 0 else "OPEN"
            print(f"  Δpos={action[0,:3].cpu().numpy().round(3)}  "
                  f"Δrot={action[0,3:6].cpu().numpy().round(3)}  {grip}")
            prev_action = action.detach()

    except KeyboardInterrupt:
        pipeline.stop()
        print("Stopped.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode in ("sim_primitives", "sim_novel"):
        run_sim(device, args.mode)
    elif args.mode == "real_camera":
        run_real_camera(device)


if __name__ == "__main__":
    main()
    simulation_app.close()
