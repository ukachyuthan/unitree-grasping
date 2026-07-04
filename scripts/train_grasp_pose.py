#!/usr/bin/env python3
"""
RL training for grasp-pose prediction (Path A).

Policy: PointNet(PC) → 3D grasp position
Reward: purely physical — how far did the object actually lift in simulation

Pre-training (optional warm start — skippable):
    ./rl_unitree/bin/python scripts/pretrain_grasp.py
    This initialises the encoder so exploration doesn't start completely blind.
    The RL then replaces those weights through physical interaction — grasps that
    work in physics are reinforced, geometric labels are discarded entirely.

RL training:
    # Cold start (random init)
    ./rl_unitree/bin/python scripts/train_grasp_pose.py --headless

    # Warm start (faster convergence, same end-point)
    ./rl_unitree/bin/python scripts/train_grasp_pose.py --headless \
        --pretrain data/grasp_weights/grasp_pretrain_best.pt

The RL reward is only lift_height — no regression loss, no geometric supervision.
Over training the policy discovers which regions of the point cloud predict
physically successful grasps, which may differ from antipodal geometry labels.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",   type=int,   default=256)
parser.add_argument("--max_iters",  type=int,   default=5000)
parser.add_argument("--log_dir",    type=str,   default="data/grasp_logs")
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--pretrain",   type=str,   default=None,
                    help="Optional: pretrained encoder .pt for warm start only")
parser.add_argument("--num_pc_points", type=int, default=128)
parser.add_argument("--pc_embed_dim",  type=int, default=128)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from grasping.grasp_pose_env_cfg import GraspPoseEnvCfg, OBS_DIM, NUM_ACTIONS
from grasping.grasp_pose_env     import GraspPoseEnv
from models.grasp_pose_actor_critic import GraspPoseActorCritic


def make_runner_cfg(max_iters: int) -> dict:
    return {
        "class_name": "OnPolicyRunner",
        "num_steps_per_env": 16,    # collect 16 grasp attempts per env per update
        "max_iterations": max_iters,
        "save_interval": 100,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",   # placeholder; swapped below
            "actor_hidden_dims":  [128, 64],
            "critic_hidden_dims": [128, 64],
            "activation": "elu",
            "init_noise_std": 0.5,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.02,    # entropy bonus keeps exploration alive early on
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[grasp-pose-train] device={device}  envs={args.num_envs}")

    env_cfg = GraspPoseEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = GraspPoseEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = make_runner_cfg(max_iters=args.max_iters)
    run_name   = f"grasp_pose_envs{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir    = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[grasp-pose-train] logging to {log_dir}")

    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=device)

    ac = GraspPoseActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=NUM_ACTIONS,
        num_pc_points=args.num_pc_points,
        pc_embed_dim=args.pc_embed_dim,
        init_noise_std=1.0,   # wider exploration; std decays via PPO adaptive lr
    ).to(device)

    # Optional warm start — only loads encoder, not the prediction head.
    # RL then overwrites these weights through physical reward signal.
    if args.pretrain and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location=device)
        ac.actor_encoder.load_state_dict(ckpt["encoder"])
        ac.critic_encoder.load_state_dict(ckpt["encoder"])
        print(f"[grasp-pose-train] warm start: encoder loaded from {args.pretrain}")
        print("  RL will now train purely on physical reward — geometric labels discarded.")
    else:
        print("[grasp-pose-train] cold start: random initialization")

    runner.alg.actor_critic = ac
    runner.alg.optimizer    = torch.optim.Adam(ac.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in ac.parameters())
    print(f"[grasp-pose-train] GraspPoseActorCritic ({n_params:,} params)")

    runner.learn(
        num_learning_iterations=runner_cfg["max_iterations"],
        init_at_random_ep_len=False,
    )

    final_path = os.path.join(log_dir, "grasp_pose_final.pt")
    runner.save(final_path)
    print(f"[grasp-pose-train] saved → {final_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
