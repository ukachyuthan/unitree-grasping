#!/usr/bin/env python3
"""
Train the G1 left-arm pick-and-place grasping policy.

Uses Isaac Lab (DirectRLEnv) + RSL-RL (PPO) + custom PointNetActorCritic.

Usage:
    ./rl_unitree/bin/python scripts/train_g1_grasp.py --headless
    ./rl_unitree/bin/python scripts/train_g1_grasp.py --headless --num_envs 256
    ./rl_unitree/bin/python scripts/train_g1_grasp.py --headless --joint_space  # skip IK

Architecture reminder:
  obs  = [point_cloud(384) | EE_pos(3) | EE_quat(4) | goal_pos(3) | goal_quat(4) | grip(1) | prev_action(7)]
  actor  = PointNet(128pt → 128d) → MLP[256,128] → 7 actions
  critic = PointNet(128pt → 128d) → MLP[256,128] → 1 value
"""

import argparse

from isaaclab.app import AppLauncher

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train G1 grasping policy")
parser.add_argument("--num_envs",   type=int,   default=512)
parser.add_argument("--max_iters",  type=int,   default=3000)
parser.add_argument("--log_dir",    type=str,   default="data/grasp_logs")
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--joint_space", action="store_true",
                    help="Use joint-space control (no IK). Simpler, good for first training run.")
parser.add_argument("--num_pc_points", type=int, default=128)
parser.add_argument("--pc_embed_dim",  type=int, default=128)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Must launch Isaac Sim before any sim-dependent imports
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Imports (inside Isaac Sim process) ───────────────────────────────────────
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from envs.g1_grasp_env_cfg import G1GraspEnvCfg, OBS_DIM, NUM_ACTIONS
from envs.g1_grasp_env import G1GraspEnv
from models.grasp_actor_critic import PointNetActorCritic


def make_runner_cfg(max_iters: int) -> dict:
    """RSL-RL OnPolicyRunner config. Uses standard ActorCritic as placeholder;
    the caller replaces runner.alg.actor_critic with PointNetActorCritic."""
    return {
        "class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "max_iterations": max_iters,
        "save_interval": 100,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",   # rsl_rl built-in; swapped out below
            "actor_hidden_dims":  [256, 128],
            "critic_hidden_dims": [256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


def main():
    # Physics runs on CPU (IOMMU incompatibility with GPU PhysX on this machine).
    # Neural network (PPO, PointNet) still runs on CUDA.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[grasp-train] device={device}  envs={args.num_envs}")

    # ── Environment ──────────────────────────────────────────────────────────
    env_cfg = G1GraspEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    # --joint_space flag forces joint-space; cfg default is already True so only override when explicitly set
    if args.joint_space:
        env_cfg.use_joint_space_control = True

    env = G1GraspEnv(cfg=env_cfg, render_mode="rgb_array" if not args.headless else None)
    env = RslRlVecEnvWrapper(env)

    # ── Runner ───────────────────────────────────────────────────────────────
    runner_cfg = make_runner_cfg(max_iters=args.max_iters)

    run_name = f"g1_grasp_envs{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir  = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[grasp-train] logging to {log_dir}")

    # Create with standard ActorCritic placeholder, then swap in PointNetActorCritic.
    # rsl_rl resolves class_name via eval() in its own module namespace, so custom
    # classes must be injected after construction.
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=device)

    pointnet_ac = PointNetActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=NUM_ACTIONS,
        num_pc_points=args.num_pc_points,
        pc_embed_dim=args.pc_embed_dim,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
    ).to(device)
    runner.alg.actor_critic = pointnet_ac
    runner.alg.optimizer = torch.optim.Adam(
        pointnet_ac.parameters(),
        lr=runner_cfg["algorithm"]["learning_rate"],
    )
    print(f"[grasp-train] PointNetActorCritic swapped in  "
          f"({sum(p.numel() for p in pointnet_ac.parameters()):,} params)")

    runner.learn(
        num_learning_iterations=runner_cfg["max_iterations"],
        init_at_random_ep_len=True,
    )

    # ── Save final policy ─────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, "g1_grasp_final.pt")
    runner.save(final_path)
    print(f"[grasp-train] saved policy to {final_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
