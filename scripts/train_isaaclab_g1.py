#!/usr/bin/env python3
"""
Train G1 flat-terrain walking policy using Isaac Lab + PPO (rsl_rl).

Usage:
    ./rl_unitree/bin/python scripts/train_isaaclab_g1.py --headless
    ./rl_unitree/bin/python scripts/train_isaaclab_g1.py --headless --num_envs 64
    ./rl_unitree/bin/python scripts/train_isaaclab_g1.py --task Isaac-Velocity-Rough-G1-v0 --headless

Isaac Lab requires AppLauncher to start Isaac Sim before any sim-dependent imports.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 walking in Isaac Lab")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v0",
                    choices=["Isaac-Velocity-Flat-G1-v0", "Isaac-Velocity-Rough-G1-v0"],
                    help="Isaac Lab task ID")
parser.add_argument("--num_envs", type=int, default=64,
                    help="Number of parallel envs (64 recommended for 8GB VRAM)")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Override max training iterations from config")
parser.add_argument("--log_dir", type=str, default="data/isaaclab_logs",
                    help="Directory for checkpoints and logs")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim — must happen before any omni/carb/isaaclab sim imports
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Everything below runs inside the Isaac Sim process ──────────────────────
import os
import gymnasium as gym
import torch

import isaaclab_tasks  # registers all Isaac-* gym envs
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from rsl_rl.runners import OnPolicyRunner


def main():
    # ── Environment ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Task: {args.task}")
    print(f"[train] Envs: {args.num_envs}  |  Device: {device}")

    env_cfg = parse_env_cfg(
        args.task,
        device=device,
        num_envs=args.num_envs,
    )
    env = gym.make(args.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    # ── PPO runner config (from the registered task) ─────────────────────────
    rsl_rl_cfg_entry = gym.spec(args.task).kwargs["rsl_rl_cfg_entry_point"]
    module_name, class_name = rsl_rl_cfg_entry.rsplit(":", 1)
    import importlib
    cfg_module = importlib.import_module(module_name)
    runner_cfg = getattr(cfg_module, class_name)()

    if args.max_iterations is not None:
        runner_cfg.max_iterations = args.max_iterations

    os.makedirs(args.log_dir, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    runner = OnPolicyRunner(
        env,
        runner_cfg.to_dict(),
        log_dir=args.log_dir,
        device=device,
    )
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    # ── Save final policy ─────────────────────────────────────────────────────
    final_path = os.path.join(args.log_dir, f"g1_flat_final.pt")
    runner.save(final_path)
    print(f"[train] Saved final policy to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
