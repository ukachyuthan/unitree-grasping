#!/usr/bin/env python3
"""
Run a trained G1 walking policy headlessly and save a joint trajectory.

The trajectory is saved as an npz file which can be replayed in MuJoCo via
scripts/visualize_g1.py — no GPU renderer required.

Usage:
    # Use latest checkpoint from most recent training run
    ./rl_unitree/bin/python scripts/play_g1.py --headless

    # Specify a run directory
    ./rl_unitree/bin/python scripts/play_g1.py --headless --run_dir data/isaaclab_logs/flat-g1-v0_envs512_20250510_123456

    # Specify a checkpoint file directly
    ./rl_unitree/bin/python scripts/play_g1.py --headless --checkpoint data/isaaclab_logs/.../g1_flat_final.pt

    # Override number of rollout steps (default 500)
    ./rl_unitree/bin/python scripts/play_g1.py --headless --steps 1000
"""
import argparse
import os
import glob

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run G1 policy headlessly and save trajectory")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v0",
                    choices=["Isaac-Velocity-Flat-G1-v0", "Isaac-Velocity-Rough-G1-v0"])
parser.add_argument("--run_dir", type=str, default=None,
                    help="Training run directory (uses latest if not set)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a specific .pt checkpoint file")
parser.add_argument("--log_dir", type=str, default="data/isaaclab_logs")
parser.add_argument("--steps", type=int, default=500,
                    help="Number of policy steps to record (at 50 Hz → 10 seconds)")
parser.add_argument("--out", type=str, default=None,
                    help="Output npz path (default: <log_dir>/trajectory.npz)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Pure headless — never set enable_cameras; any viewport creation triggers Hydra
# RTX which segfaults on 8 GB VRAM (Isaac Sim 4.5 allocates 7.5 GB TLAS).
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Everything below runs inside the Isaac Sim process ────────────────────────
import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: registers all Isaac-* envs
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from rsl_rl.runners import OnPolicyRunner


def find_checkpoint(log_dir, run_dir, checkpoint):
    if checkpoint:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    search_dir = run_dir or log_dir
    if not os.path.exists(search_dir):
        raise FileNotFoundError(
            f"No training logs found at '{search_dir}'.\n"
            "Run training first: ./rl_unitree/bin/python scripts/train_isaaclab_g1.py --headless"
        )

    candidates = glob.glob(os.path.join(search_dir, "**", "g1_flat_final.pt"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(search_dir, "**", "model_*.pt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint (.pt) found under '{search_dir}'")

    chosen = max(candidates, key=os.path.getmtime)
    print(f"[play] Using checkpoint: {chosen}")
    return chosen


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = find_checkpoint(args.log_dir, args.run_dir, args.checkpoint)

    # Play variant: no domain randomisation, no external pushes
    play_task = args.task.replace("-v0", "-Play-v0")
    print(f"[play] Task: {play_task}  |  Steps: {args.steps}  |  Device: {device}")

    env_cfg = parse_env_cfg(play_task, device=device, num_envs=1)
    env = gym.make(play_task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    # Load PPO runner config
    train_task = args.task
    rsl_rl_cfg_entry = gym.spec(train_task).kwargs["rsl_rl_cfg_entry_point"]
    module_name, class_name = rsl_rl_cfg_entry.rsplit(":", 1)
    import importlib
    cfg_module = importlib.import_module(module_name)
    runner_cfg = getattr(cfg_module, class_name)()

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)

    # Grab handles to robot articulation for state extraction
    isaac_env = env.env.unwrapped
    robot = isaac_env.scene["robot"]

    joint_names = list(robot.joint_names)
    num_joints = len(joint_names)
    print(f"[play] Robot has {num_joints} joints: {joint_names[:6]} ...")

    # Pre-allocate trajectory buffers
    joint_pos_traj = np.zeros((args.steps, num_joints), dtype=np.float32)
    root_pos_traj  = np.zeros((args.steps, 3),          dtype=np.float32)
    root_quat_traj = np.zeros((args.steps, 4),          dtype=np.float32)  # wxyz

    obs, _ = env.get_observations()

    print(f"[play] Rolling out {args.steps} steps ...")
    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        # Extract state for env 0 (we only spawned 1 env)
        joint_pos_traj[step] = robot.data.joint_pos[0].cpu().numpy()
        root_pos_traj[step]  = robot.data.root_pos_w[0].cpu().numpy()
        root_quat_traj[step] = robot.data.root_quat_w[0].cpu().numpy()

        if step % 100 == 0:
            print(f"[play] Step {step}/{args.steps}")

    out_path = args.out or os.path.join(args.log_dir, "trajectory.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        joint_names=np.array(joint_names),
        joint_pos=joint_pos_traj,
        root_pos=root_pos_traj,
        root_quat=root_quat_traj,
    )
    print(f"[play] Trajectory saved to {out_path}")
    print(f"[play] Visualize with:")
    print(f"  ./rl_unitree/bin/python scripts/visualize_g1.py --traj {out_path}")
    print(f"  ./rl_unitree/bin/python scripts/visualize_g1.py --traj {out_path} --video")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
