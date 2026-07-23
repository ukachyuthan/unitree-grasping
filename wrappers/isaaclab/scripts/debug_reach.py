#!/usr/bin/env python3
"""
Reach & gripper diagnostic:
  - dump all body names + all joint names/limits
  - report palm home position (world & base frame)
  - sweep the shoulder/elbow to find the max reachable palm z at the object x
  - identify gripper finger bodies for contact sensing

Usage:
    python wrappers/isaaclab/scripts/debug_reach.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.grasp_pose_env_cfg import GraspPoseEnvCfg
from envs.grasp_pose_env import GraspPoseEnv


def main():
    env_cfg = GraspPoseEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = GraspPoseEnv(cfg=env_cfg, render_mode=None)
    env.reset()

    robot = env._robot

    print("\n==================== BODIES ====================")
    for i, name in enumerate(robot.body_names):
        print(f"  [{i:2d}] {name}")

    print("\n==================== JOINTS ====================")
    lo = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
    hi = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
    cur = robot.data.joint_pos[0].cpu().numpy()
    for i, name in enumerate(robot.joint_names):
        tag = ""
        if i in env._arm_dof_idx:
            tag = "  <-- ARM"
        elif i in env._grip_dof_idx:
            tag = "  <-- GRIP"
        print(f"  [{i:2d}] {name:32s} cur={cur[i]:+.3f}  limits=[{lo[i]:+.3f}, {hi[i]:+.3f}]{tag}")

    root_w = robot.data.root_pos_w[0, :3].cpu()
    ee_w = robot.data.body_pos_w[0, env._ee_body_idx, :3].cpu()
    print("\n==================== EE HOME ====================")
    print(f"  ee body: {robot.body_names[env._ee_body_idx]} (idx {env._ee_body_idx})")
    print(f"  root world: {root_w.numpy().round(3)}")
    print(f"  palm world: {ee_w.numpy().round(3)}")
    print(f"  palm base : {(ee_w - root_w).numpy().round(3)}")

    obj = env._get_active_obj_pos()[0].cpu()
    print(f"  object world: {obj.numpy().round(3)}")
    print(f"  palm-to-object dz: {(obj[2] - ee_w[2]).item()*100:.1f} cm")

    # gripper finger bodies (heuristic: names containing the gripper joint stems)
    print("\n================ GRIPPER BODIES ================")
    for i, name in enumerate(robot.body_names):
        low = name.lower()
        if any(k in low for k in ["one", "two", "palm", "hand", "finger", "rubber", "wrist"]):
            if "left" in low or "palm" in low:
                bp = robot.data.body_pos_w[0, i, :3].cpu()
                print(f"  [{i:2d}] {name:28s} world={bp.numpy().round(3)}")

    # ---- Gripper sweep: find joint values that close the two fingers ----
    print("\n================ GRIPPER SWEEP (finger gap) ================")
    one_idx = env._grip_dof_idx[0]
    two_idx = env._grip_dof_idx[1]
    one_link = robot.body_names.index("left_one_link")
    two_link = robot.body_names.index("left_two_link")
    palm_link = env._ee_body_idx
    print(f"  left_one_joint idx={one_idx}  left_two_joint idx={two_idx}")
    print(f"  sweeping both grip joints; reporting |one_link - two_link| and gap to palm")
    one_lo, one_hi = -0.89, 1.09
    two_lo, two_hi = 0.092, 1.748
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # frac=0 -> "open end", frac=1 -> "closed end" (try one->lo, two->hi)
        q_one = one_hi + frac * (one_lo - one_hi)   # 1.09 -> -0.89
        q_two = two_lo + frac * (two_hi - two_lo)   # 0.092 -> 1.748
        env._robot.write_joint_state_to_sim(
            env._home_joint_pos.clone(), torch.zeros_like(env._home_joint_pos)
        )
        env.sim.forward()
        q = env._robot.data.joint_pos.clone()
        q[:, one_idx] = q_one
        q[:, two_idx] = q_two
        env._robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        for _ in range(30):
            env._robot.set_joint_position_target(
                torch.tensor([[q_one, q_two]], device=env.device),
                joint_ids=env._grip_dof_idx,
            )
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        p_one = env._robot.data.body_pos_w[0, one_link, :3].cpu()
        p_two = env._robot.data.body_pos_w[0, two_link, :3].cpu()
        p_palm = env._robot.data.body_pos_w[0, palm_link, :3].cpu()
        gap = (p_one - p_two).norm().item() * 100
        print(f"  frac={frac:.2f}  one={q_one:+.3f} two={q_two:+.3f}  "
              f"finger_gap={gap:5.1f}cm  one_link={p_one.numpy().round(3)} two_link={p_two.numpy().round(3)}")

    # ---- Reach sweep: raise palm by driving arm IK to increasing z targets ----
    print("\n================ REACH SWEEP (IK to +z) ================")
    obj_xy = obj[:2].clone()
    for target_z in [0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]:
        # reset to home first
        env._robot.write_joint_state_to_sim(
            env._home_joint_pos.clone(), torch.zeros_like(env._home_joint_pos)
        )
        env.sim.forward()
        tgt = torch.tensor([[obj_xy[0], obj_xy[1], target_z]], device=env.device)
        # run several IK steps toward this fixed target
        for _ in range(120):
            env._ik_to(tgt)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        reached = env._robot.data.body_pos_w[0, env._ee_body_idx, :3].cpu()
        err = (tgt[0].cpu() - reached).norm().item() * 100
        print(f"  target_z={target_z:.2f}  reached={reached.numpy().round(3)}  err={err:5.1f} cm")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
