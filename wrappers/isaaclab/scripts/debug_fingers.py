#!/usr/bin/env python3
"""Check whether Dex3 left-hand joints actually track position targets.

For each of the 7 left-hand joints we command it to each soft limit, step long
enough to reach steady state, and print the commanded target vs the REACHED
joint angle. If reached != target, the joint lacks drive authority (effort/
stiffness) or is otherwise constrained.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
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

HAND = [
    "left_zero_joint", "left_one_joint", "left_two_joint",
    "left_three_joint", "left_four_joint", "left_five_joint", "left_six_joint",
]


def main():
    cfg = GraspPoseEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = GraspPoseEnv(cfg=cfg, render_mode=None)
    env.reset()
    robot = env._robot

    hand_idx, _ = robot.find_joints(HAND)
    lo = robot.data.soft_joint_pos_limits[0, :, 0]
    hi = robot.data.soft_joint_pos_limits[0, :, 1]
    home = env._home_joint_pos.clone()

    # Report the actuator each hand joint belongs to and its gains.
    print("\n============ HAND JOINT LIMITS ============")
    for k, name in enumerate(HAND):
        j = hand_idx[k]
        print(f"  {name:20s} idx={j:3d}  lo={lo[j].item():+.3f}  hi={hi[j].item():+.3f}  "
              f"home={home[0, j].item():+.3f}")

    print(f"\n============ TARGET TRACKING ({args.steps} steps) ============")
    for k, name in enumerate(HAND):
        j = hand_idx[k]
        for lim_name, val in [("lo", lo[j].item()), ("hi", hi[j].item())]:
            robot.write_joint_state_to_sim(home.clone(), torch.zeros_like(home))
            env.sim.forward()
            qh = home[0, hand_idx].clone()
            qh[k] = val
            for _ in range(args.steps):
                robot.set_joint_position_target(
                    home[:, env._arm_dof_idx], joint_ids=env._arm_dof_idx
                )
                robot.set_joint_position_target(qh.unsqueeze(0), joint_ids=hand_idx)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
            reached = robot.data.joint_pos[0, j].item()
            err = abs(reached - val)
            flag = "  <-- NOT TRACKING" if err > 0.1 else ""
            print(f"  {name:20s} {lim_name} target={val:+.3f} reached={reached:+.3f} "
                  f"err={err:.3f}{flag}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
