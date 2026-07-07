#!/usr/bin/env python3
"""
Dex3 hand characterization: find the closed (enveloping) pose.

For each of the 7 left-hand joints, perturb toward each limit and measure how the
owning fingertip moves relative to a target grasp centre (a point just in front of
the palm). Then build a 'closed' pose driving every joint toward the limit that
pulls its fingertip toward the grasp centre, and report the resulting geometry.

Usage:
    python wrappers/isaaclab/scripts/debug_hand.py --headless
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

# hand joint -> owning fingertip link
FINGER_TIP = {
    "left_zero_joint": "left_two_link",   # thumb
    "left_one_joint":  "left_two_link",
    "left_two_joint":  "left_two_link",
    "left_three_joint": "left_four_link", # index
    "left_four_joint":  "left_four_link",
    "left_five_joint":  "left_six_link",  # middle
    "left_six_joint":   "left_six_link",
}


def settle(env, q_hand, hand_idx, steps=40):
    """Drive the hand joints to q_hand and step to steady state; arm held at home."""
    for _ in range(steps):
        env._robot.set_joint_position_target(
            env._home_joint_pos[:, env._arm_dof_idx], joint_ids=env._arm_dof_idx
        )
        env._robot.set_joint_position_target(
            q_hand.unsqueeze(0), joint_ids=hand_idx
        )
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)


def main():
    env_cfg = GraspPoseEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = GraspPoseEnv(cfg=env_cfg, render_mode=None)
    env.reset()
    robot = env._robot

    hand_names = list(FINGER_TIP.keys())
    hand_idx, _ = robot.find_joints(hand_names)
    lo = robot.data.soft_joint_pos_limits[0, :, 0]
    hi = robot.data.soft_joint_pos_limits[0, :, 1]

    palm_i = env._ee_body_idx
    palm_w = robot.data.body_pos_w[0, palm_i, :3].clone()
    obj = env._get_active_obj_pos()[0].clone()
    # grasp centre: where we want the object to sit relative to the palm
    grasp_centre = obj.clone()
    print(f"\npalm={palm_w.cpu().numpy().round(3)}  object={obj.cpu().numpy().round(3)}")
    print(f"grasp_centre={grasp_centre.cpu().numpy().round(3)}")

    home = env._home_joint_pos.clone()

    # Curl direction = the limit that brings the fingertip CLOSER to the palm.
    print("\n============ PER-JOINT CURL DIRECTION (dist to palm) ============")
    closed_q = home[0, hand_idx].clone()
    for k, name in enumerate(hand_names):
        j = hand_idx[k]
        tip_i = robot.body_names.index(FINGER_TIP[name])
        dists = {}
        tip_at = {}
        for lim_name, val in [("lo", lo[j].item()), ("hi", hi[j].item())]:
            q = home.clone()
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            env.sim.forward()
            qh = home[0, hand_idx].clone()
            qh[k] = val
            settle(env, qh, hand_idx, steps=30)
            tip = robot.data.body_pos_w[0, tip_i, :3].clone()
            pw = robot.data.body_pos_w[0, palm_i, :3]
            dists[lim_name] = (tip - pw).norm().item()
            tip_at[lim_name] = tip
        best = "lo" if dists["lo"] < dists["hi"] else "hi"
        closed_q[k] = lo[j] if best == "lo" else hi[j]
        moved = (tip_at["lo"] - tip_at["hi"]).norm().item() * 100
        print(f"  {name:20s} tip={FINGER_TIP[name]:16s} "
              f"palm_d_lo={dists['lo']*100:5.1f} palm_d_hi={dists['hi']*100:5.1f}  "
              f"tip_travel={moved:5.1f}cm  -> curl={best}")

    print("\n============ CLOSED POSE GEOMETRY ============")
    robot.write_joint_state_to_sim(home.clone(), torch.zeros_like(home))
    env.sim.forward()
    settle(env, closed_q, hand_idx, steps=60)
    tips = ["left_two_link", "left_four_link", "left_six_link"]
    centroid = torch.zeros(3, device=env.device)
    for t in tips:
        p = robot.data.body_pos_w[0, robot.body_names.index(t), :3]
        centroid += p
        print(f"  {t:16s} = {p.cpu().numpy().round(3)}  dist_to_grasp={((p-grasp_centre).norm()*100).item():.1f}cm")
    centroid /= len(tips)
    print(f"  fingertip centroid = {centroid.cpu().numpy().round(3)}")
    print(f"  centroid dist to grasp_centre = {((centroid-grasp_centre).norm()*100).item():.1f}cm")
    print(f"  closed_q = {closed_q.cpu().numpy().round(3)}")
    print(f"  hand joint order = {hand_names}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
