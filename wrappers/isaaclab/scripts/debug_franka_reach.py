#!/usr/bin/env python3
"""Quick Franka workspace check: EE home position vs object spawn."""
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
    cfg = GraspPoseEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = GraspPoseEnv(cfg=cfg, render_mode=None)
    env.reset()
    ee = env._robot.data.body_pos_w[0, env._ee_body_idx, :3]
    obj = env._get_active_obj_pos()[0]
    print(f"EE home   = {ee.cpu().numpy().round(3)}")
    print(f"Object    = {obj.cpu().numpy().round(3)}")
    print(f"dist (cm) = {(obj-ee).norm().item()*100:.1f}")

    tgt = obj.unsqueeze(0)
    for step in range(120):
        env._ik_to(tgt)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    ee2 = env._robot.data.body_pos_w[0, env._ee_body_idx, :3]
    print(f"EE after IK = {ee2.cpu().numpy().round(3)}")
    print(f"IK err (cm)   = {(tgt[0]-ee2).norm().item()*100:.1f}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
