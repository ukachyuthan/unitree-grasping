#!/usr/bin/env python3
"""Run one grasp episode and print EE / contact / lift stats."""
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
from envs.grasp_pose_env_cfg import GraspPoseEnvCfg, N_APPROACH, N_CLOSE
from envs.grasp_pose_env import GraspPoseEnv

def main():
    cfg = GraspPoseEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = GraspPoseEnv(cfg=cfg, render_mode=None)
    obs, _ = env.reset()
    action = torch.zeros(env.num_envs, 3, device=env.device)
    env._pre_physics_step(action)
    close_end = N_APPROACH + N_CLOSE - 1
    for s in range(env.cfg.decimation):
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        if s in (N_APPROACH - 1, close_end, env.cfg.decimation - 1):
            ee = env._robot.data.body_pos_w[0, env._ee_body_idx, :3]
            obj = env._get_active_obj_pos()[0]
            forces = env._finger_contact.data.net_forces_w[0]
            fmag = forces.norm(dim=-1)
            print(f"step {s:3d}  ee_z={ee[2]:.3f}  obj_z={obj[2]:.3f}  "
                  f"fingers={fmag.cpu().numpy().round(2)}  locked={env._grasp_locked[0].item()}")
    rew = env._get_rewards()
    print(f"reward={rew[0].item():.3f}  spawn_z={env._spawn_z[0].item():.3f}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
