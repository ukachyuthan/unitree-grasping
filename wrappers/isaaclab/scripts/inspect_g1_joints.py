#!/usr/bin/env python3
"""
Inspect the G1 USD to find the correct end-effector body name for the left arm.

Run this ONCE before training to verify ee_body_name in g1_grasp_env_cfg.py.

Usage:
    ./rl_unitree/bin/python scripts/inspect_g1_joints.py --headless

Output: prints all rigid body names and joint names of the G1.
Look for the leftmost distal link (e.g. "left_hand", "left_palm", etc.)
and update G1GraspEnvCfg.ee_body_name in grasping/g1_grasp_env_cfg.py.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class InspectSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.74)),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=100.0,
                velocity_limit=10.0,
                stiffness=40.0,
                damping=4.0,
            )
        },
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60.0)
    sim     = SimulationContext(sim_cfg)
    sim.set_camera_view([3, 3, 3], [0, 0, 1])

    scene_cfg = InspectSceneCfg(num_envs=1, env_spacing=5.0)
    scene     = InteractiveScene(scene_cfg)

    sim.reset()
    robot: Articulation = scene["robot"]

    print("\n" + "="*60)
    print("G1 JOINT NAMES")
    print("="*60)
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:3d}]  {name}")

    print("\n" + "="*60)
    print("G1 RIGID BODY NAMES")
    print("="*60)
    for i, name in enumerate(robot.body_names):
        print(f"  [{i:3d}]  {name}")

    print("\n" + "="*60)
    print("LEFT-SIDE BODIES (filter: 'left')")
    print("="*60)
    for i, name in enumerate(robot.body_names):
        if "left" in name.lower() or "L_" in name:
            print(f"  [{i:3d}]  {name}")

    print("\nUpdate G1GraspEnvCfg.ee_body_name with the last link above.")
    simulation_app.close()


if __name__ == "__main__":
    main()
