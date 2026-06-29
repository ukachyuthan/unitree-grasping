"""
Configuration for the G1 left-arm pick-and-place grasping environment.

Design principle — robot-agnostic policy interface
───────────────────────────────────────────────────
The RL policy observes only geometry:
  • Object point cloud  (expressed in robot-base frame via T_cam→robot)
  • End-effector pose relative to robot base
  • Goal pose relative to robot base
  • Gripper state + previous action

It never sees joint names, DOF counts, or robot-specific structure.
The G1 is the demonstration platform.  To port to another robot:
  1. Swap ArticulationCfg (robot asset)
  2. Update ee_body_name (G1: "left_palm_link") + left_arm_joint_names
  3. Provide a new camera-to-robot extrinsic (T_cam_robot)
  The trained PointNet weights transfer unchanged.

Action space (7 dims, use_joint_space_control=True shortcut available):
  [0:3]  Δ end-effector position  (m, in robot-base frame)
  [3:6]  Δ end-effector rotation  (axis-angle rad, in robot-base frame)
  [6]    Gripper  (−1 = open, +1 = close)
  These are converted to joint targets inside the environment via a simple
  Jacobian-transpose IK.  Set use_joint_space_control=True to bypass IK
  and directly output Δjoint (5-DOF arm + gripper) during early training.

Observation (OBS_DIM = PC_FLAT + PROPRIO_DIM):
  [0 : PC_FLAT]   Point cloud in robot-base frame  (NUM_PC_POINTS × 3)
  [PC_FLAT : ]    EE pos(3) + EE quat(4) + goal pos(3) + goal quat(4)
                  + gripper(1) + prev_action(7)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# ── Dimension constants (importable by other modules) ───────────────────────
NUM_PC_POINTS: int = 128
NUM_ACTIONS: int = 7      # Δpos(3) + Δrot_aa(3) + gripper(1)
# In joint-space mode NUM_ACTIONS = 6 (5 arm + 1 gripper); kept at 7 so
# the observation vector size doesn't change between modes.

PROPRIO_DIM: int = (
    3   # EE position (robot-base frame)
    + 4  # EE quaternion (w, x, y, z)
    + 3  # goal position (robot-base frame)
    + 4  # goal quaternion (identity for flat-place goals)
    + 1  # gripper state [0 open … 1 closed]
    + NUM_ACTIONS  # previous action
)  # = 22

PC_FLAT_DIM: int = NUM_PC_POINTS * 3   # 384
OBS_DIM: int = PC_FLAT_DIM + PROPRIO_DIM  # 406


@configclass
class G1GraspEnvCfg(DirectRLEnvCfg):
    """Full configuration for the G1 pick-and-place environment."""

    # ── Timing ──────────────────────────────────────────────────────────────
    episode_length_s: float = 15.0
    decimation: int = 4           # 60 Hz physics → 15 Hz policy
    action_space: int = NUM_ACTIONS
    observation_space: int = OBS_DIM
    state_space: int = 0

    # ── Simulation ──────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ── Scene ───────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=2.5, replicate_physics=True
    )

    # ── Robot (G1, root fixed, only left arm driven) ─────────────────────────
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                fix_root_link=True,   # arm-only / fixed-base
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={
                "left_shoulder_pitch_joint":  0.50,
                "left_shoulder_roll_joint":   0.10,
                "left_shoulder_yaw_joint":    0.00,
                "left_elbow_pitch_joint":     1.00,
                "left_elbow_roll_joint":      0.00,
                "left_one_joint":             1.00,
                "left_two_joint":             0.52,
                "right_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.16,
                "right_shoulder_yaw_joint":   0.00,
                "right_elbow_pitch_joint":    0.87,
                "right_elbow_roll_joint":     0.00,
                "right_one_joint":           -1.00,
                "right_two_joint":           -0.52,
                ".*_hip_pitch_joint":   -0.20,
                ".*_knee_joint":         0.42,
                ".*_ankle_pitch_joint": -0.23,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",   "left_elbow_pitch_joint",
                    "left_elbow_roll_joint",
                ],
                effort_limit_sim=100.0, velocity_limit_sim=10.0, stiffness=80.0, damping=8.0,
            ),
            "left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_one_joint", "left_two_joint"],
                effort_limit_sim=20.0, velocity_limit_sim=5.0, stiffness=200.0, damping=20.0,
            ),
            # Stiff hold for all inactive joints
            # Note: .*_zero/three/four/five/six match both left & right — no explicit dups
            "right_arm_and_hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",   "right_elbow_pitch_joint",
                    "right_elbow_roll_joint",
                    "right_one_joint", "right_two_joint",
                    ".*_zero_joint",   ".*_three_joint", ".*_four_joint",
                    ".*_five_joint",   ".*_six_joint",
                ],
                effort_limit_sim=300.0, velocity_limit_sim=5.0, stiffness=400.0, damping=40.0,
            ),
            "legs_and_torso": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                    ".*_knee_joint",     ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                    "torso_joint",
                ],
                effort_limit_sim=300.0, velocity_limit_sim=5.0, stiffness=400.0, damping=40.0,
            ),
        },
    )

    # ── Table: 80×60×80 cm box; surface at z = 0.4 + 0.4 = 0.80 world ───────
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.4)),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.60, 0.80),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        ),
    )

    # ── Objects: 3 primitive shapes, one activated per episode ───────────────
    object_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ObjBox",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -20.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0
            ),
        ),
    )
    object_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ObjSphere",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -20.0)),
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0
            ),
        ),
    )
    object_cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ObjCylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -20.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.18),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0
            ),
        ),
    )

    # ── Task geometry ────────────────────────────────────────────────────────
    table_surface_z: float = 0.80
    spawn_x_range: tuple = (0.35, 0.65)
    spawn_y_range: tuple = (-0.15, 0.15)
    goal_x_range: tuple  = (0.35, 0.60)
    goal_y_range: tuple  = (0.20, 0.35)

    # ── Robot-specific knobs (change these to port to another robot) ──────────
    left_arm_joint_names: list = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
    ]
    left_gripper_joint_names: list = ["left_one_joint", "left_two_joint"]
    # TODO: verify against actual G1 USD by inspecting the stage prim tree
    ee_body_name: str = "left_palm_link"

    # ── Control mode ─────────────────────────────────────────────────────────
    # True  → action[0:5] = Δjoint (arm) + action[5] = gripper (simpler to train)
    # False → action = Δ(EE pos, EE rot, gripper) → IK → joint targets (portable)
    use_joint_space_control: bool = True

    arm_action_scale: float = 0.10    # rad or m per step
    rot_action_scale: float = 0.10    # rad per step (EE mode only)

    # ── Reward ───────────────────────────────────────────────────────────────
    use_graspnet_reward: bool = False  # set True once Contact-GraspNet installed
    approach_reward_scale: float = 2.0
    lift_reward_scale: float  = 5.0
    place_reward_scale: float  = 10.0
    success_bonus: float = 50.0
    fall_penalty: float  = -5.0
    action_penalty_scale: float = 0.01

    lift_height_threshold: float = 0.12   # 12 cm above table
    place_success_radius: float  = 0.05   # 5 cm XY at goal B
