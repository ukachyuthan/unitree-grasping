"""
Configuration for the grasp-pose prediction environment.

Task: policy predicts ONE 3D grasp position per episode.
      Arm approach + gripper close + lift are scripted.
      Reward = how high the object was lifted.

This replaces the broken sequential-RL env (g1_grasp_env) with a
single-decision bandit-style env:
  obs   = object point cloud (robot-base frame)
  act   = 3D grasp position  (robot-base frame, tanh-normalized)
  exec  = scripted over decimation physics steps (IK → close → lift)
  reward= (obj_z - table_z).clamp(0) / lift_target_m
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

# ── Dimensions ────────────────────────────────────────────────────────────────
NUM_PC_POINTS: int = 128
NUM_ACTIONS: int   = 3      # (x, y, z) grasp position, tanh-normalized
OBS_DIM: int       = NUM_PC_POINTS * 3   # 384

# ── Execution phases (in physics steps, total = decimation) ──────────────────
N_APPROACH : int = 80   # steps moving EE toward grasp position
N_CLOSE    : int = 20   # steps closing gripper
N_LIFT     : int = 25   # steps lifting the arm upward
N_HOLD     : int = 10   # hold at top + measure
EXEC_STEPS : int = N_APPROACH + N_CLOSE + N_LIFT + N_HOLD   # 135


@configclass
class GraspPoseEnvCfg(DirectRLEnvCfg):

    # ── Timing ────────────────────────────────────────────────────────────────
    # episode_length_s covers exactly 1 agent step (= EXEC_STEPS physics steps)
    decimation: int        = EXEC_STEPS                       # 135
    episode_length_s: float = (EXEC_STEPS / 60.0) * 1.5      # ~3.4 s, gives room for reset
    action_space: int      = NUM_ACTIONS
    observation_space: int = OBS_DIM
    state_space: int       = 0

    # ── Simulation ────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60.0,
        render_interval=decimation,
        device="cuda:0",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ── Scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=2.5, replicate_physics=True
    )

    # ── Robot ─────────────────────────────────────────────────────────────────
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
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={
                # Pre-positioned over workspace (MuJoCo IK to x=0.30, z=0.90)
                # EE starts at ~(0.30, 0.00, 0.90); objects spawn at same x/y
                "left_shoulder_pitch_joint": -0.7840,
                "left_shoulder_roll_joint":  -0.2726,
                "left_shoulder_yaw_joint":   -0.0893,
                "left_elbow_pitch_joint":     1.05,
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
                effort_limit_sim=300.0, velocity_limit_sim=10.0, stiffness=800.0, damping=40.0,
            ),
            "left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_one_joint", "left_two_joint"],
                effort_limit_sim=20.0, velocity_limit_sim=5.0, stiffness=200.0, damping=20.0,
            ),
            "right_arm_and_hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",   "right_elbow_pitch_joint",
                    "right_elbow_roll_joint",
                    "right_one_joint", "right_two_joint",
                    ".*_zero_joint", ".*_three_joint", ".*_four_joint",
                    ".*_five_joint", ".*_six_joint",
                ],
                effort_limit_sim=300.0, velocity_limit_sim=5.0, stiffness=400.0, damping=40.0,
            ),
            "legs_and_torso": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                    ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                    "torso_joint",
                ],
                effort_limit_sim=300.0, velocity_limit_sim=5.0, stiffness=400.0, damping=40.0,
            ),
        },
    )

    # ── Table ─────────────────────────────────────────────────────────────────
    # Shifted toward the robot so the near edge starts ~2 cm in front of the base.
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.42, 0.0, 0.4)),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.60, 0.80),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        ),
    )

    # ── Objects (same 12 shape families as before) ────────────────────────────
    def _usd_obj(name: str, color: tuple) -> RigidObjectCfg:
        from envs._paths import data_path
        usd_path = str(data_path("data/objects/train", name, "000.usd"))
        return RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{name.title().replace('_','')}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -20.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        )

    object_torus:         RigidObjectCfg = _usd_obj("torus",         (0.9, 0.3, 0.1))
    object_l_shape:       RigidObjectCfg = _usd_obj("l_shape",       (0.2, 0.8, 0.2))
    object_t_shape:       RigidObjectCfg = _usd_obj("t_shape",       (0.2, 0.4, 0.9))
    object_c_shape:       RigidObjectCfg = _usd_obj("c_shape",       (0.9, 0.8, 0.1))
    object_dumbbell:      RigidObjectCfg = _usd_obj("dumbbell",      (0.8, 0.2, 0.8))
    object_wedge:         RigidObjectCfg = _usd_obj("wedge",         (0.2, 0.9, 0.8))
    object_star_prism:    RigidObjectCfg = _usd_obj("star_prism",    (0.9, 0.5, 0.2))
    object_bracket:       RigidObjectCfg = _usd_obj("bracket",       (0.4, 0.7, 0.3))
    object_stepped_cyl:   RigidObjectCfg = _usd_obj("stepped_cyl",  (0.7, 0.3, 0.9))
    object_twisted_bar:   RigidObjectCfg = _usd_obj("twisted_bar",   (0.3, 0.8, 0.5))
    object_irregular_ext: RigidObjectCfg = _usd_obj("irregular_ext", (0.8, 0.6, 0.2))
    object_convex_hull:   RigidObjectCfg = _usd_obj("convex_hull",   (0.5, 0.5, 0.9))

    # ── Workspace bounds for action de-normalization ──────────────────────────
    # policy outputs tanh in [-1,1]^3; env maps to OBJECT-LOCAL frame offsets.
    # Objects are mean-centred by trimesh so grasps are within ~±6cm of origin.
    # The env converts local → robot-base in _do_approach by adding obj_base.
    grasp_x_bounds: tuple = (-0.07, 0.07)
    grasp_y_bounds: tuple = (-0.07, 0.07)
    grasp_z_bounds: tuple = (-0.07, 0.07)

    # ── Task geometry ─────────────────────────────────────────────────────────
    # Fixed spawn under pre-positioned EE home ≈ (0.30, 0.00) robot-base.
    table_surface_z: float = 0.80
    spawn_x_range: tuple   = (0.30, 0.30)
    spawn_y_range: tuple   = (0.00, 0.00)
    spawn_z_offset: float  = 0.045   # object centre ≈ EE height (~0.845 world)
    settle_steps: int      = 30      # let object drop to true rest, then anchor there

    # ── Lift execution ────────────────────────────────────────────────────────
    lift_height_m: float    = 0.15    # commanded palm rise over N_LIFT steps
    grasp_reach_thresh: float = 0.06  # max grasp-point-to-object dist to count as grasped
    # Offset (world x,y,z) from the palm-link origin (wrist) to the finger grasp
    # zone. The arm positions this point at the object, so the object is held at
    # the fingers instead of floating at the wrist.
    grasp_point_offset: tuple = (0.06, -0.04, -0.02)

    # ── Reward ────────────────────────────────────────────────────────────────
    lift_target_m: float    = 0.12    # full reward when lifted 12 cm
    lift_threshold_m: float = 0.03    # binary success if lifted > 3 cm
    lift_reward_weight: float = 0.75  # fraction of total reward from lift
    leg_stillness_weight: float = 0.25  # fraction from keeping legs at home
    leg_dev_scale: float = 0.12       # RMS joint deviation (rad) for zero leg bonus
    leg_vel_scale: float = 0.50       # RMS joint velocity (rad/s) for zero leg bonus

    # ── Joint names (G1-specific) ─────────────────────────────────────────────
    left_arm_joint_names: list = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",   "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
    ]
    left_gripper_joint_names: list = ["left_one_joint", "left_two_joint"]
    leg_joint_names: list = [
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "torso_joint",
    ]
    ee_body_name: str = "left_palm_link"

    # IK step size — needs to be large enough to traverse ~60cm from home pose
    ik_alpha: float = 0.6    # joint step size per physics step during approach
