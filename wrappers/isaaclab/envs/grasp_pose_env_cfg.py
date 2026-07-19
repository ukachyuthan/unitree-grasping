"""
Configuration for the grasp-pose prediction environment.

Task: policy predicts ONE 3D grasp position per episode.
      Arm approach + gripper close + lift are scripted.
      Reward = how high the object was lifted.

Robot: Franka Emika Panda with parallel 2-finger gripper.
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
# Action = (gx, gy, gz, tilt, roll): 3-D grasp position + 2-D EE orientation.
#   tilt → panda_joint5 (wrist pitch): 0 = top-down, max = near-horizontal side-grasp.
#   roll → panda_joint7 (wrist roll):  rotates the jaw plane around the approach axis.
# This lets the policy learn BOTH where and how to orient the gripper per shape.
NUM_ACTIONS: int   = 5
OBS_DIM: int       = NUM_PC_POINTS * 3   # 384

# Panda hand max opening ≈ 8 cm — keep objects graspable at this scale.
OBJECT_SCALE: float = 1.5
OBJECT_MASS: float = 0.20

# ── Execution phases (in physics steps, total = decimation) ──────────────────
N_APPROACH : int = 80   # IK moves palm above grasp point (top-down)
N_CLOSE    : int = 45   # longer close for firm contact before lift
N_LIFT     : int = 40
N_HOLD     : int = 10
EXEC_STEPS : int = N_APPROACH + N_CLOSE + N_LIFT + N_HOLD

_OBJECT_PRIM_NAMES = [
    "Torus", "LShape", "TShape", "CShape", "Dumbbell", "Wedge",
    "StarPrism", "Bracket", "SteppedCyl", "TwistedBar", "IrregularExt", "ConvexHull",
]
OBJECT_CONTACT_FILTER_PATHS = [
    f"/World/envs/env_.*/{name}" for name in _OBJECT_PRIM_NAMES
]


@configclass
class GraspPoseEnvCfg(DirectRLEnvCfg):

    # ── Timing ────────────────────────────────────────────────────────────────
    decimation: int        = EXEC_STEPS
    episode_length_s: float = (EXEC_STEPS / 60.0) * 1.5
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
            static_friction=1.5,
            dynamic_friction=1.3,
            restitution=0.0,
        ),
    )

    # ── Scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=2.5, replicate_physics=True
    )

    # ── Robot: Franka Panda + parallel 2-finger gripper ───────────────────────
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                # Default Franka ready pose — reaches over table workspace.
                "panda_joint1":  0.0,
                "panda_joint2": -0.569,
                "panda_joint3":  0.0,
                "panda_joint4": -2.810,
                "panda_joint5":  0.0,
                "panda_joint6":  3.037,
                "panda_joint7":  0.741,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0, velocity_limit_sim=2.175,
                stiffness=400.0, damping=40.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0, velocity_limit_sim=2.61,
                stiffness=400.0, damping=40.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=300.0, velocity_limit_sim=0.2,
                stiffness=4000.0, damping=120.0,
            ),
        },
    )

    # ── Table ─────────────────────────────────────────────────────────────────
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, -0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.60, 0.02),
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
                scale=(OBJECT_SCALE, OBJECT_SCALE, OBJECT_SCALE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
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
    grasp_x_bounds: tuple = (-0.07, 0.07)
    grasp_y_bounds: tuple = (-0.07, 0.07)
    grasp_z_bounds: tuple = (-0.07, 0.07)
    # panda_joint5 tilt range: 0.0 = neutral (top-down), ±1.5 rad = ~85° tilt
    grasp_tilt_bounds: tuple = (-1.5, 1.5)
    # panda_joint7 roll range: full rotation of the jaw plane
    grasp_roll_bounds: tuple = (-2.5, 2.5)

    # ── Task geometry ─────────────────────────────────────────────────────────
    object_scale: float = OBJECT_SCALE
    table_surface_z: float = 0.0
    spawn_x_range: tuple   = (0.50, 0.50)
    spawn_y_range: tuple   = (0.00, 0.00)
    spawn_z_offset: float  = 0.04
    settle_steps: int      = 30

    # ── Approach / lift execution ─────────────────────────────────────────────
    approach_mode: str = "top"           # palm above object; fingers reach down
    lift_height_m: float    = 0.15
    grasp_reach_thresh: float = 0.06
    kinematic_grasp: bool = False
    contact_force_thresh: float = 0.05
    min_contact_fingers: int   = 2
    # Top-down: fingers hang below palm — negative z offset raises hand above object.
    grasp_point_offset: tuple = (0.0, 0.0, -0.10)
    # During lift, carry the object only while finger contact is active THIS step
    # (no latch — contact lost → object falls under gravity).
    contact_carry: bool = True
    # Debug: show predicted grasp point (and palm IK target) in the viewport / video.
    visualize_grasp_point: bool = False
    grasp_marker_radius_m: float = 0.015
    # Eval: cycle shapes 0..N-1 each reset instead of random (for demo videos).
    eval_cycle_shapes: bool = False
    eval_num_shapes: int = 10
    lift_target_m: float    = 0.05        # 5 cm full reward — clearer success signal
    lift_threshold_m: float = 0.03
    lift_reward_weight: float = 1.0
    leg_stillness_weight: float = 0.0   # no legs on Franka
    leg_dev_scale: float = 0.12
    leg_vel_scale: float = 0.50

    # ── Franka joint / link names ─────────────────────────────────────────────
    arm_joint_names: list = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    gripper_joint_names: list = ["panda_finger_joint1", "panda_finger_joint2"]
    hand_tuck_joint_names: list = []    # none — parallel gripper only
    leg_joint_names: list = []          # none — fixed-base arm
    ee_body_name: str = "panda_hand"
    finger_contact_prim_path: str = (
        "/World/envs/env_.*/Robot/panda_(left|right)finger"
    )
    # Ignore table contacts — only count finger forces against grasp objects.
    finger_contact_filter_paths: list = OBJECT_CONTACT_FILTER_PATHS
    gripper_open_val: float  = 0.04     # 4 cm per finger → 8 cm aperture
    gripper_close_val: float = 0.003    # tighter squeeze for side pinch

    ik_alpha: float = 0.6
    ik_jacobian_interval: int = 4
    ik_orient_weight: float = 1.0
    ik_orient_alpha: float = 0.35
    ik_orient_pos_thresh: float = 0.04   # metres — only rotate wrist once palm is near target
    ik_orient_yaw_to_object: bool = True
