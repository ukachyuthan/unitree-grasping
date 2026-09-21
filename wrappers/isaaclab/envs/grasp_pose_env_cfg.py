"""
Configuration for the grasp-pose prediction environment.

Task: policy predicts ONE 3D grasp position per episode.
      Arm approach + gripper close + lift are scripted.
      Reward = how high the object was lifted.

Robot: Franka Emika Panda with parallel 2-finger gripper.
"""

from __future__ import annotations

import colorsys
import hashlib
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from envs._paths import data_path
from envs._object_registry import PROCEDURAL_SHAPE_NAMES, ycb_shape_names, prim_name

# ── Dimensions ────────────────────────────────────────────────────────────────
NUM_PC_POINTS: int = 128
# Action = grasp center (xyz) + wrist tilt + roll (Path A).
# Contact-pair mode: two_point_grasp=True and NUM_ACTIONS=6 → (c1,c2).
NUM_ACTIONS: int   = 6
OBS_DIM: int       = NUM_PC_POINTS * 3   # 384

# Panda hand max opening ≈ 8 cm — keep objects graspable at this scale.
OBJECT_SCALE: float = 1.5
OBJECT_MASS: float = 0.20

# ── Execution phases (in physics steps, total = decimation) ──────────────────
# Order: hover+orient → descend → pinch → lift (orient above object, then lower).
N_APPROACH  : int = 120  # XY hover above contacts, set jaw yaw (open gripper)
N_DESCEND   : int = 50   # vertical lower to contacts with yaw held
N_CLOSE     : int = 45   # freeze arm, close gripper only
N_LIFT      : int = 50
N_HOLD      : int = 10
N_TRANSPORT : int = 60   # lateral move from above A to above B (pick-and-place only)
N_LOWER     : int = 30   # descend to place height
N_OPEN      : int = 15   # open gripper and release
EXEC_STEPS  : int = (
    N_APPROACH + N_DESCEND + N_CLOSE + N_LIFT + N_HOLD + N_TRANSPORT + N_LOWER + N_OPEN
)
# Union of train + eval real-object families — used only to size the contact
# filter and pre-declare RigidObjectCfg fields; harmless if some are unused
# by a given run (use_real_objects=False, or eval-only play scripts).
_ALL_YCB_NAMES = sorted(set(ycb_shape_names("train")) | set(ycb_shape_names("eval")))
_ALL_SHAPE_NAMES_FOR_CONTACT = PROCEDURAL_SHAPE_NAMES + _ALL_YCB_NAMES
OBJECT_CONTACT_FILTER_PATHS = [
    f"/World/envs/env_.*/{prim_name(name)}" for name in _ALL_SHAPE_NAMES_FOR_CONTACT
]


def _auto_color(name: str) -> tuple[float, float, float]:
    """Deterministic distinct visualization color per real-object name."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 360 / 360.0
    return colorsys.hsv_to_rgb(h, 0.6, 0.85)


def _usd_obj(name: str, color: tuple, split: str = "train") -> RigidObjectCfg:
    usd_path = str(data_path("data/objects", split, name, "000.usd"))
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name(name)}",
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
                solver_position_iteration_count=12,
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
                stiffness=1000.0, damping=100.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0, velocity_limit_sim=2.61,
                stiffness=800.0, damping=80.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=300.0, velocity_limit_sim=0.2,
                stiffness=5000.0, damping=150.0,
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

    # ── Objects: 12 procedural shape families (real YCB families are added
    #    dynamically in __post_init__ below — see _object_registry.py) ────────
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
    # Path A peak (~72%) used ±0.07. Tighter ±0.05/±0.04 was for 2-point PC projection.
    grasp_x_bounds: tuple = (-0.07, 0.07)
    grasp_y_bounds: tuple = (-0.07, 0.07)
    grasp_z_bounds: tuple = (-0.07, 0.07)
    # panda_joint5 tilt range: 0.0 = neutral (top-down), ±1.5 rad = ~85° tilt
    grasp_tilt_bounds: tuple = (-1.5, 1.5)
    # panda_joint7 roll range: full rotation of the jaw plane
    grasp_roll_bounds: tuple = (-2.5, 2.5)
    # Total pinch width clamp (or derived from ||c2-c1|| when two_point_grasp).
    grasp_width_bounds: tuple = (0.01, 0.08)
    # False = Path A (xyz + tilt + roll, pin j5/j7). True = 6D contact pairs.
    two_point_grasp: bool = True
    # Snap decoded contact(s) onto mesh surface (fallback: nearest PC sample).
    project_grasp_to_pc: bool = True
    # Top-down: cast ray from (x,y,z_top) → −Z to hit the visible upper surface.
    project_grasp_ray_down: bool = True
    surface_ray_z_margin_m: float = 0.02
    # Minimum ||c2−c1|| after projection; prevents collapsed pinch near object center.
    two_point_min_pinch_width_m: float = 0.025
    # Debug: ignore policy actions; use fixed object-local contact points below.
    use_fixed_grasp_contacts: bool = False
    fixed_contact_1: tuple = (-0.040, 0.0, 0.0)
    fixed_contact_2: tuple = (+0.040, 0.0, 0.0)
    # Debug: also draw a subsample of the object PC (grey) next to grasp markers.
    visualize_object_pc: bool = False
    visualize_object_pc_n: int = 48

    # ── Task geometry ─────────────────────────────────────────────────────────
    object_scale: float = OBJECT_SCALE
    table_surface_z: float = 0.0
    spawn_x_range: tuple   = (0.43, 0.57)
    spawn_y_range: tuple   = (-0.12, 0.12)
    spawn_yaw_range: tuple = (-math.pi, math.pi)
    randomize_object_spawn: bool = True
    spawn_z_offset: float  = 0.04
    settle_steps: int      = 30

    # ── Approach / lift execution ─────────────────────────────────────────────
    approach_mode: str = "top"           # palm above object; fingers reach down
    lift_height_m: float    = 0.15
    grasp_reach_thresh: float = 0.06
    kinematic_grasp: bool = False
    contact_force_thresh: float = 1.0    # N — ignore glancing / penetrating contacts
    min_contact_fingers: int   = 2
    # Top-down: fingers hang ~6 cm below panda_hand — use -0.06 so finger mid
    # lands on the yellow grasp marker (was -0.10 → fingers hovered ~4 cm high).
    grasp_point_offset: tuple = (0.0, 0.0, -0.06)
    # Palm retreat along approach (two_point / tilt-driven modes).
    grasp_approach_dist_m: float = 0.06
    # Two-point approach: first hover this high above contacts while setting jaw
    # yaw (clear of the object), then N_DESCEND ramps down to final clearance.
    two_point_approach_clearance_m: float = 0.060
    # Height after descend, right before pinch (0 = at contact mid).
    two_point_final_clearance_m: float = 0.0
    # Hover may early-exit only when |finger·jaw| is at least this (yaw settled).
    two_point_hover_jaw_align: float = 0.85
    # If True, policy tilt pitches the approach axis (can put palm target off to the
    # side / out of reach when tilt saturates). Keep False: top-down palm IK + pin
    # wrist joints j5/j7 from tilt/roll actions (Path A that reached ~70%+).
    approach_from_policy_tilt: bool = False
    # Kinematic weld of object→hand while contact sensors fire. KEEP FALSE: with
    # kinematic IK this teleports objects up on fake/penetrating contacts.
    contact_carry: bool = False
    # Lift reward only counts if fingers had real contact during lift/hold.
    require_contact_for_lift_reward: bool = True
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
    gripper_close_val: float = 0.003    # minimum finger joint (fully closed)
    # Close tighter than grasp half-width — solid object blocks fingers at the surface.
    gripper_squeeze_per_finger_m: float = 0.012

    ik_alpha: float = 0.80
    ik_approach_alpha: float = 1.0
    # Diff-IK: one solve per physics step (extra substeps reuse stale FK).
    ik_approach_substeps: int = 1
    ik_jacobian_interval: int = 1
    # False = honest PD reach (train and eval match). True only for IK debug scripts.
    ik_write_joint_state: bool = False
    ik_write_approach_only: bool = True
    # Isaac Lab DifferentialIK (finger-mid position + stable yaw). Prefer this path.
    use_diff_ik: bool = True
    # TCP offset from panda_hand in hand frame (+Z toward fingertips on Franka USD).
    diff_ik_body_offset: tuple = (0.0, 0.0, 0.107)
    # Include orientation in DifferentialIK (False = position-only, more stable reach).
    diff_ik_use_orientation: bool = False
    # Clamp per-step joint delta (rad) so PD / kinematic write don't explode.
    diff_ik_max_joint_step: float = 0.20
    # When True (legacy), pin j5/j7 during homemade IK — breaks reach. Keep False with use_diff_ik.
    pin_wrist_during_ik: bool = False
    # Don't start closing until finger mid is this close to the grasp point.
    approach_arrive_thresh_m: float = 0.040
    # Extra approach steps (open gripper) while waiting to arrive before pinch.
    approach_max_extra_steps: int = 40
    two_point_finger_arrive_thresh_m: float = 0.045
    # Position-only palm tracking during close/descend (no orient IK — avoids shake).
    ik_servo_close_descend: bool = True
    ik_orient_weight: float = 1.0
    ik_orient_alpha: float = 0.5
    ik_orient_pos_thresh: float = 0.08
    ik_orient_yaw_to_object: bool = True
    ik_orient_use_grasp_tilt: bool = False
    # Keep False: EE-orient IK fights pinned wrist and makes the arm shake / miss.
    ik_orient_two_point: bool = False
    ik_grasp_tilt_scale: float = 3.5
    # Path A: clamp policy tilt when pinning j5. Collapsed policies saturate at
    # ±1.5 rad (~85°) which folds the wrist and leaves palm 8–11 cm short of the
    # yellow grasp marker even when the predicted point is fine.
    path_a_max_tilt_rad: float = 0.5
    # Path A approach: position IK only + pin j5/j7 (no competing EE orient IK).
    path_a_position_only_ik: bool = True

    # ── Pick-and-place task mode ───────────────────────────────────────────────
    # 0.5 = 50% episodes are pick-and-place, 50% lift-only.
    # Transport/lower/open phases always run; lift-only episodes hold pose during them.
    # The destination "box" is NOT in the sim — no USD asset, not in point cloud.
    # It is parameterised purely as a goal position + target wrist orientation so
    # the perception pipeline only ever needs to see the grasped object.
    task_mode_prob_place: float = 0.5
    place_goal_x_range: tuple = (0.35, 0.65)
    place_goal_y_range: tuple = (-0.25, 0.25)
    place_height_above_table: float = 0.02   # target z when lowering
    place_reward_weight: float = 1.0
    place_sigma_m: float = 0.06              # exp(-dist/sigma) for place reward
    # Per-episode wrist orientation target during transport — simulates placing into
    # a container that may be angled.  Wrist smoothly rotates over N_TRANSPORT steps
    # so the grasp must learn grasps resilient to reorientation; no physical box needed.
    place_wrist_roll_range: tuple = (-1.57, 1.57)   # ±90° jaw rotation
    place_wrist_tilt_range: tuple = (-0.8,   0.8)   # ±46° wrist pitch

    # ── Contact-area reward ────────────────────────────────────────────────────
    # Bilateral finger coverage: min(left_pts, right_pts) / N_PC near each fingertip.
    contact_area_reward_weight: float = 0.3
    contact_area_radius_m: float = 0.045    # 4.5 cm radius around fingertip
    # Two-point: bonus when raw policy contacts (c1, c2) lie on the object PC.
    surface_contact_reward_weight: float = 0.4
    surface_contact_sigma_m: float = 0.015    # exp decay scale (~1.5 cm)

    # ── Point-cloud augmentation for camera-angle robustness ──────────────────
    # Random Z-axis rotation of the observed PC each episode.
    # Grasp target is inverse-rotated before IK so physics are unaffected.
    pc_augment_yaw: bool = False

    # ── Rendered camera viewpoint augmentation ────────────────────────────────
    # Spawn a jittered ring of depth cameras around each env's object per episode
    # so the policy learns to handle arbitrary viewpoints (sim→real robustness).
    # Requires --enable_cameras when launching (e.g. --headless --enable_cameras).
    # use_camera_pc=False disables the camera sensor entirely (saves render cost).
    # When True, camera_pc_prob controls the PER-EPISODE mix between the
    # rendered-camera path (real occlusion/self-shadowing) and the fast
    # pre-loaded-PC path — not an all-or-nothing switch, so both observation
    # distributions are seen during the same training run.
    use_camera_pc: bool = True
    camera_pc_prob: float = 0.5
    camera_width: int = 64
    camera_height: int = 64
    camera_horizontal_dist_range: tuple = (0.30, 0.55)  # metres from object centre
    camera_height_range: tuple = (0.15, 0.40)           # metres above table surface
    camera_fov_deg: float = 70.0                        # horizontal field of view
    camera_depth_clip: tuple = (0.05, 1.5)              # valid depth window (metres)

    # ── Real-object dataset (scripts/fetch_ycb.py + generate_ycb_meshes.py) ───
    # False reproduces the original RNG-only training distribution exactly.
    use_real_objects: bool = True
    # True: build only from ycb_shape_names("eval") (held-out real objects),
    # ignoring use_real_objects/procedural shapes — used by play scripts to
    # test zero-shot generalization on never-seen real objects.
    eval_object_mode: bool = False

    # ── Sensor-realistic point-cloud noise (Gaussian + dropout + outliers) ────
    # Severity is domain-randomized per episode within these ranges — see
    # grasping/pointcloud_utils.add_sensor_noise(). Applied identically to
    # procedural and real objects so noise level can't leak object identity.
    pc_noise_range_m: tuple        = (0.0, 0.005)
    pc_dropout_frac_range: tuple   = (0.0, 0.10)
    pc_outlier_frac_range: tuple   = (0.0, 0.02)

    # ── GraspNet-bootstrapped reward (scripts/generate_graspnet_labels.py) ────
    use_graspnet_reward: bool = True
    graspnet_reward_scale: float = 1.0
    graspnet_reward_radius_m: float = 0.03   # distance-decay radius for the quality bonus

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        for name in _ALL_YCB_NAMES:
            split = "eval" if name in ycb_shape_names("eval") else "train"
            setattr(self, f"object_{name}", _usd_obj(name, _auto_color(name), split=split))
