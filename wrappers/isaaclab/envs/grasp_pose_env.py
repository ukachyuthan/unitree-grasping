"""
Grasp-pose prediction environment.

Episode structure (one agent step = decimation physics steps):
  1. [step 0]                 policy observes PC → outputs two contact points
  2. [0–N_APPROACH]           hover above contacts + set jaw yaw (gripper open)
  3. [N_APPROACH–N_DESCEND]   vertical descend to contacts (yaw held)
  4. […–N_CLOSE]              freeze arm, close gripper
  5. […–N_LIFT]               lift; hold + measure → reward
  6. env resets

The policy only makes ONE decision per episode: where to grasp.
Everything else is scripted. This is the "Path A" architecture.
"""

from __future__ import annotations

import json
import math
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import (
    quat_rotate,
    compute_pose_error,
    quat_from_matrix,
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
    combine_frame_transforms,
    skew_symmetric_matrix,
)
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg

from envs._object_registry import (
    PROCEDURAL_SHAPE_NAMES, ycb_shape_names, shape_asset, expand_variants, prim_name,
)
from envs.grasp_pose_env_cfg import (
    GraspPoseEnvCfg,
    N_APPROACH, N_CLOSE, N_DESCEND, N_LIFT, N_HOLD, N_TRANSPORT, N_LOWER, N_OPEN, EXEC_STEPS,
    NUM_PC_POINTS,
)
from grasping.pointcloud_utils import add_sensor_noise


def _smoothstep(t: float) -> float:
    """S-curve with zero derivative at t=0 and t=1.  Prevents motion jerk."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


_PC_PRE_N   = 512   # points pre-sampled per shape on disk


class GraspPoseEnv(DirectRLEnv):
    cfg: GraspPoseEnvCfg

    def __init__(self, cfg: GraspPoseEnvCfg, render_mode=None):
        super().__init__(cfg, render_mode=render_mode)

        # Batched object-collection bookkeeping (see _setup_scene).
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._park_state_w = torch.zeros(self.num_envs, self._num_shapes, 13, device=self.device)
        self._park_state_w[:, :, :3] = self._park_pos.unsqueeze(0) + self.scene.env_origins.unsqueeze(1)
        self._park_state_w[:, :, 3] = 1.0   # identity quat (wxyz)

        # ── Joint indices (resolved after super().__init__ builds the scene) ──
        self._arm_dof_idx, _ = self._robot.find_joints(cfg.arm_joint_names)
        self._grip_dof_idx, self._grip_names = self._robot.find_joints(
            cfg.gripper_joint_names
        )
        if cfg.hand_tuck_joint_names:
            self._hand_tuck_idx, self._hand_tuck_names = self._robot.find_joints(
                cfg.hand_tuck_joint_names
            )
        else:
            self._hand_tuck_idx, self._hand_tuck_names = [], []
        if cfg.leg_joint_names:
            self._leg_dof_idx, _ = self._robot.find_joints(cfg.leg_joint_names)
        else:
            self._leg_dof_idx = []
        self._ee_body_idx, _  = self._robot.find_bodies(cfg.ee_body_name)
        self._ee_body_idx     = self._ee_body_idx[0]
        self._left_finger_idx, _ = self._robot.find_bodies("panda_leftfinger")
        self._right_finger_idx, _ = self._robot.find_bodies("panda_rightfinger")
        self._left_finger_idx = self._left_finger_idx[0]
        self._right_finger_idx = self._right_finger_idx[0]

        # ── Stored home joint positions and EE orientation ────────────────────
        self._home_joint_pos = self._robot.data.default_joint_pos.clone()
        # Reference EE orientation at the home pose (all envs identical at init).
        # Used as the base orientation before yaw-toward-object adjustment in IK.
        self._home_ee_quat = self._robot.data.body_quat_w[:, self._ee_body_idx].clone()  # (B, 4)
        # Table-plane yaw of hand +Y at home — calibrates j7 pin vs contact jaw axis.
        _hand_y = quat_rotate(
            self._home_ee_quat[:1],
            torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
        )
        self._home_jaw_yaw = torch.atan2(_hand_y[:, 0], _hand_y[:, 1])  # (1,)

        # ── Pre-load object point clouds ──────────────────────────────────────
        # self._shape_names / self._num_shapes were set in _setup_scene() (called
        # inside super().__init__() above) so the object collection already matches this order.
        pcs = []
        for name in self._shape_names:
            p = shape_asset(name, "_pc.npy")
            if p.exists():
                arr = np.load(str(p))  # (512, 3)
            else:
                arr = np.zeros((_PC_PRE_N, 3), dtype=np.float32)
                print(f"[GraspPoseEnv] WARNING: missing PC for {name}, using zeros")
            pcs.append(torch.tensor(arr, dtype=torch.float32))
        # Scale the observed point cloud to match the up-scaled spawned object.
        self._obj_pcs = (
            torch.stack(pcs, dim=0).to(self.device) * self.cfg.object_scale
        )  # (NUM_SHAPES, 512, 3)
        self._obj_meshes: list = []
        self._mesh_proximity = self._load_mesh_proximity()

        # ── Pre-load GraspNet-quality labels (scripts/generate_graspnet_labels.py) ──
        # Missing files default to all-zero quality, which contributes nothing to
        # the reward term (see _graspnet_reward) rather than erroring.
        _K_GN = 20
        gn_centers, gn_quality, n_missing = [], [], 0
        for name in self._shape_names:
            centers = np.zeros((_K_GN, 3), dtype=np.float32)
            quality = np.zeros((_K_GN,), dtype=np.float32)
            if self.cfg.use_graspnet_reward:
                p = shape_asset(name, "_graspnet.json")
                if p.exists():
                    with open(p) as f:
                        gdata = json.load(f)
                    glist = gdata["grasps"] if isinstance(gdata, dict) else gdata
                    glist = sorted(glist, key=lambda g: -g["quality"])[:_K_GN]
                    for i, g in enumerate(glist):
                        centers[i] = g["center"]
                        quality[i] = g["quality"]
                else:
                    n_missing += 1
            gn_centers.append(torch.tensor(centers))
            gn_quality.append(torch.tensor(quality))
        self._obj_graspnet_centers = (
            torch.stack(gn_centers, dim=0).to(self.device) * self.cfg.object_scale
        )  # (NUM_SHAPES, _K_GN, 3)
        self._obj_graspnet_quality = torch.stack(gn_quality, dim=0).to(self.device)  # (NUM_SHAPES, _K_GN)
        if self.cfg.use_graspnet_reward and n_missing:
            print(f"[GraspPoseEnv] WARNING: {n_missing}/{len(self._shape_names)} shapes missing "
                  f"<inst>_graspnet.json — graspnet reward term is 0 for those until "
                  f"scripts/generate_graspnet_labels.py is run.")

        # ── Per-env state ──────────────────────────────────────────────────────
        B = self.num_envs
        self._env_shape    = torch.zeros(B, dtype=torch.long, device=self.device)
        # One-shot replay overrides (see queue_replays): the next reset of a flagged
        # env spawns this shape at this pose instead of a random draw.
        self._replay_pending = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._replay_shape = torch.zeros(B, dtype=torch.long, device=self.device)
        self._replay_xy = torch.zeros(B, 2, device=self.device)
        self._replay_quat = torch.zeros(B, 4, device=self.device)
        # _grasp_target: offset from object centre in object-LOCAL frame (metres).
        # Converted to robot-base frame inside _do_approach by adding obj_base.
        self._grasp_target = torch.zeros(B, 3, device=self.device)
        # Object root pose when the policy chooses the grasp (stable, not mutated by carry).
        self._grasp_origin_w = torch.zeros(B, 3, device=self.device)
        self._grasp_origin_quat = torch.zeros(B, 4, device=self.device)
        self._grasp_origin_quat[:, 0] = 1.0
        self._spawn_z      = torch.zeros(B, device=self.device)  # world-frame z at spawn
        self._obj_anchor   = torch.zeros(B, 13, device=self.device)  # root state after settle
        self._grasp_locked = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._grasp_offset = torch.zeros(B, 3, device=self.device)   # obj - ee at close
        self._last_lift_reward     = torch.zeros(B, device=self.device)
        # The ONLY success criterion: the object reached lift_target_m (with finger
        # contact when require_contact_for_lift_reward). Every other reward term is
        # shaping and never counts towards success.
        self._last_lift_success    = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._last_leg_still       = torch.zeros(B, device=self.device)
        self._last_contact_reward  = torch.zeros(B, device=self.device)
        self._last_place_reward    = torch.zeros(B, device=self.device)
        self._last_graspnet_reward = torch.zeros(B, device=self.device)
        self._last_surface_contact_reward = torch.zeros(B, device=self.device)
        # Peak object z during lift+hold phases — used for lift reward in place mode
        # so we don't measure height AFTER the arm has lowered the object to the goal.
        self._last_hold_obj_z = torch.zeros(B, device=self.device)
        # True if fingers registered real contact at any lift/hold step.
        self._lift_had_contact = torch.zeros(B, dtype=torch.bool, device=self.device)
        # Approach finished (near clearance target) — gate the close/pinch phase.
        self._approach_arrived = torch.zeros(B, dtype=torch.bool, device=self.device)
        # Extra approach steps waiting to arrive (shared counter for vectorized envs).
        self._approach_wait = 0
        self._pinch_close_step = 0
        # Per-episode task mode: 0 = lift-only, 1 = pick-and-place.
        self._task_mode = torch.zeros(B, dtype=torch.long, device=self.device)
        # Place goal world position (sampled per episode when task_mode=1).
        self._place_goal_w = torch.zeros(B, 3, device=self.device)
        # EE world position at the start of transport (for smooth interpolation).
        self._transport_start_w = torch.zeros(B, 3, device=self.device)
        # PC augmentation yaw angle per env (0 when pc_augment_yaw=False).
        self._pc_aug_yaw = torch.zeros(B, device=self.device)
        # Pick-and-place wrist orientation targets (sampled per episode).
        # Represent the angle the destination container is at — no box in sim.
        self._place_wrist_roll = torch.zeros(B, device=self.device)
        self._place_wrist_tilt = torch.zeros(B, device=self.device)
        # Per-axis rotation flags: each axis is independently randomised per episode
        # so training sees all combinations (only roll, only tilt, both, neither).
        self._place_roll_active = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._place_tilt_active = torch.zeros(B, dtype=torch.bool, device=self.device)
        # Wrist state at transport start and computed end-targets (set at s_local==0).
        self._transport_roll_start = torch.zeros(B, device=self.device)
        self._transport_tilt_start = torch.zeros(B, device=self.device)
        self._transport_roll_tgt   = torch.zeros(B, device=self.device)
        self._transport_tilt_tgt   = torch.zeros(B, device=self.device)

        # World-frame camera poses for the rendered-depth path (set per episode).
        self._cam_pos_w  = torch.zeros(B, 3, device=self.device)
        self._cam_quat_w = torch.zeros(B, 4, device=self.device)
        self._cam_quat_w[:, 0] = 1.0  # identity
        # Per-episode choice of camera-rendered vs. fast pre-loaded PC path
        # (Bernoulli(camera_pc_prob) each reset — see _reset_idx / _synthesize_pointcloud).
        self._use_camera_this_ep = torch.zeros(B, dtype=torch.bool, device=self.device)
        # Palm world position captured at the start of the lift phase; the lift
        # target ramps up from here so it no longer chases the moving obj anchor.
        self._lift_base_w  = torch.zeros(B, 3, device=self.device)
        self._lift_quat_w  = torch.zeros(B, 4, device=self.device)
        self._lift_quat_w[:, 0] = 1.0
        # Offset from palm origin (wrist) to the finger grasp zone (world frame).
        self._grasp_pt_offset = torch.tensor(
            self.cfg.grasp_point_offset, device=self.device
        ).unsqueeze(0)
        # Cached Jacobians (recomputed every ik_jacobian_interval steps).
        self._J_cache = None
        self._J_pos_cache = None
        self._J_finger_cache = None
        self._ik_call = 0
        self._exec_step    = 0   # phase counter, reset in _pre_physics_step
        self._eval_shape_step = 0
        self._arm_q_des = self._home_joint_pos[:, self._arm_dof_idx].clone()
        self._arm_q_hold = self._arm_q_des.clone()
        self._arm_q_lift_end = self._arm_q_des.clone()
        # Filled at end of approach for IK diagnostics (play / debug).
        self._ik_finger_err = torch.zeros(B, device=self.device)
        self._ik_palm_err = torch.zeros(B, device=self.device)
        self._ik_jaw_align = torch.zeros(B, device=self.device)
        self._ik_left_contact_err = torch.zeros(B, device=self.device)
        self._ik_right_contact_err = torch.zeros(B, device=self.device)
        self._ik_left_contact_err_closed = torch.zeros(B, device=self.device)
        self._ik_right_contact_err_closed = torch.zeros(B, device=self.device)
        self._ik_contact_err_after_close = torch.zeros(B, device=self.device)
        self._ik_palm_w = torch.zeros(B, 3, device=self.device)
        self._ik_palm_tgt_w = torch.zeros(B, 3, device=self.device)
        self._ik_grasp_w = torch.zeros(B, 3, device=self.device)
        self._ik_finger_w = torch.zeros(B, 3, device=self.device)

        # Orientation targets decoded from actions 3 and 4.
        # These are applied every IK step so the policy fully controls wrist angle.
        self._tilt_target = torch.zeros(B, device=self.device)        # panda_joint5
        self._roll_target = torch.full((B,), 0.741, device=self.device)  # panda_joint7 home
        # Indices of tilt/roll joints *within* the arm joint list (0-indexed).
        # arm_joint_names = [j1, j2, j3, j4, j5, j6, j7] → j5=index 4, j7=index 6.
        self._tilt_arm_idx = 4   # panda_joint5 within _arm_dof_idx
        self._flex_arm_idx = 5   # panda_joint6 — keep top-down (DiffIK otherwise tips wrist)
        self._roll_arm_idx = 6   # panda_joint7 within _arm_dof_idx
        # Home j6 for top-down palm (Franka ready pose).
        self._flex_home = torch.tensor(3.037, device=self.device)

        # Arm + parallel gripper (and optional tuck joints) are controlled.
        controlled = set(self._arm_dof_idx) | set(self._grip_dof_idx)
        if self._hand_tuck_idx:
            controlled |= set(self._hand_tuck_idx)
        self._idle_dof_idx = [i for i in range(self._robot.num_joints) if i not in controlled]

        self._ee_jac_idx = self._ee_body_idx - 1
        self._left_finger_jac_idx = self._left_finger_idx - 1
        self._right_finger_jac_idx = self._right_finger_idx - 1

        # Isaac Lab DifferentialIK (pose mode) — all 7 arm joints free during approach.
        self._diff_ik = None
        self._diff_ik_offset_pos = None
        self._diff_ik_offset_rot = None
        if self.cfg.use_diff_ik:
            self._diff_ik = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose" if self.cfg.diff_ik_use_orientation else "position",
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={"lambda_val": 0.05},
                ),
                num_envs=self.num_envs,
                device=self.device,
            )
            self._diff_ik_offset_pos = torch.tensor(
                self.cfg.diff_ik_body_offset, device=self.device, dtype=torch.float32
            ).expand(self.num_envs, -1).clone()
            self._diff_ik_offset_rot = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32
            ).expand(self.num_envs, -1).clone()
            print(
                f"[GraspPoseEnv] DifferentialIK enabled  "
                f"offset={self.cfg.diff_ik_body_offset}  pin_wrist={self.cfg.pin_wrist_during_ik}"
            )

        # Parallel 2-finger gripper: both joints share the same open/close value.
        n_grip = len(self._grip_dof_idx)
        self._gripper_open = torch.full(
            (n_grip,), cfg.gripper_open_val, device=self.device
        )
        self._gripper_close = torch.full(
            (n_grip,), cfg.gripper_close_val, device=self.device
        )
        w_lo, w_hi = cfg.grasp_width_bounds
        self._grasp_width = torch.full((B,), 0.5 * (w_lo + w_hi), device=self.device)
        default_half = max(
            cfg.gripper_close_val,
            min(cfg.gripper_open_val, 0.25 * (w_lo + w_hi)),
        )
        self._gripper_close_target = torch.full(
            (B, n_grip), default_half, device=self.device
        )
        self._contact_left_local = torch.zeros(B, 3, device=self.device)
        self._contact_right_local = torch.zeros(B, 3, device=self.device)
        self._jaw_axis_local = torch.zeros(B, 3, device=self.device)
        self._jaw_axis_local[:, 1] = 1.0  # default jaw along +y
        if self._hand_tuck_idx:
            tuck_pose = {
                "left_zero_joint": 0.0,
                "left_five_joint":  1.0,
                "left_six_joint":   0.52,
            }
            self._hand_tuck = torch.tensor(
                [tuck_pose[n] for n in self._hand_tuck_names], device=self.device
            )

        self._grasp_marker = None
        if cfg.visualize_grasp_point:
            r = cfg.grasp_marker_radius_m
            # Dict order == marker_indices in _update_grasp_markers.
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/grasp_targets",
                markers={
                    "grasp_point": sim_utils.SphereCfg(  # 0 yellow
                        radius=r,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.85, 0.0),
                        ),
                    ),
                    "contact_left": sim_utils.SphereCfg(  # 1 orange
                        radius=r * 0.85,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.35, 0.05),
                        ),
                    ),
                    "contact_right": sim_utils.SphereCfg(  # 2 cyan
                        radius=r * 0.85,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.05, 0.9, 1.0),
                        ),
                    ),
                    "palm_ik": sim_utils.SphereCfg(  # 3 magenta
                        radius=r * 0.75,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.95, 0.2, 0.95),
                        ),
                    ),
                    "palm_actual": sim_utils.SphereCfg(  # 4 blue
                        radius=r * 0.7,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.55, 1.0),
                        ),
                    ),
                    "finger_mid": sim_utils.SphereCfg(  # 5 green
                        radius=r * 0.55,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.9, 0.3),
                        ),
                    ),
                },
            )
            self._grasp_marker = VisualizationMarkers(marker_cfg)

        self._pc_marker = None
        if cfg.visualize_object_pc or cfg.visualize_grasp_point:
            self._pc_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/object_pc",
                    markers={
                        "pc": sim_utils.SphereCfg(
                            radius=cfg.grasp_marker_radius_m * 0.22,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.75, 0.75, 0.8),
                            ),
                        ),
                    },
                )
            )

    def _setup_scene(self):
        # Resolve the active shape set FIRST — everything else (scene spawn,
        # PC preload in __init__, domain randomization bounds) depends on it.
        if self.cfg.eval_object_mode:
            self._shape_names = ycb_shape_names("eval")
            if not self._shape_names:
                print("[GraspPoseEnv] WARNING: eval_object_mode=True but no eval real "
                      "objects found — falling back to procedural shapes.")
                self._shape_names = list(PROCEDURAL_SHAPE_NAMES)
        else:
            families = list(PROCEDURAL_SHAPE_NAMES) + (
                ycb_shape_names("train") if self.cfg.use_real_objects else []
            )
            # Each family contributes instance 000 plus its mutated variants
            # (cfg.variants_per_family); every instance is a separately sampled shape.
            self._shape_names = expand_variants(families, self.cfg.variants_per_family)
            print(f"[GraspPoseEnv] {len(self._shape_names)} shapes from {len(families)} families "
                  f"(variants_per_family={self.cfg.variants_per_family or 'all'})")
        self._num_shapes = len(self._shape_names)

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._table = RigidObject(self.cfg.table)
        self.scene.rigid_objects["table"] = self._table

        # All shapes live in ONE RigidObjectCollection (one PhysX view): object
        # index i == self._shape_names[i]. Every env holds every shape; only
        # self._env_shape[env] is on the table, the rest are parked underground.
        # Reads/writes are batched over (env, shape) — with variants there are
        # hundreds of shapes, and per-shape Python loops made each physics step
        # cost hundreds of GPU syncs and PhysX calls.
        obj_cfgs = {}
        for i, name in enumerate(self._shape_names):
            base = getattr(self.cfg, f"object_{name}")
            obj_cfgs[prim_name(name)] = base.replace(
                init_state=base.init_state.replace(pos=tuple(self._park_offset(i).tolist()))
            )
        self._objects = RigidObjectCollection(RigidObjectCollectionCfg(rigid_objects=obj_cfgs))
        self.scene.rigid_object_collections["objects"] = self._objects
        self._park_pos = torch.stack(
            [self._park_offset(i) for i in range(self._num_shapes)]
        )  # (N, 3) env-local

        self._finger_contact = ContactSensor(
            ContactSensorCfg(
                prim_path=self.cfg.finger_contact_prim_path,
                history_length=0,
                track_air_time=False,
                # Only the spawned shapes: unspawned variants would match no prims.
                filter_prim_paths_expr=[
                    p for p in self.cfg.finger_contact_filter_paths
                    if p.rsplit("/", 1)[-1] in {prim_name(n) for n in self._shape_names}
                ],
            )
        )
        self.scene.sensors["finger_contact"] = self._finger_contact

        # Optional: rendered depth camera for viewpoint-diverse PC observations.
        # Camera is placed at a random pose around each object per episode.
        # Must be added before clone_environments so each env gets its own prim.
        if self.cfg.use_camera_pc:
            self._cam_sensor = TiledCamera(
                TiledCameraCfg(
                    prim_path="/World/envs/env_.*/GraspCam",
                    update_period=0,
                    history_length=1,
                    data_types=["distance_to_image_plane"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=self.cfg.camera_depth_clip,
                    ),
                    width=self.cfg.camera_width,
                    height=self.cfg.camera_height,
                )
            )
            self.scene.sensors["grasp_cam"] = self._cam_sensor
        else:
            self._cam_sensor = None

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    # ── Episode reset ──────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)
        n = len(env_ids)

        # 1. Reset robot to home pose
        joint_pos = self._home_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._arm_q_des[env_ids] = self._home_joint_pos[env_ids][:, self._arm_dof_idx]

        # 2. Pick object shape(s) for each env
        if self.cfg.eval_cycle_shapes:
            n_shapes = min(self.cfg.eval_num_shapes, self._num_shapes)
            shape_ids = (
                self._eval_shape_step + torch.arange(n, device=self.device)
            ) % n_shapes
            self._env_shape[env_ids] = shape_ids
            self._eval_shape_step += n
        else:
            self._env_shape[env_ids] = torch.randint(0, self._num_shapes, (n,), device=self.device)

        # Per-episode choice of camera-rendered vs. fast pre-loaded PC path.
        if self._cam_sensor is not None:
            self._use_camera_this_ep[env_ids] = (
                torch.rand(n, device=self.device) < self.cfg.camera_pc_prob
            )

        # 3. Spawn chosen object on table, park others underground
        if self.cfg.randomize_object_spawn:
            spawn_x = torch.empty(n, device=self.device).uniform_(*self.cfg.spawn_x_range)
            spawn_y = torch.empty(n, device=self.device).uniform_(*self.cfg.spawn_y_range)
            spawn_yaw = torch.empty(n, device=self.device).uniform_(*self.cfg.spawn_yaw_range)
            half = spawn_yaw * 0.5
            quat = torch.zeros(n, 4, device=self.device)
            quat[:, 0] = half.cos()
            quat[:, 3] = half.sin()
        else:
            cx = 0.5 * sum(self.cfg.spawn_x_range)
            cy = 0.5 * sum(self.cfg.spawn_y_range)
            spawn_x = torch.full((n,), cx, device=self.device)
            spawn_y = torch.full((n,), cy, device=self.device)
            quat = torch.zeros(n, 4, device=self.device)
            quat[:, 0] = 1.0
        # 3b. Replayed failures override the random draw (shape, xy, orientation);
        # z still comes from spawn_z_offset + settling, as for any episode.
        replay = self._replay_pending[env_ids]
        if replay.any():
            ids = env_ids[replay]
            self._env_shape[ids] = self._replay_shape[ids]
            spawn_x[replay] = self._replay_xy[ids, 0]
            spawn_y[replay] = self._replay_xy[ids, 1]
            quat[replay] = self._replay_quat[ids]
            self._replay_pending[ids] = False

        spawn_z = torch.full((n,), self.cfg.table_surface_z + self.cfg.spawn_z_offset, device=self.device)
        self._spawn_z[env_ids] = spawn_z   # record for reward computation

        # Park every shape underground, each in its own grid cell (stacked at one
        # point, parked meshes collide with each other, which with variants
        # overflows PhysX's GPU contact patch buffer), then place the chosen one.
        state = torch.zeros(n, 13, device=self.device)
        state[:, 0] = spawn_x
        state[:, 1] = spawn_y
        state[:, 2] = spawn_z
        state[:, :3] += self.scene.env_origins[env_ids]
        state[:, 3:7] = quat
        self._write_active_obj_state(state, env_ids)

        self._cache_object_anchors()
        self._spawn_z[env_ids] = self._obj_anchor[env_ids, 2]
        if self.cfg.settle_steps > 0:
            self._settle_physics()

        # Per-episode task assignment and auxiliary state reset.
        if self.cfg.task_mode_prob_place > 0.0:
            self._task_mode[env_ids] = (
                torch.rand(n, device=self.device) < self.cfg.task_mode_prob_place
            ).long()
        else:
            self._task_mode[env_ids] = 0

        goal_x = torch.empty(n, device=self.device).uniform_(*self.cfg.place_goal_x_range)
        goal_y = torch.empty(n, device=self.device).uniform_(*self.cfg.place_goal_y_range)
        goal_z = torch.full(
            (n,), self.cfg.table_surface_z + self.cfg.place_height_above_table,
            device=self.device,
        )
        self._place_goal_w[env_ids] = (
            torch.stack([goal_x, goal_y, goal_z], dim=-1)
            + self.scene.env_origins[env_ids]
        )
        self._last_hold_obj_z[env_ids] = self._spawn_z[env_ids]
        self._lift_had_contact[env_ids] = False

        # Sample random wrist orientation for place-mode episodes.
        # Each axis is independently enabled (50/50) so training sees all combinations:
        # neither rotates, only roll, only tilt, or both.  The destination container
        # has no USD asset — rotation is parameterised only, not physically simulated.
        self._place_wrist_roll[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.place_wrist_roll_range
        )
        self._place_wrist_tilt[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.place_wrist_tilt_range
        )
        self._place_roll_active[env_ids] = torch.rand(n, device=self.device) < 0.5
        self._place_tilt_active[env_ids] = torch.rand(n, device=self.device) < 0.5

        # Randomise camera positions for the reset envs and force one render
        # so _get_observations sees fresh depth at the new pose.
        if self._cam_sensor is not None:
            obj_pos = self._obj_anchor[env_ids, :3]      # world frame
            cam_pos, cam_quat = self._random_camera_poses(env_ids, obj_pos)
            self._cam_pos_w[env_ids]  = cam_pos
            self._cam_quat_w[env_ids] = cam_quat
            # Update ALL cameras at once (TiledCamera requires full-tensor update).
            self._cam_sensor.set_world_poses(
                self._cam_pos_w, self._cam_quat_w, convention="opengl"
            )
            # Force a render step so the buffer is current when _get_observations runs.
            self.scene.write_data_to_sim()
            self.sim.step(render=True)
            self.scene.update(dt=self.physics_dt)

    _PARK_SPACING_M = 0.4   # > largest scaled object diagonal (11 cm × 1.5 scale, +15% jitter)
    _PARK_GRID = 8          # cells per side per layer

    def _park_offset(self, shape_idx: int) -> torch.Tensor:
        """Env-local parking position for an inactive shape (8×8 grid per layer
        at z=-20, layers stacked downward). Cross-env collisions are filtered,
        so the grid may extend past env_spacing."""
        g, s = self._PARK_GRID, self._PARK_SPACING_M
        layer, cell = divmod(shape_idx, g * g)
        row, col = divmod(cell, g)
        return torch.tensor(
            [(col - (g - 1) / 2) * s, (row - (g - 1) / 2) * s, -20.0 - layer * s],
            device=self.device,
        )

    def queue_replays(self, env_ids, shape_ids, xy, quat_wxyz):
        """Make the next reset of each env in env_ids re-spawn a given case.

        Used by training/vr_failures.py to re-insert failed grasps into training.
        xy is relative to the env origin; quat is the settled object orientation.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._replay_shape[ids] = torch.as_tensor(shape_ids, dtype=torch.long, device=self.device)
        self._replay_xy[ids] = torch.as_tensor(xy, dtype=torch.float32, device=self.device)
        self._replay_quat[ids] = torch.as_tensor(quat_wxyz, dtype=torch.float32, device=self.device)
        self._replay_pending[ids] = True

    def _settle_physics(self, n_steps: int | None = None):
        """Let spawned objects find their true resting height before the policy acts.

        Vertical-only settle: each step we resolve physics (so a penetrating object
        pops up to rest on the table), then pin the object's x/y/orientation back to
        the spawn pose and zero its velocity. This finds the correct resting z
        without letting the object roll/drift laterally away from the spawn point.
        """
        n = self.cfg.settle_steps if n_steps is None else n_steps
        if n <= 0:
            return
        q_open = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
        for _ in range(n):
            self._robot.set_joint_position_target(
                self._home_joint_pos[:, self._arm_dof_idx], joint_ids=self._arm_dof_idx,
            )
            self._robot.set_joint_position_target(q_open, joint_ids=self._grip_dof_idx)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self._constrain_settle_xy()
        # Re-record spawn height after settle so reward baseline is accurate.
        self._spawn_z[:] = self._get_active_obj_pos()[:, 2]
        self._cache_object_anchors()

    def _constrain_settle_xy(self):
        """Keep active objects at spawn x/y/orientation; let physics update z only."""
        state = self._active_obj_state().clone()
        anchor = self._obj_anchor
        state[:, 0] = anchor[:, 0]      # pin x
        state[:, 1] = anchor[:, 1]      # pin y
        state[:, 3:7] = anchor[:, 3:7]  # pin orientation
        state[:, 7:] = 0.0              # zero velocity
        self._write_active_obj_state(state, self._all_env_ids)

    def _cache_object_anchors(self):
        """Store settled root state of each env's active object."""
        self._obj_anchor[:] = self._active_obj_state()

    def _active_obj_state(self) -> torch.Tensor:
        """(B, 13) world root state [pos, quat wxyz, lin vel, ang vel] of each env's active shape."""
        return self._objects.data.object_state_w[self._all_env_ids, self._env_shape]

    def _write_active_obj_state(self, state: torch.Tensor, env_ids: torch.Tensor):
        """Set the active shape of each env in env_ids to state (n, 13), and hold
        every other shape of those envs still at its parking cell — one batched
        write for all (env, shape) pairs."""
        full = self._park_state_w[env_ids].clone()                    # (n, N, 13)
        full[torch.arange(len(env_ids), device=self.device), self._env_shape[env_ids]] = state
        self._objects.write_object_state_to_sim(full, env_ids=env_ids)

    def _hold_hand_tuck(self):
        """Keep optional tucked joints fixed (unused on Franka parallel gripper)."""
        if not self._hand_tuck_idx:
            return
        q = self._hand_tuck.unsqueeze(0).expand(self.num_envs, -1)
        self._robot.set_joint_position_target(q, joint_ids=self._hand_tuck_idx)

    def _hold_idle_joints(self):
        """Lock uncontrolled joints at home (none on Franka — all joints driven)."""
        self._hold_hand_tuck()
        if not self._idle_dof_idx:
            return
        self._robot.set_joint_position_target(
            self._home_joint_pos[:, self._idle_dof_idx],
            joint_ids=self._idle_dof_idx,
        )

    def _anchor_objects(self):
        """Keep active objects fixed during approach/close so IK targets stay stable."""
        self._pin_objects()

    def _sync_grasped_object(self):
        """Move the object with the hand only for envs with live finger contact."""
        in_contact = self._fingers_in_contact()
        self._grasp_locked = in_contact
        if not in_contact.any():
            return
        ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
        env_ids = in_contact.nonzero(as_tuple=False).squeeze(-1)
        pinned = self._obj_anchor[env_ids].clone()
        pinned[:, :3] = ee_w[env_ids] + self._grasp_offset[env_ids]
        pinned[:, 7:] = 0.0
        self._write_active_obj_state(pinned, env_ids)
        self._obj_anchor[env_ids] = pinned

    def _pin_objects(self):
        pinned = self._obj_anchor.clone()
        pinned[:, 7:] = 0.0
        self._write_active_obj_state(pinned, self._all_env_ids)

    # ── One agent step: store grasp target, reset execution counter ───────────

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        actions: (B, 6) in tanh space [-1, 1].
          two_point_grasp (default):
            [:3] → contact point 1 (object-local)
            [3:] → contact point 2 (object-local)
            center / width / jaw axis are derived; approach stays top-down.
          legacy (two_point_grasp=False):
            [:3] grasp center, [3] tilt, [4] roll, [5] width
        """
        x_lo, x_hi = self.cfg.grasp_x_bounds
        y_lo, y_hi = self.cfg.grasp_y_bounds
        z_lo, z_hi = self.cfg.grasp_z_bounds
        w_lo, w_hi = self.cfg.grasp_width_bounds

        a = actions.clamp(-1, 1)

        def _denorm_xyz(ax, ay, az):
            gx = (ax + 1) / 2 * (x_hi - x_lo) + x_lo
            gy = (ay + 1) / 2 * (y_hi - y_lo) + y_lo
            gz = (az + 1) / 2 * (z_hi - z_lo) + z_lo
            return torch.stack([gx, gy, gz], dim=-1)

        def _undo_pc_yaw(pts: torch.Tensor) -> torch.Tensor:
            if not self.cfg.pc_augment_yaw:
                return pts
            inv_yaw = -self._pc_aug_yaw
            cos_y, sin_y = inv_yaw.cos(), inv_yaw.sin()
            x_r = cos_y * pts[:, 0] - sin_y * pts[:, 1]
            y_r = sin_y * pts[:, 0] + cos_y * pts[:, 1]
            return torch.stack([x_r, y_r, pts[:, 2]], dim=-1)

        if self.cfg.use_fixed_grasp_contacts:
            c1 = torch.tensor(self.cfg.fixed_contact_1, device=self.device).expand(
                self.num_envs, -1
            )
            c2 = torch.tensor(self.cfg.fixed_contact_2, device=self.device).expand(
                self.num_envs, -1
            )
            if self.cfg.project_grasp_to_pc:
                c1, c2, mid, width, jaw = self._project_two_contacts_to_pc(c1, c2)
                width = width.clamp(
                    min=max(w_lo, self.cfg.two_point_min_pinch_width_m), max=w_hi
                )
                self._contact_left_local = c1
                self._contact_right_local = c2
                self._grasp_target = mid
                self._grasp_width = width
                self._jaw_axis_local = jaw
                self._tilt_target.zero_()
            else:
                self._set_two_point_contacts_local(c1, c2)
            self._last_surface_contact_reward.zero_()
        elif self.cfg.two_point_grasp:
            c1 = _undo_pc_yaw(_denorm_xyz(a[:, 0], a[:, 1], a[:, 2]))
            c2 = _undo_pc_yaw(_denorm_xyz(a[:, 3], a[:, 4], a[:, 5]))
            if self.cfg.project_grasp_to_pc:
                c1s, c2s, mid, width, jaw = self._project_two_contacts_to_pc(c1, c2)
            else:
                c1s, c2s, mid, width, jaw = self._enforce_min_contact_separation(c1, c2)
            self._last_surface_contact_reward = self._surface_contact_reward(c1s, c2s)
            width = width.clamp(min=max(w_lo, self.cfg.two_point_min_pinch_width_m), max=w_hi)
            self._contact_left_local = c1s
            self._contact_right_local = c2s
            self._grasp_target = mid
            self._grasp_width = width
            self._jaw_axis_local = jaw
            self._tilt_target.zero_()
        else:
            t_lo, t_hi = self.cfg.grasp_tilt_bounds
            r_lo, r_hi = self.cfg.grasp_roll_bounds
            grasp_local = _undo_pc_yaw(_denorm_xyz(a[:, 0], a[:, 1], a[:, 2]))
            if self.cfg.project_grasp_to_pc:
                grasp_local = self._project_local_to_surface(grasp_local)
            self._grasp_target = grasp_local
            self._tilt_target = (a[:, 3] + 1) / 2 * (t_hi - t_lo) + t_lo
            self._roll_target = (a[:, 4] + 1) / 2 * (r_hi - r_lo) + r_lo
            # 5D Path A: fixed mid width. 6D legacy: policy predicts width.
            if a.shape[-1] >= 6:
                self._grasp_width = (a[:, 5] + 1) / 2 * (w_hi - w_lo) + w_lo
            else:
                self._grasp_width = torch.full(
                    (self.num_envs,), 0.5 * (w_lo + w_hi), device=self.device
                )
            half = (self._grasp_width / 2).unsqueeze(-1)
            self._jaw_axis_local = torch.tensor(
                [0.0, 1.0, 0.0], device=self.device
            ).expand(self.num_envs, -1).clone()
            self._contact_left_local = self._grasp_target - self._jaw_axis_local * half
            self._contact_right_local = self._grasp_target + self._jaw_axis_local * half

        # Freeze object pose at decision time → stable IK targets throughout episode.
        self._grasp_origin_w = self._obj_anchor[:, :3].clone()
        self._grasp_origin_quat = self._obj_anchor[:, 3:7].clone()

        if self.cfg.two_point_grasp:
            # Lock jaw *axis sign* once (no live L/R swap). j7 is solved each
            # step from current hand pose so world yaw tracks as the arm moves.
            self._lock_two_point_jaw_sign()
            self._update_roll_target_from_hand()

        # Pinch: close slightly inside geometric half-width (solid object blocks further).
        squeeze = self.cfg.gripper_squeeze_per_finger_m
        per_finger = (self._grasp_width / 2 - squeeze).clamp(
            self.cfg.gripper_close_val, self.cfg.gripper_open_val
        )
        self._gripper_close_target = per_finger.unsqueeze(-1).expand(self.num_envs, -1)

        self._grasp_locked[:] = False
        self._lift_had_contact[:] = False
        self._approach_arrived[:] = False
        self._approach_wait = 0
        self._pinch_started = False
        self._pinch_close_step = 0
        self._ik_reach_logged = False
        self._exec_step = 0
        self._J_cache = None
        self._J_pos_cache = None
        self._J_finger_cache = None
        self._ik_call = 0
        # Start every episode with the gripper fully OPEN (pinch only after arrive).
        q_open = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
        q_all = self._robot.data.joint_pos.clone()
        q_all[:, self._grip_dof_idx] = q_open
        self._robot.write_joint_state_to_sim(q_all, torch.zeros_like(q_all))
        self._robot.set_joint_position_target(q_open, joint_ids=self._grip_dof_idx)
        self._update_grasp_markers()

    def _set_two_point_contacts_local(self, c1: torch.Tensor, c2: torch.Tensor):
        """Set grasp targets from object-local contact pair (no PC snap / policy)."""
        w_lo, w_hi = self.cfg.grasp_width_bounds
        delta = c2 - c1
        mid = 0.5 * (c1 + c2)
        width = delta.norm(dim=-1).clamp(
            min=max(w_lo, self.cfg.two_point_min_pinch_width_m), max=w_hi
        )
        jaw = self._two_point_jaw_axis(delta)
        self._contact_left_local = c1
        self._contact_right_local = c2
        self._grasp_target = mid
        self._grasp_width = width
        self._jaw_axis_local = jaw
        self._tilt_target.zero_()
        self._roll_target = torch.atan2(jaw[:, 1], jaw[:, 0])

    def _load_mesh_proximity(self) -> list | None:
        """Per-shape trimesh ProximityQuery for exact surface projection."""
        try:
            import trimesh
            from trimesh.proximity import ProximityQuery
        except ImportError:
            print("[GraspPoseEnv] trimesh unavailable — contact projection uses PC snap only")
            return None
        prox: list = []
        self._obj_meshes = []
        n_missing = 0
        for name in self._shape_names:
            obj_path = shape_asset(name, ".obj")
            if not obj_path.exists():
                prox.append(None)
                self._obj_meshes.append(None)
                n_missing += 1
                continue
            mesh = trimesh.load(str(obj_path), process=False)
            if self.cfg.object_scale != 1.0:
                mesh.apply_scale(self.cfg.object_scale)
            self._obj_meshes.append(mesh)
            prox.append(ProximityQuery(mesh))
        if n_missing:
            print(
                f"[GraspPoseEnv] WARNING: {n_missing}/{len(self._shape_names)} shapes "
                "missing their .obj — those use PC snap for contact projection"
            )
        return prox

    def _project_local_to_pc(self, local: torch.Tensor) -> torch.Tensor:
        """Snap an object-local point onto the nearest preloaded PC sample."""
        pc = self._obj_pcs[self._env_shape]  # (B, 512, 3), already scaled
        d = (pc - local.unsqueeze(1)).norm(dim=-1)
        idx = d.argmin(dim=-1)
        b = torch.arange(self.num_envs, device=self.device)
        return pc[b, idx]

    def _project_local_to_surface(self, local: torch.Tensor) -> torch.Tensor:
        """Snap onto mesh surface (top-down ray when enabled, else closest point)."""
        if self._mesh_proximity is None:
            return self._project_local_to_pc(local)
        if self.cfg.project_grasp_ray_down and self.cfg.approach_mode == "top":
            return self._project_local_ray_down(local)
        out = local.clone()
        pts_np = local.detach().cpu().numpy()
        shape_np = self._env_shape.detach().cpu().numpy()
        for sid in np.unique(shape_np):
            mask = shape_np == sid
            pq = self._mesh_proximity[int(sid)]
            idx = torch.as_tensor(mask, device=local.device)
            if pq is None:
                pc = self._obj_pcs[int(sid)]
                sub = torch.as_tensor(pts_np[mask], device=local.device, dtype=local.dtype)
                d = (pc.unsqueeze(0) - sub.unsqueeze(1)).norm(dim=-1)
                out[idx] = pc[d.argmin(dim=-1)]
                continue
            closest, _, _ = pq.on_surface(pts_np[mask])
            out[idx] = torch.as_tensor(closest, device=local.device, dtype=local.dtype)
        return out

    def _project_local_ray_down(self, local: torch.Tensor) -> torch.Tensor:
        """Top-down: ray from (x,y,z_top) along −Z → first surface hit at that column."""
        out = local.clone()
        pts_np = local.detach().cpu().numpy()
        shape_np = self._env_shape.detach().cpu().numpy()
        ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        margin = self.cfg.surface_ray_z_margin_m
        for sid in np.unique(shape_np):
            mask = shape_np == sid
            mesh = self._obj_meshes[int(sid)]
            pq = self._mesh_proximity[int(sid)]
            idx = torch.as_tensor(mask, device=local.device)
            if mesh is None or pq is None:
                pc = self._obj_pcs[int(sid)]
                sub = torch.as_tensor(pts_np[mask], device=local.device, dtype=local.dtype)
                d = (pc.unsqueeze(0) - sub.unsqueeze(1)).norm(dim=-1)
                out[idx] = pc[d.argmin(dim=-1)]
                continue
            z_top = float(mesh.bounds[1, 2]) + margin
            hits = []
            for p in pts_np[mask]:
                origin = np.array([p[0], p[1], z_top], dtype=np.float64)
                locs, _, _ = mesh.ray.intersects_location([origin], [ray_dir])
                if len(locs) == 0:
                    closest, _, _ = pq.on_surface(p.reshape(1, 3))
                    hits.append(closest[0])
                else:
                    hits.append(locs[np.argmax(locs[:, 2])])
            out[idx] = torch.as_tensor(np.stack(hits), device=local.device, dtype=local.dtype)
        return out

    def _surface_distance_to_mesh(self, pts: torch.Tensor) -> torch.Tensor:
        """Per-env distance from object-local points to the nearest mesh surface."""
        if self._mesh_proximity is None:
            pc = self._obj_pcs[self._env_shape]
            return (pc - pts.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
        out = torch.zeros(self.num_envs, device=self.device, dtype=pts.dtype)
        pts_np = pts.detach().cpu().numpy()
        shape_np = self._env_shape.detach().cpu().numpy()
        for sid in np.unique(shape_np):
            mask = shape_np == sid
            pq = self._mesh_proximity[int(sid)]
            idx = torch.as_tensor(mask, device=self.device)
            if pq is None:
                pc = self._obj_pcs[int(sid)]
                d = (pc.unsqueeze(0) - pts[idx].unsqueeze(1)).norm(dim=-1).min(dim=-1).values
                out[idx] = d
            else:
                _, dist, _ = pq.on_surface(pts_np[mask])
                out[idx] = torch.as_tensor(dist, device=self.device, dtype=pts.dtype)
        return out

    def _two_point_jaw_axis(self, delta: torch.Tensor) -> torch.Tensor:
        """Unit jaw axis from c2−c1; project to xy when vertical or degenerate."""
        dist = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        jaw = delta / dist
        jaw_xy = jaw.clone()
        jaw_xy[:, 2] = 0.0
        xy_norm = jaw_xy.norm(dim=-1, keepdim=True)
        jaw_xy = torch.where(
            xy_norm > 1e-4,
            jaw_xy / xy_norm.clamp(min=1e-6),
            torch.tensor([0.0, 1.0, 0.0], device=self.device).expand_as(jaw),
        )
        bad = (dist.squeeze(-1) < 1e-4) | (jaw[:, 2].abs() > 0.9)
        return torch.where(bad.unsqueeze(-1), jaw_xy, jaw)

    def _enforce_min_contact_separation(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        jaw: "torch.Tensor | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Spread contacts to min pinch width along jaw (left = −jaw, right = +jaw)."""
        min_w = self.cfg.two_point_min_pinch_width_m
        mid = 0.5 * (c1 + c2)
        delta = c2 - c1
        if jaw is None:
            jaw = self._two_point_jaw_axis(delta)
        width = torch.maximum(
            delta.norm(dim=-1),
            torch.full((self.num_envs,), min_w, device=self.device),
        )
        half = (width / 2).unsqueeze(-1)
        c_left = mid - jaw * half
        c_right = mid + jaw * half
        return c_left, c_right, mid, width, jaw

    def _project_two_contacts_to_pc(
        self, c1_raw: torch.Tensor, c2_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project each contact to mesh surface, enforce min width, re-project."""
        c1 = self._project_local_to_surface(c1_raw)
        c2 = self._project_local_to_surface(c2_raw)
        c1, c2, _, _, jaw = self._enforce_min_contact_separation(c1, c2)
        c1 = self._project_local_to_surface(c1)
        c2 = self._project_local_to_surface(c2)
        mid = 0.5 * (c1 + c2)
        delta = c2 - c1
        width = delta.norm(dim=-1)
        jaw = self._two_point_jaw_axis(delta)
        return c1, c2, mid, width, jaw

    def _local_to_world(self, local: torch.Tensor) -> torch.Tensor:
        """Rotate an object-local offset into world frame at the grasp decision pose."""
        return self._grasp_origin_w + quat_rotate(self._grasp_origin_quat, local)

    def _grasp_point_world(self) -> torch.Tensor:
        """Policy grasp point in world frame (object-local offset at decision time)."""
        return self._local_to_world(self._grasp_target)

    def _palm_actual_w(self) -> torch.Tensor:
        return self._robot.data.body_pos_w[:, self._ee_body_idx, :3]

    def _finger_midpoint_w(self) -> torch.Tensor:
        left = self._robot.data.body_pos_w[:, self._left_finger_idx, :3]
        right = self._robot.data.body_pos_w[:, self._right_finger_idx, :3]
        return 0.5 * (left + right)

    def _jaw_axis_w(self) -> torch.Tensor:
        """Unit vector along gripper jaw opening (left→right) in world frame."""
        if self.cfg.two_point_grasp:
            return quat_rotate(self._grasp_origin_quat, self._jaw_axis_local)

        if not self.cfg.ik_orient_use_grasp_tilt and not self.cfg.ik_orient_yaw_to_object:
            return torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1)

        if self.cfg.ik_orient_use_grasp_tilt:
            approach = self._grasp_approach_dir_w()
        else:
            palm_w = self._approach_target_w()
            grasp_w = self._grasp_point_world()
            approach = grasp_w - palm_w
            approach = approach / approach.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        z_hand = -approach

        if self.cfg.ik_orient_yaw_to_object:
            grasp_w = self._grasp_point_world()
            base_xy = self._robot.data.root_pos_w[:, :2]
            to_obj = grasp_w[:, :2] - base_xy
            to_obj_h = to_obj / to_obj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            to_obj_h = torch.cat(
                [to_obj_h, torch.zeros(self.num_envs, 1, device=self.device)],
                dim=-1,
            )
            world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(
                self.num_envs, -1
            )
            y_hint = torch.cross(torch.cross(to_obj_h, world_up, dim=-1), z_hand, dim=-1)
        else:
            y_hint = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(
                self.num_envs, -1
            )

        if self.cfg.approach_from_policy_tilt:
            y_hint = self._rotate_vec_about_axis(y_hint, z_hand, self._roll_target)
        return y_hint / y_hint.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def _finger_contact_targets_w(
        self, clearance_m: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World targets for left/right fingers (fixed assignment, no live swap)."""
        c_l = self._local_to_world(self._contact_left_local)
        c_r = self._local_to_world(self._contact_right_local)
        up = self._two_point_approach_up_w() * clearance_m
        return c_l + up, c_r + up

    def _predicted_contacts_w(self, *, for_viz: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Contact targets in world frame (finger-assigned when two_point_grasp)."""
        if self.cfg.two_point_grasp:
            return self._finger_contact_targets_w(clearance_m=0.0)
        grasp_w = self._grasp_point_world()
        half = self._grasp_width / 2
        if for_viz:
            half = half.clamp(min=0.03)
        half = half.unsqueeze(-1)
        jaw = self._jaw_axis_w()
        return grasp_w - jaw * half, grasp_w + jaw * half

    def _update_grasp_markers(self):
        """0 yellow center · 1/2 orange/cyan contacts · 3 magenta IK · 4 blue palm · 5 green fingers."""
        if self._grasp_marker is None:
            return
        grasp_w = self._grasp_point_world()
        left_w, right_w = self._predicted_contacts_w(for_viz=True)
        ik_w = self._approach_target_w()
        palm_w = self._palm_actual_w()
        finger_w = self._finger_midpoint_w()
        translations = torch.cat([grasp_w, left_w, right_w, ik_w, palm_w, finger_w], dim=0)
        n = self.num_envs
        marker_indices = torch.cat([
            torch.zeros(n, dtype=torch.int, device=self.device),
            torch.ones(n, dtype=torch.int, device=self.device),
            torch.full((n,), 2, dtype=torch.int, device=self.device),
            torch.full((n,), 3, dtype=torch.int, device=self.device),
            torch.full((n,), 4, dtype=torch.int, device=self.device),
            torch.full((n,), 5, dtype=torch.int, device=self.device),
        ])
        self._grasp_marker.visualize(
            translations=translations,
            marker_indices=marker_indices,
        )
        self._update_pc_markers()

    def _update_pc_markers(self):
        """Grey subsample of the active object PC in world frame (debug overlay)."""
        if self._pc_marker is None:
            return
        n_show = min(self.cfg.visualize_object_pc_n, self._obj_pcs.shape[1])
        # Fixed stride so markers are stable across frames.
        step = max(1, self._obj_pcs.shape[1] // n_show)
        idx = torch.arange(0, step * n_show, step, device=self.device)[:n_show]
        pc_local = self._obj_pcs[self._env_shape][:, idx, :]  # (B, n_show, 3)
        B = self.num_envs
        origin = self._grasp_origin_w.unsqueeze(1).expand(-1, n_show, -1)
        quat = self._grasp_origin_quat.unsqueeze(1).expand(-1, n_show, -1).reshape(B * n_show, 4)
        pts = pc_local.reshape(B * n_show, 3)
        world = origin.reshape(B * n_show, 3) + quat_rotate(quat, pts)
        self._pc_marker.visualize(
            translations=world,
            marker_indices=torch.zeros(B * n_show, dtype=torch.int, device=self.device),
        )

    # ── Scripted execution (called each physics step within decimation) ────────

    def _apply_action(self):
        s = self._exec_step
        self._hold_idle_joints()

        # Order: hover+orient → descend → pinch → lift.
        T_hover = N_APPROACH
        T_desc = T_hover + N_DESCEND
        T_close = T_desc + N_CLOSE
        T_lift = T_close + N_LIFT
        T_hold = T_lift + N_HOLD
        T_trans = T_hold + N_TRANSPORT
        T_lower = T_trans + N_LOWER

        q_open = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
        hover_c = (
            self.cfg.two_point_approach_clearance_m
            if self.cfg.two_point_grasp
            else 0.0
        )

        if s < T_hover:
            # 1) Hover: reach XY first, then settle jaw yaw in place before descend.
            self._anchor_objects()
            if self.cfg.two_point_grasp:
                le, re_ = self._finger_contact_errors(clearance_m=hover_c)
                pos_ok = torch.max(le, re_) < self.cfg.approach_arrive_thresh_m
                # Looser gate for "close enough to start yaw-only" vs final arrive.
                pos_near = torch.max(le, re_) < max(
                    self.cfg.approach_arrive_thresh_m * 2.5, 0.04
                )
                jaw_ok = self._current_jaw_align() >= self.cfg.two_point_hover_jaw_align
                at_target = pos_ok & jaw_ok
            else:
                finger_tgt = self._grasp_point_world() + self._two_point_approach_up_w() * hover_c
                at_target = (
                    (self._finger_midpoint_w() - finger_tgt).norm(dim=-1)
                    < self.cfg.approach_arrive_thresh_m
                )
                pos_near = at_target
                jaw_ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if at_target.all():
                self._approach_arrived[:] = True
                q = self._robot.data.joint_pos[:, self._arm_dof_idx]
                self._robot.set_joint_position_target(q, joint_ids=self._arm_dof_idx)
                if self._exec_step < T_hover - 1:
                    self._exec_step = T_hover - 1
            elif self.cfg.two_point_grasp and bool(pos_near.all()) and not bool(jaw_ok.all()):
                # XY close enough: hold proximal joints, only servo wrist yaw (j7).
                self._yaw_only_settle()
                self._force_gripper_open(q_open)
            else:
                self._servo_fingers_to_grasp(clearance_m=hover_c)
                self._force_gripper_open(q_open)
        elif s < T_desc:
            # 2) Descend: lower onto contacts with yaw held (gripper still open).
            self._anchor_objects()
            self._do_descend()
            self._force_gripper_open(q_open)
            if s + 1 >= T_desc:
                self._record_ik_reach()
                self._ik_reach_logged = True
                self._freeze_arm()
        elif s < T_close:
            # 3) Pinch: freeze arm and close.
            if self._pinch_close_step >= N_CLOSE:
                self._hold_arm_frozen()
                self._robot.set_joint_position_target(
                    self._gripper_close_target, joint_ids=self._grip_dof_idx
                )
                if s + 1 >= T_close:
                    self._record_contact_reach(after_close=True)
            else:
                self._step_pinch_close()
                if s + 1 >= T_close:
                    self._record_contact_reach(after_close=True)
                if self._pinch_close_step < N_CLOSE:
                    self._exec_step -= 1
        elif s < T_lift:
            self._do_lift()
        elif s < T_hold:
            self._hold_pose()
        elif s < T_trans:
            self._do_transport(s - T_hold)
        elif s < T_lower:
            self._do_lower(s - T_trans)
        else:
            self._do_open(s - T_lower)

        if T_close <= s < T_hold:
            obj_z = self._get_active_obj_pos()[:, 2]
            self._last_hold_obj_z = torch.max(self._last_hold_obj_z, obj_z)
            self._lift_had_contact |= self._fingers_in_contact()

        self._exec_step += 1
        # Pin objects through hover+descend+close so IK targets stay stable.
        if s < T_close:
            self._pin_objects()
        elif (self.cfg.kinematic_grasp or self.cfg.contact_carry) and T_close <= s < T_lower:
            self._sync_grasped_object()

        if self._grasp_marker is not None:
            self._update_grasp_markers()

    def _step_pinch_close(self):
        """Freeze arm (once) and advance gripper close toward width-aware target."""
        if not self._pinch_started:
            self._freeze_arm()
            self._pinch_started = True
            self._pinch_close_step = 0
            if not self._ik_reach_logged:
                self._record_ik_reach()
                self._ik_reach_logged = True
        self._hold_arm_frozen()
        self._pinch_close_step += 1
        t_smooth = _smoothstep(self._pinch_close_step / max(N_CLOSE, 1))
        self._do_gripper(t_smooth, lock_at_end=(self._pinch_close_step >= N_CLOSE))

    def _force_gripper_open(self, q_open: torch.Tensor):
        """Keep aperture at fully-open during approach (visible, then pinch)."""
        if self.cfg.ik_write_joint_state:
            q_all = self._robot.data.joint_pos.clone()
            q_all[:, self._grip_dof_idx] = q_open
            # Preserve current arm pose already written by _ik_to this step.
            self._robot.write_joint_state_to_sim(q_all, torch.zeros_like(q_all))
        self._robot.set_joint_position_target(q_open, joint_ids=self._grip_dof_idx)

    def _approach_finger_target_w(self) -> torch.Tensor:
        """Where the open finger mid should sit at end of approach."""
        grasp_w = self._grasp_point_world()
        if self.cfg.two_point_grasp:
            return grasp_w + self._two_point_approach_up_w() * self.cfg.two_point_approach_clearance_m
        return grasp_w

    def _get_arm_jacobian(self) -> torch.Tensor:
        """End-effector Jacobian (B, 6, n_arm) from PhysX."""
        return self._robot.root_physx_view.get_jacobians()[
            :, self._ee_jac_idx, :, self._arm_dof_idx
        ]

    def _jacobian_rot_base(self) -> torch.Tensor:
        """Rotational EE Jacobian (B, 3, n_arm) in robot-base frame."""
        J_rot = self._get_arm_jacobian()[:, 3:, :]
        base_rot = self._robot.data.root_pose_w[:, 3:7]
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        return torch.bmm(base_rot_matrix, J_rot)

    def _jacobian_pos_base(self) -> torch.Tensor:
        """Translational EE Jacobian (B, 3, n_arm) in robot-base frame."""
        J_pos = self._get_arm_jacobian()[:, :3, :]
        base_rot = self._robot.data.root_pose_w[:, 3:7]
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        return torch.bmm(base_rot_matrix, J_pos)

    def _dls_delta(
        self,
        jacobian: torch.Tensor,
        error: torch.Tensor,
        dim: int,
        lam: float = 0.05,
    ) -> torch.Tensor:
        """Damped least-squares joint delta for a 3- or 6-DOF task."""
        jjt = jacobian @ jacobian.transpose(-1, -2)
        eye = torch.eye(dim, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        return (
            jacobian.transpose(-1, -2)
            @ torch.linalg.solve(jjt + lam * eye, error.unsqueeze(-1))
        ).squeeze(-1)

    def _two_point_approach_up_w(self) -> torch.Tensor:
        """Unit 'palm above grasp' direction for 2-point grasps.

        Table-top pinches approach from world +Z. The contact axis (jaw) still
        rotates freely with (c2−c1); only the palm retreat stays vertical so IK
        stays reachable (tilting the approach with the jaw previously pushed
        the palm off to the side and broke position convergence).
        """
        return torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(
            self.num_envs, -1
        )

    def _two_point_jaw_xy_w(self) -> torch.Tensor:
        """Jaw axis projected to the table plane (for top-down wrist yaw)."""
        jaw = self._jaw_axis_w().clone()
        jaw[:, 2] = 0.0
        n = jaw.norm(dim=-1, keepdim=True)
        return torch.where(
            n > 1e-4,
            jaw / n.clamp(min=1e-6),
            torch.tensor([0.0, 1.0, 0.0], device=self.device).expand_as(jaw),
        )

    def _approach_target_w(self) -> torch.Tensor:
        """World-frame palm IK setpoint from the frozen grasp decision pose."""
        grasp_w = self._grasp_point_world()
        if self.cfg.two_point_grasp:
            # Palm sits along approach⊥jaw (prefer +Z), not fixed object −z.
            # Fingers hang opposite this axis toward the contact midpoint.
            return grasp_w + self._two_point_approach_up_w() * self.cfg.grasp_approach_dist_m
        if self.cfg.approach_from_policy_tilt:
            approach = self._grasp_approach_dir_w()
            return grasp_w - approach * self.cfg.grasp_approach_dist_m
        # Legacy: grasp_point_offset in object-local frame (finger below palm along −z).
        offset_w = quat_rotate(
            self._grasp_origin_quat,
            self._grasp_pt_offset.expand(self.num_envs, -1),
        )
        return grasp_w - offset_w

    def _grasp_approach_dir_local(self) -> torch.Tensor:
        """Unit approach direction (palm → grasp) in object-local frame."""
        if self.cfg.approach_from_policy_tilt:
            pitch = self._tilt_target
            cy, sy = pitch.cos(), pitch.sin()
            dir_local = torch.stack([sy, torch.zeros_like(sy), -cy], dim=-1)
        else:
            g = self._grasp_target
            s = self.cfg.ik_grasp_tilt_scale
            dir_local = torch.stack(
                [g[:, 0] * s, g[:, 1] * s, -torch.ones_like(g[:, 0])],
                dim=-1,
            )
        return dir_local / dir_local.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def _grasp_approach_dir_w(self) -> torch.Tensor:
        """Unit approach direction (palm → grasp) in world frame."""
        if self.cfg.two_point_grasp:
            # Palm → grasp is opposite the "up" retreat used for the palm target.
            return -self._two_point_approach_up_w()
        return quat_rotate(self._grasp_origin_quat, self._grasp_approach_dir_local())

    def _rotate_vec_about_axis(
        self, v: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        """Rotate v about unit axis by angle (Rodrigues), batched over envs."""
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cos_a = angle.cos().unsqueeze(-1)
        sin_a = angle.sin().unsqueeze(-1)
        return (
            v * cos_a
            + torch.cross(axis, v, dim=-1) * sin_a
            + axis * (axis * v).sum(dim=-1, keepdim=True) * (1.0 - cos_a)
        )

    def _quat_from_z_and_y_hint(
        self, z_axis: torch.Tensor, y_hint: torch.Tensor
    ) -> torch.Tensor:
        """Build EE quat with hand +Z = z_axis and jaw opening near y_hint."""
        z = z_axis / z_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        y_hint = y_hint / y_hint.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x = torch.cross(y_hint, z, dim=-1)
        x_norm = x.norm(dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=z.device).expand_as(z)
        x = torch.where(
            x_norm > 1e-4,
            x / x_norm.clamp(min=1e-6),
            torch.cross(z, fallback, dim=-1),
        )
        x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        y = torch.cross(z, x, dim=-1)
        return quat_from_matrix(torch.stack([x, y, z], dim=-1))

    def _approach_target_quat(self) -> torch.Tensor:
        """EE orientation: top-down (or tilted) approach with jaw along contact axis."""
        if self.cfg.two_point_grasp:
            # Hand +Z points toward the fingers (down onto the table); jaw = c2−c1.
            return self._quat_from_z_and_y_hint(
                -self._two_point_approach_up_w(), self._two_point_jaw_xy_w()
            )

        if not self.cfg.ik_orient_use_grasp_tilt and not self.cfg.ik_orient_yaw_to_object:
            return self._home_ee_quat.clone()

        if self.cfg.ik_orient_use_grasp_tilt:
            z_hand = -self._grasp_approach_dir_w()
        else:
            palm_w = self._approach_target_w()
            grasp_w = self._grasp_point_world()
            approach = grasp_w - palm_w
            approach = approach / approach.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            z_hand = -approach

        return self._quat_from_z_and_y_hint(z_hand, self._jaw_axis_w())

    def _freeze_arm(self):
        """Lock arm at current joint angles — used before pinch and lift."""
        self._arm_q_hold = self._robot.data.joint_pos[:, self._arm_dof_idx].clone()

    def _hold_arm_frozen(self):
        self._robot.set_joint_position_target(
            self._arm_q_hold, joint_ids=self._arm_dof_idx
        )

    def _jacobian_finger_mid_pos_base(self) -> torch.Tensor:
        """Translational Jacobian of the finger midpoint in robot-base frame."""
        J = self._robot.root_physx_view.get_jacobians()
        J_l = J[:, self._left_finger_jac_idx, :3, :][:, :, self._arm_dof_idx]
        J_r = J[:, self._right_finger_jac_idx, :3, :][:, :, self._arm_dof_idx]
        J_pos = 0.5 * (J_l + J_r)
        base_rot = self._robot.data.root_pose_w[:, 3:7]
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        return torch.bmm(base_rot_matrix, J_pos)

    def _finger_contact_errors(self, clearance_m: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-finger distance to assigned contact target (+ clearance along +Z)."""
        left_tgt, right_tgt = self._finger_contact_targets_w(clearance_m=clearance_m)
        left = self._robot.data.body_pos_w[:, self._left_finger_idx, :3]
        right = self._robot.data.body_pos_w[:, self._right_finger_idx, :3]
        return (left - left_tgt).norm(dim=-1), (right - right_tgt).norm(dim=-1)

    def _servo_fingers_to_grasp(self, clearance_m: float = 0.0):
        """Move fingertips to grasp target(s)."""
        if self.cfg.two_point_grasp:
            left_tgt, right_tgt = self._finger_contact_targets_w(clearance_m=clearance_m)
            if self.cfg.use_diff_ik and self._diff_ik is not None:
                mid_tgt = 0.5 * (left_tgt + right_tgt)
                quat_tgt = self._approach_target_quat()
                for _ in range(self.cfg.ik_approach_substeps):
                    self._diff_ik_to_pose_w(mid_tgt, quat_tgt)
                return
            # Yaw already locked at grasp decision — only servo position.
            left = self._robot.data.body_pos_w[:, self._left_finger_idx, :3]
            right = self._robot.data.body_pos_w[:, self._right_finger_idx, :3]
            palm = self._palm_actual_w()
            palm_tgt = palm + 0.5 * ((left_tgt - left) + (right_tgt - right))
            for _ in range(self.cfg.ik_approach_substeps):
                self._ik_to(palm_tgt, None)
            return
        finger_tgt = self._grasp_point_world() + self._two_point_approach_up_w() * clearance_m
        if self.cfg.use_diff_ik and self._diff_ik is not None:
            quat_tgt = self._approach_target_quat()
            for _ in range(self.cfg.ik_approach_substeps):
                self._diff_ik_to_pose_w(finger_tgt, quat_tgt)
            return
        for _ in range(self.cfg.ik_approach_substeps):
            self._ik_finger_mid_to(finger_tgt)

    def _diff_ik_hand_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current panda_hand pose in robot root frame (no TCP offset)."""
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_body_idx]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        return subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

    def _diff_ik_frame_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current TCP pose (panda_hand + offset) in robot root frame."""
        ee_pos_b, ee_quat_b = self._diff_ik_hand_pose_b()
        return combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._diff_ik_offset_pos, self._diff_ik_offset_rot
        )

    def _diff_ik_jacobian_b(self) -> torch.Tensor:
        """Geometric Jacobian of TCP (hand + offset) in robot root frame (B, 6, 7).

        Offset is expressed in the hand frame; rotate into root before the
        translational cross-product correction (Isaac Lab applies body-frame
        offset directly — wrong when hand ≠ identity).
        """
        J = self._robot.root_physx_view.get_jacobians()[
            :, self._ee_jac_idx, :, self._arm_dof_idx
        ].clone()
        base_rot = self._robot.data.root_quat_w
        R_inv = matrix_from_quat(quat_inv(base_rot))
        J[:, :3, :] = torch.bmm(R_inv, J[:, :3, :])
        J[:, 3:, :] = torch.bmm(R_inv, J[:, 3:, :])
        # Hand orientation in root → offset vector in root.
        _, ee_quat_b = self._diff_ik_hand_pose_b()
        R_ee_b = matrix_from_quat(ee_quat_b)
        offset_b = torch.bmm(R_ee_b, self._diff_ik_offset_pos.unsqueeze(-1)).squeeze(-1)
        # v_tcp = v_hand + ω × r_b
        J[:, 0:3, :] += torch.bmm(-skew_symmetric_matrix(offset_b), J[:, 3:, :])
        J[:, 3:, :] = torch.bmm(
            matrix_from_quat(self._diff_ik_offset_rot), J[:, 3:, :]
        )
        return J

    def _diff_ik_finger_pos_b(self) -> torch.Tensor:
        """Finger-midpoint position in robot root frame."""
        finger_w = self._finger_midpoint_w()
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        R_inv = matrix_from_quat(quat_inv(root_quat_w))
        return torch.bmm(R_inv, (finger_w - root_pos_w).unsqueeze(-1)).squeeze(-1)

    def _diff_ik_to_pose_w(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor):
        """DifferentialIK → joint targets.

        Position-only: drive the *finger midpoint* with its PhysX Jacobian (matches
        what we measure). Pose mode: drive panda_hand+offset TCP with corrected
        offset Jacobian.
        """
        if self._exec_step < N_APPROACH + N_DESCEND + N_CLOSE:
            self._anchor_objects()

        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )
        hand_pos_b, hand_quat_b = self._diff_ik_hand_pose_b()

        if self.cfg.diff_ik_use_orientation:
            ee_pos_b, ee_quat_b = self._diff_ik_frame_pose_b()
            cmd = torch.cat([tgt_pos_b, tgt_quat_b], dim=-1)
            jacobian = self._diff_ik_jacobian_b()
        else:
            # Position-only on the real contact frame (finger mid).
            ee_pos_b = self._diff_ik_finger_pos_b()
            ee_quat_b = hand_quat_b
            cmd = tgt_pos_b
            jacobian = self._jacobian_finger_mid_pos_base()  # (B, 3, 7)

        self._diff_ik.set_command(cmd, ee_pos_b, ee_quat_b)
        joint_pos = self._robot.data.joint_pos[:, self._arm_dof_idx]

        # Two-point top-down: pin wrist flex/tilt out of position DLS; set j7 from jaw.
        pin_two_pt = (
            self.cfg.two_point_grasp and not self.cfg.diff_ik_use_orientation
        )
        if pin_two_pt:
            jacobian = jacobian.clone()
            jacobian[:, :, self._tilt_arm_idx] = 0.0
            jacobian[:, :, self._flex_arm_idx] = 0.0
            jacobian[:, :, self._roll_arm_idx] = 0.0

        q_tgt = self._diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # Cap step size — raw DLS on large pose errors yields multi-radian jumps.
        max_step = self.cfg.diff_ik_max_joint_step
        if max_step > 0.0:
            dq = (q_tgt - joint_pos).clamp(-max_step, max_step)
            q_tgt = joint_pos + dq

        if pin_two_pt:
            ti, fi, ri = self._tilt_arm_idx, self._flex_arm_idx, self._roll_arm_idx
            lo_a = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 0]
            hi_a = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 1]
            # Keep palm top-down (j5=0, j6=home). Free j6 was tipping some reaches
            # ~90° in the table plane even with correct j7.
            self._update_roll_target_from_hand()
            q_tgt[:, ti] = torch.zeros_like(q_tgt[:, ti]).clamp(lo_a[:, ti], hi_a[:, ti])
            q_tgt[:, fi] = self._flex_home.expand(self.num_envs).clamp(lo_a[:, fi], hi_a[:, fi])
            q_tgt[:, ri] = self._roll_target.clamp(lo_a[:, ri], hi_a[:, ri])

        if not getattr(self, "_diff_ik_dbg", False):
            self._diff_ik_dbg = True
            pos_err = (self._diff_ik.ee_pos_des - ee_pos_b).norm(dim=-1)[0].item()
            print(
                f"[diff-ik-dbg] pos_err_b={pos_err*100:.1f}cm  "
                f"des={self._diff_ik.ee_pos_des[0].detach().cpu().numpy().round(3)}  "
                f"cur={ee_pos_b[0].detach().cpu().numpy().round(3)}  "
                f"dq_norm={(q_tgt-joint_pos).norm(dim=-1)[0].item():.3f}  "
                f"J_norm={jacobian.norm().item():.3f}  "
                f"mode={'pose' if self.cfg.diff_ik_use_orientation else 'finger-mid'}"
            )

        lo = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 0]
        hi = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 1]
        q_tgt = q_tgt.clamp(lo, hi)
        self._arm_q_des = q_tgt

        in_approach = self._exec_step < N_APPROACH + N_DESCEND
        do_write = self.cfg.ik_write_joint_state and (
            in_approach if self.cfg.ik_write_approach_only else True
        )
        if do_write:
            q_all = self._robot.data.joint_pos.clone()
            q_all[:, self._arm_dof_idx] = q_tgt
            if in_approach:
                q_all[:, self._grip_dof_idx] = self._gripper_open.unsqueeze(0).expand(
                    self.num_envs, -1
                )
            self._robot.write_joint_state_to_sim(q_all, torch.zeros_like(q_all))
            # Refresh FK so the next solve (or markers) see the new pose.
            if hasattr(self.sim, "forward"):
                self.sim.forward()
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._arm_dof_idx)

    def _ik_finger_mid_to(self, target_w: torch.Tensor):
        """Position IK on the finger midpoint (not the palm)."""
        if self.cfg.use_diff_ik and self._diff_ik is not None:
            self._diff_ik_to_pose_w(target_w, self._approach_target_quat())
            return
        if self._exec_step < N_APPROACH + N_DESCEND + N_CLOSE:
            self._anchor_objects()

        finger_w = self._finger_midpoint_w()
        recompute = (
            self._J_finger_cache is None
            or self._ik_call % self.cfg.ik_jacobian_interval == 0
        )
        if recompute:
            self._J_finger_cache = self._jacobian_finger_mid_pos_base()
        self._ik_call += 1

        err_w = target_w - finger_w
        base_rot = self._robot.data.root_pose_w[:, 3:7]
        R_inv = matrix_from_quat(quat_inv(base_rot))
        err_b = torch.bmm(R_inv, err_w.unsqueeze(-1)).squeeze(-1)

        J_pos = self._J_finger_cache.clone()
        ti, ri = self._tilt_arm_idx, self._roll_arm_idx
        in_approach = self._exec_step < N_APPROACH + N_DESCEND + N_CLOSE
        pin_wrist = self.cfg.pin_wrist_during_ik and self.cfg.two_point_grasp

        # Legacy path only: pin wrist when explicitly requested.
        if pin_wrist:
            J_pos[:, :, ti] = 0.0
            J_pos[:, :, ri] = 0.0

        dq = self._dls_delta(J_pos, err_b, dim=3, lam=0.008)
        if pin_wrist:
            dq[:, ti] = 0.0
            dq[:, ri] = 0.0

        alpha = (
            self.cfg.ik_approach_alpha
            if in_approach
            else self.cfg.ik_alpha
        )
        q_cur = self._robot.data.joint_pos[:, self._arm_dof_idx]
        lo = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 0]
        hi = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 1]
        q_tgt = (q_cur + alpha * dq).clamp(lo, hi)

        if pin_wrist:
            q_tgt[:, ti] = torch.zeros_like(q_tgt[:, ti]).clamp(lo[:, ti], hi[:, ti])
            q_tgt[:, ri] = self._roll_target.clamp(lo[:, ri], hi[:, ri])

        self._arm_q_des = q_tgt
        in_approach_phase = self._exec_step < N_APPROACH + N_DESCEND
        do_write = self.cfg.ik_write_joint_state and (
            in_approach_phase if self.cfg.ik_write_approach_only else True
        )
        if do_write:
            q_all = self._robot.data.joint_pos.clone()
            q_all[:, self._arm_dof_idx] = q_tgt
            if in_approach_phase:
                q_open = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
                q_all[:, self._grip_dof_idx] = q_open
            self._robot.write_joint_state_to_sim(q_all, torch.zeros_like(q_all))
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._arm_dof_idx)

    def _do_approach(self):
        """Legacy name — approach = finger midpoint to grasp."""
        clearance = (
            self.cfg.two_point_approach_clearance_m
            if self.cfg.two_point_grasp
            else 0.0
        )
        self._servo_fingers_to_grasp(clearance_m=clearance)

    def _update_approach_arrived(self):
        """Gate close on finger-mid + (two-point) per-finger contact reach."""
        finger_tgt = self._approach_finger_target_w()
        near_mid = (
            (self._finger_midpoint_w() - finger_tgt).norm(dim=-1)
            < self.cfg.approach_arrive_thresh_m
        )
        if self.cfg.two_point_grasp:
            self._record_contact_reach(after_close=False)
            finger_ok = torch.max(
                self._ik_left_contact_err, self._ik_right_contact_err
            ) < self.cfg.two_point_finger_arrive_thresh_m
            self._approach_arrived |= near_mid & finger_ok
        else:
            self._approach_arrived |= near_mid

    def _yaw_only_settle(self):
        """Hold arm pose; only update j7 toward table-plane jaw (hover yaw phase)."""
        self._update_roll_target_from_hand()
        q = self._robot.data.joint_pos[:, self._arm_dof_idx].clone()
        lo = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 0]
        hi = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 1]
        ti, fi, ri = self._tilt_arm_idx, self._flex_arm_idx, self._roll_arm_idx
        q[:, ti] = torch.zeros_like(q[:, ti]).clamp(lo[:, ti], hi[:, ti])
        q[:, fi] = self._flex_home.expand(self.num_envs).clamp(lo[:, fi], hi[:, fi])
        # Step j7 toward target (honest PD needs a finite rate).
        max_step = max(self.cfg.diff_ik_max_joint_step, 0.15)
        j7 = q[:, ri]
        j7_tgt = self._roll_target.clamp(lo[:, ri], hi[:, ri])
        q[:, ri] = j7 + (j7_tgt - j7).clamp(-max_step, max_step)
        self._arm_q_des = q
        in_approach = self._exec_step < N_APPROACH + N_DESCEND
        do_write = self.cfg.ik_write_joint_state and (
            in_approach if self.cfg.ik_write_approach_only else True
        )
        if do_write:
            q_all = self._robot.data.joint_pos.clone()
            q_all[:, self._arm_dof_idx] = q
            q_all[:, self._grip_dof_idx] = self._gripper_open.unsqueeze(0).expand(
                self.num_envs, -1
            )
            self._robot.write_joint_state_to_sim(q_all, torch.zeros_like(q_all))
            if hasattr(self.sim, "forward"):
                self.sim.forward()
        self._robot.set_joint_position_target(q, joint_ids=self._arm_dof_idx)

    def _table_yaw_error(self) -> torch.Tensor:
        """Signed angle (rad) from finger_xy → jaw_xy about world +Z."""
        left = self._robot.data.body_pos_w[:, self._left_finger_idx, :3]
        right = self._robot.data.body_pos_w[:, self._right_finger_idx, :3]
        finger_xy = (right - left).clone()
        finger_xy[:, 2] = 0.0
        n = finger_xy.norm(dim=-1, keepdim=True)
        ee_quat = self._robot.data.body_quat_w[:, self._ee_body_idx]
        hand_y = quat_rotate(
            ee_quat,
            torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1),
        ).clone()
        hand_y[:, 2] = 0.0
        hand_y = hand_y / hand_y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        finger_xy = torch.where(n > 1e-4, finger_xy / n.clamp(min=1e-6), hand_y)
        jaw = self._two_point_jaw_xy_w()
        cross_z = finger_xy[:, 0] * jaw[:, 1] - finger_xy[:, 1] * jaw[:, 0]
        dot = (finger_xy * jaw).sum(dim=-1)
        return torch.atan2(cross_z, dot)

    def _current_jaw_align(self) -> torch.Tensor:
        """|finger_xy · jaw_xy| — 1 = table-plane opening aligned with contacts."""
        left = self._robot.data.body_pos_w[:, self._left_finger_idx, :3]
        right = self._robot.data.body_pos_w[:, self._right_finger_idx, :3]
        finger_xy = (right - left).clone()
        finger_xy[:, 2] = 0.0
        finger_xy = finger_xy / finger_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        jaw = self._two_point_jaw_xy_w()
        return (finger_xy * jaw).sum(dim=-1).abs()

    def _lock_two_point_jaw_sign(self):
        """Choose jaw / −jaw once (closer to current finger opening); swap contacts if flipped."""
        jaw_w = self._two_point_jaw_xy_w()
        left = self._robot.data.body_pos_w[:, self._left_finger_idx, :3]
        right = self._robot.data.body_pos_w[:, self._right_finger_idx, :3]
        finger_xy = (right - left).clone()
        finger_xy[:, 2] = 0.0
        n = finger_xy.norm(dim=-1, keepdim=True)
        finger_xy = torch.where(
            n > 1e-4,
            finger_xy / n.clamp(min=1e-6),
            quat_rotate(
                self._robot.data.body_quat_w[:, self._ee_body_idx],
                torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1),
            ),
        )
        finger_xy = finger_xy.clone()
        finger_xy[:, 2] = 0.0
        finger_xy = finger_xy / finger_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        flip = (jaw_w * finger_xy).sum(dim=-1) < 0.0
        if not flip.any():
            return
        self._jaw_axis_local = torch.where(
            flip.unsqueeze(-1), -self._jaw_axis_local, self._jaw_axis_local
        )
        left_c, right_c = self._contact_left_local, self._contact_right_local
        self._contact_left_local = torch.where(flip.unsqueeze(-1), right_c, left_c)
        self._contact_right_local = torch.where(flip.unsqueeze(-1), left_c, right_c)

    def _update_roll_target_from_hand(self):
        """Set j7 so table-plane finger opening aligns with locked jaw.

        Uses the real finger axis (not hand +Y) and maps the world-Z yaw error
        through hand +Z so tipped wrists don't command a 90°-wrong j7.
        """
        delta_w = self._table_yaw_error()
        ee_quat = self._robot.data.body_quat_w[:, self._ee_body_idx]
        hand_z = quat_rotate(
            ee_quat,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1),
        )
        # +j7 about hand +Z; with top-down Franka hand +Z ≈ −world +Z.
        j7_delta = -delta_w * torch.sign(hand_z[:, 2]).clamp(min=-1.0)
        flat = hand_z[:, 2].abs() < 0.3
        jaw = self._two_point_jaw_xy_w()
        jaw_h = quat_rotate(quat_inv(ee_quat), jaw)
        err_h = torch.atan2(jaw_h[:, 0], jaw_h[:, 1])
        j7_delta = torch.where(flat, err_h, j7_delta)
        j7 = self._robot.data.joint_pos[:, self._arm_dof_idx][:, self._roll_arm_idx]
        self._roll_target = j7 + j7_delta

    def _lock_two_point_wrist_yaw(self):
        """Deprecated: lock jaw sign + solve j7 from current hand."""
        self._lock_two_point_jaw_sign()
        self._update_roll_target_from_hand()

    def _refresh_two_point_wrist_yaw(self):
        """Deprecated alias — prefer _update_roll_target_from_hand each IK step."""
        self._update_roll_target_from_hand()

    def _servo_two_point_contacts(self, clearance_m: float = 0.0):
        """Position-only IK: finger midpoint → grasp mid (+ clearance along +Z)."""
        finger_tgt = self._grasp_point_world() + self._two_point_approach_up_w() * clearance_m
        finger_now = self._finger_midpoint_w()
        palm_tgt = self._palm_actual_w() + (finger_tgt - finger_now)
        self._ik_to(palm_tgt, None)

    def _servo_two_point_finger_mid(self, clearance_m: float = 0.0):
        """Legacy finger-mid servo (Path A / fallback)."""
        finger_tgt = (
            self._grasp_point_world()
            + self._two_point_approach_up_w() * clearance_m
        )
        finger_now = self._finger_midpoint_w()
        palm_tgt = self._palm_actual_w() + (finger_tgt - finger_now)
        self._ik_to(palm_tgt, None)

    def _do_descend(self):
        """Pre-pinch: ramp hover → final clearance while holding jaw yaw (gripper open)."""
        if not self.cfg.two_point_grasp:
            self._hold_pose()
            return
        s_local = self._exec_step - N_APPROACH
        t = float(s_local) / max(N_DESCEND - 1, 1)
        t = min(max(t, 0.0), 1.0)
        hover = self.cfg.two_point_approach_clearance_m
        final = self.cfg.two_point_final_clearance_m
        clearance = hover * (1.0 - t) + final * t
        self._servo_fingers_to_grasp(clearance_m=clearance)

    def _record_ik_reach(self):
        """Snapshot approach-end reach errors (finger mid / palm / jaw / contacts)."""
        grasp_w = self._grasp_point_world()
        finger_w = self._finger_midpoint_w()
        palm_w = self._palm_actual_w()
        # For two-point, palm setpoint is dynamic (finger-mid servo); report the
        # geometric magenta marker for viz, but palm_err vs that is less meaningful.
        palm_tgt = self._approach_target_w()
        self._ik_finger_err = (finger_w - grasp_w).norm(dim=-1)
        self._ik_palm_err = (palm_w - palm_tgt).norm(dim=-1)
        self._ik_palm_w = palm_w.detach().clone()
        self._ik_palm_tgt_w = palm_tgt.detach().clone()
        self._ik_grasp_w = grasp_w.detach().clone()
        self._ik_finger_w = finger_w.detach().clone()

        left = self._robot.data.body_pos_w[:, self._left_finger_idx, :3]
        right = self._robot.data.body_pos_w[:, self._right_finger_idx, :3]
        if self.cfg.two_point_grasp:
            self._ik_jaw_align = self._current_jaw_align()
        else:
            finger_axis = right - left
            finger_axis = finger_axis / finger_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self._ik_jaw_align = (finger_axis * self._jaw_axis_w()).sum(dim=-1).abs()
        self._record_contact_reach(after_close=False)

    def _record_contact_reach(self, *, after_close: bool):
        """Per-finger distance to the two contact targets (optimal assignment)."""
        c_l, c_r = self._predicted_contacts_w(for_viz=False)
        left = self._robot.data.body_pos_w[:, self._left_finger_idx, :3]
        right = self._robot.data.body_pos_w[:, self._right_finger_idx, :3]
        # Match fingers→contacts with the lower total distance (handles wrist flip).
        d_ll = (left - c_l).norm(dim=-1)
        d_lr = (left - c_r).norm(dim=-1)
        d_rl = (right - c_l).norm(dim=-1)
        d_rr = (right - c_r).norm(dim=-1)
        cost_keep = d_ll + d_rr
        cost_swap = d_lr + d_rl
        keep = cost_keep <= cost_swap
        left_err = torch.where(keep, d_ll, d_lr)
        right_err = torch.where(keep, d_rr, d_rl)
        if after_close:
            self._ik_left_contact_err_closed = left_err
            self._ik_right_contact_err_closed = right_err
            self._ik_contact_err_after_close = 0.5 * (left_err + right_err)
        else:
            self._ik_left_contact_err = left_err
            self._ik_right_contact_err = right_err
    def _ik_to(
        self,
        target_w: torch.Tensor,
        target_quat: torch.Tensor | None = None,
    ):
        """Hybrid IK: reliable numeric position, optional PhysX orientation."""
        # Jacobian / anchors: keep objects fixed through approach+close+descend.
        if self._exec_step < N_APPROACH + N_DESCEND + N_CLOSE:
            self._anchor_objects()
        ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
        ee_q = self._robot.data.body_quat_w[:, self._ee_body_idx]

        recompute = (
            self._J_pos_cache is None
            or self._ik_call % self.cfg.ik_jacobian_interval == 0
        )
        if recompute:
            self._J_pos_cache = self._jacobian_pos_base()
            if target_quat is not None and self.cfg.ik_orient_alpha > 0.0:
                self._J_cache = self._jacobian_rot_base()
            if self._exec_step < N_APPROACH + N_DESCEND + N_CLOSE:
                self._anchor_objects()
        self._ik_call += 1

        # Hold pinned wrist joints fixed so position IK doesn't fight the pin.
        # CRITICAL: mask Jacobian columns *before* DLS. Zeroing dq after a full
        # solve discards the motion the solver relied on and leaves ~5 cm residual.
        pin_wrist = self.cfg.pin_wrist_during_ik and (
            self.cfg.two_point_grasp
            or (not self.cfg.two_point_grasp and self.cfg.path_a_position_only_ik)
        )

        # Jacobian is in robot-base frame — express position error there too.
        err_w = target_w - ee_w
        base_rot = self._robot.data.root_pose_w[:, 3:7]
        R_inv = matrix_from_quat(quat_inv(base_rot))
        err_b = torch.bmm(R_inv, err_w.unsqueeze(-1)).squeeze(-1)

        J_pos = self._J_pos_cache
        if pin_wrist:
            J_pos = J_pos.clone()
            J_pos[:, :, self._tilt_arm_idx] = 0.0
            J_pos[:, :, self._roll_arm_idx] = 0.0
            if self.cfg.two_point_grasp:
                J_pos[:, :, self._flex_arm_idx] = 0.0

        dq = self._dls_delta(J_pos, err_b, dim=3, lam=0.02)

        if target_quat is not None and self.cfg.ik_orient_alpha > 0.0:
            _, rot_err = compute_pose_error(
                ee_w, ee_q, target_w, target_quat, rot_error_type="axis_angle"
            )
            rot_err = rot_err * self.cfg.ik_orient_weight
            J_rot = self._J_cache
            if pin_wrist:
                J_rot = J_rot.clone()
                J_rot[:, :, self._tilt_arm_idx] = 0.0
                J_rot[:, :, self._roll_arm_idx] = 0.0
                if self.cfg.two_point_grasp:
                    J_rot[:, :, self._flex_arm_idx] = 0.0
            dq_rot = self._dls_delta(J_rot, rot_err, dim=3, lam=0.08)
            # Two-point: allow wrist yaw earlier so jaw lines up before contact,
            # but still gate on a loose position threshold (always-on orient
            # previously blew up palm reach to ~30–40 cm).
            thresh = (
                max(self.cfg.ik_orient_pos_thresh, 0.20)
                if self.cfg.two_point_grasp
                else self.cfg.ik_orient_pos_thresh
            )
            near = (err_w.norm(dim=-1) < thresh).unsqueeze(-1)
            dq = dq + self.cfg.ik_orient_alpha * near * dq_rot

        if pin_wrist:
            dq[:, self._tilt_arm_idx] = 0.0
            dq[:, self._roll_arm_idx] = 0.0
            if self.cfg.two_point_grasp:
                dq[:, self._flex_arm_idx] = 0.0

        alpha = self.cfg.ik_alpha
        q_cur = self._robot.data.joint_pos[:, self._arm_dof_idx]
        lo = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 0]
        hi = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 1]
        q_tgt = (q_cur + alpha * dq).clamp(lo, hi)

        # Position IK. Legacy Path A may pin j5/j7. Diff-IK path leaves all joints free.
        if pin_wrist and self.cfg.two_point_grasp:
            ti, fi, ri = self._tilt_arm_idx, self._flex_arm_idx, self._roll_arm_idx
            self._update_roll_target_from_hand()
            q_tgt[:, ti] = torch.zeros_like(q_tgt[:, ti]).clamp(lo[:, ti], hi[:, ti])
            q_tgt[:, fi] = self._flex_home.expand(self.num_envs).clamp(lo[:, fi], hi[:, fi])
            q_tgt[:, ri] = self._roll_target.clamp(lo[:, ri], hi[:, ri])
        elif pin_wrist:
            ti, ri = self._tilt_arm_idx, self._roll_arm_idx
            max_t = self.cfg.path_a_max_tilt_rad
            tilt = self._tilt_target.clamp(-max_t, max_t)
            q_tgt[:, ti] = tilt.clamp(lo[:, ti], hi[:, ti])
            q_tgt[:, ri] = self._roll_target.clamp(lo[:, ri], hi[:, ri])

        self._arm_q_des = q_tgt
        # Kinematic write only during approach (reach). Close/lift use PD so the
        # gripper can pinch with real contact forces instead of teleporting.
        in_approach = self._exec_step < N_APPROACH + N_DESCEND
        do_write = self.cfg.ik_write_joint_state and (
            in_approach if self.cfg.ik_write_approach_only else True
        )
        if do_write:
            q_all = self._robot.data.joint_pos.clone()
            q_all[:, self._arm_dof_idx] = q_tgt
            if in_approach:
                # Never freeze fingers closed while reaching.
                q_all[:, self._grip_dof_idx] = self._gripper_open.unsqueeze(0).expand(
                    self.num_envs, -1
                )
            self._robot.write_joint_state_to_sim(
                q_all, torch.zeros_like(q_all)
            )
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._arm_dof_idx)

    def _hold_pose(self):
        """Keep arm frozen and gripper closed."""
        self._hold_arm_frozen()
        self._robot.set_joint_position_target(
            self._gripper_close_target, joint_ids=self._grip_dof_idx
        )

    def _do_transport(self, s_local: int):
        """Lateral move from above A to above B, with optional per-axis wrist rotation.

        For place-mode envs the arm moves to above the goal while the wrist may
        rotate to simulate depositing into an angled container.  Each episode
        independently randomises which axes rotate (roll / tilt / both / neither)
        so the policy must learn grasps resilient to any reorientation combination.
        The container has no USD asset — rotation is parameterised, not simulated.
        For lift-only envs the arm holds position and keeps the policy's orientation.
        """
        if s_local == 0:
            self._transport_start_w = (
                self._robot.data.body_pos_w[:, self._ee_body_idx, :3].clone()
            )
            # Capture the wrist configuration the policy chose (set in _pre_physics_step).
            self._transport_roll_start = self._roll_target.clone()
            self._transport_tilt_start = self._tilt_target.clone()
            # Effective end-target per axis: sampled angle if this axis is active for
            # this episode, otherwise hold the policy's own orientation (no rotation).
            place_mask = self._task_mode.bool()
            self._transport_roll_tgt = torch.where(
                place_mask & self._place_roll_active,
                self._place_wrist_roll,
                self._roll_target,    # no rotation: keep policy's jaw angle
            )
            self._transport_tilt_tgt = torch.where(
                place_mask & self._place_tilt_active,
                self._place_wrist_tilt,
                self._tilt_target,    # no rotation: keep policy's tilt angle
            )

        t   = (s_local + 1) / N_TRANSPORT
        t_s = _smoothstep(t)

        place_above = self._place_goal_w.clone()
        place_above[:, 2] = self._transport_start_w[:, 2]
        target_w = torch.where(
            self._task_mode.unsqueeze(-1).bool(),
            self._transport_start_w * (1 - t) + place_above * t,
            self._transport_start_w,
        )

        # Smoothly rotate wrist toward the episode's place targets.
        # _ik_to reads _roll_target / _tilt_target for the wrist joint overrides,
        # so updating them here drives rotation without touching position IK math.
        self._roll_target = (1 - t_s) * self._transport_roll_start + t_s * self._transport_roll_tgt
        self._tilt_target = (1 - t_s) * self._transport_tilt_start + t_s * self._transport_tilt_tgt

        self._ik_to(target_w)

    def _do_lower(self, s_local: int):
        """Descend EE from transport height to the place goal height.

        For lift-only envs the arm stays at the transport height.
        """
        t = (s_local + 1) / N_LOWER
        place_above = self._place_goal_w.clone()
        place_above[:, 2] = self._transport_start_w[:, 2]   # transport height
        target_w = torch.where(
            self._task_mode.unsqueeze(-1).bool(),
            place_above * (1 - t) + self._place_goal_w * t,
            self._transport_start_w,
        )
        self._ik_to(target_w)

    def _do_open(self, s_local: int):
        """Open gripper to release object at place location (place mode only)."""
        t = (s_local + 1) / max(N_OPEN, 1)
        self._robot.set_joint_position_target(self._arm_q_des, joint_ids=self._arm_dof_idx)
        q_open  = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
        t_per = torch.where(
            self._task_mode.bool(),
            torch.full((self.num_envs,), t, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
        ).unsqueeze(-1)
        q_tgt = (1.0 - t_per) * self._gripper_close_target + t_per * q_open
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._grip_dof_idx)

    def _do_gripper(self, t: float, lock_at_end: bool = False):
        """Interpolate gripper from OPEN → pinch width; arm stays frozen."""
        q_open = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
        q_grip = (1 - t) * q_open + t * self._gripper_close_target
        self._hold_arm_frozen()
        self._robot.set_joint_position_target(q_grip, joint_ids=self._grip_dof_idx)
        if lock_at_end:
            ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
            obj_w = self._get_active_obj_pos()
            contact = self._fingers_in_contact()
            if contact.any():
                self._grasp_offset[contact] = obj_w[contact] - ee_w[contact]
            if self.cfg.kinematic_grasp:
                grasp_pt = ee_w + self._grasp_pt_offset.expand(self.num_envs, -1)
                self._grasp_locked = (
                    (obj_w - grasp_pt).norm(dim=-1) <= self.cfg.grasp_reach_thresh
                )

    def _fingers_in_contact(self) -> torch.Tensor:
        """True per-env when enough fingertips register real contact force."""
        forces = self._finger_contact.data.net_forces_w  # (B, n_fingers, 3)
        fmag = forces.norm(dim=-1)                        # (B, n_fingers)
        n_touch = (fmag > self.cfg.contact_force_thresh).sum(dim=-1)
        return n_touch >= self.cfg.min_contact_fingers

    def _do_lift(self):
        """Lift: one-shot finger-mid IK to raised height, then joint interpolation."""
        lift_start = N_APPROACH + N_DESCEND + N_CLOSE
        if self._exec_step == lift_start:
            self._arm_q_hold = self._robot.data.joint_pos[:, self._arm_dof_idx].clone()
            lift_tgt = self._grasp_point_world().clone()
            lift_tgt[:, 2] += self.cfg.lift_height_m
            self._ik_finger_mid_to(lift_tgt)
            self._arm_q_lift_end = self._arm_q_des.clone()
            lo = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 0]
            hi = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 1]
            self._arm_q_lift_end = self._arm_q_lift_end.clamp(lo, hi)
        k = (self._exec_step - lift_start) + 1
        t = _smoothstep(k / max(N_LIFT, 1))
        q = (1.0 - t) * self._arm_q_hold + t * self._arm_q_lift_end
        self._robot.set_joint_position_target(q, joint_ids=self._arm_dof_idx)
        self._robot.set_joint_position_target(
            self._gripper_close_target, joint_ids=self._grip_dof_idx
        )

    # ── Camera helpers ────────────────────────────────────────────────────────

    def _lookat_quat(self, eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Quaternion (w,x,y,z) for a camera at `eye` looking toward `target`.

        Uses the OpenGL camera convention: +X right, +Y up, -Z forward.
        World is Z-up.  Result is suitable for TiledCamera.set_world_poses
        with convention="opengl".
        """
        fwd = target - eye                                              # (B, 3)
        fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        world_up = torch.zeros_like(fwd); world_up[:, 2] = 1.0        # Z-up world
        alt_up   = torch.zeros_like(fwd); alt_up[:, 1]   = 1.0        # fallback Y
        degenerate = fwd[:, 2].abs() > 0.98
        wup = torch.where(degenerate.unsqueeze(-1), alt_up, world_up)

        right = torch.linalg.cross(fwd, wup)
        right = right / right.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        up    = torch.linalg.cross(right, fwd)                         # camera +Y

        # Camera-to-world rotation matrix: cols = [right, up, -fwd]
        R = torch.stack([right, up, -fwd], dim=-1)                     # (B, 3, 3)

        # Shepperd's method: rotation matrix → quaternion (w, x, y, z)
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        B  = eye.shape[0]
        q  = torch.zeros(B, 4, device=eye.device, dtype=eye.dtype)

        m0 = tr > 0
        if m0.any():
            s = (tr[m0] + 1.0).sqrt() * 2
            q[m0, 0] = 0.25 * s
            q[m0, 1] = (R[m0, 2, 1] - R[m0, 1, 2]) / s
            q[m0, 2] = (R[m0, 0, 2] - R[m0, 2, 0]) / s
            q[m0, 3] = (R[m0, 1, 0] - R[m0, 0, 1]) / s

        m1 = ~m0 & (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2])
        if m1.any():
            s = (1.0 + R[m1, 0, 0] - R[m1, 1, 1] - R[m1, 2, 2]).clamp(0).sqrt() * 2
            q[m1, 0] = (R[m1, 2, 1] - R[m1, 1, 2]) / s.clamp(1e-8)
            q[m1, 1] = 0.25 * s
            q[m1, 2] = (R[m1, 0, 1] + R[m1, 1, 0]) / s.clamp(1e-8)
            q[m1, 3] = (R[m1, 0, 2] + R[m1, 2, 0]) / s.clamp(1e-8)

        m2 = ~m0 & ~m1 & (R[:, 1, 1] >= R[:, 2, 2])
        if m2.any():
            s = (1.0 + R[m2, 1, 1] - R[m2, 0, 0] - R[m2, 2, 2]).clamp(0).sqrt() * 2
            q[m2, 0] = (R[m2, 0, 2] - R[m2, 2, 0]) / s.clamp(1e-8)
            q[m2, 1] = (R[m2, 0, 1] + R[m2, 1, 0]) / s.clamp(1e-8)
            q[m2, 2] = 0.25 * s
            q[m2, 3] = (R[m2, 1, 2] + R[m2, 2, 1]) / s.clamp(1e-8)

        m3 = ~m0 & ~m1 & ~m2
        if m3.any():
            s = (1.0 + R[m3, 2, 2] - R[m3, 0, 0] - R[m3, 1, 1]).clamp(0).sqrt() * 2
            q[m3, 0] = (R[m3, 1, 0] - R[m3, 0, 1]) / s.clamp(1e-8)
            q[m3, 1] = (R[m3, 0, 2] + R[m3, 2, 0]) / s.clamp(1e-8)
            q[m3, 2] = (R[m3, 1, 2] + R[m3, 2, 1]) / s.clamp(1e-8)
            q[m3, 3] = 0.25 * s

        return q / q.norm(dim=-1, keepdim=True).clamp(1e-8)

    def _random_camera_poses(
        self, env_ids: torch.Tensor, obj_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one camera pose per env on a randomised ring around the object.

        Azimuth is uniform in [0, 2π].  Horizontal radius and height are drawn
        from the configured ranges.  The camera is oriented to look at obj_pos.

        Returns:
            cam_pos  (n, 3) — world-frame camera positions
            cam_quat (n, 4) — world-frame camera quaternions (OpenGL convention)
        """
        n = len(env_ids)
        az    = torch.empty(n, device=self.device).uniform_(0.0, 2.0 * math.pi)
        h_r   = torch.empty(n, device=self.device).uniform_(*self.cfg.camera_horizontal_dist_range)
        elev  = torch.empty(n, device=self.device).uniform_(*self.cfg.camera_height_range)

        cam_x = obj_pos[:, 0] + h_r * az.cos()
        cam_y = obj_pos[:, 1] + h_r * az.sin()
        cam_z = torch.full((n,), self.cfg.table_surface_z, device=self.device) + elev

        cam_pos  = torch.stack([cam_x, cam_y, cam_z], dim=-1)   # (n, 3)
        cam_quat = self._lookat_quat(cam_pos, obj_pos)           # (n, 4)
        return cam_pos, cam_quat

    def _depth_to_pc_local(self, depth: torch.Tensor) -> torch.Tensor:
        """Unproject depth image to a point cloud in object-local frame.

        Args:
            depth: (B, H, W) distance_to_image_plane in metres (positive, NaN = invalid).

        Returns:
            pc: (B, NUM_PC_POINTS, 3) sampled point cloud in object-local frame.

        Pipeline:
            1. Unproject each pixel → camera frame (OpenGL: X right, Y up, -Z forward).
            2. Rotate + translate → world frame using stored camera pose.
            3. Filter: valid depth AND above table surface (strips table background).
            4. Transform → object-local frame (inverse of settled object pose).
            5. Random sample NUM_PC_POINTS per env; repeat-sample if too few points.
        """
        B, H, W = depth.shape
        fov_rad = self.cfg.camera_fov_deg * (math.pi / 180.0)
        fx = W / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx                                             # square pixels
        cx, cy = W / 2.0, H / 2.0
        d_min, d_max = self.cfg.camera_depth_clip

        # Pixel grid — built once, reused across calls (move to __init__ if profiling shows cost)
        u = torch.arange(W, device=self.device, dtype=torch.float32)  # (W,)
        v = torch.arange(H, device=self.device, dtype=torch.float32)  # (H,)
        uu, vv = torch.meshgrid(u, v, indexing="xy")                  # (H, W) each

        # Unproject: OpenGL camera frame (X right, Y up, -Z forward)
        d = depth                                            # (B, H, W)
        x_c =  (uu - cx).unsqueeze(0) / fx * d             # (B, H, W)
        y_c = -(vv - cy).unsqueeze(0) / fy * d             # flip image-v → camera-Y
        z_c = -d                                            # -Z is forward in OpenGL
        pts_cam = torch.stack([x_c, y_c, z_c], dim=-1)    # (B, H, W, 3)
        pts_cam = pts_cam.reshape(B, H * W, 3)

        # Depth validity mask
        valid = (d > d_min) & (d < d_max) & d.isfinite()
        valid = valid.reshape(B, H * W)                     # (B, H*W)

        # Camera → world: rotate by stored camera quaternion, then translate
        HW  = H * W
        q_e = self._cam_quat_w.unsqueeze(1).expand(-1, HW, -1).reshape(B * HW, 4)
        p_e = self._cam_pos_w.unsqueeze(1).expand(-1, HW, -1).reshape(B * HW, 3)
        pts_world = quat_rotate(q_e, pts_cam.reshape(B * HW, 3)) + p_e
        pts_world = pts_world.reshape(B, HW, 3)             # (B, H*W, 3)

        # Strip table: only keep points above the table surface + small margin
        table_z = self.cfg.table_surface_z + 0.015          # 1.5 cm clearance
        valid   = valid & (pts_world[:, :, 2] > table_z)

        # World → object-local frame
        obj_w = self._obj_anchor[:, :3]                     # (B, 3)
        inv_q = quat_inv(self._obj_anchor[:, 3:7])          # (B, 4)
        diff  = (pts_world - obj_w.unsqueeze(1)).reshape(B * HW, 3)
        iq_e  = inv_q.unsqueeze(1).expand(-1, HW, -1).reshape(B * HW, 4)
        pts_local = quat_rotate(iq_e, diff).reshape(B, HW, 3)

        # Vectorised random sampling: high score for valid pixels, -inf otherwise
        scores = torch.where(
            valid,
            torch.rand(B, HW, device=self.device),
            torch.full((B, HW), float("-inf"), device=self.device),
        )
        _, top_idx = scores.topk(NUM_PC_POINTS, dim=-1, sorted=False)  # (B, N)
        pc = pts_local.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 3))

        # Fix envs where depth sees fewer than NUM_PC_POINTS object pixels.
        # For those envs repeat-sample from whatever valid points exist; fall back
        # to the pre-loaded PC if the camera sees nothing at all.
        n_valid = valid.sum(dim=-1)  # (B,)
        for bi in (n_valid < NUM_PC_POINTS).nonzero(as_tuple=False).view(-1):
            nv = n_valid[bi].item()
            if nv == 0:
                fallback = self._obj_pcs[self._env_shape[bi]]
                ridx = torch.randint(0, _PC_PRE_N, (NUM_PC_POINTS,), device=self.device)
                pc[bi] = fallback[ridx]
            else:
                vpts = pts_local[bi][valid[bi]]              # (nv, 3)
                ridx = torch.randint(0, int(nv), (NUM_PC_POINTS,), device=self.device)
                pc[bi] = vpts[ridx]

        return pc   # (B, NUM_PC_POINTS, 3) in object-local frame

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """Return point cloud of the current object (object-local frame)."""
        pc_flat = self._synthesize_pointcloud()   # (B, NUM_PC_POINTS * 3)
        return {"policy": pc_flat}

    def _synthesize_pointcloud(self) -> torch.Tensor:
        """Point cloud observation in object-local frame.

        Two source paths, mixed PER-ENV PER-EPISODE (self._use_camera_this_ep,
        drawn in _reset_idx from Bernoulli(camera_pc_prob)) rather than one
        global on/off switch — both the idealized canonical-mesh-sample
        distribution and the rendered-depth distribution (real occlusion/
        self-shadowing from a randomised viewpoint, sim→real robustness) are
        seen within the same training run:
        • camera path — rendered depth from this episode's randomised camera
          position (requires --enable_cameras at launch).
        • fast path   — sub-sample pre-loaded 512-pt PC + optional random Z
          rotation (pc_augment_yaw).

        Sensor-realistic noise (Gaussian + dropout + outliers, severity
        domain-randomized per episode) is then applied identically to both
        paths — see grasping.pointcloud_utils.add_sensor_noise.

        Returns (B, NUM_PC_POINTS * 3) flattened, always in object-local frame.
        _pc_aug_yaw is zeroed for envs using the camera path this episode so
        _pre_physics_step's inverse rotation is a no-op for them (the
        projection pipeline already gives true local coords).
        """
        B = self.num_envs

        # Fast pre-loaded-PC path — always computed (cheap); used outright when
        # the camera sensor is disabled, or blended per-env otherwise.
        shape_pcs = self._obj_pcs[self._env_shape]     # (B, 512, 3)
        idx       = torch.randint(0, _PC_PRE_N, (B, NUM_PC_POINTS), device=self.device)
        pc_fast   = shape_pcs.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

        if self.cfg.pc_augment_yaw:
            aug_yaw = torch.empty(B, device=self.device).uniform_(-math.pi, math.pi)
            self._pc_aug_yaw = aug_yaw
            cos_y = aug_yaw.cos().unsqueeze(-1)   # (B, 1)
            sin_y = aug_yaw.sin().unsqueeze(-1)
            x_new = cos_y * pc_fast[:, :, 0] - sin_y * pc_fast[:, :, 1]
            y_new = sin_y * pc_fast[:, :, 0] + cos_y * pc_fast[:, :, 1]
            pc_fast = torch.stack([x_new, y_new, pc_fast[:, :, 2]], dim=-1)
        else:
            self._pc_aug_yaw.zero_()

        if self.cfg.use_camera_pc and self._cam_sensor is not None:
            # Rendered depth branch — camera was already moved and rendered in _reset_idx.
            depth = self._cam_sensor.data.output["distance_to_image_plane"]  # (B, H, W, 1)
            depth = depth[..., 0]                                             # (B, H, W)
            pc_cam = self._depth_to_pc_local(depth)                          # (B, N, 3)

            use_cam = self._use_camera_this_ep.view(B, 1, 1)
            pc_local = torch.where(use_cam, pc_cam, pc_fast)
            # Camera-path envs already give true local coords — zero their yaw
            # augment so _pre_physics_step's inverse rotation is a no-op for them.
            self._pc_aug_yaw = torch.where(
                self._use_camera_this_ep, torch.zeros_like(self._pc_aug_yaw), self._pc_aug_yaw
            )
        else:
            pc_local = pc_fast

        g_std  = torch.empty(B, device=self.device).uniform_(*self.cfg.pc_noise_range_m)
        drop_p = torch.empty(B, device=self.device).uniform_(*self.cfg.pc_dropout_frac_range)
        out_p  = torch.empty(B, device=self.device).uniform_(*self.cfg.pc_outlier_frac_range)
        pc_local = add_sensor_noise(pc_local, gaussian_std=g_std, dropout_frac=drop_p, outlier_frac=out_p)

        return pc_local.view(B, -1)   # (B, N*3) in object-local frame

    def _get_active_obj_pos(self) -> torch.Tensor:
        """Return world-frame position of the active object in each env."""
        return self._active_obj_state()[:, :3]

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _surface_contact_reward(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """Reward raw (pre-projection) contacts on the object surface with spread."""
        d1 = self._surface_distance_to_mesh(c1)
        d2 = self._surface_distance_to_mesh(c2)
        sigma = self.cfg.surface_contact_sigma_m
        surface = 0.5 * (torch.exp(-d1 / sigma) + torch.exp(-d2 / sigma))
        width = (c1 - c2).norm(dim=-1)
        min_w = self.cfg.two_point_min_pinch_width_m
        spread = torch.exp(-torch.relu(min_w - width) / sigma)
        return surface * spread

    def _contact_area_reward(self) -> torch.Tensor:
        """Bilateral finger coverage: min(left, right) fraction of PC points within radius.

        Both fingertip positions are transformed into object-local frame and compared
        against the pre-loaded 512-point PC. The bilateral min penalises one-sided
        grasps and encourages both fingers to be in contact with the object surface.
        """
        pc_local = self._obj_pcs[self._env_shape]   # (B, 512, 3) object-local, scaled
        lf_w = self._robot.data.body_pos_w[:, self._left_finger_idx, :3]
        rf_w = self._robot.data.body_pos_w[:, self._right_finger_idx, :3]
        obj_w = self._get_active_obj_pos()          # (B, 3)

        inv_q = quat_inv(self._grasp_origin_quat)   # (B, 4)
        lf_local = quat_rotate(inv_q, lf_w - obj_w)   # (B, 3)
        rf_local = quat_rotate(inv_q, rf_w - obj_w)

        r = self.cfg.contact_area_radius_m
        lf_dist = (pc_local - lf_local.unsqueeze(1)).norm(dim=-1)  # (B, 512)
        rf_dist = (pc_local - rf_local.unsqueeze(1)).norm(dim=-1)
        lf_cov = (lf_dist < r).float().mean(dim=-1)   # (B,)
        rf_cov = (rf_dist < r).float().mean(dim=-1)
        return torch.min(lf_cov, rf_cov)

    def _graspnet_reward(self) -> torch.Tensor:
        """Distance-decayed GraspNet quality bonus at the policy's chosen grasp point.

        Offline-precomputed candidates only (scripts/generate_graspnet_labels.py) —
        no live model inference here, so this is as cheap as the other reward terms.
        Shapes without labels yet contribute 0 (see the __init__ preload).
        """
        centers = self._obj_graspnet_centers[self._env_shape]   # (B, K, 3) object-local, scaled
        quality = self._obj_graspnet_quality[self._env_shape]   # (B, K)
        dist = (centers - self._grasp_target.unsqueeze(1)).norm(dim=-1)   # (B, K)
        weight = torch.exp(-dist / self.cfg.graspnet_reward_radius_m)
        return (weight * quality).max(dim=-1).values

    def _get_rewards(self) -> torch.Tensor:
        # Lift reward: peak object height during lift+hold (tracked in _apply_action),
        # so it's unaffected by the arm lowering the object during pick-and-place.
        delta = (self._last_hold_obj_z - self._spawn_z).clamp(min=0.0)
        lift_r = (delta / self.cfg.lift_target_m).clamp(max=1.0)
        if self.cfg.require_contact_for_lift_reward:
            # Zero height credit unless fingers actually pinched during lift/hold.
            lift_r = lift_r * self._lift_had_contact.float()

        # Contact-area reward: bilateral fingertip coverage of the object PC.
        contact_r = self._contact_area_reward()

        # Place reward: proximity to goal at episode end (zero for lift-only envs).
        obj_w = self._get_active_obj_pos()
        goal_dist = (obj_w - self._place_goal_w).norm(dim=-1)
        place_r = torch.exp(-goal_dist / self.cfg.place_sigma_m) * self._task_mode.float()

        # GraspNet-bootstrapped reward: learned grasp-quality bonus at the chosen point.
        if self.cfg.use_graspnet_reward:
            graspnet_r = self._graspnet_reward()
        else:
            graspnet_r = torch.zeros(self.num_envs, device=self.device)

        surface_r = (
            self._last_surface_contact_reward
            if self.cfg.two_point_grasp
            else torch.zeros(self.num_envs, device=self.device)
        )

        leg_still = torch.ones(self.num_envs, device=self.device)  # no legs on Franka

        self._last_lift_reward     = lift_r
        self._last_lift_success    = lift_r >= 1.0   # lift_r is capped at 1.0 = target reached
        self._last_contact_reward  = contact_r
        self._last_place_reward    = place_r
        self._last_graspnet_reward = graspnet_r
        self._last_leg_still       = leg_still

        w_lift     = self.cfg.lift_reward_weight
        w_contact  = self.cfg.contact_area_reward_weight
        w_place    = self.cfg.place_reward_weight
        w_graspnet = self.cfg.graspnet_reward_scale
        w_surface  = self.cfg.surface_contact_reward_weight
        return (
            w_lift * lift_r + w_contact * contact_r + w_place * place_r
            + w_graspnet * graspnet_r + w_surface * surface_r
        )

    # ── Dones ─────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Every episode is exactly one agent step; always terminal
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return ones, ones   # (terminated, truncated)
