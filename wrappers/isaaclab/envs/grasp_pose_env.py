"""
Grasp-pose prediction environment.

Episode structure (one agent step = decimation physics steps):
  1. [step 0]          policy observes point cloud → outputs 3D grasp position
  2. [steps 0–N_APPROACH]  scripted pose IK moves EE toward grasp position + orientation
  3. [steps N_APPROACH–N_CLOSE]  scripted gripper close
  4. [steps N_CLOSE–N_LIFT]  scripted lift (EE target +z)
  5. [steps N_LIFT–end]   hold + measure object height → reward
  6. env resets with a new random object + random pose

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
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    quat_mul,
    matrix_from_quat,
    quat_inv,
)

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg

from envs._paths import data_path
from envs._object_registry import PROCEDURAL_SHAPE_NAMES, ycb_shape_names, shape_split
from envs.grasp_pose_env_cfg import (
    GraspPoseEnvCfg,
    N_APPROACH, N_CLOSE, N_LIFT, N_HOLD, N_TRANSPORT, N_LOWER, N_OPEN, EXEC_STEPS,
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

        # ── Pre-load object point clouds ──────────────────────────────────────
        # self._shape_names / self._num_shapes were set in _setup_scene() (called
        # inside super().__init__() above) so self._objs already matches this order.
        pcs = []
        for name in self._shape_names:
            p = data_path("data/objects", shape_split(name), name, "000_pc.npy")
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

        # ── Pre-load GraspNet-quality labels (scripts/generate_graspnet_labels.py) ──
        # Missing files default to all-zero quality, which contributes nothing to
        # the reward term (see _graspnet_reward) rather than erroring.
        _K_GN = 20
        gn_centers, gn_quality, n_missing = [], [], 0
        for name in self._shape_names:
            centers = np.zeros((_K_GN, 3), dtype=np.float32)
            quality = np.zeros((_K_GN,), dtype=np.float32)
            if self.cfg.use_graspnet_reward:
                p = data_path("data/objects", shape_split(name), name, "000_graspnet.json")
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
                  f"000_graspnet.json — graspnet reward term is 0 for those until "
                  f"scripts/generate_graspnet_labels.py is run.")

        # ── Per-env state ──────────────────────────────────────────────────────
        B = self.num_envs
        self._env_shape    = torch.zeros(B, dtype=torch.long, device=self.device)
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
        self._last_leg_still       = torch.zeros(B, device=self.device)
        self._last_contact_reward  = torch.zeros(B, device=self.device)
        self._last_place_reward    = torch.zeros(B, device=self.device)
        self._last_graspnet_reward = torch.zeros(B, device=self.device)
        # Peak object z during lift+hold phases — used for lift reward in place mode
        # so we don't measure height AFTER the arm has lowered the object to the goal.
        self._last_hold_obj_z = torch.zeros(B, device=self.device)
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
        self._ik_call = 0
        self._exec_step    = 0   # phase counter, reset in _pre_physics_step
        self._eval_shape_step = 0
        self._arm_q_des = self._home_joint_pos[:, self._arm_dof_idx].clone()

        # Orientation targets decoded from actions 3 and 4.
        # These are applied every IK step so the policy fully controls wrist angle.
        self._tilt_target = torch.zeros(B, device=self.device)        # panda_joint5
        self._roll_target = torch.full((B,), 0.741, device=self.device)  # panda_joint7 home
        # Indices of tilt/roll joints *within* the arm joint list (0-indexed).
        # arm_joint_names = [j1, j2, j3, j4, j5, j6, j7] → j5=index 4, j7=index 6.
        self._tilt_arm_idx = 4   # panda_joint5 within _arm_dof_idx
        self._roll_arm_idx = 6   # panda_joint7 within _arm_dof_idx

        # Arm + parallel gripper (and optional tuck joints) are controlled.
        controlled = set(self._arm_dof_idx) | set(self._grip_dof_idx)
        if self._hand_tuck_idx:
            controlled |= set(self._hand_tuck_idx)
        self._idle_dof_idx = [i for i in range(self._robot.num_joints) if i not in controlled]

        self._ee_jac_idx = self._ee_body_idx - 1

        # Parallel 2-finger gripper: both joints share the same open/close value.
        n_grip = len(self._grip_dof_idx)
        self._gripper_open = torch.full(
            (n_grip,), cfg.gripper_open_val, device=self.device
        )
        self._gripper_close = torch.full(
            (n_grip,), cfg.gripper_close_val, device=self.device
        )
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
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/grasp_targets",
                markers={
                    "grasp_point": sim_utils.SphereCfg(
                        radius=r,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.85, 0.0),
                        ),
                    ),
                    "palm_actual": sim_utils.SphereCfg(
                        radius=r * 0.7,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.55, 1.0),
                        ),
                    ),
                    "finger_mid": sim_utils.SphereCfg(
                        radius=r * 0.55,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.9, 0.3),
                        ),
                    ),
                },
            )
            self._grasp_marker = VisualizationMarkers(marker_cfg)

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
            self._shape_names = list(PROCEDURAL_SHAPE_NAMES) + (
                ycb_shape_names("train") if self.cfg.use_real_objects else []
            )
        self._num_shapes = len(self._shape_names)

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._table = RigidObject(self.cfg.table)
        self.scene.rigid_objects["table"] = self._table

        self._objs: list[RigidObject] = []
        for name in self._shape_names:
            obj_cfg = getattr(self.cfg, f"object_{name}")
            obj = RigidObject(obj_cfg)
            self.scene.rigid_objects[name] = obj
            self._objs.append(obj)

        self._finger_contact = ContactSensor(
            ContactSensorCfg(
                prim_path=self.cfg.finger_contact_prim_path,
                history_length=0,
                track_air_time=False,
                filter_prim_paths_expr=self.cfg.finger_contact_filter_paths,
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
        spawn_x = torch.empty(n, device=self.device).uniform_(*self.cfg.spawn_x_range)
        spawn_y = torch.empty(n, device=self.device).uniform_(*self.cfg.spawn_y_range)
        spawn_z = torch.full((n,), self.cfg.table_surface_z + self.cfg.spawn_z_offset, device=self.device)
        self._spawn_z[env_ids] = spawn_z   # record for reward computation

        for i, obj in enumerate(self._objs):
            pos = torch.zeros(n, 3, device=self.device)
            mask = (self._env_shape[env_ids] == i)
            if mask.any():
                pos[mask, 0] = spawn_x[mask]
                pos[mask, 1] = spawn_y[mask]
                pos[mask, 2] = spawn_z[mask]
            pos[~mask, 2] = -20.0   # park underground

            quat = torch.zeros(n, 4, device=self.device)
            quat[:, 0] = 1.0   # identity quaternion (w=1)
            default_state = obj.data.default_root_state[env_ids].clone()
            default_state[:, :3] = (
                pos + self.scene.env_origins[env_ids]
            )
            default_state[:, 3:7] = quat
            default_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(default_state, env_ids=env_ids)

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
        for i, obj in enumerate(self._objs):
            mask = (self._env_shape == i)
            if not mask.any():
                continue
            state = obj.data.root_state_w[mask].clone()
            anchor = self._obj_anchor[mask]
            state[:, 0] = anchor[:, 0]      # pin x
            state[:, 1] = anchor[:, 1]      # pin y
            state[:, 3:7] = anchor[:, 3:7]  # pin orientation
            state[:, 7:] = 0.0              # zero velocity
            env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            obj.write_root_state_to_sim(state, env_ids=env_ids)

    def _cache_object_anchors(self):
        """Store settled root state of each env's active object."""
        for i, obj in enumerate(self._objs):
            mask = (self._env_shape == i)
            if mask.any():
                state = obj.data.root_state_w[mask]
                self._obj_anchor[mask] = state

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
        for i, obj in enumerate(self._objs):
            mask = in_contact & (self._env_shape == i)
            if not mask.any():
                continue
            pinned = self._obj_anchor[mask].clone()
            pinned[:, :3] = ee_w[mask] + self._grasp_offset[mask]
            pinned[:, 7:] = 0.0
            env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            obj.write_root_state_to_sim(pinned, env_ids=env_ids)
            self._obj_anchor[mask] = pinned

    def _pin_objects(self):
        for i, obj in enumerate(self._objs):
            mask = (self._env_shape == i)
            if not mask.any():
                continue
            pinned = self._obj_anchor[mask].clone()
            pinned[:, 7:] = 0.0
            env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            obj.write_root_state_to_sim(pinned, env_ids=env_ids)

    # ── One agent step: store grasp target, reset execution counter ───────────

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        actions: (B, 5) in tanh space [-1, 1].
          [:3] → object-local grasp position offset
          [3]  → panda_joint5 tilt (approach angle from vertical)
          [4]  → panda_joint7 roll (jaw rotation around approach axis)
        """
        x_lo, x_hi = self.cfg.grasp_x_bounds
        y_lo, y_hi = self.cfg.grasp_y_bounds
        z_lo, z_hi = self.cfg.grasp_z_bounds
        t_lo, t_hi = self.cfg.grasp_tilt_bounds
        r_lo, r_hi = self.cfg.grasp_roll_bounds

        a = actions.clamp(-1, 1)
        gx = (a[:, 0] + 1) / 2 * (x_hi - x_lo) + x_lo
        gy = (a[:, 1] + 1) / 2 * (y_hi - y_lo) + y_lo
        gz = (a[:, 2] + 1) / 2 * (z_hi - z_lo) + z_lo
        grasp_local = torch.stack([gx, gy, gz], dim=-1)  # (B, 3) in observed (maybe augmented) frame

        # If PC was randomly rotated for augmentation, rotate grasp target BACK to
        # the true object-local frame before using it for IK.
        if self.cfg.pc_augment_yaw:
            inv_yaw = -self._pc_aug_yaw   # (B,)
            cos_y = inv_yaw.cos()
            sin_y = inv_yaw.sin()
            gx_r = cos_y * grasp_local[:, 0] - sin_y * grasp_local[:, 1]
            gy_r = sin_y * grasp_local[:, 0] + cos_y * grasp_local[:, 1]
            grasp_local = torch.stack([gx_r, gy_r, grasp_local[:, 2]], dim=-1)

        self._grasp_target = grasp_local  # (B, 3) true object-local frame

        # Freeze object pose at decision time → stable IK targets throughout episode.
        self._grasp_origin_w    = self._obj_anchor[:, :3].clone()
        self._grasp_origin_quat = self._obj_anchor[:, 3:7].clone()

        # Orientation: policy controls wrist tilt and jaw roll per episode.
        self._tilt_target = (a[:, 3] + 1) / 2 * (t_hi - t_lo) + t_lo  # (B,)
        self._roll_target = (a[:, 4] + 1) / 2 * (r_hi - r_lo) + r_lo  # (B,)

        self._grasp_locked[:] = False
        self._exec_step = 0
        self._J_cache = None
        self._J_pos_cache = None
        self._ik_call = 0
        self._update_grasp_markers()

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

    def _update_grasp_markers(self):
        """Yellow = policy grasp, blue = actual palm, green = finger midpoint."""
        if self._grasp_marker is None:
            return
        grasp_w = self._grasp_point_world()
        palm_w = self._palm_actual_w()
        finger_w = self._finger_midpoint_w()
        translations = torch.cat([grasp_w, palm_w, finger_w], dim=0)
        n = self.num_envs
        marker_indices = torch.cat([
            torch.zeros(n, dtype=torch.int, device=self.device),
            torch.ones(n, dtype=torch.int, device=self.device),
            torch.full((n,), 2, dtype=torch.int, device=self.device),
        ])
        self._grasp_marker.visualize(
            translations=translations,
            marker_indices=marker_indices,
        )

    # ── Scripted execution (called each physics step within decimation) ────────

    def _apply_action(self):
        s = self._exec_step
        self._hold_idle_joints()

        T0 = N_APPROACH
        T1 = T0 + N_CLOSE
        T2 = T1 + N_LIFT
        T3 = T2 + N_HOLD
        T4 = T3 + N_TRANSPORT
        T5 = T4 + N_LOWER
        # T6 = T5 + N_OPEN = EXEC_STEPS

        if s < T0:
            self._anchor_objects()
            self._do_approach()
            q_open = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
            self._robot.set_joint_position_target(q_open, joint_ids=self._grip_dof_idx)
        elif s < T1:
            # Arm HOLDS its final approach configuration — do NOT keep running IK.
            # Running IK during close displaces the palm while fingers are animating,
            # breaking contact before it can be established (same principle as the
            # reference demo: joint targets are fixed while fingers close).
            t_smooth = _smoothstep((s - T0) / N_CLOSE)
            self._do_gripper(t_smooth, lock_at_end=(s + 1 >= T1))
        elif s < T2:
            self._do_lift()
        elif s < T3:
            self._hold_pose()
        elif s < T4:
            self._do_transport(s - T3)
        elif s < T5:
            self._do_lower(s - T4)
        else:
            self._do_open(s - T5)

        # Track peak object height during lift+hold for clean lift reward.
        if T1 <= s < T3:
            obj_z = self._get_active_obj_pos()[:, 2]
            self._last_hold_obj_z = torch.max(self._last_hold_obj_z, obj_z)

        self._exec_step += 1
        # Pin during approach; carry during lift/hold/transport/lower; release at open.
        if s < T0:
            self._pin_objects()
        elif (self.cfg.kinematic_grasp or self.cfg.contact_carry) and T1 <= s < T5:
            self._sync_grasped_object()

        if self._grasp_marker is not None:
            self._update_grasp_markers()

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

    def _approach_target_w(self) -> torch.Tensor:
        """World-frame palm IK setpoint from the frozen grasp decision pose."""
        grasp_w = self._grasp_point_world()
        # grasp_point_offset is in object-local frame (e.g. finger below palm along -z).
        offset_w = quat_rotate(
            self._grasp_origin_quat,
            self._grasp_pt_offset.expand(self.num_envs, -1),
        )
        return grasp_w - offset_w

    def _approach_target_quat(self) -> torch.Tensor:
        """Top-down EE orientation with optional yaw toward the grasp point."""
        if not self.cfg.ik_orient_yaw_to_object:
            return self._home_ee_quat.clone()
        _, _, home_yaw = euler_xyz_from_quat(self._home_ee_quat)
        grasp_w = self._grasp_point_world()
        base_xy = self._robot.data.root_pos_w[:, :2]
        yaw = torch.atan2(grasp_w[:, 1] - base_xy[:, 1], grasp_w[:, 0] - base_xy[:, 0])
        dyaw = yaw - home_yaw
        zero = torch.zeros_like(dyaw)
        dquat = quat_from_euler_xyz(zero, zero, dyaw)
        return quat_mul(dquat, self._home_ee_quat)

    def _do_approach(self):
        """Position IK toward grasp, then wrist orientation correction."""
        self._ik_to(self._approach_target_w(), self._approach_target_quat())

    def _ik_to(
        self,
        target_w: torch.Tensor,
        target_quat: torch.Tensor | None = None,
    ):
        """Hybrid IK: reliable numeric position, optional PhysX orientation."""
        if self._exec_step < N_APPROACH + N_CLOSE:
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
            if self._exec_step < N_APPROACH + N_CLOSE:
                self._anchor_objects()
        self._ik_call += 1

        dq = self._dls_delta(self._J_pos_cache, target_w - ee_w, dim=3)

        if target_quat is not None and self.cfg.ik_orient_alpha > 0.0:
            _, rot_err = compute_pose_error(
                ee_w, ee_q, target_w, target_quat, rot_error_type="axis_angle"
            )
            rot_err = rot_err * self.cfg.ik_orient_weight
            dq_rot = self._dls_delta(self._J_cache, rot_err, dim=3, lam=0.08)
            near = ((target_w - ee_w).norm(dim=-1) < self.cfg.ik_orient_pos_thresh).unsqueeze(-1)
            dq = dq + self.cfg.ik_orient_alpha * near * dq_rot

        alpha = self.cfg.ik_alpha
        q_cur = self._robot.data.joint_pos[:, self._arm_dof_idx]
        lo = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 0]
        hi = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 1]
        q_tgt = (q_cur + alpha * dq).clamp(lo, hi)

        # Override wrist orientation joints with policy-predicted targets.
        # IK solved joints 1-4 for position; joints 5 and 7 control orientation
        # and are roughly decoupled from EE position — pinning them here lets the
        # policy learn to rotate the gripper based on the object's point cloud.
        ti, ri = self._tilt_arm_idx, self._roll_arm_idx
        q_tgt[:, ti] = self._tilt_target.clamp(lo[:, ti], hi[:, ti])
        q_tgt[:, ri] = self._roll_target.clamp(lo[:, ri], hi[:, ri])

        self._arm_q_des = q_tgt
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._arm_dof_idx)

    def _hold_pose(self):
        """Keep arm and gripper fixed during measurement phase."""
        self._robot.set_joint_position_target(self._arm_q_des, joint_ids=self._arm_dof_idx)
        q_close = self._gripper_close.unsqueeze(0).expand(self.num_envs, -1)
        self._robot.set_joint_position_target(q_close, joint_ids=self._grip_dof_idx)

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
        q_close = self._gripper_close.unsqueeze(0).expand(self.num_envs, -1)
        t_per = torch.where(
            self._task_mode.bool(),
            torch.full((self.num_envs,), t, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
        ).unsqueeze(-1)
        q_tgt = (1.0 - t_per) * q_close + t_per * q_open
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._grip_dof_idx)

    def _do_gripper(self, t: float, lock_at_end: bool = False):
        """Interpolate gripper from open to closed; hold arm at last IK solution."""
        self._robot.set_joint_position_target(self._arm_q_des, joint_ids=self._arm_dof_idx)
        q_open  = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
        q_close = self._gripper_close.unsqueeze(0).expand(self.num_envs, -1)
        q_tgt   = (1 - t) * q_open + t * q_close
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._grip_dof_idx)
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
        """Raise the palm along +z toward a fixed world target set at lift start.

        The target is anchored to the palm position captured when the gripper
        finished closing, then ramped up linearly over N_LIFT steps. It no longer
        depends on the (moving) object anchor or the accumulating grasp target,
        which previously double-counted and made the target accelerate past the
        arm's reach (runaway feedback loop).
        """
        lift_start = N_APPROACH + N_CLOSE
        if self._exec_step == lift_start:
            self._lift_base_w = (
                self._robot.data.body_pos_w[:, self._ee_body_idx, :3].clone()
            )
        k = (self._exec_step - lift_start) + 1   # 1 … N_LIFT
        target_w = self._lift_base_w.clone()
        # Smoothstep: gentle start and landing — avoids jerking the grasped object loose.
        target_w[:, 2] += self.cfg.lift_height_m * _smoothstep(k / N_LIFT)
        # Position-only during lift — orientation correction here breaks finger contact.
        self._ik_to(target_w)

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
        positions = torch.zeros(self.num_envs, 3, device=self.device)
        for i, obj in enumerate(self._objs):
            mask = (self._env_shape == i)
            if mask.any():
                positions[mask] = obj.data.root_pos_w[mask, :3]
        return positions

    # ── Rewards ───────────────────────────────────────────────────────────────

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

        leg_still = torch.ones(self.num_envs, device=self.device)  # no legs on Franka

        self._last_lift_reward     = lift_r
        self._last_contact_reward  = contact_r
        self._last_place_reward    = place_r
        self._last_graspnet_reward = graspnet_r
        self._last_leg_still       = leg_still

        w_lift     = self.cfg.lift_reward_weight
        w_contact  = self.cfg.contact_area_reward_weight
        w_place    = self.cfg.place_reward_weight
        w_graspnet = self.cfg.graspnet_reward_scale
        return (
            w_lift * lift_r + w_contact * contact_r + w_place * place_r
            + w_graspnet * graspnet_r
        )

    # ── Dones ─────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Every episode is exactly one agent step; always terminal
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return ones, ones   # (terminated, truncated)
