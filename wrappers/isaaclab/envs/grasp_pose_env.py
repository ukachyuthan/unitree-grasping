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
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from envs._paths import data_path
from envs.grasp_pose_env_cfg import (
    GraspPoseEnvCfg,
    N_APPROACH, N_CLOSE, N_LIFT, N_HOLD, EXEC_STEPS,
    NUM_PC_POINTS,
)

_SHAPE_NAMES = [
    "torus", "l_shape", "t_shape", "c_shape", "dumbbell",
    "wedge", "star_prism", "bracket", "stepped_cyl",
    "twisted_bar", "irregular_ext", "convex_hull",
]
NUM_SHAPES  = len(_SHAPE_NAMES)
_PC_PRE_N   = 512   # points pre-sampled per shape on disk
_PC_NOISE_M = 0.003  # 3 mm Gaussian noise on point cloud


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

        # ── Stored home joint positions (for reset) ───────────────────────────
        self._home_joint_pos = self._robot.data.default_joint_pos.clone()

        # ── Pre-load object point clouds ──────────────────────────────────────
        pcs = []
        for name in _SHAPE_NAMES:
            p = data_path("data/objects/train", name, "000_pc.npy")
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
        self._last_lift_reward = torch.zeros(B, device=self.device)
        self._last_leg_still = torch.zeros(B, device=self.device)
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
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._table = RigidObject(self.cfg.table)
        self.scene.rigid_objects["table"] = self._table

        self._objs: list[RigidObject] = []
        for name in _SHAPE_NAMES:
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
            n_shapes = min(self.cfg.eval_num_shapes, NUM_SHAPES)
            shape_ids = (
                self._eval_shape_step + torch.arange(n, device=self.device)
            ) % n_shapes
            self._env_shape[env_ids] = shape_ids
            self._eval_shape_step += n
        else:
            self._env_shape[env_ids] = torch.randint(0, NUM_SHAPES, (n,), device=self.device)

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
        self._grasp_target = torch.stack([gx, gy, gz], dim=-1)  # (B, 3)

        # Orientation: policy controls wrist tilt and jaw roll per episode.
        self._tilt_target = (a[:, 3] + 1) / 2 * (t_hi - t_lo) + t_lo  # (B,)
        self._roll_target = (a[:, 4] + 1) / 2 * (r_hi - r_lo) + r_lo  # (B,)

        self._grasp_locked[:] = False
        self._exec_step = 0
        self._J_cache = None
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

        if s < N_APPROACH:
            self._anchor_objects()
            self._do_approach()
            q_open = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
            self._robot.set_joint_position_target(q_open, joint_ids=self._grip_dof_idx)
        elif s < N_APPROACH + N_CLOSE:
            # Keep IK on the grasp point while closing so jaws stay around the object.
            self._do_approach()
            t = (s - N_APPROACH) / N_CLOSE   # 0 → 1
            self._do_gripper(t, lock_at_end=(s + 1 >= N_APPROACH + N_CLOSE))
        elif s < N_APPROACH + N_CLOSE + N_LIFT:
            self._do_lift()
        else:
            self._hold_pose()

        self._exec_step += 1
        # Pin the object only during approach (stable IK target). During close /
        # lift / hold the object is fully simulated: gravity, mass, finger contact.
        if s < N_APPROACH:
            self._pin_objects()
        elif (self.cfg.kinematic_grasp or self.cfg.contact_carry) and s >= N_APPROACH + N_CLOSE:
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
        target_w[:, 2] += self.cfg.lift_height_m * (k / N_LIFT)
        # Position-only during lift — orientation correction here breaks finger contact.
        self._ik_to(target_w)

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """Return point cloud of the current object (object-local frame)."""
        pc_flat = self._synthesize_pointcloud()   # (B, NUM_PC_POINTS * 3)
        return {"policy": pc_flat}

    def _synthesize_pointcloud(self) -> torch.Tensor:
        """
        Sub-sample pre-loaded PCs + add 3mm Gaussian noise.
        Returns PC in OBJECT-LOCAL frame (centred at object origin).
        This matches the pre-training frame so encoder weights transfer correctly.
        The policy predicts grasp offsets in the same local frame;
        _do_approach converts local → robot-base by adding obj_base.
        """
        B  = self.num_envs
        shape_pcs = self._obj_pcs[self._env_shape]   # (B, 512, 3)

        idx      = torch.randint(0, _PC_PRE_N, (B, NUM_PC_POINTS), device=self.device)
        pc_local = shape_pcs.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, N, 3)
        pc_local = pc_local + torch.randn_like(pc_local) * _PC_NOISE_M

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

    def _get_rewards(self) -> torch.Tensor:
        # Lift reward uses the object's REAL simulated height only. No welded proxy —
        # if the fingers don't hold it, it stays on the table and reward is zero.
        actual_z = self._get_active_obj_pos()[:, 2]
        if self.cfg.kinematic_grasp:
            ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
            welded_z = (ee_w + self._grasp_offset)[:, 2]
            obj_z = torch.where(self._grasp_locked, welded_z, actual_z)
        else:
            obj_z = actual_z
        delta = (obj_z - self._spawn_z).clamp(min=0.0)
        lift_r = (delta / self.cfg.lift_target_m).clamp(max=1.0)

        leg_pos = self._robot.data.joint_pos[:, self._leg_dof_idx]
        leg_home = self._home_joint_pos[:, self._leg_dof_idx]
        leg_vel = self._robot.data.joint_vel[:, self._leg_dof_idx]
        if len(self._leg_dof_idx) == 0:
            leg_still = torch.ones(self.num_envs, device=self.device)
        else:
            leg_dev = (leg_pos - leg_home).pow(2).mean(dim=-1).sqrt()
            leg_spd = leg_vel.pow(2).mean(dim=-1).sqrt()
            leg_still = torch.exp(-leg_dev / self.cfg.leg_dev_scale)
            leg_still = leg_still * torch.exp(-leg_spd / self.cfg.leg_vel_scale)

        self._last_lift_reward = lift_r
        self._last_leg_still = leg_still

        w_lift = self.cfg.lift_reward_weight
        w_leg = self.cfg.leg_stillness_weight
        return w_lift * lift_r + w_leg * leg_still

    # ── Dones ─────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Every episode is exactly one agent step; always terminal
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return ones, ones   # (terminated, truncated)
