"""
Grasp-pose prediction environment.

Episode structure (one agent step = decimation physics steps):
  1. [step 0]          policy observes point cloud → outputs 3D grasp position
  2. [steps 0–N_APPROACH]  scripted IK moves EE toward grasp position
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

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject

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
        self._arm_dof_idx, _ = self._robot.find_joints(cfg.left_arm_joint_names)
        self._grip_dof_idx, _ = self._robot.find_joints(cfg.left_gripper_joint_names)
        self._leg_dof_idx, _ = self._robot.find_joints(cfg.leg_joint_names)
        self._ee_body_idx, _  = self._robot.find_bodies(cfg.ee_body_name)
        self._ee_body_idx     = self._ee_body_idx[0]

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
        self._obj_pcs = torch.stack(pcs, dim=0).to(self.device)  # (NUM_SHAPES, 512, 3)

        # ── Per-env state ──────────────────────────────────────────────────────
        B = self.num_envs
        self._env_shape    = torch.zeros(B, dtype=torch.long, device=self.device)
        # _grasp_target: offset from object centre in object-LOCAL frame (metres).
        # Converted to robot-base frame inside _do_approach by adding obj_base.
        self._grasp_target = torch.zeros(B, 3, device=self.device)
        self._spawn_z      = torch.zeros(B, device=self.device)  # world-frame z at spawn
        self._obj_anchor   = torch.zeros(B, 13, device=self.device)  # root state after settle
        self._grasp_locked = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._grasp_offset = torch.zeros(B, 3, device=self.device)   # obj - ee at close
        self._last_lift_reward = torch.zeros(B, device=self.device)
        self._last_leg_still = torch.zeros(B, device=self.device)
        self._exec_step    = 0   # phase counter, reset in _pre_physics_step
        self._arm_q_des = self._home_joint_pos[:, self._arm_dof_idx].clone()

        # Hold right arm / legs / torso at home — only left arm + gripper move.
        controlled = set(self._arm_dof_idx) | set(self._grip_dof_idx)
        self._idle_dof_idx = [i for i in range(self._robot.num_joints) if i not in controlled]

        self._ee_jac_idx = self._ee_body_idx - 1  # fixed-base robot (unused with numeric IK)

        # Gripper open/close limits (radians, from joint limits)
        self._gripper_open  = torch.tensor([ 1.00,  0.52], device=self.device)
        self._gripper_close = torch.tensor([-0.40, -0.20], device=self.device)

    # ── Scene setup ────────────────────────────────────────────────────────────

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

        # 2. Pick a random shape for each env
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
        """Let spawned objects settle on the table before the policy acts."""
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
        # Re-record spawn height after settle so reward baseline is accurate.
        self._spawn_z[:] = self._get_active_obj_pos()[:, 2]
        self._cache_object_anchors()

    def _cache_object_anchors(self):
        """Store settled root state of each env's active object."""
        for i, obj in enumerate(self._objs):
            mask = (self._env_shape == i)
            if mask.any():
                state = obj.data.root_state_w[mask]
                self._obj_anchor[mask] = state

    def _hold_idle_joints(self):
        """Lock legs, torso, and right arm at the home pose."""
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
        """After close, move the object with the palm (simulated successful grasp)."""
        if not self._grasp_locked.any():
            return
        ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
        for i, obj in enumerate(self._objs):
            mask = self._grasp_locked & (self._env_shape == i)
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
        actions: (B, 3) in tanh space [-1, 1].
        De-normalize to object-local workspace offsets and store.
        """
        x_lo, x_hi = self.cfg.grasp_x_bounds
        y_lo, y_hi = self.cfg.grasp_y_bounds
        z_lo, z_hi = self.cfg.grasp_z_bounds

        a = actions.clamp(-1, 1)
        gx = (a[:, 0] + 1) / 2 * (x_hi - x_lo) + x_lo
        gy = (a[:, 1] + 1) / 2 * (y_hi - y_lo) + y_lo
        gz = (a[:, 2] + 1) / 2 * (z_hi - z_lo) + z_lo
        self._grasp_target = torch.stack([gx, gy, gz], dim=-1)  # (B, 3)
        self._grasp_locked[:] = False
        self._exec_step = 0

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
            self._anchor_objects()
            t = (s - N_APPROACH) / N_CLOSE   # 0 → 1
            self._do_gripper(t, lock_at_end=(s + 1 >= N_APPROACH + N_CLOSE))
        elif s < N_APPROACH + N_CLOSE + N_LIFT:
            self._do_lift()
        else:
            self._hold_pose()

        self._exec_step += 1
        if s < N_APPROACH + N_CLOSE:
            self._pin_objects()
        elif s < N_APPROACH + N_CLOSE + N_LIFT + N_HOLD:
            self._sync_grasped_object()

    def _numeric_jacobian_pos(self) -> torch.Tensor:
        """Positional Jacobian (B, 3, n_arm) from batched finite differences."""
        n_arm = len(self._arm_dof_idx)
        J = torch.zeros(self.num_envs, 3, n_arm, device=self.device)
        q_full = self._robot.data.joint_pos.clone()
        v_full = self._robot.data.joint_vel.clone()
        ee0 = self._robot.data.body_pos_w[:, self._ee_body_idx, :3].clone()
        eps = 1e-3
        for j, dof in enumerate(self._arm_dof_idx):
            q_pert = q_full.clone()
            q_pert[:, dof] += eps
            self._robot.write_joint_state_to_sim(q_pert, v_full)
            self.sim.forward()
            ee1 = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
            J[:, :, j] = (ee1 - ee0) / eps
        self._robot.write_joint_state_to_sim(q_full, v_full)
        self.sim.forward()
        return J

    def _do_approach(self):
        """Damped least-squares IK using a numeric palm Jacobian."""
        self._anchor_objects()
        ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
        target_w = self._obj_anchor[:, :3] + self._grasp_target
        error = target_w - ee_w

        J = self._numeric_jacobian_pos()
        self._anchor_objects()

        lam = 0.05
        JJT = J @ J.transpose(-1, -2)
        eye3 = torch.eye(3, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        dq = (
            J.transpose(-1, -2)
            @ torch.linalg.solve(JJT + lam * eye3, error.unsqueeze(-1))
        ).squeeze(-1)

        alpha = self.cfg.ik_alpha
        q_cur = self._robot.data.joint_pos[:, self._arm_dof_idx]
        lo = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 0]
        hi = self._robot.data.soft_joint_pos_limits[:, self._arm_dof_idx, 1]
        q_tgt = (q_cur + alpha * dq).clamp(lo, hi)
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
            self._grasp_offset = self._obj_anchor[:, :3] - ee_w
            self._grasp_locked[:] = True

    def _do_lift(self):
        """Lift EE straight up by shifting the local-frame z offset."""
        delta_z = 0.15 / N_LIFT   # total 15 cm over N_LIFT steps
        self._grasp_target[:, 2] += delta_z   # moves world-z up since +z=up
        self._do_approach()

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
        if self._grasp_locked.any():
            ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
            obj_z = (ee_w + self._grasp_offset)[:, 2]
        else:
            obj_z = self._get_active_obj_pos()[:, 2]
        delta = (obj_z - self._spawn_z).clamp(min=0.0)
        lift_r = (delta / self.cfg.lift_target_m).clamp(max=1.0)

        leg_pos = self._robot.data.joint_pos[:, self._leg_dof_idx]
        leg_home = self._home_joint_pos[:, self._leg_dof_idx]
        leg_vel = self._robot.data.joint_vel[:, self._leg_dof_idx]
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
