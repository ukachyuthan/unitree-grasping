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
        self._exec_step    = 0   # phase counter, reset in _pre_physics_step

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

        # 2. Pick a random shape for each env
        self._env_shape[env_ids] = torch.randint(0, NUM_SHAPES, (n,), device=self.device)

        # 3. Spawn chosen object on table, park others underground
        spawn_x = torch.empty(n, device=self.device).uniform_(*self.cfg.spawn_x_range)
        spawn_y = torch.empty(n, device=self.device).uniform_(*self.cfg.spawn_y_range)
        spawn_z = torch.full((n,), self.cfg.table_surface_z + 0.05, device=self.device)
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
        self._exec_step = 0

    # ── Scripted execution (called each physics step within decimation) ────────

    def _apply_action(self):
        s = self._exec_step

        if s < N_APPROACH:
            self._do_approach()
        elif s < N_APPROACH + N_CLOSE:
            t = (s - N_APPROACH) / N_CLOSE   # 0 → 1
            self._do_gripper(t)
        elif s < N_APPROACH + N_CLOSE + N_LIFT:
            self._do_lift()
        # else: hold — no joint updates

        self._exec_step += 1

    def _do_approach(self):
        """Jacobian DLS IK step toward grasp target (object-local offset)."""
        # EE position in robot-base frame
        ee_w    = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
        root_w  = self._robot.data.root_pos_w[:, :3]
        ee_base = ee_w - root_w              # (B, 3)

        # Convert object-local grasp offset → robot-base frame target
        obj_world = self._get_active_obj_pos()         # (B, 3) world frame
        obj_base  = obj_world - root_w                 # (B, 3) robot-base frame
        target_base = obj_base + self._grasp_target    # (B, 3) robot-base frame

        error   = target_base - ee_base      # (B, 3)

        # Jacobian (num_envs, num_bodies-1, 6, num_dofs)
        J_full  = self._robot.root_physx_view.get_jacobians()
        # Linear-velocity rows (0:3), EE body, arm DOFs only
        J = J_full[:, self._ee_body_idx - 1, :3, :]
        J = J[:, :, self._arm_dof_idx]   # (B, 3, 5)

        # Damped least squares: dq = J^T (J J^T + λI)^{-1} error
        lam    = 0.05
        JJT    = J @ J.transpose(-1, -2)   # (B, 3, 3)
        eye3   = torch.eye(3, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        JJT_r  = JJT + lam * eye3
        dq     = (J.transpose(-1, -2) @ torch.linalg.solve(JJT_r, error.unsqueeze(-1))).squeeze(-1)

        alpha  = self.cfg.ik_alpha
        q_cur  = self._robot.data.joint_pos[:, self._arm_dof_idx]
        q_tgt  = q_cur + alpha * dq
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._arm_dof_idx)

    def _do_gripper(self, t: float):
        """Interpolate gripper from open to closed.  t ∈ [0, 1]."""
        q_open  = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
        q_close = self._gripper_close.unsqueeze(0).expand(self.num_envs, -1)
        q_tgt   = (1 - t) * q_open + t * q_close
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._grip_dof_idx)

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
        obj_z    = self._get_active_obj_pos()[:, 2]
        # Measure how far the object moved UP from its spawn position.
        delta    = (obj_z - self._spawn_z).clamp(min=0.0)
        reward   = (delta >= self.cfg.lift_threshold_m).float()
        return reward

    # ── Dones ─────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Every episode is exactly one agent step; always terminal
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return ones, ones   # (terminated, truncated)
