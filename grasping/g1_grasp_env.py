"""
G1 left-arm pick-and-place environment  (DirectRLEnv).

Key design choices
──────────────────
• Training uses pre-sampled surface point clouds (loaded from .npy at init) +
  3 mm Gaussian noise — identical to DexPoint / UniDexGrasp.  No rendering needed.
• 12 procedural shape families; one random shape per episode per env.
• At deployment, replace _synthesize_pointcloud() with depth_to_pointcloud_world()
  from pointcloud_utils.py.  Policy weights transfer unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate, quat_conjugate, subtract_frame_transforms
import isaaclab.sim as sim_utils

from grasping.g1_grasp_env_cfg import G1GraspEnvCfg, NUM_PC_POINTS

# ── Shape index ↔ cfg attribute name ↔ PC file ──────────────────────────────
# Order must match the RigidObjectCfg field names in G1GraspEnvCfg.
_SHAPE_NAMES = [
    "torus", "l_shape", "t_shape", "c_shape",
    "dumbbell", "wedge", "star_prism", "bracket",
    "stepped_cyl", "twisted_bar", "irregular_ext", "convex_hull",
]
NUM_SHAPES = len(_SHAPE_NAMES)

_PC_NOISE    = 0.003   # sensor noise std (3 mm)
_PC_PRE_N    = 512     # points in the pre-sampled numpy PCs


class G1GraspEnv(DirectRLEnv):
    cfg: G1GraspEnvCfg

    def __init__(self, cfg: G1GraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Joint indices ──────────────────────────────────────────────────
        self._arm_dof_idx, _  = self._robot.find_joints(self.cfg.left_arm_joint_names)
        self._grip_dof_idx, _ = self._robot.find_joints(self.cfg.left_gripper_joint_names)
        self._num_arm_dofs = len(self._arm_dof_idx)

        ee_bodies, _ = self._robot.find_bodies(self.cfg.ee_body_name)
        self._ee_body_idx = ee_bodies[0]

        self._arm_lo = self._robot.data.soft_joint_pos_limits[0, self._arm_dof_idx, 0].to(self.device)
        self._arm_hi = self._robot.data.soft_joint_pos_limits[0, self._arm_dof_idx, 1].to(self.device)
        g_lo = self._robot.data.soft_joint_pos_limits[0, self._grip_dof_idx, 0].to(self.device)
        g_hi = self._robot.data.soft_joint_pos_limits[0, self._grip_dof_idx, 1].to(self.device)
        self._grip_open  = g_hi
        self._grip_close = g_lo

        self._joint_targets = self._robot.data.default_joint_pos.clone()

        # ── Per-env state ──────────────────────────────────────────────────
        self._env_shape   = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_pos_w  = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # ── Pre-load object point clouds ───────────────────────────────────
        # Shape (NUM_SHAPES, _PC_PRE_N, 3) — stored on GPU for fast sampling
        obj_dir = self.cfg._OBJ_DIR
        pcs = []
        for name in _SHAPE_NAMES:
            pc_path = os.path.join(obj_dir, name, "000_pc.npy")
            if os.path.exists(pc_path):
                pc = torch.from_numpy(np.load(pc_path)).float().to(self.device)  # (512, 3)
            else:
                # Fallback: unit sphere if file missing
                pc = torch.randn(_PC_PRE_N, 3, device=self.device)
                pc = pc / pc.norm(dim=-1, keepdim=True) * 0.04
                print(f"[GraspEnv] WARNING: PC file not found for {name}, using sphere fallback")
            pcs.append(pc)
        self._obj_pcs = torch.stack(pcs, dim=0)  # (NUM_SHAPES, 512, 3)

    # ── Scene ────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._table = RigidObject(self.cfg.table)
        self.scene.rigid_objects["table"] = self._table

        # Load all 12 shape objects
        self._objs: list[RigidObject] = []
        for name in _SHAPE_NAMES:
            cfg_attr = f"object_{name}"
            obj = RigidObject(getattr(self.cfg, cfg_attr))
            self._objs.append(obj)
            self.scene.rigid_objects[f"obj_{name}"] = obj

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── Action application ────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_action = actions.clone()
        actions = actions.clamp(-1.0, 1.0)

        if self.cfg.use_joint_space_control:
            arm_delta = actions[:, :self._num_arm_dofs] * self.cfg.arm_action_scale
            new_arm = (self._joint_targets[:, self._arm_dof_idx] + arm_delta).clamp(
                self._arm_lo, self._arm_hi
            )
            self._joint_targets[:, self._arm_dof_idx] = new_arm
        else:
            self._apply_ee_delta(actions[:, :6])

        gripper_cmd = actions[:, -1]
        close = gripper_cmd > 0.0
        g_tgt = torch.where(
            close.unsqueeze(-1),
            self._grip_close.unsqueeze(0).expand(self.num_envs, -1),
            self._grip_open.unsqueeze(0).expand(self.num_envs, -1),
        )
        self._joint_targets[:, self._grip_dof_idx] = g_tgt

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets)

    def _apply_ee_delta(self, ee_delta: torch.Tensor) -> None:
        jac_full = self._robot.root_physx_view.get_jacobians()
        row0 = self._ee_body_idx * 6
        J = jac_full[:, row0:row0 + 6, :][:, :, self._arm_dof_idx]
        delta_ee = torch.cat([
            ee_delta[:, :3] * self.cfg.arm_action_scale,
            ee_delta[:, 3:] * self.cfg.rot_action_scale,
        ], dim=-1)
        dq = torch.bmm(J.transpose(1, 2), delta_ee.unsqueeze(-1)).squeeze(-1)
        new_arm = (self._joint_targets[:, self._arm_dof_idx] + dq).clamp(
            self._arm_lo, self._arm_hi
        )
        self._joint_targets[:, self._arm_dof_idx] = new_arm

    # ── Observations ─────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        pc_robot = self._synthesize_pointcloud()   # (B, NUM_PC_POINTS*3)

        ee_pos_w  = self._robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_body_idx]
        base_pos  = self._robot.data.root_pos_w
        base_quat = self._robot.data.root_quat_w

        ee_pos_b, ee_quat_b = subtract_frame_transforms(base_pos, base_quat, ee_pos_w, ee_quat_w)
        goal_quat_w = torch.tensor([[1., 0., 0., 0.]], device=self.device).expand(self.num_envs, -1)
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
            base_pos, base_quat, self._goal_pos_w, goal_quat_w
        )

        g_pos = self._robot.data.joint_pos[:, self._grip_dof_idx].mean(dim=-1, keepdim=True)
        gripper_state = (
            (g_pos - self._grip_open[0]) / (self._grip_close[0] - self._grip_open[0] + 1e-6)
        ).clamp(0., 1.)

        proprio = torch.cat(
            [ee_pos_b, ee_quat_b, goal_pos_b, goal_quat_b, gripper_state, self._prev_action],
            dim=-1,
        )
        obs = torch.cat([pc_robot, proprio], dim=-1)
        return {"policy": obs}

    def _synthesize_pointcloud(self) -> torch.Tensor:
        """
        Per-env: randomly subsample NUM_PC_POINTS from the pre-loaded surface PC,
        rotate to world frame using object orientation, add noise, transform to robot base frame.
        """
        B, N = self.num_envs, NUM_PC_POINTS
        device = self.device

        obj_pos_w  = self._active_obj_pos()   # (B, 3)
        obj_quat_w = self._active_obj_quat()  # (B, 4)

        # Gather pre-loaded PC for each env's active shape, subsample to N
        # self._obj_pcs: (NUM_SHAPES, _PC_PRE_N, 3)
        shape_pcs = self._obj_pcs[self._env_shape]  # (B, 512, 3)
        idx = torch.randint(0, _PC_PRE_N, (B, N), device=device)
        pc_local = shape_pcs.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, N, 3)

        # Rotate local → world
        q_rep    = obj_quat_w.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 4)
        pc_world = quat_rotate(q_rep, pc_local.reshape(B * N, 3)).reshape(B, N, 3)
        pc_world = pc_world + obj_pos_w.unsqueeze(1)

        # Sensor noise
        pc_world = pc_world + torch.randn_like(pc_world) * _PC_NOISE

        # Transform to robot-base frame
        base_pos      = self._robot.data.root_pos_w
        base_quat     = self._robot.data.root_quat_w
        base_quat_inv = quat_conjugate(base_quat)

        rel    = pc_world - base_pos.unsqueeze(1)
        q_inv  = base_quat_inv.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 4)
        pc_rob = quat_rotate(q_inv, rel.reshape(B * N, 3)).reshape(B, N, 3)

        return pc_rob.reshape(B, -1)

    # ── Reward ────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self._active_obj_pos()
        ee_pos  = self._robot.data.body_pos_w[:, self._ee_body_idx]

        dist_ee_obj   = torch.norm(ee_pos - obj_pos, dim=-1)
        lift_h        = (obj_pos[:, 2] - self.cfg.table_surface_z).clamp(0)
        dist_obj_goal = torch.norm(obj_pos[:, :2] - self._goal_pos_w[:, :2], dim=-1)

        success = (dist_obj_goal < self.cfg.place_success_radius) & (lift_h > 0)
        fell    = obj_pos[:, 2] < (self.cfg.table_surface_z - 0.10)
        lifted  = lift_h > 0.05

        return (
            - dist_ee_obj * self.cfg.approach_reward_scale
            + lift_h      * self.cfg.lift_reward_scale
            - dist_obj_goal * self.cfg.place_reward_scale * lifted.float()
            + success.float() * self.cfg.success_bonus
            + fell.float()    * self.cfg.fall_penalty
            - self._prev_action.pow(2).sum(-1) * self.cfg.action_penalty_scale
        )

    # ── Termination ───────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self._active_obj_pos()
        dist    = torch.norm(obj_pos[:, :2] - self._goal_pos_w[:, :2], dim=-1)
        lift_h  = obj_pos[:, 2] - self.cfg.table_surface_z

        success    = (dist < self.cfg.place_success_radius) & (lift_h > 0)
        fell       = obj_pos[:, 2] < (self.cfg.table_surface_z - 0.20)
        terminated = success | fell
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        ids     = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n       = ids.numel()
        origins = self.scene.env_origins[ids]

        # Robot reset
        def_pos = self._robot.data.default_joint_pos[ids]
        def_vel = self._robot.data.default_joint_vel[ids]
        self._robot.write_joint_state_to_sim(def_pos, def_vel, env_ids=ids)
        self._robot.set_joint_position_target(def_pos, env_ids=ids)
        self._joint_targets[ids] = def_pos

        # Randomly assign one shape per resetting env
        self._env_shape[ids] = torch.randint(0, NUM_SHAPES, (n,), device=self.device)

        # Randomise spawn position and goal
        def rnd(lo, hi, count):
            return torch.rand(count, device=self.device) * (hi - lo) + lo

        spawn_x = rnd(*self.cfg.spawn_x_range, n)
        spawn_y = rnd(*self.cfg.spawn_y_range, n)
        goal_x  = rnd(*self.cfg.goal_x_range,  n)
        goal_y  = rnd(*self.cfg.goal_y_range,  n)

        self._goal_pos_w[ids, 0] = origins[:, 0] + goal_x
        self._goal_pos_w[ids, 1] = origins[:, 1] + goal_y
        self._goal_pos_w[ids, 2] = self.cfg.table_surface_z

        obj_spawn_w = torch.stack([
            origins[:, 0] + spawn_x,
            origins[:, 1] + spawn_y,
            torch.full((n,), self.cfg.table_surface_z + 0.08, device=self.device),
        ], dim=-1)

        ident_quat = torch.tensor([[1., 0., 0., 0.]], device=self.device).expand(n, -1)
        zero_vel   = torch.zeros(n, 6, device=self.device)

        for shape_idx, obj in enumerate(self._objs):
            active_mask = self._env_shape[ids] == shape_idx
            active_ids  = ids[active_mask]
            hidden_ids  = ids[~active_mask]

            if active_ids.numel() > 0:
                na = active_ids.numel()
                pos = obj_spawn_w[active_mask]
                obj.write_root_pose_to_sim(
                    torch.cat([pos, ident_quat[:na]], dim=-1), env_ids=active_ids
                )
                obj.write_root_velocity_to_sim(zero_vel[:na], env_ids=active_ids)

            if hidden_ids.numel() > 0:
                nh = hidden_ids.numel()
                h_pos = origins[~active_mask].clone()
                h_pos[:, 2] = -20.0
                obj.write_root_pose_to_sim(
                    torch.cat([h_pos, ident_quat[:nh]], dim=-1), env_ids=hidden_ids
                )
                obj.write_root_velocity_to_sim(zero_vel[:nh], env_ids=hidden_ids)

        self._prev_action[ids] = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _active_obj_pos(self) -> torch.Tensor:
        pos = torch.zeros(self.num_envs, 3, device=self.device)
        for shape_idx, obj in enumerate(self._objs):
            mask = self._env_shape == shape_idx
            if mask.any():
                pos[mask] = obj.data.root_pos_w[mask]
        return pos

    def _active_obj_quat(self) -> torch.Tensor:
        quat = torch.zeros(self.num_envs, 4, device=self.device)
        quat[:, 0] = 1.0
        for shape_idx, obj in enumerate(self._objs):
            mask = self._env_shape == shape_idx
            if mask.any():
                quat[mask] = obj.data.root_quat_w[mask]
        return quat
