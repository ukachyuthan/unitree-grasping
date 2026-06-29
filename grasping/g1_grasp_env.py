"""
G1 left-arm pick-and-place environment  (DirectRLEnv).

Key design choices
──────────────────
• During training, point clouds are synthesized from ground-truth object geometry
  + 3 mm Gaussian noise — identical to DexPoint / UniDexGrasp.  This removes the
  RTX rendering dependency and cuts wall-clock iteration time by ~3×.
• At deployment, swap _synthesize_pointcloud() for depth_to_pointcloud_world() in
  pointcloud_utils.py with the real camera's K and T_cam_robot.  Policy weights
  transfer unchanged because the observation format is identical.
• Action interface: joint-space Δ (5 arm DOF + gripper) via use_joint_space_control,
  or EE-space Δ(pos, rot) → Jacobian-transpose IK for robot-agnostic deployment.

Episode:
  1. Spawn one random primitive (box / sphere / cylinder) on table at A.
  2. Sample a random goal B on same surface.
  3. Phase 0 – approach: reward = −‖EE − object‖
     Phase 1 – lift:     reward = clip(object_z − table_z, 0, max)
     Phase 2 – place:    reward = −‖object_xy − goal_xy‖ + success_bonus
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_rotate,
    quat_conjugate,
    subtract_frame_transforms,
)
import isaaclab.sim as sim_utils

from grasping.g1_grasp_env_cfg import G1GraspEnvCfg, NUM_PC_POINTS

SHAPE_BOX      = 0
SHAPE_SPHERE   = 1
SHAPE_CYLINDER = 2
NUM_SHAPES     = 3

# Object dimensions matching spawn configs in G1GraspEnvCfg
_BOX_HALF = (0.02, 0.02, 0.05)   # CuboidCfg size=(0.04, 0.04, 0.10)
_SPH_RAD  = 0.04                  # SphereCfg  radius=0.04
_CYL_RAD  = 0.03                  # CylinderCfg radius=0.03
_CYL_H    = 0.12                  # CylinderCfg height=0.12
_PC_NOISE = 0.003                 # sensor noise std (3 mm)


# ── Surface samplers ─────────────────────────────────────────────────────────

def _sample_box_surface(n_envs: int, n_pts: int,
                        hx: float, hy: float, hz: float,
                        device: torch.device) -> torch.Tensor:
    """Uniform surface sample from box with half-extents (hx,hy,hz). Returns (B,N,3)."""
    ax, ay, az = hy * hz, hx * hz, hx * hy
    probs = torch.tensor([ax, ax, ay, ay, az, az], dtype=torch.float32, device=device)
    probs = probs / probs.sum()
    face  = torch.multinomial(probs.unsqueeze(0).expand(n_envs * n_pts, -1), 1).squeeze(-1)
    u = torch.rand(n_envs * n_pts, device=device) * 2 - 1
    v = torch.rand(n_envs * n_pts, device=device) * 2 - 1
    pts = torch.zeros(n_envs * n_pts, 3, device=device)
    m = face == 0; pts[m, 0] =  hx; pts[m, 1] = hy * u[m]; pts[m, 2] = hz * v[m]
    m = face == 1; pts[m, 0] = -hx; pts[m, 1] = hy * u[m]; pts[m, 2] = hz * v[m]
    m = face == 2; pts[m, 1] =  hy; pts[m, 0] = hx * u[m]; pts[m, 2] = hz * v[m]
    m = face == 3; pts[m, 1] = -hy; pts[m, 0] = hx * u[m]; pts[m, 2] = hz * v[m]
    m = face == 4; pts[m, 2] =  hz; pts[m, 0] = hx * u[m]; pts[m, 1] = hy * v[m]
    m = face == 5; pts[m, 2] = -hz; pts[m, 0] = hx * u[m]; pts[m, 1] = hy * v[m]
    return pts.reshape(n_envs, n_pts, 3)


def _sample_sphere_surface(n_envs: int, n_pts: int,
                            radius: float, device: torch.device) -> torch.Tensor:
    """Uniform surface sample from sphere. Returns (B,N,3)."""
    pts = torch.randn(n_envs * n_pts, 3, device=device)
    pts = pts / (pts.norm(dim=-1, keepdim=True) + 1e-8) * radius
    return pts.reshape(n_envs, n_pts, 3)


def _sample_cylinder_surface(n_envs: int, n_pts: int,
                              radius: float, height: float,
                              device: torch.device) -> torch.Tensor:
    """Uniform surface sample from cylinder (barrel + 2 caps). Returns (B,N,3)."""
    barrel_a = 2 * math.pi * radius * height
    cap_a    = math.pi * radius ** 2
    total    = barrel_a + 2 * cap_a
    probs = torch.tensor([barrel_a, cap_a, cap_a], dtype=torch.float32, device=device)
    probs = probs / total
    part  = torch.multinomial(probs.unsqueeze(0).expand(n_envs * n_pts, -1), 1).squeeze(-1)

    pts = torch.zeros(n_envs * n_pts, 3, device=device)

    # Barrel
    m = part == 0
    if m.any():
        theta = torch.rand(m.sum(), device=device) * 2 * math.pi
        z     = torch.rand(m.sum(), device=device) * height - height / 2
        pts[m, 0] = radius * torch.cos(theta)
        pts[m, 1] = radius * torch.sin(theta)
        pts[m, 2] = z

    # Top cap
    m = part == 1
    if m.any():
        r_  = torch.sqrt(torch.rand(m.sum(), device=device)) * radius
        th_ = torch.rand(m.sum(), device=device) * 2 * math.pi
        pts[m, 0] = r_ * torch.cos(th_)
        pts[m, 1] = r_ * torch.sin(th_)
        pts[m, 2] = height / 2

    # Bottom cap
    m = part == 2
    if m.any():
        r_  = torch.sqrt(torch.rand(m.sum(), device=device)) * radius
        th_ = torch.rand(m.sum(), device=device) * 2 * math.pi
        pts[m, 0] = r_ * torch.cos(th_)
        pts[m, 1] = r_ * torch.sin(th_)
        pts[m, 2] = -height / 2

    return pts.reshape(n_envs, n_pts, 3)


# ── Environment ──────────────────────────────────────────────────────────────

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

        # Arm / gripper joint limits
        self._arm_lo = self._robot.data.soft_joint_pos_limits[0, self._arm_dof_idx, 0].to(self.device)
        self._arm_hi = self._robot.data.soft_joint_pos_limits[0, self._arm_dof_idx, 1].to(self.device)
        g_lo = self._robot.data.soft_joint_pos_limits[0, self._grip_dof_idx, 0].to(self.device)
        g_hi = self._robot.data.soft_joint_pos_limits[0, self._grip_dof_idx, 1].to(self.device)
        self._grip_open  = g_hi
        self._grip_close = g_lo

        self._joint_targets = self._robot.data.default_joint_pos.clone()  # (B, J)

        self._env_shape   = torch.arange(self.num_envs, device=self.device) % NUM_SHAPES
        self._goal_pos_w  = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    # ── Scene ────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._table        = RigidObject(self.cfg.table)
        self._obj_box      = RigidObject(self.cfg.object_box)
        self._obj_sphere   = RigidObject(self.cfg.object_sphere)
        self._obj_cylinder = RigidObject(self.cfg.object_cylinder)
        for name, obj in [
            ("table",        self._table),
            ("obj_box",      self._obj_box),
            ("obj_sphere",   self._obj_sphere),
            ("obj_cylinder", self._obj_cylinder),
        ]:
            self.scene.rigid_objects[name] = obj

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── Action application ────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_action = actions.clone()
        actions = actions.clamp(-1.0, 1.0)

        if self.cfg.use_joint_space_control:
            arm_delta = actions[:, : self._num_arm_dofs] * self.cfg.arm_action_scale
            new_arm = (self._joint_targets[:, self._arm_dof_idx] + arm_delta).clamp(
                self._arm_lo, self._arm_hi
            )
            self._joint_targets[:, self._arm_dof_idx] = new_arm
        else:
            self._apply_ee_delta(actions[:, :6])

        gripper_cmd = actions[:, -1]
        close = (gripper_cmd > 0.0)
        g_tgt = torch.where(
            close.unsqueeze(-1),
            self._grip_close.unsqueeze(0).expand(self.num_envs, -1),
            self._grip_open.unsqueeze(0).expand(self.num_envs, -1),
        )
        self._joint_targets[:, self._grip_dof_idx] = g_tgt

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets)

    def _apply_ee_delta(self, ee_delta: torch.Tensor) -> None:
        """Jacobian-transpose IK: Δ(pos, rot) → Δjoint.
        Isaac Lab jacobian shape: (B, num_bodies * 6, num_dofs).
        EE body rows are at indices [ee_body_idx*6 : ee_body_idx*6+6].
        """
        jac_full = self._robot.root_physx_view.get_jacobians()  # (B, L*6, D)
        row0 = self._ee_body_idx * 6
        J = jac_full[:, row0:row0 + 6, :][:, :, self._arm_dof_idx]  # (B, 6, num_arm_dofs)
        delta_ee = torch.cat([
            ee_delta[:, :3] * self.cfg.arm_action_scale,
            ee_delta[:, 3:] * self.cfg.rot_action_scale,
        ], dim=-1)  # (B, 6)
        dq = torch.bmm(J.transpose(1, 2), delta_ee.unsqueeze(-1)).squeeze(-1)  # (B, num_arm_dofs)
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

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pos, base_quat, ee_pos_w, ee_quat_w
        )
        goal_quat_w = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=self.device
        ).expand(self.num_envs, -1)
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
            base_pos, base_quat, self._goal_pos_w, goal_quat_w
        )

        g_pos = self._robot.data.joint_pos[:, self._grip_dof_idx].mean(dim=-1, keepdim=True)
        gripper_state = (
            (g_pos - self._grip_open[0]) / (self._grip_close[0] - self._grip_open[0] + 1e-6)
        ).clamp(0.0, 1.0)

        proprio = torch.cat(
            [ee_pos_b, ee_quat_b, goal_pos_b, goal_quat_b, gripper_state, self._prev_action],
            dim=-1,
        )   # (B, 22)

        obs = torch.cat([pc_robot, proprio], dim=-1)   # (B, 406)
        return {"policy": obs}

    def _synthesize_pointcloud(self) -> torch.Tensor:
        """
        Synthesize object point cloud from ground-truth geometry + Gaussian noise.

        Samples NUM_PC_POINTS from each active object's surface in object-local frame,
        rotates to world frame, adds 3 mm noise, then transforms to robot-base frame.
        Returns (B, NUM_PC_POINTS * 3).

        To switch to a real depth camera at deployment, replace this method with:
            pts_w = depth_to_pointcloud_world(depth, K, cam_pos_w, cam_quat_w)
            # then sample + transform to robot frame as below
        """
        B = self.num_envs
        N = NUM_PC_POINTS
        device = self.device

        obj_pos_w  = self._active_obj_pos()   # (B, 3)
        obj_quat_w = self._active_obj_quat()  # (B, 4) [w,x,y,z]

        # Sample in object-local frame per shape type
        pc_local = torch.zeros(B, N, 3, device=device)

        box_mask = self._env_shape == SHAPE_BOX
        if box_mask.any():
            n = box_mask.sum().item()
            pc_local[box_mask] = _sample_box_surface(n, N, *_BOX_HALF, device)

        sph_mask = self._env_shape == SHAPE_SPHERE
        if sph_mask.any():
            n = sph_mask.sum().item()
            pc_local[sph_mask] = _sample_sphere_surface(n, N, _SPH_RAD, device)

        cyl_mask = self._env_shape == SHAPE_CYLINDER
        if cyl_mask.any():
            n = cyl_mask.sum().item()
            pc_local[cyl_mask] = _sample_cylinder_surface(n, N, _CYL_RAD, _CYL_H, device)

        # Rotate local → world using object orientation
        q_rep = obj_quat_w.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 4)
        pc_world = quat_rotate(q_rep, pc_local.reshape(B * N, 3)).reshape(B, N, 3)
        pc_world = pc_world + obj_pos_w.unsqueeze(1)

        # Sensor noise
        pc_world = pc_world + torch.randn_like(pc_world) * _PC_NOISE

        # Transform to robot-base frame
        base_pos      = self._robot.data.root_pos_w       # (B, 3)
        base_quat     = self._robot.data.root_quat_w      # (B, 4)
        base_quat_inv = quat_conjugate(base_quat)         # (B, 4)

        rel   = pc_world - base_pos.unsqueeze(1)          # (B, N, 3)
        q_inv = base_quat_inv.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 4)
        pc_rob = quat_rotate(q_inv, rel.reshape(B * N, 3)).reshape(B, N, 3)

        return pc_rob.reshape(B, -1)   # (B, N*3)

    # ── Reward ────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self._active_obj_pos()
        ee_pos  = self._robot.data.body_pos_w[:, self._ee_body_idx]

        dist_ee_obj   = torch.norm(ee_pos  - obj_pos, dim=-1)
        lift_h        = (obj_pos[:, 2] - self.cfg.table_surface_z).clamp(0)
        dist_obj_goal = torch.norm(obj_pos[:, :2] - self._goal_pos_w[:, :2], dim=-1)

        success = (dist_obj_goal < self.cfg.place_success_radius) & (lift_h > 0)
        fell    = obj_pos[:, 2] < (self.cfg.table_surface_z - 0.10)
        lifted  = lift_h > 0.05

        reward = (
            # Phase 0 — always active: clear gradient signal to move EE toward object
            - dist_ee_obj * self.cfg.approach_reward_scale
            # Phase 1 — lift
            + lift_h * self.cfg.lift_reward_scale
            # Phase 2 — place penalty only once object is off the table.
            # Applying it before lift gives a constant negative signal that kills
            # the advantage gradient and prevents any policy learning.
            - dist_obj_goal * self.cfg.place_reward_scale * lifted.float()
            + success.float() * self.cfg.success_bonus
            + fell.float()    * self.cfg.fall_penalty
            - self._prev_action.pow(2).sum(-1) * self.cfg.action_penalty_scale
        )
        return reward

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
        origins = self.scene.env_origins[ids]   # (n, 3)

        # Robot reset
        def_pos = self._robot.data.default_joint_pos[ids]
        def_vel = self._robot.data.default_joint_vel[ids]
        self._robot.write_joint_state_to_sim(def_pos, def_vel, env_ids=ids)
        self._robot.set_joint_position_target(def_pos, env_ids=ids)
        self._joint_targets[ids] = def_pos

        # Randomise spawn A and goal B
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

        ident_quat = torch.tensor([[1.0, 0, 0, 0]], device=self.device).expand(n, -1)
        zero_vel   = torch.zeros(n, 6, device=self.device)
        hide_pos   = torch.zeros(n, 3, device=self.device)
        hide_pos[:, 2] = -20.0

        for shape_idx, obj in enumerate([self._obj_box, self._obj_sphere, self._obj_cylinder]):
            active = ids[self._env_shape[ids] == shape_idx]
            hidden = ids[self._env_shape[ids] != shape_idx]

            if active.numel() > 0:
                mask = self._env_shape[ids] == shape_idx
                pos  = obj_spawn_w[mask]
                obj.write_root_pose_to_sim(
                    torch.cat([pos, ident_quat[:active.numel()]], dim=-1), env_ids=active
                )
                obj.write_root_velocity_to_sim(zero_vel[:active.numel()], env_ids=active)

            if hidden.numel() > 0:
                mask  = self._env_shape[ids] != shape_idx
                h_pos = (hide_pos[mask] + origins[mask]).clone()
                h_pos[:, 2] = -20.0
                obj.write_root_pose_to_sim(
                    torch.cat([h_pos, ident_quat[:hidden.numel()]], dim=-1), env_ids=hidden
                )
                obj.write_root_velocity_to_sim(zero_vel[:hidden.numel()], env_ids=hidden)

        self._prev_action[ids] = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _active_obj_pos(self) -> torch.Tensor:
        """World position of each env's active object. (B, 3)"""
        pos = torch.zeros(self.num_envs, 3, device=self.device)
        for shape_idx, obj in enumerate([self._obj_box, self._obj_sphere, self._obj_cylinder]):
            mask = self._env_shape == shape_idx
            if mask.any():
                pos[mask] = obj.data.root_pos_w[mask]
        return pos

    def _active_obj_quat(self) -> torch.Tensor:
        """World quaternion [w,x,y,z] of each env's active object. (B, 4)"""
        quat = torch.zeros(self.num_envs, 4, device=self.device)
        quat[:, 0] = 1.0   # identity default
        for shape_idx, obj in enumerate([self._obj_box, self._obj_sphere, self._obj_cylinder]):
            mask = self._env_shape == shape_idx
            if mask.any():
                quat[mask] = obj.data.root_quat_w[mask]
        return quat
