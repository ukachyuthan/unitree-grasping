"""
Residual predictive control environment.

Extends Path A with closed-loop corrections:
  obs  = point cloud + proprio (EE pose, phase, grasp target, contact)
  action = [grasp(5) on first step | residual Δpos Δrot Δgrip every step]

Scripted IK provides the baseline target; the policy adds a scaled residual each
policy step (every cfg.decimation physics steps).

Method 2: optional scripted antipodal grasp base — policy residual corrects reach.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from isaaclab.utils.math import (
    quat_from_angle_axis,
    quat_mul,
    quat_inv,
    quat_rotate,
    compute_pose_error,
)

from envs._object_registry import shape_asset
from envs.grasp_pose_env import GraspPoseEnv
from envs.grasp_pose_env_cfg import N_APPROACH, N_CLOSE, N_DESCEND, N_LIFT, N_HOLD, N_TRANSPORT, N_LOWER, N_OPEN, EXEC_STEPS
from envs.grasp_pose_residual_env_cfg import (
    GraspPoseResidualEnvCfg,
    NUM_GRASP_ACTIONS,
    PROPRIO_DIM,
)


class GraspPoseResidualEnv(GraspPoseEnv):
    cfg: GraspPoseResidualEnvCfg

    def __init__(self, cfg: GraspPoseResidualEnvCfg, render_mode=None):
        super().__init__(cfg, render_mode=render_mode)
        B = self.num_envs
        self._exec_step = torch.zeros(B, dtype=torch.long, device=self.device)
        self._policy_step = torch.zeros(B, dtype=torch.long, device=self.device)
        self._residual_pos = torch.zeros(B, 3, device=self.device)
        self._residual_rot = torch.zeros(B, 3, device=self.device)
        self._residual_grip = torch.zeros(B, 1, device=self.device)
        self._prev_obj_z = torch.zeros(B, device=self.device)
        self._load_antipodal_bases()

    def _load_antipodal_bases(self):
        """Preload one horizontal antipodal (center, roll, width) per shape."""
        n = self._num_shapes
        centers = torch.zeros(n, 3, device=self.device)
        rolls = torch.full((n,), 0.741, device=self.device)
        widths = torch.full(
            (n,),
            0.5 * (self.cfg.grasp_width_bounds[0] + self.cfg.grasp_width_bounds[1]),
            device=self.device,
        )
        scale = float(self.cfg.object_scale)
        z_max = abs(self.cfg.antipodal_max_abs_approach_z)
        w_lo, w_hi = self.cfg.grasp_width_bounds
        x_lo, x_hi = self.cfg.grasp_x_bounds
        y_lo, y_hi = self.cfg.grasp_y_bounds
        z_lo, z_hi = self.cfg.grasp_z_bounds
        n_ok = 0

        for i, name in enumerate(self._shape_names):
            g_path = shape_asset(name, "_grasps.json")
            pc_path = shape_asset(name, "_pc.npy")
            chosen = None
            if g_path.exists():
                with open(g_path) as f:
                    data = json.load(f)
                grasps = data["grasps"] if isinstance(data, dict) else data
                grasps = sorted(grasps, key=lambda g: g["quality"], reverse=True)
                for g in grasps:
                    approach = np.asarray(g["approach"], dtype=np.float64)
                    an = np.linalg.norm(approach) + 1e-9
                    a = approach / an
                    if abs(a[2]) > z_max:
                        continue
                    xy = math.sqrt(a[0] * a[0] + a[1] * a[1])
                    if xy < 1e-4:
                        continue
                    jaw = np.array([a[0] / xy, a[1] / xy, 0.0], dtype=np.float64)
                    width = float(np.clip(g["width"] * scale, w_lo, w_hi))
                    center = np.asarray(g["center"], dtype=np.float64) * scale
                    center[2] = float(np.clip(center[2], z_lo + 0.005, z_hi - 0.005))
                    if not (
                        x_lo <= center[0] <= x_hi
                        and y_lo <= center[1] <= y_hi
                        and z_lo <= center[2] <= z_hi
                    ):
                        continue
                    yaw = math.atan2(jaw[0], jaw[1])
                    chosen = (center, 0.741 + yaw, width)
                    break

            if chosen is None and pc_path.exists():
                pc = np.load(str(pc_path)).astype(np.float64) * scale
                center = pc.mean(axis=0)
                center[2] = float(np.clip(center[2], z_lo + 0.005, z_hi - 0.005))
                chosen = (center, 0.741, 0.5 * (w_lo + w_hi))

            if chosen is not None:
                centers[i] = torch.tensor(chosen[0], dtype=torch.float32, device=self.device)
                rolls[i] = float(chosen[1])
                widths[i] = float(chosen[2])
                n_ok += 1

        self._antipodal_center = centers
        self._antipodal_roll = rolls
        self._antipodal_width = widths
        print(
            f"[GraspPoseResidualEnv] antipodal bases: {n_ok}/{n} shapes "
            f"(use_antipodal_base={self.cfg.use_antipodal_base}, "
            f"freeze={self.cfg.freeze_grasp_to_antipodal})"
        )

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._exec_step[env_ids] = 0
        self._policy_step[env_ids] = 0
        self._residual_pos[env_ids] = 0.0
        self._residual_rot[env_ids] = 0.0
        self._residual_grip[env_ids] = 0.0
        self._prev_obj_z[env_ids] = self._spawn_z[env_ids]
        self._last_hold_obj_z[env_ids] = self._spawn_z[env_ids]

    # ── Policy step ─────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store grasp (first step per episode) + residual deltas for this interval."""
        a = actions.clamp(-1, 1)
        first = self._policy_step == 0
        if first.any():
            self._decode_grasp_action(a[:, :NUM_GRASP_ACTIONS], first)

        self._residual_pos = a[:, NUM_GRASP_ACTIONS : NUM_GRASP_ACTIONS + 3] * self.cfg.residual_pos_scale
        self._residual_rot = a[:, NUM_GRASP_ACTIONS + 3 : NUM_GRASP_ACTIONS + 6] * self.cfg.residual_rot_scale
        self._residual_grip = a[:, NUM_GRASP_ACTIONS + 6 : NUM_GRASP_ACTIONS + 7] * self.cfg.residual_grip_scale

        self._policy_step += 1
        self._J_cache = None
        self._J_pos_cache = None
        self._ik_call = 0

    def _decode_grasp_action(self, a: torch.Tensor, mask: torch.Tensor):
        """Decode 5-D grasp action for envs at the start of an episode."""
        x_lo, x_hi = self.cfg.grasp_x_bounds
        y_lo, y_hi = self.cfg.grasp_y_bounds
        z_lo, z_hi = self.cfg.grasp_z_bounds
        t_lo, t_hi = self.cfg.grasp_tilt_bounds
        r_lo, r_hi = self.cfg.grasp_roll_bounds
        w_lo, w_hi = self.cfg.grasp_width_bounds

        if self.cfg.use_antipodal_base:
            shape = self._env_shape
            base = self._antipodal_center[shape]
            base_roll = self._antipodal_roll[shape]
            base_width = self._antipodal_width[shape]
            if self.cfg.freeze_grasp_to_antipodal:
                grasp_local = base
                tilt = torch.zeros(self.num_envs, device=self.device)
                roll = base_roll
                width = base_width
            else:
                s = self.cfg.antipodal_grasp_delta_scale
                dx = a[:, 0] * s * 0.5 * (x_hi - x_lo)
                dy = a[:, 1] * s * 0.5 * (y_hi - y_lo)
                dz = a[:, 2] * s * 0.5 * (z_hi - z_lo)
                grasp_local = base + torch.stack([dx, dy, dz], dim=-1)
                grasp_local = torch.stack(
                    [
                        grasp_local[:, 0].clamp(x_lo, x_hi),
                        grasp_local[:, 1].clamp(y_lo, y_hi),
                        grasp_local[:, 2].clamp(z_lo, z_hi),
                    ],
                    dim=-1,
                )
                tilt = a[:, 3] * s * 0.5 * (t_hi - t_lo)
                roll = base_roll + a[:, 4] * s * 0.5 * (r_hi - r_lo)
                width = base_width
        else:
            gx = (a[:, 0] + 1) / 2 * (x_hi - x_lo) + x_lo
            gy = (a[:, 1] + 1) / 2 * (y_hi - y_lo) + y_lo
            gz = (a[:, 2] + 1) / 2 * (z_hi - z_lo) + z_lo
            grasp_local = torch.stack([gx, gy, gz], dim=-1)
            tilt = (a[:, 3] + 1) / 2 * (t_hi - t_lo) + t_lo
            roll = (a[:, 4] + 1) / 2 * (r_hi - r_lo) + r_lo
            width = torch.full(
                (self.num_envs,), 0.5 * (w_lo + w_hi), device=self.device
            )

        if self.cfg.pc_augment_yaw and not self.cfg.use_antipodal_base:
            inv_yaw = -self._pc_aug_yaw
            cos_y, sin_y = inv_yaw.cos(), inv_yaw.sin()
            gx_r = cos_y * grasp_local[:, 0] - sin_y * grasp_local[:, 1]
            gy_r = sin_y * grasp_local[:, 0] + cos_y * grasp_local[:, 1]
            grasp_local = torch.stack([gx_r, gy_r, grasp_local[:, 2]], dim=-1)

        m = mask.unsqueeze(-1)
        self._grasp_target = torch.where(m, grasp_local, self._grasp_target)
        self._grasp_origin_w = torch.where(m, self._obj_anchor[:, :3], self._grasp_origin_w)
        self._grasp_origin_quat = torch.where(m, self._obj_anchor[:, 3:7], self._grasp_origin_quat)

        self._tilt_target = torch.where(mask, tilt, self._tilt_target)
        self._roll_target = torch.where(mask, roll.clamp(r_lo, r_hi), self._roll_target)
        self._grasp_width = torch.where(mask, width.clamp(w_lo, w_hi), self._grasp_width)
        per_finger = (self._grasp_width / 2).clamp(
            self.cfg.gripper_close_val, self.cfg.gripper_open_val
        )
        n_grip = self._gripper_close_target.shape[-1]
        # Keep `_gripper_close` as the shared (n_grip,) template; per-env width
        # lives only in `_gripper_close_target` (matches Path A).
        self._gripper_close_target = torch.where(
            m, per_finger.unsqueeze(-1).expand(-1, n_grip), self._gripper_close_target
        )
        self._grasp_locked = torch.where(mask, torch.zeros_like(self._grasp_locked), self._grasp_locked)

    # ── Masked scripted execution ─────────────────────────────────────────────

    def _apply_action(self):
        s = self._exec_step
        self._hold_idle_joints()

        T0 = N_APPROACH
        T1 = T0 + N_CLOSE
        T2 = T1 + N_LIFT
        T3 = T2 + N_HOLD
        T5 = T3 + N_TRANSPORT + N_LOWER

        m_app = s < T0
        m_close = (s >= T0) & (s < T1)
        m_lift = (s >= T1) & (s < T2)
        m_hold = (s >= T2) & (s < T3)
        m_rest = s >= T3

        if (m_app | m_close).any():
            self._anchor_objects()

        ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
        target_w = ee_w.clone()

        m_ac = s < T1
        target_w = torch.where(m_ac.unsqueeze(-1), self._approach_target_w(), target_w)

        lift_start = T1
        just_lift = s == lift_start
        if just_lift.any():
            self._lift_base_w = torch.where(just_lift.unsqueeze(-1), ee_w, self._lift_base_w)
        k = (s - lift_start).clamp(min=0).float() + 1.0
        lift_tgt = self._lift_base_w.clone()
        lift_tgt[:, 2] += self.cfg.lift_height_m * (k / N_LIFT)
        target_w = torch.where(m_lift.unsqueeze(-1), lift_tgt, target_w)

        hold_tgt = self._lift_base_w.clone()
        hold_tgt[:, 2] += self.cfg.lift_height_m
        target_w = torch.where((m_hold | m_rest).unsqueeze(-1), hold_tgt, target_w)

        ee_q = self._robot.data.body_quat_w[:, self._ee_body_idx]
        quat_tgt = self._approach_target_quat()
        quat_tgt = torch.where(m_ac.unsqueeze(-1), quat_tgt, ee_q)
        self._ik_to(target_w, quat_tgt)

        q_open = self._gripper_open.unsqueeze(0).expand(self.num_envs, -1)
        q_close = self._gripper_close_target  # (num_envs, n_grip)
        q_arm = self._arm_q_des

        t_close = ((s - T0).float() / N_CLOSE).clamp(0, 1).unsqueeze(-1)
        grip_delta = self._residual_grip * (1.0 / float(self.cfg.decimation))
        t_close = (t_close + grip_delta * m_close.unsqueeze(-1).float()).clamp(0, 1)
        q_from_close = (1 - t_close) * q_open + t_close * q_close

        q_grip = q_close.clone()
        q_grip = torch.where(m_app.unsqueeze(-1), q_open, q_grip)
        q_grip = torch.where(m_close.unsqueeze(-1), q_from_close, q_grip)

        self._robot.set_joint_position_target(q_arm, joint_ids=self._arm_dof_idx)
        self._robot.set_joint_position_target(q_grip, joint_ids=self._grip_dof_idx)

        end_close = (s + 1 >= T1) & m_close
        if end_close.any():
            ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
            obj_w = self._get_active_obj_pos()
            contact = self._fingers_in_contact() & end_close
            if contact.any():
                self._grasp_offset[contact] = obj_w[contact] - ee_pos[contact]

        track = (s >= T1) & (s < T3)
        if track.any():
            obj_z = self._get_active_obj_pos()[:, 2]
            self._last_hold_obj_z = torch.where(
                track, torch.max(self._last_hold_obj_z, obj_z), self._last_hold_obj_z
            )

        self._exec_step += 1

        if m_app.any():
            self._pin_objects()
        carry = (self.cfg.kinematic_grasp or self.cfg.contact_carry) & (s >= T1) & (s < T5)
        if carry.any():
            self._sync_grasped_object()

        if self._grasp_marker is not None:
            self._update_grasp_markers()

    def _ik_to(self, target_w: torch.Tensor, target_quat: torch.Tensor | None = None):
        """Scripted IK target + policy residual (pos + axis-angle rot)."""
        T1 = N_APPROACH + N_CLOSE
        active = self._exec_step < T1 + N_LIFT
        # Spread residual over physics sub-steps within one policy interval.
        sub_scale = 1.0 / float(self.cfg.decimation)
        pos_delta = self._residual_pos * sub_scale
        rot_delta = self._residual_rot * sub_scale
        target_w = target_w + active.unsqueeze(-1) * pos_delta
        if target_quat is not None:
            rot = rot_delta
            angle = rot.norm(dim=-1).clamp(max=0.5)
            axis = rot / angle.unsqueeze(-1).clamp(min=1e-6)
            delta_q = quat_from_angle_axis(angle, axis)
            target_quat = quat_mul(delta_q, target_quat)

        in_ac = self._exec_step < T1
        if in_ac.any():
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
            if in_ac.any():
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

        ti, ri = self._tilt_arm_idx, self._roll_arm_idx
        q_tgt[:, ti] = self._tilt_target.clamp(lo[:, ti], hi[:, ti])
        q_tgt[:, ri] = self._roll_target.clamp(lo[:, ri], hi[:, ri])

        self._arm_q_des = q_tgt
        self._robot.set_joint_position_target(q_tgt, joint_ids=self._arm_dof_idx)

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_proprio(self) -> torch.Tensor:
        ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
        obj_w = self._get_active_obj_pos()
        inv_q = quat_inv(self._grasp_origin_quat)
        ee_local = quat_rotate(inv_q, ee_w - obj_w)

        grip = self._robot.data.joint_pos[:, self._grip_dof_idx].mean(dim=-1, keepdim=True)
        phase = (self._exec_step.float() / EXEC_STEPS).unsqueeze(-1)
        t_lo, t_hi = self.cfg.grasp_tilt_bounds
        r_lo, r_hi = self.cfg.grasp_roll_bounds
        tilt_n = ((self._tilt_target - t_lo) / (t_hi - t_lo + 1e-6)).unsqueeze(-1)
        roll_n = ((self._roll_target - r_lo) / (r_hi - r_lo + 1e-6)).unsqueeze(-1)
        contact = self._fingers_in_contact().float().unsqueeze(-1)
        lift_d = (obj_w[:, 2:3] - self._spawn_z.unsqueeze(-1)).clamp(min=0.0) / self.cfg.lift_target_m

        return torch.cat(
            [ee_local, grip, phase, self._grasp_target, tilt_n, roll_n, contact, lift_d],
            dim=-1,
        )

    def _get_observations(self) -> dict:
        pc = self._synthesize_pointcloud()
        prop = self._get_proprio()
        assert prop.shape[-1] == PROPRIO_DIM, f"proprio dim {prop.shape[-1]} != {PROPRIO_DIM}"
        return {"policy": torch.cat([pc, prop], dim=-1)}

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        sparse = super()._get_rewards()

        ee_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :3]
        grasp_w = self._grasp_point_world()
        reach = (ee_w - grasp_w).norm(dim=-1)
        reach_r = torch.exp(-reach / self.cfg.reach_sigma_m)

        contact_r = self._contact_area_reward()
        obj_z = self._get_active_obj_pos()[:, 2]
        lift_step = (obj_z - self._prev_obj_z).clamp(min=0.0) / self.cfg.lift_target_m
        self._prev_obj_z = obj_z.clone()

        dense = (
            self.cfg.reach_reward_weight * reach_r
            + self.cfg.step_contact_reward_weight * contact_r
            + self.cfg.step_lift_reward_weight * lift_step
        )
        terminal = self._exec_step >= EXEC_STEPS
        return torch.where(terminal, sparse, dense)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._exec_step >= EXEC_STEPS
        truncated = torch.zeros_like(terminated)
        return terminated, truncated

    @property
    def terminal_lift_reward(self) -> torch.Tensor:
        """Sparse lift reward (same as Path A terminal reward)."""
        delta = (self._last_hold_obj_z - self._spawn_z).clamp(min=0.0)
        return (delta / self.cfg.lift_target_m).clamp(max=1.0)
