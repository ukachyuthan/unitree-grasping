"""
G1 left-arm pick-and-place environment  (DirectRLEnv).

Key design choices
──────────────────
• Training uses pre-sampled surface point clouds (loaded from .npy at init),
  optionally mixed per-episode with rendered-depth from a randomised camera
  ring (use_camera_pc/camera_pc_prob — see grasp_pose_env.py for the original
  implementation of this mechanism), plus sensor-realistic noise (Gaussian +
  dropout + outliers, grasping.pointcloud_utils.add_sensor_noise).
• 12 procedural + (optionally) real YCB-derived shape families — see
  envs._object_registry — one random shape per episode per env.
• At deployment, replace _synthesize_pointcloud() with depth_to_pointcloud_world()
  from pointcloud_utils.py.  Policy weights transfer unchanged.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils.math import quat_rotate, quat_conjugate, subtract_frame_transforms
import isaaclab.sim as sim_utils

from envs._paths import data_path
from envs._object_registry import PROCEDURAL_SHAPE_NAMES, ycb_shape_names, shape_split
from envs.g1_grasp_env_cfg import G1GraspEnvCfg, NUM_PC_POINTS
from grasping.pointcloud_utils import add_sensor_noise

_PC_PRE_N = 512     # points in the pre-sampled numpy PCs


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
        # World-frame camera poses for the rendered-depth path (set per episode).
        self._cam_pos_w  = torch.zeros(self.num_envs, 3, device=self.device)
        self._cam_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._cam_quat_w[:, 0] = 1.0  # identity
        # Per-episode choice of camera-rendered vs. fast pre-loaded PC path
        # (Bernoulli(camera_pc_prob) each reset — see _reset_idx / _synthesize_pointcloud).
        self._use_camera_this_ep = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── Pre-load object point clouds ───────────────────────────────────
        # self._shape_names / self._num_shapes were set in _setup_scene() (called
        # inside super().__init__() above) so self._objs already matches this order.
        pcs = []
        for name in self._shape_names:
            pc_path = os.path.join(data_path("data/objects", shape_split(name), name), "000_pc.npy")
            if os.path.exists(pc_path):
                pc = torch.from_numpy(np.load(pc_path)).float().to(self.device)  # (512, 3)
            else:
                # Fallback: unit sphere if file missing
                pc = torch.randn(_PC_PRE_N, 3, device=self.device)
                pc = pc / pc.norm(dim=-1, keepdim=True) * 0.04
                print(f"[GraspEnv] WARNING: PC file not found for {name}, using sphere fallback")
            pcs.append(pc)
        self._obj_pcs = torch.stack(pcs, dim=0)  # (NUM_SHAPES, 512, 3)

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
        self._obj_graspnet_centers = torch.stack(gn_centers, dim=0).to(self.device)  # (NUM_SHAPES, _K_GN, 3)
        self._obj_graspnet_quality = torch.stack(gn_quality, dim=0).to(self.device)  # (NUM_SHAPES, _K_GN)
        if self.cfg.use_graspnet_reward and n_missing:
            print(f"[GraspEnv] WARNING: {n_missing}/{len(self._shape_names)} shapes missing "
                  f"000_graspnet.json — graspnet reward term is 0 for those until "
                  f"scripts/generate_graspnet_labels.py is run.")

    # ── Scene ────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        # Resolve the active shape set FIRST — everything else (scene spawn,
        # PC preload in __init__, domain randomization bounds) depends on it.
        if self.cfg.eval_object_mode:
            self._shape_names = ycb_shape_names("eval")
            if not self._shape_names:
                print("[G1GraspEnv] WARNING: eval_object_mode=True but no eval real "
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

        # Load all shape objects (procedural + real)
        self._objs: list[RigidObject] = []
        for name in self._shape_names:
            cfg_attr = f"object_{name}"
            obj = RigidObject(getattr(self.cfg, cfg_attr))
            self._objs.append(obj)
            self.scene.rigid_objects[f"obj_{name}"] = obj

        # Optional: rendered depth camera for viewpoint-diverse PC observations
        # (ported from grasp_pose_env.py — ring randomised around the object each
        # episode). Must be added before clone_environments so each env gets its
        # own prim.
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

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── Camera helpers (ported from grasp_pose_env.py) ─────────────────────────

    def _lookat_quat(self, eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Quaternion (w,x,y,z) for a camera at `eye` looking toward `target`.

        OpenGL camera convention (+X right, +Y up, -Z forward), Z-up world.
        Suitable for TiledCamera.set_world_poses with convention="opengl".
        """
        fwd = target - eye
        fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        world_up = torch.zeros_like(fwd); world_up[:, 2] = 1.0
        alt_up   = torch.zeros_like(fwd); alt_up[:, 1]   = 1.0
        degenerate = fwd[:, 2].abs() > 0.98
        wup = torch.where(degenerate.unsqueeze(-1), alt_up, world_up)

        right = torch.linalg.cross(fwd, wup)
        right = right / right.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        up    = torch.linalg.cross(right, fwd)

        R = torch.stack([right, up, -fwd], dim=-1)   # (B, 3, 3)

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
        self, n: int, obj_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one camera pose per env on a randomised ring around the object."""
        az   = torch.empty(n, device=self.device).uniform_(0.0, 2.0 * math.pi)
        h_r  = torch.empty(n, device=self.device).uniform_(*self.cfg.camera_horizontal_dist_range)
        elev = torch.empty(n, device=self.device).uniform_(*self.cfg.camera_height_range)

        cam_x = obj_pos[:, 0] + h_r * az.cos()
        cam_y = obj_pos[:, 1] + h_r * az.sin()
        cam_z = torch.full((n,), self.cfg.table_surface_z, device=self.device) + elev

        cam_pos  = torch.stack([cam_x, cam_y, cam_z], dim=-1)
        cam_quat = self._lookat_quat(cam_pos, obj_pos)
        return cam_pos, cam_quat

    def _depth_to_pc_world(self, depth: torch.Tensor, obj_pos_w: torch.Tensor,
                            pc_fast_world: torch.Tensor) -> torch.Tensor:
        """Unproject rendered depth to a WORLD-frame point cloud near the object.

        Unlike grasp_pose_env.py's object-local variant, G1's downstream pipeline
        already transforms world-frame points into robot-base frame itself (see
        _synthesize_pointcloud), so this stops at world frame.
        """
        B, H, W = depth.shape
        device = self.device
        fov_rad = self.cfg.camera_fov_deg * (math.pi / 180.0)
        fx = W / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx
        cx, cy = W / 2.0, H / 2.0
        d_min, d_max = self.cfg.camera_depth_clip

        u = torch.arange(W, device=device, dtype=torch.float32)
        v = torch.arange(H, device=device, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        x_c =  (uu - cx).unsqueeze(0) / fx * depth
        y_c = -(vv - cy).unsqueeze(0) / fy * depth
        z_c = -depth
        pts_cam = torch.stack([x_c, y_c, z_c], dim=-1).reshape(B, H * W, 3)

        valid = (depth > d_min) & (depth < d_max) & depth.isfinite()
        valid = valid.reshape(B, H * W)

        HW = H * W
        q_e = self._cam_quat_w.unsqueeze(1).expand(-1, HW, -1).reshape(B * HW, 4)
        p_e = self._cam_pos_w.unsqueeze(1).expand(-1, HW, -1).reshape(B * HW, 3)
        pts_world = (quat_rotate(q_e, pts_cam.reshape(B * HW, 3)) + p_e).reshape(B, HW, 3)

        table_z = self.cfg.table_surface_z + 0.015
        valid = valid & (pts_world[:, :, 2] > table_z)

        scores = torch.where(
            valid, torch.rand(B, HW, device=device),
            torch.full((B, HW), float("-inf"), device=device),
        )
        _, top_idx = scores.topk(NUM_PC_POINTS, dim=-1, sorted=False)
        pc = pts_world.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 3))

        n_valid = valid.sum(dim=-1)
        for bi in (n_valid < NUM_PC_POINTS).nonzero(as_tuple=False).view(-1):
            nv = n_valid[bi].item()
            if nv == 0:
                pc[bi] = pc_fast_world[bi]   # camera saw nothing — reuse the fast-path cloud
            else:
                vpts = pts_world[bi][valid[bi]]
                ridx = torch.randint(0, int(nv), (NUM_PC_POINTS,), device=device)
                pc[bi] = vpts[ridx]
        return pc

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
        Two source paths, mixed PER-ENV PER-EPISODE (self._use_camera_this_ep,
        drawn in _reset_idx from Bernoulli(camera_pc_prob)) — see
        grasp_pose_env.py's _synthesize_pointcloud for the original mechanism:
        • fast path   — pre-loaded surface PC rotated object-local → world.
        • camera path — rendered depth from this episode's randomised camera ring.
        Sensor-realistic noise (Gaussian + dropout + outliers, severity
        domain-randomized per episode) is applied in WORLD frame identically to
        both paths, then everything is transformed into robot-base frame.
        """
        B, N = self.num_envs, NUM_PC_POINTS
        device = self.device

        obj_pos_w  = self._active_obj_pos()   # (B, 3)
        obj_quat_w = self._active_obj_quat()  # (B, 4)

        # Fast pre-loaded-PC path: object-local → world
        shape_pcs = self._obj_pcs[self._env_shape]  # (B, 512, 3)
        idx = torch.randint(0, _PC_PRE_N, (B, N), device=device)
        pc_local = shape_pcs.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, N, 3)

        q_rep = obj_quat_w.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 4)
        pc_fast_world = quat_rotate(q_rep, pc_local.reshape(B * N, 3)).reshape(B, N, 3)
        pc_fast_world = pc_fast_world + obj_pos_w.unsqueeze(1)

        if self.cfg.use_camera_pc and self._cam_sensor is not None:
            depth = self._cam_sensor.data.output["distance_to_image_plane"][..., 0]  # (B, H, W)
            pc_cam_world = self._depth_to_pc_world(depth, obj_pos_w, pc_fast_world)
            use_cam = self._use_camera_this_ep.view(B, 1, 1)
            pc_world = torch.where(use_cam, pc_cam_world, pc_fast_world)
        else:
            pc_world = pc_fast_world

        # Sensor-realistic noise (severity domain-randomized per episode)
        g_std  = torch.empty(B, device=device).uniform_(*self.cfg.pc_noise_range_m)
        drop_p = torch.empty(B, device=device).uniform_(*self.cfg.pc_dropout_frac_range)
        out_p  = torch.empty(B, device=device).uniform_(*self.cfg.pc_outlier_frac_range)
        pc_world = add_sensor_noise(pc_world, gaussian_std=g_std, dropout_frac=drop_p, outlier_frac=out_p)

        # Transform to robot-base frame
        base_pos      = self._robot.data.root_pos_w
        base_quat     = self._robot.data.root_quat_w
        base_quat_inv = quat_conjugate(base_quat)

        rel    = pc_world - base_pos.unsqueeze(1)
        q_inv  = base_quat_inv.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 4)
        pc_rob = quat_rotate(q_inv, rel.reshape(B * N, 3)).reshape(B, N, 3)

        return pc_rob.reshape(B, -1)

    # ── Reward ────────────────────────────────────────────────────────────────
    def _graspnet_reward(self) -> torch.Tensor:
        """Distance-decayed GraspNet quality bonus for the EE's current position
        relative to learned-good grasp points (object-local frame).

        Unlike grasp_pose_env.py's single-shot lookup at a frozen decision point,
        G1 uses continuous multi-step control, so this is a per-step shaping term
        that rewards the EE for approaching/holding near a good grasp region.
        Offline-precomputed candidates only (scripts/generate_graspnet_labels.py);
        shapes without labels yet contribute 0 (see the __init__ preload).
        """
        centers = self._obj_graspnet_centers[self._env_shape]   # (B, K, 3) object-local
        quality = self._obj_graspnet_quality[self._env_shape]   # (B, K)
        obj_pos_w  = self._active_obj_pos()
        obj_quat_w = self._active_obj_quat()
        ee_pos_w   = self._robot.data.body_pos_w[:, self._ee_body_idx]
        ee_local   = quat_rotate(quat_conjugate(obj_quat_w), ee_pos_w - obj_pos_w)   # (B, 3)
        dist = (centers - ee_local.unsqueeze(1)).norm(dim=-1)   # (B, K)
        weight = torch.exp(-dist / self.cfg.graspnet_reward_radius_m)
        return (weight * quality).max(dim=-1).values

    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self._active_obj_pos()
        ee_pos  = self._robot.data.body_pos_w[:, self._ee_body_idx]

        dist_ee_obj   = torch.norm(ee_pos - obj_pos, dim=-1)
        lift_h        = (obj_pos[:, 2] - self.cfg.table_surface_z).clamp(0)
        dist_obj_goal = torch.norm(obj_pos[:, :2] - self._goal_pos_w[:, :2], dim=-1)

        success = (dist_obj_goal < self.cfg.place_success_radius) & (lift_h > 0)
        fell    = obj_pos[:, 2] < (self.cfg.table_surface_z - 0.10)
        lifted  = lift_h > 0.05

        if self.cfg.use_graspnet_reward:
            graspnet_r = self._graspnet_reward() * self.cfg.graspnet_reward_scale
        else:
            graspnet_r = torch.zeros(self.num_envs, device=self.device)

        return (
            - dist_ee_obj * self.cfg.approach_reward_scale
            + lift_h      * self.cfg.lift_reward_scale
            - dist_obj_goal * self.cfg.place_reward_scale * lifted.float()
            + success.float() * self.cfg.success_bonus
            + fell.float()    * self.cfg.fall_penalty
            - self._prev_action.pow(2).sum(-1) * self.cfg.action_penalty_scale
            + graspnet_r
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
        self._env_shape[ids] = torch.randint(0, self._num_shapes, (n,), device=self.device)

        # Per-episode choice of camera-rendered vs. fast pre-loaded PC path.
        if self._cam_sensor is not None:
            self._use_camera_this_ep[ids] = (
                torch.rand(n, device=self.device) < self.cfg.camera_pc_prob
            )

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

        # Randomise camera positions for the reset envs and force one render so
        # _get_observations sees fresh depth at the new pose (ported from
        # grasp_pose_env.py — obj_spawn_w is this reset batch's object target).
        if self._cam_sensor is not None:
            cam_pos, cam_quat = self._random_camera_poses(n, obj_spawn_w)
            self._cam_pos_w[ids]  = cam_pos
            self._cam_quat_w[ids] = cam_quat
            # Update ALL cameras at once (TiledCamera requires full-tensor update).
            self._cam_sensor.set_world_poses(
                self._cam_pos_w, self._cam_quat_w, convention="opengl"
            )
            self.scene.write_data_to_sim()
            self.sim.step(render=True)
            self.scene.update(dt=self.physics_dt)

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
