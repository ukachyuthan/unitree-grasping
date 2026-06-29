"""
Camera-agnostic point cloud utilities.

All functions are pure PyTorch and work identically in simulation (Isaac Lab
depth camera) and on real hardware (Intel RealSense, ZED, Azure Kinect …).

Portability checklist for a new camera / robot pair:
  1. Provide intrinsics K  (3×3) — from `camera.get_intrinsics()` or YAML calibration.
  2. Provide extrinsic T_cam_robot (4×4) — from hand-eye calibration.
  3. Call `depth_to_pointcloud_world(depth, K, cam_pos, cam_quat)` as usual.
  4. Call `transform_pointcloud(pc, T)` with your T_cam_robot if needed.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_rotate


def depth_to_pointcloud_world(
    depth: torch.Tensor,       # (B, H, W)  perpendicular depth in metres; 0/inf = invalid
    intrinsics: torch.Tensor,  # (B, 3, 3)  camera intrinsic matrix
    cam_pos_w: torch.Tensor,   # (B, 3)     camera world position
    cam_quat_w: torch.Tensor,  # (B, 4)     camera world orientation (w,x,y,z)
    min_depth: float = 0.05,
    max_depth: float = 2.45,
) -> torch.Tensor:             # (B, H*W, 3) world-frame XYZ; zeros for invalid pixels
    """
    Back-project a batch of depth images to 3-D world-frame point clouds.

    Uses OpenGL camera convention (forward = −Z_cam, Y_cam up = image down):
        X_cam =  (u − cx) / fx × Z
        Y_cam = −(v − cy) / fy × Z      ← negative because image V ↓ but Y_cam ↑
        Z_cam = −Z                       ← camera looks along −Z
    Then rotates + translates to world frame using cam_pos_w / cam_quat_w.

    Invalid pixels (depth ≤ min_depth or ≥ max_depth) are returned as
    world-frame zero-vectors so they can be filtered downstream by callers.
    """
    B, H, W = depth.shape
    device   = depth.device

    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)

    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )

    valid = (depth > min_depth) & (depth < max_depth) & torch.isfinite(depth)
    Z_d = depth.clone()
    Z_d[~valid] = 0.0

    # Camera-frame coordinates (OpenGL)
    X_c =  ((u_grid - cx) / fx) * Z_d
    Y_c = -((v_grid - cy) / fy) * Z_d
    Z_c = -Z_d

    pts_cam = torch.stack([X_c, Y_c, Z_c], dim=-1).reshape(B, H * W, 3)

    # Rotate to world frame
    q_rep   = cam_quat_w.unsqueeze(1).expand(-1, H * W, -1).reshape(B * H * W, 4)
    p_flat  = pts_cam.reshape(B * H * W, 3)
    pts_world = (quat_rotate(q_rep, p_flat)
                 + cam_pos_w.unsqueeze(1).expand(-1, H * W, -1).reshape(B * H * W, 3))

    pts_world = pts_world.reshape(B, H * W, 3)

    # Zero out invalid pixels so they are easy to mask later
    valid_flat = valid.reshape(B, H * W)
    pts_world[~valid_flat] = 0.0

    return pts_world  # (B, H*W, 3)


def transform_pointcloud(
    pointcloud: torch.Tensor,   # (B, N, 3) or (N, 3)
    T: torch.Tensor,            # (B, 4, 4) or (4, 4) homogeneous transform
) -> torch.Tensor:
    """
    Apply a rigid SE(3) transform to a point cloud.

    Used to convert a point cloud from camera frame to robot-base frame
    given a calibrated camera-to-robot extrinsic matrix T_cam_robot.

    Works for both batched (B, N, 3) and unbatched (N, 3) inputs.

    Example (real-robot deployment):
        T = load_hand_eye_calibration()         # (4, 4) numpy → torch
        pc_robot = transform_pointcloud(pc_cam, T)
    """
    batched = pointcloud.dim() == 3
    if not batched:
        pointcloud = pointcloud.unsqueeze(0)
        if T.dim() == 2:
            T = T.unsqueeze(0)

    R = T[:, :3, :3]   # (B, 3, 3)
    t = T[:, :3,  3]   # (B, 3)

    # (B, N, 3) = (B, N, 3) @ (B, 3, 3)^T + (B, 1, 3)
    transformed = torch.bmm(pointcloud, R.transpose(1, 2)) + t.unsqueeze(1)

    return transformed if batched else transformed.squeeze(0)


def farthest_point_sample(
    pointcloud: torch.Tensor,   # (B, N, 3)
    num_samples: int,
) -> torch.Tensor:              # (B, num_samples, 3)
    """
    Farthest-point sampling (FPS) — spreads samples maximally over the cloud.
    Use instead of random sampling for better spatial coverage at inference.
    Slightly slower than random sampling; not needed during training.
    """
    B, N, _ = pointcloud.shape
    device  = pointcloud.device

    idx = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)

    # Initialise from a random point per env
    current = torch.randint(0, N, (B,), device=device)

    for i in range(num_samples):
        idx[:, i] = current
        cur_pts = pointcloud[torch.arange(B, device=device), current].unsqueeze(1)  # (B,1,3)
        d = torch.norm(pointcloud - cur_pts, dim=-1)   # (B, N)
        dist = torch.minimum(dist, d)
        current = dist.argmax(dim=-1)                  # (B,)

    batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(-1, num_samples)
    return pointcloud[batch_idx, idx]
