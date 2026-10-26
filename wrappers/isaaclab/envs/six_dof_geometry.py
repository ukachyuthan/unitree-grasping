"""Approach-direction geometry for 6-DoF contact-pair grasps (torch only, no Isaac).

A parallel-jaw grasp is fixed by its two contacts up to one rotation: about the
jaw axis u = (c2 − c1)/|c2 − c1|. The policy's ψ picks that rotation:

    a0 = most-downward unit vector ⟂ u   (projection of world −Z onto the plane ⟂ u)
    b  = u × a0                           (horizontal, ⟂ u and a0)
    approach(ψ) = cos ψ · a0 + sin ψ · b

ψ = 0 is the most top-down approach the jaw allows; |ψ| = π/2 is horizontal.
Since b has no vertical component, approach.z = cos ψ · a0.z ≤ 0 for |ψ| ≤ π/2:
the hand never comes from below the table. For a vertical jaw every ⟂ direction is
horizontal, and a0 falls back to the caller's `fallback` (e.g. radial from the robot).
"""

from __future__ import annotations

import torch


def approach_from_jaw(u: torch.Tensor, psi: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """u: (B, 3) unit jaw axis, psi: (B,) radians, fallback: (B, 3) unit horizontal
    direction used when u is vertical. Returns (B, 3) unit approach directions
    (pointing from the hand toward the object, i.e. hand +Z)."""
    down = torch.zeros_like(u)
    down[:, 2] = -1.0
    a0 = down - (down * u).sum(-1, keepdim=True) * u
    n = a0.norm(dim=-1, keepdim=True)
    a0 = torch.where(n > 1e-3, a0 / n.clamp(min=1e-6), fallback)
    b = torch.cross(u, a0, dim=-1)
    psi = psi.unsqueeze(-1)
    return psi.cos() * a0 + psi.sin() * b


def euler_zyx_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) rotation → (B, 3) angles (a, b, c) with R = Rz(a)·Ry(b)·Rx(c).
    At gimbal lock (b = ±90°, e.g. a horizontal approach with a horizontal jaw)
    only c − a (or c + a) is defined; a = 0 is chosen."""
    sb = (-R[:, 2, 0]).clamp(-1.0, 1.0)
    b = torch.asin(sb)
    gimbal = torch.cos(b).abs() < 1e-4
    a = torch.where(gimbal, torch.zeros_like(b), torch.atan2(R[:, 1, 0], R[:, 0, 0]))
    c = torch.where(gimbal, torch.atan2(sb * R[:, 0, 1], R[:, 1, 1]), torch.atan2(R[:, 2, 1], R[:, 2, 2]))
    return torch.stack([a, b, c], dim=-1)
