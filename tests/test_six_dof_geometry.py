"""6-DoF approach geometry: ⟂ jaw, never from below, ψ spans top-down → horizontal."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrappers" / "isaaclab"))
from envs.six_dof_geometry import approach_from_jaw, euler_zyx_from_matrix  # noqa: E402


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    return v / v.norm(dim=-1, keepdim=True)


def random_jaws(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return unit(torch.randn(n, 3, generator=g))


def radial_for(n):
    return unit(torch.tensor([[1.0, 0.0, 0.0]]).expand(n, -1).clone())


def test_unit_and_perpendicular_to_jaw():
    u = random_jaws()
    psi = torch.linspace(-math.pi / 2, math.pi / 2, len(u))
    a = approach_from_jaw(u, psi, radial_for(len(u)))
    assert torch.allclose(a.norm(dim=-1), torch.ones(len(u)), atol=1e-5)
    assert (a * u).sum(-1).abs().max() < 1e-5


def test_never_from_below_the_table():
    u = random_jaws()
    for psi_v in torch.linspace(-math.pi / 2, math.pi / 2, 13):
        a = approach_from_jaw(u, torch.full((len(u),), float(psi_v)), radial_for(len(u)))
        assert a[:, 2].max() <= 1e-5          # approach z ≤ 0: hand above / beside, never below


def test_psi_zero_is_most_downward_and_psi_90_horizontal():
    u = random_jaws()
    zero = approach_from_jaw(u, torch.zeros(len(u)), radial_for(len(u)))
    side = approach_from_jaw(u, torch.full((len(u),), math.pi / 2), radial_for(len(u)))
    for psi_v in torch.linspace(-math.pi / 2, math.pi / 2, 9):
        other = approach_from_jaw(u, torch.full((len(u),), float(psi_v)), radial_for(len(u)))
        assert (zero[:, 2] <= other[:, 2] + 1e-5).all()
    assert side[:, 2].abs().max() < 1e-5


def test_horizontal_jaw_top_down_and_side():
    u = unit(torch.tensor([[0.0, 1.0, 0.0]]))
    top = approach_from_jaw(u, torch.zeros(1), radial_for(1))
    side = approach_from_jaw(u, torch.full((1,), math.pi / 2), radial_for(1))
    assert torch.allclose(top, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6)
    assert torch.allclose(side.abs(), torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_vertical_jaw_uses_fallback_direction():
    u = unit(torch.tensor([[0.0, 0.0, 1.0]]))
    fb = unit(torch.tensor([[0.6, 0.8, 0.0]]))
    a = approach_from_jaw(u, torch.zeros(1), fb)
    assert torch.allclose(a, fb, atol=1e-6)


def rot_zyx(a, b, c):
    ca, sa, cb, sb, cc, sc = torch.cos(a), torch.sin(a), torch.cos(b), torch.sin(b), torch.cos(c), torch.sin(c)
    Rz = torch.stack([torch.stack([ca, -sa, 0 * a], -1), torch.stack([sa, ca, 0 * a], -1),
                      torch.stack([0 * a, 0 * a, 1 + 0 * a], -1)], -2)
    Ry = torch.stack([torch.stack([cb, 0 * b, sb], -1), torch.stack([0 * b, 1 + 0 * b, 0 * b], -1),
                      torch.stack([-sb, 0 * b, cb], -1)], -2)
    Rx = torch.stack([torch.stack([1 + 0 * c, 0 * c, 0 * c], -1), torch.stack([0 * c, cc, -sc], -1),
                      torch.stack([0 * c, sc, cc], -1)], -2)
    return Rz @ Ry @ Rx


def test_euler_zyx_round_trip_including_gimbal_lock():
    g = torch.Generator().manual_seed(1)
    a = (torch.rand(400, generator=g) * 2 - 1) * math.pi
    b = (torch.rand(400, generator=g) * 2 - 1) * (math.pi / 2)
    c = (torch.rand(400, generator=g) * 2 - 1) * math.pi
    b[:50] = math.pi / 2          # gimbal lock: horizontal approach, horizontal jaw
    b[50:100] = -math.pi / 2
    R = rot_zyx(a, b, c)
    ang = euler_zyx_from_matrix(R)
    R2 = rot_zyx(ang[:, 0], ang[:, 1], ang[:, 2])
    assert (R - R2).abs().max() < 1e-4


def test_pretrain_psi_inverts_env_approach():
    """scripts/pretrain_grasp_6dof.psi_of must recover the env's ψ from an approach."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pretrain_grasp_6dof import psi_of

    g = torch.Generator().manual_seed(2)
    c1 = torch.randn(300, 3, generator=g) * 0.03
    c2 = c1 + unit(torch.randn(300, 3, generator=g)) * 0.04
    u = unit(c2 - c1)
    keep = u[:, 2].abs() < 0.98
    psi = (torch.rand(300, generator=g) * 2 - 1) * (math.pi / 2)
    a = approach_from_jaw(u, psi, radial_for(300))
    assert (psi_of(c1, c2, a) - psi)[keep].abs().max() < 1e-4
