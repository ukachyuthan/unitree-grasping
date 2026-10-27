"""Demo loading (VR + sim teacher), ψ from a VR hand, and demo replay scheduling."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wrappers" / "isaaclab"))
from training.demo_replay import (  # noqa: E402
    DemoReplayer, SOURCE_TEACHER, SOURCE_VR, load_demos, psi_from_approach, quat_rotate_wxyz,
)
from envs.six_dof_geometry import approach_from_jaw  # noqa: E402


def yaw_quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def vr_demo(wrist_w, obj_pos, grasp_quat, c1=(0.0, -0.02, 0.0), c2=(0.0, 0.02, 0.0), success=True):
    names = ["wrist", "thumb-tip"]
    return {
        "schema": "unitree-grasp-demo/v1", "run": "r", "case_id": "case_a", "shape": "cube",
        "source": SOURCE_VR, "success": success,
        "grasp": {"t": 1.0, "hand": "right", "mode": "opposed", "c1_local": list(c1), "c2_local": list(c2),
                  "object_pose": {"pos": list(obj_pos), "quat_wxyz": list(grasp_quat)}},
        "joint_names": names,
        "frames": [{"t": 1.0, "hands": {"right": {"joints": list(wrist_w) + [0.0, 0.0, 0.0],
                                                  "wrist_quat_wxyz": [1, 0, 0, 0]}, "left": None}}],
    }


def write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_psi_matches_env_geometry():
    g = np.random.default_rng(0)
    for _ in range(200):
        c1 = g.normal(size=3) * 0.03
        u = g.normal(size=3)
        u[2] *= 0.5
        u /= np.linalg.norm(u)
        c2 = c1 + 0.04 * u
        psi = g.uniform(-math.pi / 2, math.pi / 2)
        a = approach_from_jaw(torch.tensor(u[None], dtype=torch.float32), torch.tensor([psi]),
                              torch.tensor([[1.0, 0.0, 0.0]]))[0].double().numpy()
        assert abs(psi_from_approach(c1, c2, a) - psi) < 1e-4


def test_vr_wrist_above_is_top_down_and_beside_is_side(tmp_path):
    obj = (0.5, 0.0, 0.03)
    write(tmp_path / "r/demos/case_a/top.json", vr_demo((0.5, 0.0, 0.25), obj, [1, 0, 0, 0]))
    write(tmp_path / "r/demos/case_b/side.json", vr_demo((0.3, 0.0, 0.03), obj, [1, 0, 0, 0]))
    demos = {Path(d.path).stem: d for d in load_demos(tmp_path)}
    assert abs(demos["top"].psi) < 1e-6
    assert abs(abs(demos["side"].psi) - math.pi / 2) < 1e-6


def test_vr_demo_replayed_at_case_pose_keeps_object_frame_approach(tmp_path):
    # Grasped after the object was yawed 90° from its resting pose, hand coming from -x
    # in the world; at the resting pose that approach is along +y... still horizontal,
    # and still perpendicular to the jaw, so |ψ| stays 90°, while c1/c2 are unchanged.
    obj = (0.5, 0.0, 0.03)
    write(tmp_path / "r/demos/case_a/d.json", vr_demo((0.3, 0.0, 0.03), obj, yaw_quat(math.pi / 2),
                                                      c1=(-0.02, 0.0, 0.0), c2=(0.02, 0.0, 0.0)))
    write(tmp_path / "r/cases/case_a.json", {"object_pose": {"pos": [0.45, 0.05, 0.03], "quat_wxyz": [1, 0, 0, 0]}})
    (d,) = load_demos(tmp_path)
    assert np.allclose(d.pos, [0.45, 0.05, 0.03]) and np.allclose(d.quat_wxyz, [1, 0, 0, 0])
    assert np.allclose(d.c1_local, [-0.02, 0.0, 0.0])
    assert abs(abs(d.psi) - math.pi / 2) < 1e-6


def test_teacher_demo_psi_is_taken_as_is_and_failures_skipped(tmp_path):
    t = vr_demo((0, 0, 1), (0.5, 0, 0.03), [1, 0, 0, 0])
    t["source"] = SOURCE_TEACHER
    t["grasp"]["psi_rad"] = 0.4
    write(tmp_path / "r/demos/case_a/teacher.json", t)
    write(tmp_path / "r/demos/case_a/failed.json", vr_demo((0, 0, 1), (0.5, 0, 0.03), [1, 0, 0, 0], success=False))
    demos = load_demos(tmp_path)
    assert [d.psi for d in demos] == [0.4]
    assert load_demos(tmp_path, sources=(SOURCE_VR,)) == []


def test_quat_rotate():
    v = quat_rotate_wxyz(np.array(yaw_quat(math.pi / 2)), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v, [0.0, 1.0, 0.0])


def test_replayer_plans_and_returns_targets(tmp_path):
    for i, shape in enumerate(["cube", "cube", "unknown_shape"]):
        d = vr_demo((0.5, 0.0, 0.25), (0.4 + 0.01 * i, 0.0, 0.03), [1, 0, 0, 0])
        d["shape"], d["case_id"] = shape, f"c{i}"
        write(tmp_path / f"r/demos/c{i}/d.json", d)
    rep = DemoReplayer(load_demos(tmp_path), ["sphere", "cube"], num_envs=64, replay_prob=0.5, seed=1)
    assert len(rep) == 2 and rep.num_skipped == 1
    plan = rep.plan(exclude_env_ids=[0, 1, 2])
    assert len(plan["env_ids"]) > 0 and not set(plan["env_ids"]) & {0, 1, 2}
    assert set(plan["shape_ids"]) == {1}
    ids, c1, c2, psi = rep.take_started()
    assert list(ids) == plan["env_ids"] and c1.shape == (len(ids), 3) and psi.shape == (len(ids),)
    assert len(rep.take_started()[0]) == 0       # consumed
