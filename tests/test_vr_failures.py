"""FailureCurator: the failure replay queue, promotion to cases, and the dataset format."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.vr_failures import (  # noqa: E402
    CASE_SCHEMA,
    FailureCurator,
    FailureCuratorConfig,
    yaw_bin,
    yaw_from_quat_wxyz,
)

SHAPES = ["torus", "wedge"]


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def make_curator(tmp_path, num_envs=8, mesh_dir=None, **overrides) -> FailureCurator:
    cfg = FailureCuratorConfig(
        out_dir=str(tmp_path), run_name="run1", **{"warmup_iters": 0, "replay_prob": 0.5, **overrides}
    )
    pcs = np.random.default_rng(0).normal(size=(len(SHAPES), 512, 3)).astype(np.float32) * 0.03

    def mesh_path(shape):
        if mesh_dir is None:
            return None
        p = Path(mesh_dir) / f"{shape}.obj"
        return p if p.exists() else None

    scene = {"object_scale": 1.5, "lift_target_m": 0.05}
    return FailureCurator(cfg, num_envs, SHAPES, pcs, mesh_path, scene)


class FakeEnv:
    """Mimics GraspPoseEnv's episode flow: random episodes unless a replay was queued."""

    def __init__(self, n, seed=1):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.pending = {}
        self.shape = np.zeros(n, dtype=np.int64)
        self.pos = np.zeros((n, 3))
        self.quat = np.zeros((n, 4))
        self.replayed = np.zeros(n, dtype=bool)
        for b in range(n):
            self._reset(b)

    def queue_replays(self, env_ids, shape_ids, xy, quat_wxyz):
        for b, s, p, q in zip(env_ids, shape_ids, xy, quat_wxyz):
            self.pending[int(b)] = (int(s), p, q)

    def _reset(self, b):
        if b in self.pending:
            s, xy, q = self.pending.pop(b)
            self.shape[b], self.pos[b], self.quat[b] = s, [xy[0], xy[1], 0.04], q
            self.replayed[b] = True
        else:
            self.shape[b] = self.rng.integers(len(SHAPES))
            self.pos[b] = [self.rng.uniform(0.43, 0.57), self.rng.uniform(-0.12, 0.12), 0.04]
            self.quat[b] = yaw_quat(self.rng.uniform(-math.pi, math.pi))
            self.replayed[b] = False

    def step(self, curator, it, lift_fn):
        """One training step, in the order train_grasp_pose.py runs it."""
        ctx = {"shape_ids": self.shape.copy(), "obj_pos": self.pos.copy(), "obj_quat_wxyz": self.quat.copy()}
        was_replay = self.replayed.copy()
        replays = curator.plan_replays(it)
        if len(replays["env_ids"]):
            self.queue_replays(**replays)
        lifts = np.array([lift_fn(int(s), bool(r)) for s, r in zip(ctx["shape_ids"], was_replay)], dtype=float)
        for b in range(self.n):
            self._reset(b)
        zeros = np.zeros((self.n, 3))
        return curator.observe(
            iteration=it, **ctx, lift_frac=lifts, reward=lifts * 0.8,
            reward_terms={"lift": lifts},
            contact_left_local=zeros - [0, 0.02, 0], contact_right_local=zeros + [0, 0.02, 0],
            grasp_center_local=zeros, grasp_width=np.full(self.n, 0.04),
        ), was_replay


def run(curator, env, iters, lift_fn, start=1):
    cases = []
    for it in range(start, start + iters):
        new, _ = env.step(curator, it, lift_fn)
        cases += new
    return cases


def test_yaw_roundtrip_and_binning():
    yaws = np.array([0.0, 0.5, -2.0, 3.1])
    got = yaw_from_quat_wxyz(np.array([yaw_quat(y) for y in yaws]))
    assert np.allclose(got, yaws, atol=1e-6)
    assert list(yaw_bin(np.array([-0.1, 0.1, math.pi / 2, math.pi]), 4)) == [0, 0, 1, 2]


def test_failures_are_queued_not_exported(tmp_path):
    c = make_curator(tmp_path, replay_prob=0.0)
    env = FakeEnv(8)
    assert run(c, env, 3, lambda s, r: 0.0) == []
    assert c.queue_size == 24
    assert c.num_cases == 0


def test_replays_run_at_the_queued_shape_and_pose(tmp_path):
    c = make_curator(tmp_path, num_envs=4, replay_prob=1.0)
    env = FakeEnv(4)
    env.step(c, 1, lambda s, r: 0.0)                 # 4 failures queued
    queued = {(e.shape_id, tuple(np.round(e.pos[:2], 6))) for e in c._queue.values()}
    env.step(c, 2, lambda s, r: 0.0)                 # replays scheduled here run next
    replayed = {(int(env.shape[b]), tuple(np.round(env.pos[b, :2], 6))) for b in range(4) if env.replayed[b]}
    assert len(replayed) == 4
    assert replayed <= queued


def test_exported_after_threshold_failed_replays(tmp_path):
    c = make_curator(tmp_path, num_envs=1, replay_prob=1.0, fail_threshold=4)
    env = FakeEnv(1)
    # Only the torus fails, so only torus episodes enter the queue.
    cases = run(c, env, 40, lambda s, r: 0.0 if s == 0 else 1.0)
    assert cases, "a torus that keeps failing on replay must be exported"
    case = json.loads((tmp_path / "run1/cases" / f"{cases[0]}.json").read_text())
    assert case["schema"] == CASE_SCHEMA
    attempts = case["failed_attempts"]
    # The original failure, then 4 failed replays, all at the queued pose.
    assert [a["replay"] for a in attempts] == [False, True, True, True, True]
    assert all(a["object_pose"]["pos"][:2] == case["object_pose"]["pos"][:2] for a in attempts)
    assert case["object"]["shape"] == "torus"
    assert len(case["object"]["point_cloud_local"]) == 256
    manifest = json.loads((tmp_path / "run1/manifest.json").read_text())
    assert cases[0] in [m["case_id"] for m in manifest["cases"]]


def test_successful_replay_removes_entry(tmp_path):
    c = make_curator(tmp_path, num_envs=1, replay_prob=1.0)
    env = FakeEnv(1)
    # Fresh episodes fail, replays succeed: the policy "learned" them.
    assert run(c, env, 30, lambda s, r: 1.0 if r else 0.0) == []
    assert c.stats["replay_successes"] > 0
    assert c.stats["replay_failures"] == 0


def test_replays_are_a_random_fraction_of_the_batch(tmp_path):
    c = make_curator(tmp_path, num_envs=200, replay_prob=0.1, queue_capacity=10_000)
    env = FakeEnv(200)
    run(c, env, 5, lambda s, r: 0.0)
    _, was_replay = env.step(c, 6, lambda s, r: 0.0)
    assert 5 <= was_replay.sum() <= 40


def test_an_entry_never_replays_in_two_envs_at_once(tmp_path):
    c = make_curator(tmp_path, num_envs=16, replay_prob=1.0)
    env = FakeEnv(16)
    env.step(c, 1, lambda s, r: 1.0 if s else 0.0)
    for it in range(2, 10):
        env.step(c, it, lambda s, r: 0.0 if s == 0 else 1.0)
        running = [id(e) for e in c._running if e is not None]
        assert len(running) == len(set(running))


def test_warmup_capacity_and_max_cases(tmp_path):
    c = make_curator(tmp_path, warmup_iters=5, queue_capacity=10, max_cases=1, replay_prob=0.5)
    env = FakeEnv(8)
    run(c, env, 5, lambda s, r: 0.0)
    assert c.queue_size == 0
    cases = run(c, env, 40, lambda s, r: 0.0, start=6)
    # Full queue turns new failures away; queued ones stay until resolved.
    assert c.queue_size <= 10 and c.stats["rejected_full"] > 0
    assert len(cases) == 1


def test_mesh_copied_and_referenced(tmp_path):
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "torus.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    c = make_curator(tmp_path / "out", num_envs=2, mesh_dir=mesh_dir, replay_prob=1.0)
    cases = run(c, FakeEnv(2), 40, lambda s, r: 0.0 if s == 0 else 1.0)
    case = json.loads((tmp_path / "out/run1/cases" / f"{cases[0]}.json").read_text())
    assert case["object"]["mesh"] == "objects/torus.obj"
    assert case["object"]["mesh_scale"] == 1.5
    assert (tmp_path / "out/run1/objects/torus.obj").exists()


def test_monitor_snapshot(tmp_path):
    c = make_curator(tmp_path, num_envs=4, replay_prob=0.0)
    FakeEnv(4).step(c, 1, lambda s, r: 0.0)
    monitor = json.loads((tmp_path / "run1/monitor.json").read_text())
    assert monitor["queue_size"] == 4
    assert monitor["queue_by_replay_failures"] == [4, 0, 0, 0]
    assert sum(monitor["queue_per_shape"].values()) == 4


def test_invalid_threshold(tmp_path):
    with pytest.raises(ValueError):
        make_curator(tmp_path, fail_threshold=0)


def test_load_vr_demos_reads_successful_demos(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from load_vr_demos import load_vr_demos

    demos = tmp_path / "run1/demos/torus__it000004__q0"
    demos.mkdir(parents=True)
    base = {"schema": "unitree-grasp-demo/v1", "run": "run1", "case_id": "torus__it000004__q0",
            "shape": "torus", "success": True}
    grasp = {"mode": "opposed", "c1_local": [0, -0.02, 0], "c2_local": [0, 0.02, 0]}
    (demos / "a.json").write_text(json.dumps({**base, "grasp": grasp}))
    (demos / "b.json").write_text(json.dumps({**base, "grasp": {**grasp, "mode": "pinch"}}))
    (demos / "c.json").write_text(json.dumps({**base, "success": False, "grasp": grasp}))

    pairs = load_vr_demos(tmp_path)
    assert [p["mode"] for p in pairs] == ["opposed", "pinch"]
    assert pairs[0]["width"] == pytest.approx(0.04)
    assert len(load_vr_demos(tmp_path, modes=("opposed",))) == 1
