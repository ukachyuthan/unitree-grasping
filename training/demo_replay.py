"""
Demonstrations of exported failure cases, replayed during RL with an imitation loss.

Closes the failure loop of training/vr_failures.py: cases the policy keeps
failing are exported, someone demonstrates them, and the demos come back here.
Two sources write the same demo schema ("unitree-grasp-demo/v1"):

    webxr-hands   a human in the VR app (vr-grasping-project)
    sim-teacher   scripts/teach_failure_cases.py: the first Contact-GraspNet grasp
                  that lifts the case's object in simulation

Layout, for either source (the VR app writes into the failure dataset itself;
teacher demos go to a separate root so the VR app never sees them as done):

    <root>/<run_name>/demos/<case_id>/<name>.json

A demo becomes a six_dof training target (c1, c2, ψ):

    c1, c2  the two contacts, object-local and scaled — the frame of the env's
            _contact_left_local / _contact_right_local
    ψ       the approach angle about the jaw (envs/six_dof_geometry.py). Teacher
            demos store it; for a VR demo the approach is wrist → contact
            midpoint at the grasp, which is where a parallel-jaw hand would come from.

During training, DemoReplayer spawns demo cases at their recorded pose (via
GraspPoseEnv.queue_replays, like failure replays). The policy acts on them as on
any episode, and the trainer adds an imitation loss toward the demo's action.

All poses are in the robot-base frame: z-up, metres, quaternions as wxyz.
This module depends on numpy only, so it can be unit-tested without Isaac Lab.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEMO_SCHEMA = "unitree-grasp-demo/v1"
SOURCE_VR = "webxr-hands"
SOURCE_TEACHER = "sim-teacher"


@dataclass
class GraspDemo:
    run: str
    case_id: str
    shape: str
    source: str
    pos: np.ndarray          # (3,) object position at the grasp
    quat_wxyz: np.ndarray    # (4,)
    c1_local: np.ndarray     # (3,)
    c2_local: np.ndarray     # (3,)
    psi: float               # rad
    path: str


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (..., 3) by unit quaternion q (4,) given as wxyz."""
    w, xyz = q[0], np.asarray(q[1:4])
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def psi_from_approach(c1: np.ndarray, c2: np.ndarray, approach: np.ndarray) -> float:
    """Approach angle about the jaw c1→c2 (world frame) such that
    approach_from_jaw(u, ψ) points along `approach` (its component ⟂ the jaw).
    Same convention as scripts/pretrain_grasp_6dof.psi_of."""
    u = c2 - c1
    u = u / max(np.linalg.norm(u), 1e-9)
    down = np.array([0.0, 0.0, -1.0])
    a0 = down - (down @ u) * u
    a0 = a0 / max(np.linalg.norm(a0), 1e-9)
    b = np.cross(u, a0)
    return float(np.arctan2(approach @ b, approach @ a0))


def vr_approach_world(demo: dict) -> np.ndarray | None:
    """Unit approach direction of a VR demo: grasping hand's wrist → contact
    midpoint, at the recorded frame nearest the grasp. None if no such frame."""
    g = demo["grasp"]
    hand, t = g["hand"], g.get("t", 0.0)
    names = demo.get("joint_names") or []
    wrist = names.index("wrist") if "wrist" in names else 0
    frames = [f for f in demo.get("frames", []) if (f.get("hands") or {}).get(hand)]
    if not frames:
        return None
    f = min(frames, key=lambda fr: abs(fr["t"] - t))
    j = np.asarray(f["hands"][hand]["joints"], dtype=np.float64)
    wrist_w = j[3 * wrist : 3 * wrist + 3]
    pose = g["object_pose"]
    mid_local = 0.5 * (np.asarray(g["c1_local"]) + np.asarray(g["c2_local"]))
    mid_w = np.asarray(pose["pos"]) + quat_rotate_wxyz(np.asarray(pose["quat_wxyz"]), mid_local)
    a = mid_w - wrist_w
    n = np.linalg.norm(a)
    return a / n if n > 1e-6 else None


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def demo_from_json(demo: dict, path: str, case_pose: dict | None = None) -> GraspDemo | None:
    """One demo file → GraspDemo, or None if it is unusable (failed, no grasp,
    no approach for a VR demo).

    The demo is replayed at case_pose (the case's resting pose) when given: in VR
    the object may have been nudged before the grasp, so the approach is carried
    over in the object frame and ψ recomputed for the resting pose."""
    if demo.get("schema") != DEMO_SCHEMA or not demo.get("success"):
        return None
    g = demo.get("grasp")
    if not g:
        return None
    pose = case_pose or g["object_pose"]
    pos = np.asarray(pose["pos"], dtype=np.float64)
    q = np.asarray(pose["quat_wxyz"], dtype=np.float64)
    c1 = np.asarray(g["c1_local"], dtype=np.float64)
    c2 = np.asarray(g["c2_local"], dtype=np.float64)
    if "psi_rad" in g:
        psi = float(g["psi_rad"])
    else:
        a_w = vr_approach_world(demo)
        if a_w is None:
            return None
        q_grasp = np.asarray(g["object_pose"]["quat_wxyz"], dtype=np.float64)
        a = quat_rotate_wxyz(q, quat_rotate_wxyz(quat_conj_wxyz(q_grasp), a_w))
        psi = psi_from_approach(quat_rotate_wxyz(q, c1), quat_rotate_wxyz(q, c2), a)
    return GraspDemo(
        run=demo["run"], case_id=demo["case_id"], shape=demo["shape"],
        source=demo.get("source", SOURCE_VR), pos=pos, quat_wxyz=q,
        c1_local=c1, c2_local=c2, psi=psi, path=path,
    )


def load_demos(roots, sources: tuple[str, ...] | None = None) -> list[GraspDemo]:
    """Usable demos under each root (<root>/<run>/demos/<case>/*.json), oldest first."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    out = []
    for root in roots:
        for path in sorted(Path(root).glob("*/demos/*/*.json")):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            # <run>/demos/<case_id>/x.json → <run>/cases/<case_id>.json (VR dataset layout)
            case_path = path.parents[2] / "cases" / f"{path.parent.name}.json"
            case_pose = None
            if case_path.exists():
                with open(case_path, encoding="utf-8") as f:
                    case_pose = json.load(f).get("object_pose")
            d = demo_from_json(raw, str(path), case_pose)
            if d is not None and (sources is None or d.source in sources):
                out.append(d)
    return out


def load_verdicts(roots) -> dict[tuple[str, str], dict]:
    """Latest demonstrator verdict per case, {(run, case_id): verdict record}, from
    <root>/<run>/verdicts/<case_id>.json (written by the VR app's IMPOSSIBLE button)."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    out = {}
    for root in roots:
        for path in sorted(Path(root).glob("*/verdicts/*.json")):
            with open(path, encoding="utf-8") as f:
                v = json.load(f)
            out[(path.parents[1].name, path.stem)] = v
    return out


def impossible_cases(roots) -> set[tuple[str, str]]:
    """(run, case_id) of the cases currently marked impossible."""
    return {k for k, v in load_verdicts(roots).items() if v.get("verdict") == "impossible"}


def impossible_failure_keys(roots) -> set[tuple[str, int]]:
    """(shape, yaw_bin) failure keys of the cases marked impossible, so the curator
    can stop exporting the same situation again (FailureCurator.impossible_keys)."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    keys = set()
    for root in roots:
        for run, case_id in impossible_cases(root):
            case_path = Path(root) / run / "cases" / f"{case_id}.json"
            if case_path.exists():
                k = json.loads(case_path.read_text(encoding="utf-8"))["failure_key"]
                keys.add((k["shape"], int(k["yaw_bin"])))
    return keys


class DemoReplayer:
    """Schedules demo cases into training episodes and hands back their targets.

    Per step: plan() picks envs whose next episode replays a demo (pass them to
    GraspPoseEnv.queue_replays before env.step); after env.step() those episodes
    have been spawned, and take_started() returns (env_ids, c1, c2, ψ) for them.
    """

    def __init__(self, demos: list[GraspDemo], shape_names: list[str], num_envs: int,
                 replay_prob: float = 0.1, seed: int = 0):
        index = {s: i for i, s in enumerate(shape_names)}
        self.demos = [d for d in demos if d.shape in index]
        self.num_skipped = len(demos) - len(self.demos)
        self._shape_ids = np.array([index[d.shape] for d in self.demos], dtype=np.int64)
        self.num_envs = num_envs
        self.replay_prob = replay_prob
        self._rng = np.random.default_rng(seed)
        self._pending: np.ndarray = np.zeros(0, dtype=np.int64)       # env ids
        self._pending_demo: np.ndarray = np.zeros(0, dtype=np.int64)  # demo index per env

    def __len__(self) -> int:
        return len(self.demos)

    def plan(self, exclude_env_ids=()) -> dict:
        """Envs whose next episode replays a demo, in GraspPoseEnv.queue_replays form."""
        if not self.demos:
            self._pending = np.zeros(0, dtype=np.int64)
            return {"env_ids": [], "shape_ids": [], "xy": np.zeros((0, 2)), "quat_wxyz": np.zeros((0, 4))}
        pick = self._rng.random(self.num_envs) < self.replay_prob
        pick[np.asarray(exclude_env_ids, dtype=np.int64)] = False
        env_ids = np.nonzero(pick)[0]
        demo_ids = self._rng.integers(0, len(self.demos), size=len(env_ids))
        self._pending, self._pending_demo = env_ids, demo_ids
        return {
            "env_ids": env_ids.tolist(),
            "shape_ids": self._shape_ids[demo_ids].tolist(),
            "xy": np.stack([self.demos[i].pos[:2] for i in demo_ids]) if len(demo_ids) else np.zeros((0, 2)),
            "quat_wxyz": np.stack([self.demos[i].quat_wxyz for i in demo_ids]) if len(demo_ids) else np.zeros((0, 4)),
        }

    def take_started(self):
        """(env_ids, c1 (n,3), c2 (n,3), ψ (n,)) of the demo episodes just spawned."""
        ids, dem = self._pending, self._pending_demo
        self._pending = np.zeros(0, dtype=np.int64)
        c1 = np.stack([self.demos[i].c1_local for i in dem]) if len(dem) else np.zeros((0, 3))
        c2 = np.stack([self.demos[i].c2_local for i in dem]) if len(dem) else np.zeros((0, 3))
        psi = np.array([self.demos[i].psi for i in dem], dtype=np.float64)
        return ids, c1, c2, psi
