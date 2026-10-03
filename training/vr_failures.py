"""
Replays failed grasps during RL training; the ones that keep failing become VR cases.

Every failed training episode (one grasp attempt on one object at one pose) is
put in a *replay queue*. On later steps, a random fraction of the batch is
spent re-running queued failures with their exact shape and pose instead of a
fresh random draw, so the policy keeps meeting its hard cases in the middle of
normal training. Each replay resolves the entry:

    replay succeeds          → the policy has learned it; entry leaves the queue
    replay fails             → entry.replay_failures += 1
    replay_failures == N (4) → exported as a VR dataset case; entry leaves the queue

The original failure that queued an entry does not count towards N — only
failures on replay, i.e. after the policy had more training in between.

Dataset layout (one folder per training run, self-contained so it can be
rsync'd to whatever machine serves the VR app):

    <out_dir>/<run_name>/
        manifest.json          index of exported cases
        monitor.json           live snapshot of the replay queue
        cases/<case_id>.json   everything needed to replay one case in VR
        objects/<shape>.obj    object mesh, unscaled (apply case.object.mesh_scale)
        demos/<case_id>/*.json written by the VR app, never by this module

All poses are in the robot-base frame: z-up, metres, quaternions as wxyz —
exactly the frame the env's anchors and contact targets use.

This module depends on numpy only, so it can be unit-tested without Isaac Lab.
The env side is GraspPoseEnv.queue_replays().
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

CASE_SCHEMA = "unitree-grasp-failure/v1"
MANIFEST_SCHEMA = "unitree-grasp-failure-manifest/v1"


@dataclass
class FailureCuratorConfig:
    out_dir: str
    run_name: str
    # Failed replays before a queued failure becomes a dataset case.
    fail_threshold: int = 4
    # An attempt succeeds when lift_reward (fraction of lift_target_m) reaches this.
    success_lift: float = 0.5
    # Per env, per step: probability the next episode is a replay (when the queue has one).
    replay_prob: float = 0.1
    # When full, new failures are turned away rather than evicting queued ones:
    # eviction would churn entries out before any collects fail_threshold replays.
    # Entries free their slot when a replay resolves them (success or export).
    queue_capacity: int = 512
    # Iterations to skip before queueing; an untrained policy fails everything.
    warmup_iters: int = 50
    # Hard cap so a collapsed policy cannot flood the dataset.
    max_cases: int = 200
    # Spawn yaw is bucketed into this many bins, as case metadata.
    yaw_bins: int = 8
    # Object point-cloud samples embedded in each case (VR fallback when no mesh).
    pc_points: int = 256
    seed: int = 0


@dataclass
class QueueEntry:
    entry_id: int
    shape_id: int
    pos: np.ndarray
    quat_wxyz: np.ndarray
    queued_at: int
    # Original failure first, then every failed replay.
    attempts: list = field(default_factory=list)
    replay_failures: int = 0


def yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Rotation about +z of each (w, x, y, z) quaternion, in radians."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_bin(yaw: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index in [0, n_bins) with bin 0 centred on yaw = 0."""
    width = 2.0 * math.pi / n_bins
    shifted = np.mod(np.asarray(yaw) + width / 2.0, 2.0 * math.pi)
    return np.minimum((shifted // width).astype(np.int64), n_bins - 1)


def _vec(a) -> list[float]:
    return [round(float(v), 6) for v in np.asarray(a).reshape(-1)]


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Readers (the VR server) must never see a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


class FailureCurator:
    """Failure replay queue, with repeat failures promoted to VR dataset cases.

    Per training step the caller does, in order:
        replays = curator.plan_replays(it)   # before env.step(): the resets inside
        env.queue_replays(...replays)        #   step() spawn the next episodes
        env.step(actions)
        curator.observe(it, ...)             # results of the episodes just run

    Args:
        cfg: thresholds and output location.
        num_envs: batch size; env indices are 0..num_envs-1.
        shape_names: env shape index → name (``GraspPoseEnv._shape_names``).
        object_pcs: (num_shapes, P, 3) object-local point clouds, already scaled
            by ``object_scale`` (``GraspPoseEnv._obj_pcs``).
        mesh_path_fn: shape name → path of its unscaled OBJ, or None if absent.
        scene: static scene description copied into every case (table, robot
            base, object scale, lift target).
    """

    def __init__(
        self,
        cfg: FailureCuratorConfig,
        num_envs: int,
        shape_names: list[str],
        object_pcs: np.ndarray,
        mesh_path_fn: Callable[[str], Path | None],
        scene: dict,
    ):
        if cfg.fail_threshold < 1:
            raise ValueError("fail_threshold must be >= 1")
        self.cfg = cfg
        self.num_envs = num_envs
        self.shape_names = list(shape_names)
        self.object_pcs = np.asarray(object_pcs, dtype=np.float32)
        self.mesh_path_fn = mesh_path_fn
        self.scene = scene
        self._rng = np.random.default_rng(cfg.seed)

        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.cases_dir = self.run_dir / "cases"
        self.objects_dir = self.run_dir / "objects"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)

        self._queue: dict[int, QueueEntry] = {}   # insertion-ordered: oldest first
        self._next_id = 0
        # Entry replaying in each env: for the episode now running, and the next one.
        self._running: list[QueueEntry | None] = [None] * num_envs
        self._planned: list[QueueEntry | None] = [None] * num_envs

        self.stats = {"queued": 0, "rejected_full": 0, "replays": 0, "replay_successes": 0,
                      "replay_failures": 0}
        self._window = {"replays": 0, "replay_successes": 0}
        self._mesh_copied: dict[str, str | None] = {}
        self._manifest_cases: list[dict] = []
        self.last_iteration = 0
        self._write_manifest()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def num_cases(self) -> int:
        return len(self._manifest_cases)

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    def pop_replay_success_rate(self) -> float:
        """Success rate of replays since the last call (NaN if there were none)."""
        n, ok = self._window["replays"], self._window["replay_successes"]
        self._window = {"replays": 0, "replay_successes": 0}
        return ok / n if n else float("nan")

    def plan_replays(self, iteration: int) -> dict:
        """Choose which envs' next episodes replay a queued failure.

        Returns arrays for GraspPoseEnv.queue_replays (env_ids, shape_ids, xy,
        quat_wxyz); env_ids is empty when nothing is replayed this step.
        """
        self._planned = [None] * self.num_envs
        busy = {id(e) for e in self._running if e is not None}
        free = [e for e in self._queue.values() if id(e) not in busy]
        if iteration > self.cfg.warmup_iters and free:
            envs = np.flatnonzero(self._rng.random(self.num_envs) < self.cfg.replay_prob)
            picks = self._rng.permutation(len(free))[: len(envs)]
            for env, pick in zip(envs, picks):
                self._planned[int(env)] = free[int(pick)]

        chosen = [(b, e) for b, e in enumerate(self._planned) if e is not None]
        return {
            "env_ids": np.array([b for b, _ in chosen], dtype=np.int64),
            "shape_ids": np.array([e.shape_id for _, e in chosen], dtype=np.int64),
            "xy": np.array([e.pos[:2] for _, e in chosen], dtype=np.float32).reshape(-1, 2),
            "quat_wxyz": np.array([e.quat_wxyz for _, e in chosen], dtype=np.float32).reshape(-1, 4),
        }

    def observe(
        self,
        iteration: int,
        shape_ids: np.ndarray,
        obj_pos: np.ndarray,
        obj_quat_wxyz: np.ndarray,
        lift_frac: np.ndarray,
        reward: np.ndarray,
        reward_terms: dict[str, np.ndarray],
        contact_left_local: np.ndarray,
        contact_right_local: np.ndarray,
        grasp_center_local: np.ndarray,
        grasp_width: np.ndarray,
    ) -> list[str]:
        """Record the batch of episodes that just finished; returns newly exported case ids.

        Every array is indexed by env (leading dimension num_envs). ``obj_pos``
        must be relative to the env origin, i.e. in the robot-base frame.
        """
        self.last_iteration = iteration
        shape_ids = np.asarray(shape_ids).reshape(-1).astype(np.int64)
        lift_frac = np.asarray(lift_frac, dtype=np.float64).reshape(-1)
        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
        obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(-1, 3)
        obj_quat_wxyz = np.asarray(obj_quat_wxyz, dtype=np.float64).reshape(-1, 4)

        def attempt(b: int, replay: bool) -> dict:
            return {
                "iteration": int(iteration),
                "replay": replay,
                "reward": round(float(reward[b]), 6),
                "reward_terms": {
                    k: round(float(np.asarray(v).reshape(-1)[b]), 6) for k, v in reward_terms.items()
                },
                "object_pose": {"pos": _vec(obj_pos[b]), "quat_wxyz": _vec(obj_quat_wxyz[b])},
                "grasp": {
                    "center_local": _vec(grasp_center_local[b]),
                    "width": round(float(np.asarray(grasp_width).reshape(-1)[b]), 6),
                    "contact_left_local": _vec(contact_left_local[b]),
                    "contact_right_local": _vec(contact_right_local[b]),
                },
            }

        new_cases: list[str] = []
        for b in range(len(shape_ids)):
            success = lift_frac[b] >= self.cfg.success_lift
            entry = self._running[b]

            if entry is not None:
                self.stats["replays"] += 1
                self._window["replays"] += 1
                if success:
                    self.stats["replay_successes"] += 1
                    self._window["replay_successes"] += 1
                    del self._queue[entry.entry_id]
                    continue
                self.stats["replay_failures"] += 1
                entry.replay_failures += 1
                entry.attempts.append(attempt(b, replay=True))
                if entry.replay_failures >= self.cfg.fail_threshold:
                    del self._queue[entry.entry_id]
                    if self.num_cases < self.cfg.max_cases:
                        new_cases.append(self._export_case(entry))
                continue

            if success or iteration <= self.cfg.warmup_iters:
                continue
            self._enqueue(QueueEntry(
                entry_id=self._next_id,
                shape_id=int(shape_ids[b]),
                pos=obj_pos[b].copy(),
                quat_wxyz=obj_quat_wxyz[b].copy(),
                queued_at=int(iteration),
                attempts=[attempt(b, replay=False)],
            ))
            self._next_id += 1

        # The episodes planned before this step's resets are the ones now running.
        self._running = self._planned
        self._planned = [None] * self.num_envs
        self._write_monitor()
        return new_cases

    def snapshot(self) -> dict:
        """Live view of the replay queue, for monitoring during training."""
        by_failures = [0] * self.cfg.fail_threshold
        per_shape: dict[str, int] = {}
        for e in self._queue.values():
            by_failures[e.replay_failures] += 1
            name = self.shape_names[e.shape_id]
            per_shape[name] = per_shape.get(name, 0) + 1
        return {
            "iteration": self.last_iteration,
            "fail_threshold": self.cfg.fail_threshold,
            "replay_prob": self.cfg.replay_prob,
            "warmup_iters": self.cfg.warmup_iters,
            "queue_size": self.queue_size,
            "queue_capacity": self.cfg.queue_capacity,
            # Index k: entries that have failed k replays so far.
            "queue_by_replay_failures": by_failures,
            "queue_per_shape": dict(sorted(per_shape.items(), key=lambda kv: -kv[1])),
            "cases_exported": self.num_cases,
            "max_cases": self.cfg.max_cases,
            **self.stats,
        }

    # ── Internals ───────────────────────────────────────────────────────────

    def _enqueue(self, entry: QueueEntry) -> None:
        if len(self._queue) >= self.cfg.queue_capacity:
            self.stats["rejected_full"] += 1
            return
        self._queue[entry.entry_id] = entry
        self.stats["queued"] += 1

    def _export_case(self, entry: QueueEntry) -> str:
        shape = self.shape_names[entry.shape_id]
        ybin = int(yaw_bin(yaw_from_quat_wxyz(entry.quat_wxyz), self.cfg.yaw_bins)[0])
        case_id = f"{shape}__it{entry.queued_at:06d}__q{entry.entry_id}"
        # Never overwrite a case already on disk (e.g. a resumed run reusing the dir).
        suffix = 1
        while (self.cases_dir / f"{case_id}.json").exists():
            suffix += 1
            case_id = f"{shape}__it{entry.queued_at:06d}__q{entry.entry_id}_{suffix}"

        pc = self.object_pcs[entry.shape_id]
        n = min(self.cfg.pc_points, len(pc))
        pc_idx = self._rng.choice(len(pc), size=n, replace=False)
        last_iteration = entry.attempts[-1]["iteration"]

        case = {
            "schema": CASE_SCHEMA,
            "case_id": case_id,
            "run": self.cfg.run_name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "iteration": last_iteration,
            "frame": "robot_base: z-up, metres, quaternions wxyz",
            "failure_key": {
                "shape": shape,
                "yaw_bin": ybin,
                "yaw_bins": self.cfg.yaw_bins,
                "fail_threshold": self.cfg.fail_threshold,
                "queued_at": entry.queued_at,
                "queue_entry": entry.entry_id,
            },
            "object": {
                "shape": shape,
                "mesh": self._copy_mesh(shape),
                "mesh_scale": self.scene.get("object_scale", 1.0),
                "point_cloud_local": [_vec(p) for p in pc[pc_idx]],
            },
            # The pose that was queued and replayed.
            "object_pose": {"pos": _vec(entry.pos), "quat_wxyz": _vec(entry.quat_wxyz)},
            "scene": self.scene,
            "task": {
                "type": "lift",
                "lift_target_m": self.scene.get("lift_target_m", 0.05),
                "success_lift_frac": self.cfg.success_lift,
                "description": "Grasp the object and raise it lift_target_m above its resting height.",
            },
            # The original failure, then fail_threshold failed replays.
            "failed_attempts": entry.attempts,
        }
        _write_json_atomic(self.cases_dir / f"{case_id}.json", case)

        self._manifest_cases.append({
            "case_id": case_id,
            "shape": shape,
            "yaw_bin": ybin,
            "iteration": last_iteration,
            "mean_failed_reward": round(float(np.mean([a["reward"] for a in entry.attempts])), 6),
        })
        self._write_manifest()
        return case_id

    def _copy_mesh(self, shape: str) -> str | None:
        """Copies the shape's OBJ into the run once; returns its run-relative path."""
        if shape in self._mesh_copied:
            return self._mesh_copied[shape]
        rel = None
        src = self.mesh_path_fn(shape)
        if src is not None and Path(src).exists():
            dst = self.objects_dir / f"{shape}.obj"
            shutil.copyfile(src, dst)
            rel = f"objects/{shape}.obj"
        self._mesh_copied[shape] = rel
        return rel

    def _write_manifest(self) -> None:
        _write_json_atomic(self.run_dir / "manifest.json", {
            "schema": MANIFEST_SCHEMA,
            "run": self.cfg.run_name,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cases": self._manifest_cases,
        })

    def _write_monitor(self) -> None:
        _write_json_atomic(self.run_dir / "monitor.json", self.snapshot())
