#!/usr/bin/env python3
"""
Simulated teacher for the failure loop: label failure cases with the best-ranked
Contact-GraspNet grasp that lifts the object in simulation.

Stands in for the human VR demonstrator, so the whole loop can be run, ablated
and repeated without anyone in VR. Writes the VR app's demo schema
(training/demo_replay.py) to a separate root, so the VR app never sees teacher
demos as already done:

    <out>/<run>/demos/<case_id>/sim-teacher.json
    <out>/<run>/teacher_summary.json        per case: grasps tried, which lifted

Each case's top-k grasps run in parallel, one env each, at the case's exact pose,
through the same six_dof decode and floating hand as a policy action, without
execution noise. The best-ranked one that reaches the lift target is the demo.

    # failure-targeted: the exported cases of a training run
    python scripts/teach_failure_cases.py --headless \\
        --cases ../../data/vr_failures/<run> --out ../../data/teacher_demos

    # control: random training objects at random poses
    python scripts/teach_failure_cases.py --headless --random_cases 300 \\
        --out ../../data/teacher_demos_random

    # check recorded VR demos: does the Franka hand reproduce each one?
    python scripts/teach_failure_cases.py --headless --verify ../../data/vr_failures
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--cases", type=str, help="A failure dataset run dir (<vr_failures>/<run>) to label.")
mode.add_argument("--random_cases", type=int, help="Label this many random (object, pose) cases instead.")
mode.add_argument("--verify", type=str, help="Demo root to verify: execute each demo, write verify.json.")
mode.add_argument("--eval_cases", type=str,
                  help="A failure dataset run dir: run --checkpoint's policy (action mean) on each "
                       "case, --repeats times, and report its success on those cases.")
parser.add_argument("--checkpoint", type=str, default=None, help="Policy for --eval_cases.")
parser.add_argument("--repeats", type=int, default=3,
                    help="--eval_cases: attempts per case (the observation noise differs each time).")
parser.add_argument("--exec_noise", action="store_true",
                    help="--eval_cases: keep the env's execution noise (default off).")
parser.add_argument("--out", type=str, default=None, help="Demo root to write (--cases / --random_cases).")
parser.add_argument("--top_k", type=int, default=8, help="Contact-GraspNet grasps tried per case.")
parser.add_argument("--teacher_repeats", type=int, default=2,
                    help="A grasp counts only if it lifts in every one of this many executions "
                         "(physics is not bit-exact; one-off lucky lifts make poor demos).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.cases or args.random_cases) and not args.out:
    parser.error("--out is required with --cases / --random_cases")
if args.eval_cases and not args.checkpoint:
    parser.error("--eval_cases needs --checkpoint")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.grasp_pose_env_cfg import GraspPoseEnvCfg, OBS_DIM
from envs.grasp_pose_env import GraspPoseEnv
from models.grasp_pose_actor_critic import GraspPoseActorCritic
from envs._object_registry import shape_asset
from training.demo_replay import (
    DEMO_SCHEMA, SOURCE_TEACHER, impossible_cases, load_demos, psi_from_approach, quat_rotate_wxyz,
)


def load_cases(run_dir: Path, halves=("demo",)) -> list[dict]:
    """The run's cases in the given halves of its split.json (scripts/split_failure_cases.py;
    every case is "demo" when there is no split), minus those marked impossible."""
    impossible = {case_id for run, case_id in impossible_cases(run_dir.parent) if run == run_dir.name}
    split_path = run_dir / "split.json"
    heldout = set(json.loads(split_path.read_text())["heldout"]) if split_path.exists() else set()
    cases = []
    for p in sorted((run_dir / "cases").glob("*.json")):
        half = "heldout" if p.stem in heldout else "demo"
        if p.stem in impossible or half not in halves:
            continue
        c = json.loads(p.read_text())
        cases.append({"case_id": c["case_id"], "run": c["run"], "shape": c["object"]["shape"], "half": half,
                      "pos": c["object_pose"]["pos"], "quat_wxyz": c["object_pose"]["quat_wxyz"]})
    return cases


def random_cases(n: int, shape_names: list[str], cfg: GraspPoseEnvCfg, rng) -> list[dict]:
    cases = []
    for i in range(n):
        shape = shape_names[rng.integers(len(shape_names))]
        yaw = rng.uniform(*cfg.spawn_yaw_range)
        cases.append({
            "case_id": f"random_{i:04d}__{shape}", "run": f"random_seed{args.seed}", "shape": shape,
            "pos": [rng.uniform(*cfg.spawn_x_range), rng.uniform(*cfg.spawn_y_range), 0.0],
            "quat_wxyz": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        })
    return cases


def candidate_grasps(case: dict, k: int, object_scale: float) -> list[dict]:
    """Top-k Contact-GraspNet grasps of the case's object as (c1, c2 object-local scaled, ψ).
    ψ is taken in the world at the case's pose; grasps from below the table or with a
    vertical jaw (no ψ reference) are skipped, as in scripts/pretrain_grasp_6dof.py."""
    path = shape_asset(case["shape"], "_graspnet.json")
    if not path.exists():
        return []
    q = np.asarray(case["quat_wxyz"], dtype=np.float64)
    out = []
    for rank, g in enumerate(json.loads(path.read_text())["grasps"][:k]):
        c1 = np.asarray(g["c1"]) * object_scale
        c2 = np.asarray(g["c2"]) * object_scale
        c1w, c2w = quat_rotate_wxyz(q, c1), quat_rotate_wxyz(q, c2)
        u = (c2w - c1w) / max(np.linalg.norm(c2w - c1w), 1e-9)
        psi = psi_from_approach(c1w, c2w, quat_rotate_wxyz(q, np.asarray(g["approach"])))
        if abs(u[2]) >= 0.98 or abs(psi) > math.pi / 2 + 1e-3:
            continue
        out.append({"rank": rank, "c1": c1, "c2": c2, "psi": psi, "quality": g.get("quality")})
    return out


def run_jobs(env: GraspPoseEnv, jobs: list[dict], shape_index: dict, policy=None) -> list[bool]:
    """Execute jobs ({shape, pos, quat_wxyz} + {c1, c2, psi}, or the policy's action
    when given), num_envs at a time → lifted?"""
    B = env.num_envs
    results = []
    for start in range(0, len(jobs), B):
        batch = jobs[start : start + B]
        n = len(batch)
        env.queue_replays(
            env_ids=list(range(n)),
            shape_ids=[shape_index[j["shape"]] for j in batch],
            xy=np.array([j["pos"][:2] for j in batch], dtype=np.float32),
            quat_wxyz=np.array([j["quat_wxyz"] for j in batch], dtype=np.float32),
        )
        # This step's own attempts are discarded: its resets spawn the batch's cases.
        obs, *_ = env.step(torch.zeros(B, int(env.cfg.action_space), device=env.device))
        if policy is not None:
            with torch.no_grad():
                env.step(policy.act_inference(obs["policy"]))
            results += env._last_lift_success[:n].cpu().tolist()
            continue
        c1 = torch.zeros(B, 3, device=env.device)
        c2 = torch.zeros(B, 3, device=env.device)
        psi = torch.zeros(B, device=env.device)
        c1[:n] = torch.tensor(np.stack([j["c1"] for j in batch]), dtype=torch.float32, device=env.device)
        c2[:n] = torch.tensor(np.stack([j["c2"] for j in batch]), dtype=torch.float32, device=env.device)
        psi[:n] = torch.tensor([j["psi"] for j in batch], dtype=torch.float32, device=env.device)
        env.step(env.encode_six_dof_action(c1, c2, psi))
        results += env._last_lift_success[:n].cpu().tolist()
        print(f"  executed {min(start + B, len(jobs))}/{len(jobs)}", flush=True)
    return results


def write_demo(out_root: Path, case: dict, g: dict, tried: int, lifted: list[int]):
    d = out_root / case["run"] / "demos" / case["case_id"]
    d.mkdir(parents=True, exist_ok=True)
    demo = {
        "schema": DEMO_SCHEMA, "run": case["run"], "case_id": case["case_id"], "shape": case["shape"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE_TEACHER,
        "frame": "robot_base: z-up, metres, quaternions wxyz; grasp contacts object-local (scaled)",
        "success": True,
        "grasp": {
            "t": 0.0, "hand": "franka", "mode": "parallel-jaw",
            "c1_local": [round(float(v), 6) for v in g["c1"]],
            "c2_local": [round(float(v), 6) for v in g["c2"]],
            "psi_rad": round(float(g["psi"]), 6),
            "width": round(float(np.linalg.norm(g["c2"] - g["c1"])), 6),
            "object_pose": {"pos": list(case["pos"]), "quat_wxyz": list(case["quat_wxyz"])},
        },
        "teacher": {"label_source": "contact_graspnet", "rank": g["rank"], "quality": g["quality"],
                    "tried": tried, "lifted_ranks": lifted},
    }
    (d / "sim-teacher.json").write_text(json.dumps(demo, indent=1))


def main():
    cfg = GraspPoseEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.grasp_mode = "six_dof"
    cfg.variants_per_family = 0
    cfg.use_camera_pc = False
    cfg.six_dof_exec_noise_pos_m = 0.0
    cfg.six_dof_exec_noise_rot_rad = 0.0
    rng = np.random.default_rng(args.seed)

    if args.exec_noise:
        defaults = GraspPoseEnvCfg()
        cfg.six_dof_exec_noise_pos_m = defaults.six_dof_exec_noise_pos_m
        cfg.six_dof_exec_noise_rot_rad = defaults.six_dof_exec_noise_rot_rad

    if args.verify:
        demos = load_demos(args.verify)
        cfg.shape_filter = tuple(sorted({d.shape for d in demos}))
    elif args.cases:
        cases = load_cases(Path(args.cases))
        cfg.shape_filter = tuple(sorted({c["shape"] for c in cases}))
    elif args.eval_cases:
        cases = load_cases(Path(args.eval_cases), halves=("demo", "heldout"))
        cfg.shape_filter = tuple(sorted({c["shape"] for c in cases}))
    if not cfg.shape_filter and not args.random_cases:
        print("[teach] nothing to do: no cases / demos found")
        return

    env = GraspPoseEnv(cfg=cfg, render_mode=None)
    env.seed(args.seed)
    env.reset()
    shape_index = {s: i for i, s in enumerate(env._shape_names)}

    if args.verify:
        demos = [d for d in demos if d.shape in shape_index]
        jobs = [{"shape": d.shape, "pos": d.pos, "quat_wxyz": d.quat_wxyz,
                 "c1": d.c1_local, "c2": d.c2_local, "psi": d.psi} for d in demos]
        ok = run_jobs(env, jobs, shape_index)
        report = {d.path: bool(s) for d, s in zip(demos, ok)}
        by_source = {}
        for d, s in zip(demos, ok):
            n, k = by_source.get(d.source, (0, 0))
            by_source[d.source] = (n + 1, k + int(s))
        path = Path(args.verify) / "verify.json"
        path.write_text(json.dumps({"by_source": by_source, "demos": report}, indent=1))
        for src, (n, k) in by_source.items():
            print(f"[teach] verify {src}: {k}/{n} demos lift with the Franka hand")
        print(f"[teach] → {path}")
        return

    if args.eval_cases:
        policy = GraspPoseActorCritic(num_actor_obs=OBS_DIM, num_critic_obs=OBS_DIM,
                                      num_actions=int(cfg.action_space)).to(args.device)
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        policy.load_state_dict(ckpt.get("model", ckpt))
        policy.eval()
        cases = [c for c in cases if c["shape"] in shape_index]
        jobs = [c for c in cases for _ in range(args.repeats)]
        ok = run_jobs(env, jobs, shape_index, policy=policy)
        per_case, by_half = {}, {}
        for c, s in zip(jobs, ok):
            per_case.setdefault(c["case_id"], []).append(bool(s))
            by_half.setdefault(c["half"], []).append(bool(s))
        rates = {h: sum(v) / len(v) for h, v in by_half.items()}
        tag = "noise" if args.exec_noise else "nonoise"
        path = Path(args.eval_cases) / f"eval_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}_{tag}.json"
        path.write_text(json.dumps({"checkpoint": args.checkpoint, "success_rate": rates,
                                    "exec_noise": args.exec_noise, "repeats": args.repeats,
                                    "cases": per_case}, indent=1))
        for h, v in sorted(by_half.items()):
            print(f"[teach] policy success on {h} cases: {rates[h] * 100:.1f}% of {len(v)} attempts")
        print(f"[teach] → {path}")
        return

    if args.random_cases:
        cases = random_cases(args.random_cases, env._shape_names, cfg, rng)
    cases = [c for c in cases if c["shape"] in shape_index]
    cands, owner = [], []
    for ci, c in enumerate(cases):
        for g in candidate_grasps(c, args.top_k, float(cfg.object_scale)):
            cands.append({**g, "shape": c["shape"], "pos": c["pos"], "quat_wxyz": c["quat_wxyz"]})
            owner.append(ci)
    R = max(args.teacher_repeats, 1)
    print(f"[teach] {len(cases)} cases, {len(cands)} candidate grasps (top {args.top_k}) × {R} executions")
    ok = run_jobs(env, [j for j in cands for _ in range(R)], shape_index)
    ok = [all(ok[i * R : (i + 1) * R]) for i in range(len(cands))]

    out_root = Path(args.out)
    per_case = {ci: [] for ci in range(len(cases))}
    for j, ci, s in zip(cands, owner, ok):
        per_case[ci].append((j, bool(s)))
    summary, n_demo = {}, 0
    for ci, c in enumerate(cases):
        tried = per_case[ci]
        lifted = [j["rank"] for j, s in tried if s]
        summary[c["case_id"]] = {"shape": c["shape"], "tried": len(tried), "lifted_ranks": lifted}
        if lifted:
            best = min((j for j, s in tried if s), key=lambda j: j["rank"])
            write_demo(out_root, c, best, len(tried), lifted)
            n_demo += 1
    run = cases[0]["run"] if cases else "none"
    (out_root / run).mkdir(parents=True, exist_ok=True)
    (out_root / run / "teacher_summary.json").write_text(json.dumps(summary, indent=1))
    no_labels = sum(1 for v in summary.values() if v["tried"] == 0)
    print(f"[teach] {n_demo}/{len(cases)} cases solved → {out_root / run}  "
          f"({no_labels} had no usable GraspNet grasp; "
          f"top-1 lifted on {sum(1 for v in summary.values() if 0 in v['lifted_ranks'])})")


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang for minutes after headless runs; every output
    # is already on disk, so leave without it.
    sys.stdout.flush()
    os._exit(0)
