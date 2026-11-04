#!/usr/bin/env python3
"""
Live robot check for VR demos: can the Franka hand grasp at the two points just
demonstrated?

The demonstrator gives grasp points only; the approach is the robot's to choose.
The VR app (vr-grasping-project, POST /api/check) forwards each finished demo here
before saving it. The demo's two contacts run at the case's exact pose, through
the same six_dof decode and floating hand as a policy action, over a grid of
approach angles without noise; up to 3 lifting approaches are then re-run under
training's execution noise and the best is reported. The headset shows the
verdict, so the demonstrator can pick other points while still on the case.

    scripts/check_server.sh start      # serves http://127.0.0.1:8765/check

POST /check   body: a unitree-grasp-demo/v1 demo (as the VR app saves it)
  → {"lifted": some approach lifted at these points, "lifted_psi_deg": [...],
     "noisy_lifts": k, "noisy_tries": n, "robust": k >= n/2, "width_m": jaw width,
     "reason": why the points do not work (null when robust), "ms": time,
     "attempts": [{psi_deg, lifted, tcp_local, quat_local_wxyz, width_m, table_raise_m}]}
     — the best lifting approach (empty if none), in the object's frame, for
     drawing a ghost gripper on the case's object.
GET  /health  → {"ok": true, "shapes": n}
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--cases_root", type=str, default="../../data/vr_failures",
                    help="Failure dataset root (<root>/<run>/cases/*.json): its shapes are loaded, and a "
                         "demo is checked at its case's resting pose.")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--psi_grid", type=float, nargs="*", default=[-80, -60, -40, -20, 0, 20, 40, 60, 80],
                    help="Approach angles (deg) the robot may use at the demo's grasp points; the "
                         "demonstrator's own hand angle plays no part.")
parser.add_argument("--noisy_tries", type=int, default=4)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import math
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.grasp_pose_env_cfg import GraspPoseEnvCfg
from envs.grasp_pose_env import GraspPoseEnv
from envs._object_registry import shape_asset
from isaaclab.utils.math import matrix_from_quat
from training.demo_replay import demo_from_json, quat_conj_wxyz, quat_rotate_wxyz


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def case_poses(root: Path) -> dict:
    """(run, case_id) → (shape, object_pose) for every case under the root."""
    out = {}
    for p in root.glob("*/cases/*.json"):
        c = json.loads(p.read_text())
        out[(p.parents[1].name, c["case_id"])] = (c["object"]["shape"], c["object_pose"])
    return out


class Checker:
    def __init__(self):
        self.cases = case_poses(Path(args.cases_root))
        cfg = GraspPoseEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.grasp_mode = "six_dof"
        cfg.variants_per_family = 0
        cfg.use_camera_pc = False
        cfg.shape_filter = tuple(sorted({s for s, _ in self.cases.values()}))
        self.noise = (cfg.six_dof_exec_noise_pos_m, cfg.six_dof_exec_noise_rot_rad)
        cfg.six_dof_exec_noise_pos_m = 0.0
        cfg.six_dof_exec_noise_rot_rad = 0.0
        self.width_bounds = cfg.grasp_width_bounds
        self.env = GraspPoseEnv(cfg=cfg, render_mode=None)
        self.env.seed(args.seed)
        self.env.reset()
        self.shape_index = {s: i for i, s in enumerate(self.env._shape_names)}
        self.meshes = {}
        print(f"[check] {len(self.shape_index)} shapes from {len(self.cases)} cases in {args.cases_root}",
              flush=True)

    def _run(self, shape: str, pos, quat, c1, c2, psis: list[float]) -> list[bool]:
        env, B, n = self.env, self.env.num_envs, len(psis)
        assert n <= B
        env.queue_replays(env_ids=list(range(n)), shape_ids=[self.shape_index[shape]] * n,
                          xy=np.array([pos[:2]] * n, dtype=np.float32),
                          quat_wxyz=np.array([quat] * n, dtype=np.float32))
        env.step(torch.zeros(B, int(env.cfg.action_space), device=env.device))   # spawns the case
        t = lambda v: torch.tensor(np.asarray(v), dtype=torch.float32, device=env.device)
        c1s = torch.zeros(B, 3, device=env.device)
        c2s = torch.zeros(B, 3, device=env.device)
        psi = torch.zeros(B, device=env.device)
        c1s[:n], c2s[:n], psi[:n] = t(c1), t(c2), t(psis)
        env.step(env.encode_six_dof_action(c1s, c2s, psi))
        # What the hand did, for drawing it in VR: its orientation, and how far the
        # planner raised it so the fingertips clear the table (it then grips higher).
        R = matrix_from_quat(env._six_quat_w[:n])
        planned = env._six_mid_w[:n] - R[:, :, 2] * (env._FLOAT_TCP_OFFSET_M + env.cfg.finger_pad_offset_m)
        executed = env._float_q_final[:n, :3] + env._robot.data.root_pos_w[:n]
        self.last_hand = [{"quat_w": q, "raise_m": r} for q, r in
                          zip(env._six_quat_w[:n].cpu().numpy(), (executed - planned)[:, 2].cpu().tolist())]
        return env._last_lift_success[:n].cpu().tolist()

    def _attempt(self, i: int, psi: float, lifted: bool, c1, c2, q_obj) -> dict:
        """Attempt i of the last _run, in the object's frame (as the VR ghost is drawn)."""
        h = self.last_hand[i]
        q_local = quat_mul_wxyz(quat_conj_wxyz(q_obj), h["quat_w"])
        raise_local = quat_rotate_wxyz(quat_conj_wxyz(q_obj), np.array([0.0, 0.0, h["raise_m"]]))
        return {"psi_deg": round(math.degrees(psi)), "lifted": bool(lifted),
                "tcp_local": [round(float(v), 5) for v in (c1 + c2) / 2 + raise_local],
                "quat_local_wxyz": [round(float(v), 5) for v in q_local],
                "width_m": round(float(np.linalg.norm(c2 - c1)), 4),
                "table_raise_m": round(h["raise_m"], 4)}

    def _reason(self, shape: str, c1, c2, width: float) -> str:
        """Why a grasp cannot work, from the geometry alone (best guess)."""
        lo, hi = self.width_bounds
        if width > hi:
            return f"too wide for the gripper: {width * 100:.1f} cm apart, it opens {hi * 100:.0f} cm"
        if width < lo:
            return f"contacts too close together ({width * 100:.1f} cm)"
        if shape not in self.meshes:
            import trimesh
            m = trimesh.load(shape_asset(shape, ".obj"), force="mesh")
            m.apply_scale(float(self.env.cfg.object_scale))
            self.meshes[shape] = m
        m = self.meshes[shape]
        _, _, tri = m.nearest.on_surface(np.stack([c1, c2]))
        n1, n2 = m.face_normals[tri]
        u = (c2 - c1) / max(width, 1e-9)
        a1 = math.degrees(math.acos(np.clip(-n1 @ u, -1, 1)))
        a2 = math.degrees(math.acos(np.clip(n2 @ u, -1, 1)))
        if max(a1, a2) > 40:
            return ("the gripper pads would slide off, not squeeze: pinch a wall, rim or handle from "
                    "both sides (e.g. a bowl: one finger inside the wall, one outside, below the rim)")
        return "no approach works at these points: the jaw cannot reach them or the grip does not hold"

    def check(self, demo: dict) -> dict:
        t0 = time.time()
        key = (demo.get("run"), demo.get("case_id"))
        if key not in self.cases:
            return {"error": f"unknown case {key[0]}/{key[1]}"}
        shape, pose = self.cases[key]
        d = demo_from_json(demo, "live", pose)
        if d is None or d.shape not in self.shape_index:
            return {"error": "no grasp in the demo"}
        c1, c2 = np.asarray(d.c1_local), np.asarray(d.c2_local)
        grid = [math.radians(p) for p in args.psi_grid]
        # Only the two grasp points are the demonstrator's: the robot picks its own approach.
        res = self._run(d.shape, d.pos, d.quat_wxyz, c1, c2, grid)
        lifting = [i for i, s in enumerate(res) if s]
        out = {"lifted": bool(lifting), "lifted_psi_deg": [round(math.degrees(grid[i])) for i in lifting],
               "noisy_lifts": 0, "noisy_tries": args.noisy_tries, "robust": False,
               "width_m": round(float(np.linalg.norm(c2 - c1)), 4), "reason": None, "attempts": []}
        if lifting:
            # Re-run the lifting approaches (up to 3, one batch) with placement error; keep the best.
            tried = lifting[:: max(1, len(lifting) // 3)][:3]
            hands = [self._attempt(i, grid[i], True, c1, c2, d.quat_wxyz) for i in tried]
            cfg = self.env.cfg
            cfg.six_dof_exec_noise_pos_m, cfg.six_dof_exec_noise_rot_rad = self.noise
            try:
                noisy = self._run(d.shape, d.pos, d.quat_wxyz, c1, c2,
                                  [grid[i] for i in tried for _ in range(args.noisy_tries)])
            finally:
                cfg.six_dof_exec_noise_pos_m = cfg.six_dof_exec_noise_rot_rad = 0.0
            T = args.noisy_tries
            k, best = max((sum(noisy[j * T : (j + 1) * T]), j) for j in range(len(tried)))
            out.update(noisy_lifts=int(k), robust=2 * k >= T, attempts=[hands[best]])
            if not out["robust"]:
                out["reason"] = (f"lifts only when placed exactly ({k}/{T} with the robot's placement error): "
                                 "move the points to a sturdier spot")
        else:
            out["reason"] = self._reason(d.shape, c1, c2, out["width_m"])
            raise_m = min(h["raise_m"] for h in self.last_hand)
            if raise_m > 0.005 and out["reason"].startswith("no approach"):
                out["reason"] = (f"too close to the table: from any side the fingertips would hit it "
                                 f"(it has to grip {raise_m * 100:.1f} cm higher). Pick points higher up")
        out["ms"] = round((time.time() - t0) * 1000)
        return out


class Handler(BaseHTTPRequestHandler):
    checker: Checker = None

    def _send(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "shapes": len(self.checker.shape_index)})
        self._send(404, {"error": "no such endpoint"})

    def do_POST(self):
        if self.path != "/check":
            return self._send(404, {"error": "no such endpoint"})
        try:
            demo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            out = self.checker.check(demo)
        except Exception as e:                            # report, keep serving
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        print(f"[check] {demo.get('case_id')}: {out}", flush=True)
        self._send(400 if "error" in out else 200, out)

    def log_message(self, *_):
        pass


def main():
    Handler.checker = Checker()
    server = HTTPServer((args.host, args.port), Handler)      # one request at a time: one sim
    print(f"[check] serving http://{args.host}:{args.port}/check", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
