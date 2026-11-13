#!/usr/bin/env python3
"""
Table-sorting demo: the trained point-cloud grasp policy on a full Franka arm.

Three depth cameras look at the table (GPU ray casting against the simulated
scene, robot included; RTX rendering crashes on this machine) and each pixel keeps
the id of the object it hit, which splits the fused cloud into objects (a real
robot would use an instance segmenter). For each
object the policy gets 128 points of its segment as in training (world axes,
centred on the observed centroid) and returns two contact points and an approach
angle. A scripted arm (differential IK) executes that grasp, lifts, carries the
object to the bin for its category and lets go. The object's pose and mesh are
never used for grasping: contacts snap to the observed cloud, not to the mesh.
Objects are not pinned while the hand approaches (training pins them).

Scenes:
  easy  six fruits (banana and pear never seen in training) → citrus / other fruit
  hard  six families from the round-3 held-out hard cases (never demonstrated in VR),
        original meshes (not the mutated training instances) at the yaw where RL
        alone kept failing → kitchen / tools

  showcase  six irregular test objects (never seen in training), picked for the demo
        video because the policy handles them well in the paper's test → tools / household

From wrappers/isaaclab, after `source ../../scripts/v2_env.sh`:
    python scripts/sort_demo.py --headless --scene easy --checkpoint <ckpt> --record ../../data/sort/easy.npz
    python scripts/render_sort.py ../../data/sort/easy.npz             # → easy.mp4
    python scripts/sort_demo.py --headless --scene hard --checkpoint <ckpt> --trials 10 --json ../../data/sort/hard.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scene", choices=["easy", "hard", "showcase"], default="easy")
parser.add_argument("--checkpoint", default=None, help="Policy checkpoint (not needed for --method graspnet)")
parser.add_argument("--method", choices=["policy", "graspnet"], default="policy",
                    help="graspnet: Contact-GraspNet labels (<instance>_graspnet.json) placed at the object's true "
                         "pose; each try takes the next best-ranked grasp of the top 10 that passes the checks of "
                         "the paper's filtered baseline and is reachable")
parser.add_argument("--record", default=None, help=".npz of the first trial for scripts/render_sort.py")
parser.add_argument("--title", default=None, help="Overlay title, e.g. which policy this is")
parser.add_argument("--trials", type=int, default=1, help="Scene resets (positions jittered after the first)")
parser.add_argument("--max_tries", type=int, default=3, help="Grasp attempts per object (first = policy mean)")
parser.add_argument("--json", default=None, help="Write per-object results here")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cams", type=int, choices=[3, 6], default=6, help="Depth views fused per look")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

REPO = bootstrap()

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_from_matrix, quat_slerp

from envs._object_registry import shape_asset
from envs.grasp_pose_env_cfg import CONTACT_BOUND_M, NUM_PC_POINTS, OBS_DIM, OBJECT_MASS, OBJECT_SCALE, \
    SIX_DOF_NUM_ACTIONS, GraspPoseEnvCfg
from envs.six_dof_geometry import approach_from_jaw
from models.grasp_pose_actor_critic import GraspPoseActorCritic
from state_recorder import _link_visual_meshes

DT = 1 / 60.0
DEV = "cuda:0"
TABLE_CLEARANCE_M = float(os.environ.get("GRASP_TABLE_CLEARANCE_M", 0.003))
# Same constants as the training executor (grasp_pose_env.py / grasp_pose_env_cfg.py).
TCP_OFFSET_M = 0.0584 + 0.045        # hand frame → finger-pad centre
FINGERTIP_Z_M, FINGERTIP_HALF_X_M, FINGERTIP_OUT_M = 0.0545, 0.00875, 0.0152
MIN_WIDTH_M, MAX_WIDTH_M, SQUEEZE_M = 0.025, 0.08, 0.012
STANDOFF_M, HOVER_M, LIFT_M, LIFT_OK_M = 0.10, 0.08, 0.15, 0.05

# ── Scenes ────────────────────────────────────────────────────────────────────
# Bins: name, centre xy, inner size, colour. Objects: label, shape, category, colour, xy, quat.
BIN_WALL_H, BIN_WALL_T = 0.08, 0.01

FRUIT = {   # shape, category, colour
    "apple":  ("ycb_013_apple",  "other",  (0.72, 0.06, 0.08)),
    "lemon":  ("ycb_014_lemon",  "citrus", (0.96, 0.85, 0.18)),
    "banana": ("ycb_011_banana", "other",  (0.98, 0.82, 0.30)),
    "orange": ("ycb_017_orange", "citrus", (0.98, 0.50, 0.06)),
    "pear":   ("ycb_016_pear",   "other",  (0.55, 0.72, 0.20)),
    "plum":   ("ycb_018_plum",   "other",  (0.35, 0.08, 0.32)),
}
EASY_SLOTS = [(0.45, -0.15), (0.45, 0.0), (0.45, 0.15), (0.61, -0.15), (0.61, 0.0), (0.61, 0.15)]

HARD = {    # family, category, colour
    "fork":     ("ycb_030_fork",              "kitchen", (0.70, 0.72, 0.76)),
    "sponge":   ("ycb_026_sponge",            "kitchen", (0.85, 0.80, 0.25)),
    "pudding":  ("ycb_008_pudding_box",       "kitchen", (0.55, 0.33, 0.18)),
    "scissors": ("ycb_037_scissors",          "tools",   (0.80, 0.12, 0.15)),
    "clamp":    ("ycb_052_extra_large_clamp", "tools",   (0.95, 0.45, 0.05)),
    "wrench":   ("ycb_042_adjustable_wrench", "tools",   (0.30, 0.45, 0.70)),
}
HARD_SLOTS = [(0.42, -0.20), (0.42, 0.0), (0.42, 0.20), (0.64, -0.20), (0.64, 0.0), (0.64, 0.20)]
SHOWCASE = {   # shape, category, colour (all held-out test objects)
    "knife":   ("ycb_032_knife",           "tools",     (0.55, 0.57, 0.62)),
    "clamp":   ("ycb_050_medium_clamp",    "tools",     (0.95, 0.45, 0.05)),
    "marker":  ("ycb_040_large_marker",    "tools",     (0.12, 0.12, 0.14)),
    "spray":   ("ycb_022_windex_bottle",   "household", (0.25, 0.55, 0.90)),
    "bleach":  ("ycb_021_bleach_cleanser", "household", (0.92, 0.92, 0.88)),
    "cup":     ("ycb_065-c_cups",          "household", (0.85, 0.25, 0.30)),
}
CASES = REPO / "data" / "vr_round3" / "round3"


def hard_cases() -> dict[str, dict]:
    """First held-out case per family (sorted ids) that rests flat: yaw-only quaternion.
    Deterministic, so the scene is not picked by how either policy does on it."""
    held = sorted(json.load(open(CASES / "split.json"))["heldout"])
    out = {}
    for label, (family, _, _) in HARD.items():
        for cid in held:
            if cid.split("__")[0].split(".v")[0] != family:
                continue
            case = json.load(open(CASES / "cases" / f"{cid}.json"))
            q = case["object_pose"]["quat_wxyz"]
            if abs(q[1]) < 1e-3 and abs(q[2]) < 1e-3:
                out[label] = {"case": cid, "shape": family, "quat": q}   # original mesh, case yaw
                break
    return out


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def scene_spec(scene: str, rng: np.random.Generator, jitter: bool) -> tuple[list, list]:
    j = (lambda: rng.uniform(-0.015, 0.015, 2)) if jitter else (lambda: np.zeros(2))
    if scene == "easy":
        bins = [("Citrus", "citrus", (0.27, -0.42), (0.24, 0.24), (0.98, 0.78, 0.25)),
                ("Other fruit", "other", (0.27, 0.42), (0.24, 0.24), (0.45, 0.70, 0.35))]
        order = list(FRUIT) if not jitter else list(rng.permutation(list(FRUIT)))
        objs = []
        for label, (x, y) in zip(order, EASY_SLOTS):
            shape, cat, col = FRUIT[label]
            yaw = rng.uniform(-math.pi, math.pi) if jitter else 0.6 * len(objs)
            objs.append(dict(label=label, shape=shape, category=cat, color=col,
                             xy=np.array([x, y]) + j(), quat=yaw_quat(yaw)))
        return bins, objs
    if scene == "showcase":
        bins = [("Tools", "tools", (0.24, -0.47), (0.30, 0.30), (0.75, 0.45, 0.30)),
                ("Household", "household", (0.24, 0.47), (0.30, 0.30), (0.35, 0.55, 0.75))]
        objs = []
        for label, (x, y) in zip(SHOWCASE, HARD_SLOTS):
            shape, cat, col = SHOWCASE[label]
            yaw = rng.uniform(-math.pi, math.pi) if jitter else 0.7 * len(objs) - 1.0
            objs.append(dict(label=label, shape=shape, category=cat, color=col,
                             xy=np.array([x, y]) + j(), quat=yaw_quat(yaw)))
        return bins, objs
    bins = [("Kitchen", "kitchen", (0.24, -0.47), (0.30, 0.30), (0.35, 0.55, 0.75)),
            ("Tools", "tools", (0.24, 0.47), (0.30, 0.30), (0.75, 0.45, 0.30))]
    cases = hard_cases()
    objs = []
    for label, (x, y) in zip(HARD, HARD_SLOTS):
        _, cat, col = HARD[label]
        c = cases[label]
        objs.append(dict(label=label, shape=c["shape"], category=cat, color=col, case=c["case"],
                         xy=np.array([x, y]) + j(), quat=c["quat"]))
    return bins, objs


# ── Scene construction ────────────────────────────────────────────────────────
def mesh_min_z(shape: str, quat: list[float]) -> float:
    import trimesh
    m = trimesh.load(str(shape_asset(shape, ".obj")), process=False)
    v = torch.tensor(np.asarray(m.vertices) * OBJECT_SCALE, dtype=torch.float32)
    R = matrix_from_quat(torch.tensor([quat], dtype=torch.float32))[0]
    return float((v @ R.T)[:, 2].min())


def build_scene(bins, objs):
    sim_utils.GroundPlaneCfg(color=(0.86, 0.86, 0.84)).func("/World/ground", sim_utils.GroundPlaneCfg(
        color=(0.86, 0.86, 0.84)), translation=(0, 0, -0.75))
    sim_utils.DomeLightCfg(intensity=1800.0, color=(1.0, 1.0, 1.0)).func(
        "/World/dome", sim_utils.DomeLightCfg(intensity=1800.0, color=(1.0, 1.0, 1.0)))
    light = sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0)
    light.func("/World/sun", light, orientation=(0.92, 0.2, 0.3, 0.1))

    def box(path, size, pos, color, collide=True):
        cfg = sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) if collide else None,
            collision_props=sim_utils.CollisionPropertiesCfg() if collide else None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
        )
        cfg.func(path, cfg, translation=pos)

    box("/World/table", (1.00, 1.40, 0.75), (0.45, 0.0, -0.375), (0.62, 0.50, 0.38))
    xs = [o["xy"][0] for o in objs]
    ys = [o["xy"][1] for o in objs]
    box("/World/tray", (max(xs) - min(xs) + 0.24, max(ys) - min(ys) + 0.26, 0.002),
        ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0.0004), (0.93, 0.92, 0.88), collide=False)
    for k, (_, _, (cx, cy), (sx, sy), col) in enumerate(bins):
        h, t = BIN_WALL_H, BIN_WALL_T
        box(f"/World/bin{k}_floor", (sx, sy, 0.002), (cx, cy, 0.0004), col, collide=False)
        box(f"/World/bin{k}_n", (sx + 2 * t, t, h), (cx, cy + sy / 2 + t / 2, h / 2), col)
        box(f"/World/bin{k}_s", (sx + 2 * t, t, h), (cx, cy - sy / 2 - t / 2, h / 2), col)
        box(f"/World/bin{k}_e", (t, sy, h), (cx + sx / 2 + t / 2, cy, h / 2), col)
        box(f"/World/bin{k}_w", (t, sy, h), (cx - sx / 2 - t / 2, cy, h / 2), col)

    robot_cfg = GraspPoseEnvCfg().robot.replace(prim_path="/World/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = True        # as the franka_arm executor
    robot_cfg.actuators["panda_hand"].effort_limit_sim = 140.0   # as the trained floating hand
    robot = Articulation(robot_cfg)

    rigid = []
    for i, o in enumerate(objs):
        z = -mesh_min_z(o["shape"], o["quat"]) + 0.002
        cfg = RigidObjectCfg(
            prim_path=f"/World/obj{i}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(float(o["xy"][0]), float(o["xy"][1]), z),
                                                      rot=tuple(o["quat"])),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(shape_asset(o["shape"], ".usd")),
                scale=(OBJECT_SCALE,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=o["color"], roughness=0.45),
                semantic_tags=[("class", f"obj{i}")],
            ),
        )
        rigid.append(RigidObject(cfg))

    return robot, rigid


def box_mesh(size, center):
    import trimesh
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)


def static_boxes(bins):
    """Table top and bin walls as (verts, faces) in the world, for ray casting and rendering."""
    out = [box_mesh((1.00, 1.40, 0.02), (0.45, 0.0, -0.01))]
    for _, _, (cx, cy), (sx, sy), _ in bins:
        h, t = BIN_WALL_H, BIN_WALL_T
        out += [box_mesh((sx + 2 * t, t, h), (cx, cy + sy / 2 + t / 2, h / 2)),
                box_mesh((sx + 2 * t, t, h), (cx, cy - sy / 2 - t / 2, h / 2)),
                box_mesh((t, sy, h), (cx + sx / 2 + t / 2, cy, h / 2)),
                box_mesh((t, sy, h), (cx - sx / 2 - t / 2, cy, h / 2))]
    return out


# ── Depth cameras (Warp ray casting) ──────────────────────────────────────────
import warp as wp

wp.init()


@wp.kernel
def _raycast(mesh: wp.uint64, origin: wp.vec3, dirs: wp.array(dtype=wp.vec3), max_t: float,
             out_t: wp.array(dtype=float), out_f: wp.array(dtype=int)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, origin, dirs[i], max_t)
    if q.result:
        out_t[i] = q.t
        out_f[i] = q.face
    else:
        out_t[i] = -1.0
        out_f[i] = -1


class DepthCams:
    """Pinhole depth cameras that ray-cast the current scene. Each hit keeps the label
    of the triangle it hit: object index, or -1 for robot / table / bins."""

    def __init__(self, eyes, target, width=320, height=240, hfov_deg=60.0, noise_m=0.001):
        self.noise = noise_m
        f = width / (2 * math.tan(math.radians(hfov_deg) / 2))
        v, u = torch.meshgrid(torch.arange(height, device=DEV), torch.arange(width, device=DEV), indexing="ij")
        ray_c = torch.stack([(u + 0.5 - width / 2) / f, (v + 0.5 - height / 2) / f, torch.ones_like(u, dtype=torch.float32)], -1)
        self.cams = []
        for e in eyes:
            e = torch.tensor(e, device=DEV, dtype=torch.float32)
            fwd = torch.tensor(target, device=DEV, dtype=torch.float32) - e
            fwd = fwd / fwd.norm()
            right = torch.linalg.cross(fwd, torch.tensor([0.0, 0.0, 1.0], device=DEV))
            right = right / right.norm()
            down = torch.linalg.cross(fwd, right)
            R = torch.stack([right, down, fwd], -1)               # ROS camera axes → world
            d = ray_c.reshape(-1, 3) @ R.T
            self.cams.append((e, d / d.norm(dim=-1, keepdim=True), ray_c.reshape(-1, 3)[:, 2] / ray_c.reshape(-1, 3).norm(dim=-1)))

    def capture(self, verts: torch.Tensor, faces: torch.Tensor, labels: torch.Tensor) -> dict[int, torch.Tensor]:
        mesh = wp.Mesh(points=wp.from_torch(verts.contiguous(), dtype=wp.vec3),
                       indices=wp.from_torch(faces.reshape(-1).int().contiguous(), dtype=wp.int32))
        clouds: dict[int, list] = {}
        for e, dirs, cos in self.cams:
            n = len(dirs)
            t = wp.zeros(n, dtype=float, device=DEV)
            f = wp.zeros(n, dtype=int, device=DEV)
            wp.launch(_raycast, dim=n, inputs=[mesh.id, wp.vec3(*e.tolist()), wp.from_torch(dirs, dtype=wp.vec3), 3.0, t, f],
                      device=DEV)
            t, f = wp.to_torch(t), wp.to_torch(f)
            hit = f >= 0
            depth = t * cos + torch.randn_like(t) * self.noise         # depth noise along the optical axis
            pts = e + dirs * (depth / cos).unsqueeze(-1)
            lab = torch.where(hit, labels[f.clamp(min=0).long()], torch.full_like(f, -1))
            for k in torch.unique(lab[lab >= 0]).tolist():
                clouds.setdefault(int(k), []).append(pts[lab == k])
        return {k: torch.cat(v) for k, v in clouds.items()}


# ── Perception + policy ───────────────────────────────────────────────────────
def voxel_down(p: torch.Tensor, v: float) -> torch.Tensor:
    keys = torch.floor(p / v).long()
    _, inv = torch.unique(keys, dim=0, return_inverse=True)
    out = torch.zeros(int(inv.max()) + 1, 3, device=p.device).index_add_(0, inv, p)
    cnt = torch.zeros(len(out), device=p.device).index_add_(0, inv, torch.ones(len(p), device=p.device))
    return out / cnt.unsqueeze(-1)


def load_policy(path: str) -> GraspPoseActorCritic:
    ckpt = torch.load(path, map_location=DEV, weights_only=False)
    if "encoder" in ckpt and "head" in ckpt and "model" not in ckpt:   # BC warm start
        state = {f"actor_encoder.{k}": v for k, v in ckpt["encoder"].items()}
        state.update({f"actor_head.{k}": v for k, v in ckpt["head"].items()})
    else:
        state = ckpt.get("model", ckpt)
    model = GraspPoseActorCritic(OBS_DIM, OBS_DIM, SIX_DOF_NUM_ACTIONS, NUM_PC_POINTS, 128).to(DEV)
    missing, _ = model.load_state_dict(state, strict=False)
    assert not [k for k in missing if k.startswith("actor")], f"actor weights missing: {missing}"
    return model.eval()


@torch.no_grad()
def plan_grasp(policy, cloud: torch.Tensor, sample: bool, gen) -> list[dict]:
    """Policy grasp on one object's observed cloud → candidate executions, in order of
    preference: the policy's approach (both jaw-sign hand poses), then its mirror (-psi,
    same contacts). The policy was trained on a floating hand; the arm takes the first
    candidate it can reach."""
    sparse = voxel_down(cloud, 0.004)
    dense = voxel_down(cloud, 0.002)
    idx = torch.randint(0, len(sparse), (NUM_PC_POINTS,), device=DEV, generator=gen)
    pts = sparse[idx]
    centroid = pts.mean(0)
    obs = (pts - centroid).reshape(1, -1)
    if sample:
        policy.update_distribution(obs)
        a = policy.distribution.sample()[0].clamp(-1, 1)
    else:
        a = policy.act_inference(obs)[0].clamp(-1, 1)
    B = CONTACT_BOUND_M
    c1 = (a[0:3] + 1) / 2 * 2 * B - B + centroid
    c2 = (a[3:6] + 1) / 2 * 2 * B - B + centroid

    def snap(x):
        return dense[(dense - x).norm(dim=-1).argmin()]

    c1, c2 = snap(c1), snap(c2)
    d = (c2 - c1).norm()
    u = (c2 - c1) / d if d > 1e-4 else torch.tensor([0.0, 1.0, 0.0], device=DEV)
    mid = 0.5 * (c1 + c2)
    half = max(float(d), MIN_WIDTH_M) / 2
    c1, c2 = snap(mid - u * half), snap(mid + u * half)
    d = (c2 - c1).norm()
    u = (c2 - c1) / d if d > 1e-4 else u
    width = float(d.clamp(MIN_WIDTH_M, MAX_WIDTH_M))
    mid = 0.5 * (c1 + c2)
    psi0 = a[6] * (math.pi / 2)
    radial = mid.clone()
    radial[2] = 0.0
    radial = radial / radial.norm().clamp(min=1e-6)
    corners = torch.tensor([[x, s * y, 0.0584 + FINGERTIP_Z_M] for x in (-FINGERTIP_HALF_X_M, FINGERTIP_HALF_X_M)
                            for y in (0.04, 0.04 + FINGERTIP_OUT_M) for s in (1.0, -1.0)], device=DEV)
    down = torch.tensor([0.0, 1.0, 0.0, 0.0], device=DEV)
    out = []
    for mirrored in (False, True):
        psi = -psi0 if mirrored else psi0
        approach = approach_from_jaw(u[None], psi[None], radial[None])[0]
        qs = [quat_from_matrix(torch.stack([torch.linalg.cross(y, approach), y, approach], dim=-1)[None])[0]
              for y in (u, -u)]
        qs.sort(key=lambda q: -float((q * down).sum().abs()))      # less wrist roll from hand-down first
        for q in qs:
            R = matrix_from_quat(q[None])[0]
            z = R[:, 2]
            hand_final = mid - z * TCP_OFFSET_M
            # Lowest open-fingertip corner no lower than the table (+ clearance), as in training.
            low = (hand_final[2] + corners @ R[2, :]).min()
            hand_final[2] += (TABLE_CLEARANCE_M - low).clamp(min=0.0)
            out.append(dict(c1=c1, c2=c2, mid=mid, width=width, psi_deg=float(psi) * 180 / math.pi, quat=q,
                            mirrored=mirrored, outward=float((approach[:2] * radial[:2]).sum()),
                            approach=approach, hand_final=hand_final, hand_pre=hand_final - z * STANDOFF_M,
                            obs_pts=pts))
    return out


def hand_candidates(c1, c2, approach, width, psi, mirrored, pts) -> list[dict]:
    """Both jaw-sign hand poses for one grasp (contacts + approach), table clearance applied."""
    u = (c2 - c1) / (c2 - c1).norm().clamp(min=1e-6)
    mid = 0.5 * (c1 + c2)
    corners = torch.tensor([[x, s * y, 0.0584 + FINGERTIP_Z_M] for x in (-FINGERTIP_HALF_X_M, FINGERTIP_HALF_X_M)
                            for y in (0.04, 0.04 + FINGERTIP_OUT_M) for s in (1.0, -1.0)], device=DEV)
    down = torch.tensor([0.0, 1.0, 0.0, 0.0], device=DEV)
    radial = mid.clone()
    radial[2] = 0.0
    radial = radial / radial.norm().clamp(min=1e-6)
    qs = [quat_from_matrix(torch.stack([torch.linalg.cross(y, approach), y, approach], dim=-1)[None])[0] for y in (u, -u)]
    qs.sort(key=lambda q: -float((q * down).sum().abs()))
    out = []
    for q in qs:
        R = matrix_from_quat(q[None])[0]
        z = R[:, 2]
        hand_final = mid - z * TCP_OFFSET_M
        low = (hand_final[2] + corners @ R[2, :]).min()
        hand_final[2] += (TABLE_CLEARANCE_M - low).clamp(min=0.0)
        out.append(dict(c1=c1, c2=c2, mid=mid, width=width, psi_deg=psi, quat=q, mirrored=mirrored,
                        outward=float((approach[:2] * radial[:2]).sum()), approach=approach,
                        hand_final=hand_final, hand_pre=hand_final - z * STANDOFF_M, obs_pts=pts))
    return out


def graspnet_grasps(shape: str, pos: torch.Tensor, quat: torch.Tensor, top_k: int = 10) -> list[tuple]:
    """Contact-GraspNet's top_k labelled grasps at the object's true pose, in rank order, keeping
    those that pass the filtered baseline's checks (play_grasp_pose.py graspnet_filtered): jaw
    opening within the gripper's range, approach not from below, jaw not vertical, palm above the table."""
    path = shape_asset(shape, "_graspnet.json")
    if not path.exists():
        return []
    R = matrix_from_quat(quat[None])[0]
    out = []
    for g in json.load(open(path))["grasps"][:top_k]:
        c1 = pos + R @ (torch.tensor(g["c1"], device=DEV) * OBJECT_SCALE)
        c2 = pos + R @ (torch.tensor(g["c2"], device=DEV) * OBJECT_SCALE)
        a = R @ torch.tensor(g["approach"], device=DEV)
        width = float((c2 - c1).norm())
        jaw = (c2 - c1) / max(width, 1e-9)
        a = a - (a * jaw).sum() * jaw                     # exactly perpendicular to the jaw
        a = a / a.norm().clamp(min=1e-6)
        palm_z = float((0.5 * (c1 + c2) - a * TCP_OFFSET_M)[2])
        if not (0.01 <= width <= MAX_WIDTH_M and float(a[2]) <= 1e-3 and abs(float(jaw[2])) < 0.98 and palm_z >= 0.02):
            continue
        psi = math.degrees(math.acos(max(-1.0, min(1.0, -float(a[2])))))
        out.append((c1, c2, a, max(width, MIN_WIDTH_M), psi))
    return out


# ── Arm IK (pinocchio, multi-seed, joint limits) ──────────────────────────────
class PandaIK:
    """Full-pose IK for panda_hand on the 7 arm joints, from several seeds; returns the
    solution closest to the first seed, or None if the pose is out of reach."""

    def __init__(self):
        import pinocchio as pin
        self.pin = pin
        urdf = Path.home() / ".cache/robot_descriptions/example-robot-data/robots/panda_description/urdf/panda.urdf"
        full = pin.buildModelFromUrdf(str(urdf))
        lock = [full.getJointId(n) for n in ("panda_finger_joint1", "panda_finger_joint2") if full.existJointName(n)]
        self.m = pin.buildReducedModel(full, lock, pin.neutral(full))
        self.d = self.m.createData()
        self.fid = self.m.getFrameId("panda_hand")
        self.lo = self.m.lowerPositionLimit + 0.02
        self.hi = self.m.upperPositionLimit - 0.02
        self.rng = np.random.default_rng(0)

    def fk(self, q):
        self.pin.framesForwardKinematics(self.m, self.d, q)
        return self.d.oMf[self.fid]

    def solve(self, pos, quat_wxyz, seeds, n_random=8):
        pin = self.pin
        w, x, y, z = [float(v) for v in quat_wxyz]
        T = pin.SE3(pin.Quaternion(w, x, y, z).toRotationMatrix(), np.asarray(pos, dtype=float))
        seeds = list(seeds) + [self.rng.uniform(self.lo, self.hi) for _ in range(n_random)]
        best = None
        for q0 in seeds:
            q = np.clip(np.asarray(q0, dtype=float), self.lo, self.hi)
            for _ in range(150):
                err = pin.log6(self.fk(q).actInv(T)).vector
                if np.linalg.norm(err[:3]) < 2e-3 and np.linalg.norm(err[3:]) < 0.02:
                    if best is None or np.linalg.norm(q - seeds[0]) < np.linalg.norm(best - seeds[0]):
                        best = q.copy()
                    break
                J = pin.computeFrameJacobian(self.m, self.d, q, self.fid, pin.LOCAL)
                q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err), self.lo, self.hi)
            if best is not None and q0 is seeds[0]:
                break      # the nearest seed converged: take the smooth solution
        return best


# ── Arm control + recording ───────────────────────────────────────────────────
class Runner:
    def __init__(self, sim, robot, rigid, bins, objs, record: bool):
        self.sim, self.robot, self.rigid = sim, robot, rigid
        self.bins, self.objs = bins, objs
        self.arm_ids = robot.find_joints("panda_joint.*")[0]
        self.grip_ids = robot.find_joints("panda_finger_joint.*")[0]
        self.ee_idx = robot.find_bodies("panda_hand")[0][0]
        self.jac_idx = self.ee_idx - 1
        self.pik = PandaIK()
        T = self.pik.fk(self.q_now())
        p, _ = self.ee()
        print(f"[sort] pinocchio vs sim hand position: {np.round(T.translation, 4)} vs {np.round(p.cpu().numpy(), 4)}",
              flush=True)
        self.grip = 0.04
        self.overlay = {}
        self.step_i = 0
        self.record = record
        self.frames: list[dict] = []
        self.attempt_spans: list[tuple] = []     # (first frame, end frame, sorted) per grasp attempt
        self.body_names = list(robot.body_names)
        self.link_meshes = _link_visual_meshes("/World/Robot", self.body_names)
        self._init_raycast_scene()
        eyes = [(1.10, 0.0, 0.60), (0.55, 0.75, 0.62), (0.55, -0.75, 0.62)]
        if args.cams == 6:      # plus views from behind and above, as a wrist camera would give
            eyes += [(0.05, 0.45, 0.85), (0.05, -0.45, 0.85), (0.53, 0.02, 1.10)]
        self.cams = DepthCams(eyes=eyes, target=(0.52, 0.0, 0.02))

    def _init_raycast_scene(self):
        """Meshes the depth cameras see: objects (labelled), robot links and static boxes (-1)."""
        import trimesh
        self.obj_meshes = []
        for o in self.objs:
            m = trimesh.load(str(shape_asset(o["shape"], ".obj")), process=False, force="mesh")
            self.obj_meshes.append((torch.tensor(np.asarray(m.vertices) * OBJECT_SCALE, dtype=torch.float32, device=DEV),
                                    torch.tensor(np.asarray(m.faces), dtype=torch.int64, device=DEV)))
        self.link_t = {i: (torch.tensor(v, device=DEV), torch.tensor(f, dtype=torch.int64, device=DEV))
                       for i, (v, f) in self.link_meshes.items()}
        self.static = [(torch.tensor(v, device=DEV), torch.tensor(f, dtype=torch.int64, device=DEV))
                       for v, f in static_boxes(self.bins)]

    def scene_mesh(self):
        verts, faces, labels, off = [], [], [], 0

        def add(v, f, lab):
            nonlocal off
            verts.append(v)
            faces.append(f + off)
            labels.append(torch.full((len(f),), lab, dtype=torch.int32, device=DEV))
            off += len(v)

        for i, (v, f) in enumerate(self.obj_meshes):
            st = self.rigid[i].data.root_state_w[0]
            add(v @ matrix_from_quat(st[3:7][None])[0].T + st[:3], f, i)
        bp, bq = self.robot.data.body_pos_w[0], self.robot.data.body_quat_w[0]
        for i, (v, f) in self.link_t.items():
            add(v @ matrix_from_quat(bq[i][None])[0].T + bp[i], f, -1)
        for v, f in self.static:
            add(v, f, -1)
        return torch.cat(verts), torch.cat(faces), torch.cat(labels)

    # -- low level
    def ee(self):
        p = self.robot.data.body_pos_w[0, self.ee_idx]
        q = self.robot.data.body_quat_w[0, self.ee_idx]
        return p.clone(), q.clone()

    def physics_step(self, q_arm):
        self.robot.set_joint_position_target(q_arm[None], joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(torch.full((1, 2), self.grip, device=DEV), joint_ids=self.grip_ids)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(DT)
        for r in self.rigid:
            r.update(DT)
        if self.record and self.step_i % 2 == 0:
            self.capture()
        self.step_i += 1

    def q_now(self):
        return self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy().astype(float)

    def solve_path(self, poses, seed=None):
        """IK for a chain of (pos, quat) waypoints, each seeded from the previous; None if any fails."""
        qs, q = [], self.q_now() if seed is None else seed
        for pos, quat in poses:
            q = self.pik.solve(torch.as_tensor(pos).tolist(), torch.as_tensor(quat).tolist(), [q, self.home_np])
            if q is None:
                return None
            qs.append(q)
        return qs

    def follow(self, qs, n, settle=10):
        """Joint-space moves through IK waypoints (n physics steps in total)."""
        for q in qs:
            self.move_joints(torch.tensor(q, device=DEV, dtype=torch.float32), max(2, n // len(qs)), smooth=len(qs) == 1)
        self.hold(settle)

    def line(self, p0, p1, quat, k=6):
        p0, p1 = torch.as_tensor(p0, device=DEV), torch.as_tensor(p1, device=DEV)
        return [(p0 + (p1 - p0) * (j / k), quat) for j in range(1, k + 1)]

    def hold(self, n):
        q = self._q_cmd if getattr(self, "_q_cmd", None) is not None else self.robot.data.joint_pos[0, self.arm_ids].clone()
        for _ in range(n):
            self.physics_step(q)

    def move_joints(self, q_goal, n, smooth=True):
        q0 = self._q_cmd if getattr(self, "_q_cmd", None) is not None else self.robot.data.joint_pos[0, self.arm_ids].clone()
        for k in range(n):
            t = (k + 1) / n
            self.physics_step(q0 + (q_goal - q0) * ((t * t * (3 - 2 * t)) if smooth else t))
        self._q_cmd = q_goal.clone()

    def set_grip(self, w, n):
        self.grip = w
        self.hold(n)

    def obj_pos(self, i):
        return self.rigid[i].data.root_pos_w[0].clone()

    # -- recording (rendered offline by scripts/render_sort.py)
    def capture(self, plan=None, repeat=1):
        if not self.record:
            return
        f = dict(body_pos=self.robot.data.body_pos_w[0].cpu().numpy(),
                 body_quat=self.robot.data.body_quat_w[0].cpu().numpy(),
                 obj=np.stack([r.data.root_state_w[0, :7].cpu().numpy() for r in self.rigid]),
                 overlay=json.dumps(self.overlay), plan=plan)
        self.frames.extend([f] * repeat)

    def show_plan(self, cloud, g, seconds=2.0):
        """Hold the frame showing what the policy saw (its 128 points) and chose."""
        plan = dict(obs_pts=g["obs_pts"].cpu().numpy().tolist(), cloud=voxel_down(cloud, 0.006).cpu().numpy().tolist(),
                    c1=g["c1"].tolist(), c2=g["c2"].tolist(), mid=g["mid"].tolist(),
                    approach=g["approach"].tolist())
        self.capture(plan, repeat=int(seconds * 30))

    def save(self, path, title):
        import trimesh
        arrays = dict(
            body_pos=np.stack([f["body_pos"] for f in self.frames]).astype(np.float32),
            body_quat=np.stack([f["body_quat"] for f in self.frames]).astype(np.float32),
            obj=np.stack([f["obj"] for f in self.frames]).astype(np.float32),
        )
        meta = dict(title=title, body_names=self.body_names, attempt_spans=self.attempt_spans,
                    overlays=[f["overlay"] for f in self.frames],
                    plans={str(k): f["plan"] for k, f in enumerate(self.frames) if f["plan"] is not None},
                    objects=[dict(label=o["label"], shape=o["shape"], color=list(o["color"]), category=o["category"])
                             for o in self.objs],
                    bins=[dict(name=b[0], center=list(b[2]), size=list(b[3]), color=list(b[4]),
                               wall_h=BIN_WALL_H, wall_t=BIN_WALL_T) for b in self.bins],
                    cams=[c[0].tolist() for c in self.cams.cams])
        # Plans repeat for whole holds; keep one copy per hold.
        dedup, last = {}, None
        for k, p in meta["plans"].items():
            key = json.dumps(p["c1"])
            if key != last:
                dedup[k] = p
                last = key
            else:
                dedup[k] = "same"
        meta["plans"] = dedup
        arrays["meta"] = np.array(json.dumps(meta))
        for i, (v, f) in self.link_meshes.items():
            arrays[f"link_{i}_v"], arrays[f"link_{i}_f"] = v, f
        for j, o in enumerate(self.objs):
            m = trimesh.load(str(shape_asset(o["shape"], ".obj")), process=False, force="mesh")
            arrays[f"obj_{j}_v"] = (np.asarray(m.vertices) * OBJECT_SCALE).astype(np.float32)
            arrays[f"obj_{j}_f"] = np.asarray(m.faces, dtype=np.int32)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **arrays)
        print(f"[sort] recorded {len(self.frames)} frames → {path}", flush=True)

    # -- task
    def perceive(self):
        return self.cams.capture(*self.scene_mesh())

    def bin_of(self, cat):
        return next(b for b in self.bins if b[1] == cat)

    def in_bin(self, i):
        p = self.obj_pos(i)
        _, _, (cx, cy), (sx, sy), _ = self.bin_of(self.objs[i]["category"])
        # Centre over the bin floor and not held up (tall objects stand with their centre above the wall).
        return abs(float(p[0]) - cx) < sx / 2 and abs(float(p[1]) - cy) < sy / 2 and float(p[2]) < 0.20

    def run(self, policy, gen, title):
        n = len(self.objs)
        home_q = self.robot.data.default_joint_pos[0, self.arm_ids].clone()
        self.home_np = home_q.cpu().numpy().astype(float)
        results = [dict(label=o["label"], shape=o["shape"], category=o["category"], case=o.get("case"),
                        tries=[], sorted=False) for o in self.objs]
        done = set()
        self.gn_next = {}
        self.overlay = dict(title=title)
        self.update_tally(results)
        self.hold(60)                                            # let objects settle
        for _ in range(n * self.max_tries):
            todo = [i for i in range(n) if i not in done and len(results[i]["tries"]) < self.max_tries]
            if not todo:
                break
            span_start = len(self.frames)
            self.overlay["phase"] = "Looking: 3 depth cameras, segment objects"
            self.move_joints(home_q, 40)
            clouds = self.perceive()
            seen = [i for i in todo if i in clouds and len(clouds[i]) >= 30]
            if not seen:
                break
            i = min(seen, key=lambda k: float(self.obj_pos(k)[:2].norm()))   # nearest first
            o, r = self.objs[i], results[i]
            k = len(r["tries"])
            self.overlay["status"] = (f"{o['label']}  ({o['category']})   try {k + 1}"
                                      + (" (sampled grasp)" if k else ""))
            up = torch.tensor([0.0, 0.0, 1.0], device=DEV)
            g, path = None, None
            if self.method == "graspnet":
                st = self.rigid[i].data.root_state_w[0]
                gn = graspnet_grasps(o["shape"], st[:3], st[3:7])[self.gn_next.get(i, 0):]
                pts = voxel_down(clouds[i], 0.004)
                pts = pts[torch.randint(0, len(pts), (NUM_PC_POINTS,), device=DEV, generator=gen)]
                cands = []
                for j, (c1, c2, a_w, width, psi) in enumerate(gn):
                    for c in hand_candidates(c1, c2, a_w, width, psi, False, pts):
                        c["gn_skip"] = j + 1
                        cands.append(c)
            else:
                cands = plan_grasp(policy, clouds[i], sample=k > 0, gen=gen)
            for cand in cands:
                path = self.solve_path([(cand["hand_pre"] + up * HOVER_M, cand["quat"]), (cand["hand_pre"], cand["quat"])]
                                       + self.line(cand["hand_pre"], cand["hand_final"], cand["quat"]))
                if path is not None:
                    g = cand
                    break
            if g is not None and "gn_skip" in g:          # next try starts after this grasp
                self.gn_next[i] = self.gn_next.get(i, 0) + g["gn_skip"]
            if g is None:      # no reachable execution of this grasp: counts as a failed try
                r["tries"].append(dict(lifted=False, sorted=False, reachable=False))
                print(f"[sort] {o['label']:9s} try {k + 1}: unreachable", flush=True)
                self.attempt_spans.append((span_start, len(self.frames), False))
                continue
            self.overlay["phase"] = (f"Policy: 2 contacts + approach (psi {g['psi_deg']:+.0f} deg) from point cloud"
                                     if self.method == "policy" else "Contact-GraspNet grasp (labels at true pose)")
            self.show_plan(clouds[i], g)
            z0 = float(self.obj_pos(i)[2])
            self.overlay["phase"] = "Arm: approach"
            self.follow(path[:1], 80)
            self.follow(path[1:2], 40)
            self.overlay["phase"] = "Arm: insert along the approach"
            self.follow(path[2:], 90)          # slow, so the fingers don't shove the object
            p, q = self.ee()
            pos_err, rot_err = compute_pose_error(p[None], q[None], g["hand_final"][None], g["quat"][None],
                                                  rot_error_type="axis_angle")
            reach = dict(pos_cm=100 * float(pos_err.norm()), rot_deg=float(rot_err.norm()) * 180 / math.pi)
            self.overlay["phase"] = "Close gripper"
            self.set_grip(max(0.003, g["width"] / 2 - SQUEEZE_M), 45)
            self.overlay["phase"] = "Lift"
            q_grasp = self.q_now()
            lift = self.solve_path(self.line(g["hand_final"], g["hand_final"] + up * LIFT_M, g["quat"], k=4), seed=q_grasp)
            self.follow(lift or [q_grasp], 50)
            lifted = float(self.obj_pos(i)[2]) - z0 >= LIFT_OK_M
            attempt = dict(lifted=lifted, reachable=True, psi_deg=g["psi_deg"], width_cm=100 * g["width"],
                           mirrored=g["mirrored"], outward=g["outward"], **reach)
            if lifted:
                name, _, (cx, cy), _, _ = self.bin_of(o["category"])
                self.overlay["phase"] = f"Carry to bin: {name}"
                carry = None
                for yaw in (0, 1, -1, 2, -2, 3, -3, 4):            # turn the hand about vertical if needed
                    qz = torch.tensor([math.cos(yaw * math.pi / 8), 0, 0, math.sin(yaw * math.pi / 8)], device=DEV)
                    from isaaclab.utils.math import quat_mul
                    qc = quat_mul(qz[None], g["quat"][None])[0]
                    lift_top = g["hand_final"] + up * max(LIFT_M, 0.30 - float(g["hand_final"][2]))
                    carry = self.solve_path([(lift_top, qc),
                                             (torch.tensor([cx, cy, float(lift_top[2])], device=DEV), qc),
                                             (torch.tensor([cx, cy, float(lift_top[2]) - 0.08], device=DEV), qc)])
                    if carry is not None:
                        break
                if carry is not None:
                    self.follow(carry[:1], 40)
                    self.follow(carry[1:2], 120)
                    self.follow(carry[2:], 40)
                self.overlay["phase"] = "Release"
                self.set_grip(0.04, 30)
                self.follow([self.q_now()], 1)
                p, q = self.ee()
                up_q = self.solve_path([(p + up * 0.10, q)])
                self.follow(up_q or [self.q_now()], 30, settle=5)
            else:
                self.overlay["phase"] = "Grasp failed: release and look again"
                self.set_grip(0.04, 20)
                p, q = self.ee()
                up_q = self.solve_path([(p + up * 0.10, q)])
                self.follow(up_q or [self.q_now()], 30, settle=5)
            r["tries"].append(attempt)
            r["sorted"] = self.in_bin(i)
            attempt["sorted"] = r["sorted"]
            if r["sorted"]:
                done.add(i)
            self.update_tally(results)
            ok = r["sorted"]
            self.overlay["card"] = (("SORTED" if ok else "MISSED"), (20, 150, 60) if ok else (200, 40, 40))
            self.capture(repeat=30)
            self.overlay.pop("card")
            self.attempt_spans.append((span_start, len(self.frames), bool(ok)))
            print(f"[sort] {o['label']:9s} try {k + 1}: lifted={lifted} sorted={r['sorted']} "
                  f"psi={g['psi_deg']:+.0f}{' (mirrored)' if g['mirrored'] else ''} reach={reach['pos_cm']:.1f}cm/{reach['rot_deg']:.0f}deg", flush=True)
        self.overlay["phase"] = "Done"
        self.overlay["status"] = f"Sorted {len(done)}/{n}"
        self.move_joints(home_q, 40)
        self.capture(repeat=60)
        return results

    def update_tally(self, results):
        rows = [(f"Sorted {sum(r['sorted'] for r in results)}/{len(results)}", (20, 20, 30))]
        for r in results:
            mark = "done" if r["sorted"] else (f"try {len(r['tries'])}" if r["tries"] else "")
            col = (20, 140, 60) if r["sorted"] else ((200, 40, 40) if r["tries"] else (90, 90, 100))
            rows.append((f"{r['label']:<10s} {mark}", col))
        self.overlay["tally"] = rows


def main():
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    policy = load_policy(args.checkpoint) if args.method == "policy" else None
    title = args.title or {"easy": "Fruit sorting", "hard": "Hard cases: tidy the table",
                           "showcase": "Unseen objects: tidy the workbench"}[args.scene]
    all_results = []
    bins, objs = scene_spec(args.scene, rng, jitter=False)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, device=DEV, physics_material=GraspPoseEnvCfg().sim.physics_material))
    robot, rigid = build_scene(bins, objs)
    sim.reset()
    runner = Runner(sim, robot, rigid, bins, objs, record=bool(args.record))
    runner.max_tries = args.max_tries
    runner.method = args.method
    for trial in range(args.trials):
        if trial > 0:
            _, objs_t = scene_spec(args.scene, rng, jitter=True)
            reset_scene(runner, objs_t)
        res = runner.run(policy, gen, title)
        n_ok = sum(r["sorted"] for r in res)
        first = sum(bool(r["tries"]) and r["tries"][0]["lifted"] for r in res)
        print(f"[sort] trial {trial}: sorted {n_ok}/{len(res)}, first-try lifts {first}/{len(res)}", flush=True)
        all_results.append(res)
        if runner.record:
            path = args.record if args.trials == 1 else args.record.replace(".npz", f"_t{trial}.npz")
            runner.save(path, title)
            runner.frames, runner.attempt_spans = [], []
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            json.dump(dict(scene=args.scene, checkpoint=args.checkpoint, seed=args.seed, max_tries=args.max_tries,
                           trials=all_results), open(args.json, "w"), indent=1)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(dict(scene=args.scene, checkpoint=args.checkpoint, seed=args.seed, max_tries=args.max_tries,
                       trials=all_results), open(args.json, "w"), indent=1)
    tot = sum(len(r) for r in all_results)
    print(f"[sort] TOTAL sorted {sum(x['sorted'] for r in all_results for x in r)}/{tot}, "
          f"first-try lifts {sum(bool(x['tries']) and x['tries'][0]['lifted'] for r in all_results for x in r)}/{tot}")


def reset_scene(runner, objs_t):
    """New positions (and, for fruit, labels/order) for the same spawned objects."""
    runner.robot.write_joint_state_to_sim(runner.robot.data.default_joint_pos, runner.robot.data.default_joint_vel)
    runner.grip = 0.04
    runner._q_cmd = None
    for i, (r, o_new) in enumerate(zip(runner.rigid, objs_t)):
        o = runner.objs[i]
        if args.scene in ("easy", "showcase"):   # same bodies; only the xy and yaw change
            o_new = dict(o, xy=o_new["xy"], quat=o_new["quat"])
            slot = [x["label"] for x in objs_t].index(o["label"])
            o_new["xy"] = objs_t[slot]["xy"]
        else:
            o_new = dict(o, xy=o_new["xy"])
        z = -mesh_min_z(o_new["shape"], o_new["quat"]) + 0.002
        pose = torch.tensor([[*o_new["xy"], z, *o_new["quat"]]], device=DEV, dtype=torch.float32)
        r.write_root_pose_to_sim(pose)
        r.write_root_velocity_to_sim(torch.zeros(1, 6, device=DEV))
        runner.objs[i] = o_new


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)      # app.close() hangs after this script; results are already written
