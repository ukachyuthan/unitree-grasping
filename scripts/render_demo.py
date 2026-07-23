"""
Render an offline demo MP4 of the grasping pipeline — no GUI, no Isaac Lab.

Shows (per object):
  1. Object on table with point cloud
  2. Predicted antipodal grasp pose highlighted
  3. Gripper (two box jaws) animating from pre-grasp → grasp → lift

Saves: data/viz/grasp_demo.mp4

Usage:
    ./rl_unitree/bin/python scripts/render_demo.py
    ./rl_unitree/bin/python scripts/render_demo.py --shape dumbbell --seed 1
"""

import argparse, os, sys, math, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import trimesh
import trimesh.creation as creation
import trimesh.sample

from scripts.generate_objects import (
    make_superquadric, make_torus, make_l_shape, make_t_shape,
    make_c_shape, make_dumbbell, make_wedge, make_star_prism,
    make_bracket, make_stepped_cylinder, make_twisted_bar,
    make_irregular_extrusion, make_convex_hull,
)

MAKERS = {
    "superquadric": make_superquadric, "torus": make_torus,
    "l_shape": make_l_shape, "t_shape": make_t_shape,
    "c_shape": make_c_shape, "dumbbell": make_dumbbell,
    "wedge": make_wedge, "star": make_star_prism,
    "bracket": make_bracket, "stepped": make_stepped_cylinder,
    "twisted": make_twisted_bar, "irregular": make_irregular_extrusion,
    "convex": make_convex_hull,
}

TABLE_Z  = 0.0
TABLE_HH = 0.005    # half-thickness of table slab drawn in scene
PC_N     = 128
BG       = "#0d0d1a"
FPS      = 30


# ── Grasp labeler (same as visualize_pipeline) ───────────────────────────────

def compute_best_grasp(mesh, gripper_w=0.08):
    pts, fidx = trimesh.sample.sample_surface(mesh, 500)
    normals = mesh.face_normals[fidx]
    best = None
    for p1, n1 in zip(pts, normals):
        origin = p1 - n1 * 1e-4
        locs, _, _ = mesh.ray.intersects_location([origin], [-n1])
        if len(locs) == 0:
            continue
        dists = np.linalg.norm(locs - p1, axis=1)
        p2 = locs[np.argmin(dists)]
        w = np.linalg.norm(p2 - p1)
        if w > gripper_w or w < 0.005:
            continue
        q = abs(np.dot(n1 / (np.linalg.norm(n1)+1e-9),
                       (p1 - p2) / (w + 1e-9)))
        if best is None or q > best["quality"]:
            up = np.array([0., 0., 1.])
            bn = np.cross(n1, up)
            if np.linalg.norm(bn) < 1e-6:
                bn = np.array([1., 0., 0.])
            bn /= np.linalg.norm(bn)
            best = dict(center=(p1+p2)/2, approach=n1/np.linalg.norm(n1),
                        binormal=bn, width=w, quality=q, p1=p1, p2=p2)
    return best


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _poly(ax, verts, faces, color, alpha):
    col = Poly3DCollection(verts[faces], alpha=alpha,
                           facecolor=color, edgecolor="none")
    ax.add_collection3d(col)


def _draw_table(ax):
    hw, hd, hh = 0.25, 0.18, TABLE_HH
    verts = np.array([[-hw,-hd,-hh],[ hw,-hd,-hh],[ hw, hd,-hh],[-hw, hd,-hh],
                      [-hw,-hd, hh],[ hw,-hd, hh],[ hw, hd, hh],[-hw, hd, hh]])
    faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    _poly(ax, verts, faces, "#5a4a3a", 0.6)


def _draw_mesh(ax, mesh, color="#3a3a6a", alpha=0.25):
    _poly(ax, mesh.vertices, mesh.faces, color, alpha)


def _draw_pc(ax, pts, t=1.0):
    alpha = min(t, 1.0)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
               c="royalblue", s=6, alpha=alpha*0.8, depthshade=True, zorder=3)


def _draw_gripper(ax, center, approach, binormal, width, t=1.0):
    """Draw two jaw boxes + palm bar at given pose."""
    sc  = 0.020          # jaw length
    ap  = approach * sc
    bn  = binormal * width / 2

    jaw_color = "#00ff88"
    alpha = min(t, 1.0)

    # Jaw lines
    for side in [bn, -bn]:
        ja = center + side - ap * 0.5
        jb = center + side + ap * 0.5
        ax.plot(*zip(ja, jb), color=jaw_color, linewidth=3.0, alpha=alpha, zorder=5)
    # Palm bar
    ax.plot(*zip(center+bn, center-bn), color="#ffffff", linewidth=1.5,
            alpha=alpha*0.7, zorder=5)
    # Approach arrow
    ax.quiver(*center, *(ap*1.5), color="gold", linewidth=2,
              alpha=alpha, arrow_length_ratio=0.35, zorder=6)


def _set_lims(ax, center, rng=0.14):
    ax.set_xlim(center[0]-rng, center[0]+rng)
    ax.set_ylim(center[1]-rng, center[1]+rng)
    ax.set_zlim(center[2]-rng, center[2]+rng)
    ax.set_axis_off()
    ax.set_facecolor(BG)


def _ease(t):
    return t * t * (3 - 2 * t)   # smoothstep


# ── Animation phases ──────────────────────────────────────────────────────────

PHASES = [
    ("Scanning object...",        0,   60),
    ("Computing point cloud...",  60,  100),
    ("Predicting grasp pose...",  100, 140),
    ("Executing grasp...",        140, 210),
    ("Lift!",                     210, 270),
]
TOTAL_FRAMES = 270


def make_frames(mesh, pc, grasp, shape_name):
    """Return (fig, update_fn) for matplotlib FuncAnimation."""
    center = np.array([0., 0., mesh.extents[2]/2 + TABLE_Z + TABLE_HH])
    gc = grasp["center"] + center   # grasp center in world

    # Pre-grasp: 6cm above grasp center along approach axis
    pre_grasp = gc + grasp["approach"] * 0.06

    fig = plt.figure(figsize=(10, 5.6), facecolor=BG)
    fig.patch.set_facecolor(BG)

    ax3d = fig.add_axes([0.02, 0.08, 0.58, 0.84], projection="3d")
    ax3d.set_facecolor(BG)

    ax_info = fig.add_axes([0.60, 0.08, 0.38, 0.84])
    ax_info.set_facecolor(BG)
    ax_info.set_axis_off()

    # Lifted object position (10 cm above table)
    lift_target = center + np.array([0., 0., 0.12])

    def update(frame):
        ax3d.cla(); ax_info.cla()
        ax3d.set_facecolor(BG)
        ax_info.set_facecolor(BG)
        ax_info.set_axis_off()

        # Determine phase
        phase_name = PHASES[0][0]
        for pname, p0, p1 in PHASES:
            if p0 <= frame < p1:
                phase_name = pname
                t_phase = (frame - p0) / (p1 - p0)
                break
        else:
            t_phase = 1.0

        # Rotating camera: full 360° over all frames
        elev = 18 + 5 * math.sin(frame * 2 * math.pi / TOTAL_FRAMES)
        azim = frame * 360 / TOTAL_FRAMES
        ax3d.view_init(elev=elev, azim=azim)

        # Table
        _draw_table(ax3d)

        # Phase 0 (0-60): object appears, rotates
        if frame < 60:
            obj_center = center
            _draw_mesh(ax3d, mesh.copy().apply_translation(obj_center), "#4a5aaa", 0.35)

        # Phase 1 (60-100): PC fades in
        elif frame < 100:
            obj_center = center
            _draw_mesh(ax3d, mesh.copy().apply_translation(obj_center), "#4a5aaa", 0.20)
            _draw_pc(ax3d, pc + obj_center, t=_ease(t_phase))

        # Phase 2 (100-140): grasp pose fades in
        elif frame < 140:
            obj_center = center
            _draw_mesh(ax3d, mesh.copy().apply_translation(obj_center), "#4a5aaa", 0.20)
            _draw_pc(ax3d, pc + obj_center)
            # gripper at pre-grasp, fades in
            _draw_gripper(ax3d, pre_grasp,
                          grasp["approach"], grasp["binormal"], grasp["width"],
                          t=_ease(t_phase))

        # Phase 3 (140-210): gripper moves from pre-grasp → grasp → closes
        elif frame < 210:
            obj_center = center
            _draw_mesh(ax3d, mesh.copy().apply_translation(obj_center), "#4a5aaa", 0.20)
            _draw_pc(ax3d, pc + obj_center)
            t_e = _ease(t_phase)
            gpos = pre_grasp + (gc - pre_grasp) * t_e
            # Width closes from full → 0 in last 30% of phase
            open_t = max(0., 1. - (t_phase - 0.7) / 0.3) if t_phase > 0.7 else 1.
            w = grasp["width"] * open_t
            _draw_gripper(ax3d, gpos, grasp["approach"], grasp["binormal"], w)

        # Phase 4 (210-270): lift
        else:
            t_e = _ease(t_phase)
            obj_center = center + (lift_target - center) * t_e
            _draw_mesh(ax3d, mesh.copy().apply_translation(obj_center), "#55dd88", 0.35)
            gpos = gc + (lift_target - center) * t_e
            _draw_gripper(ax3d, gpos, grasp["approach"], grasp["binormal"],
                          grasp["width"] * 0.05)

        _set_lims(ax3d, center)

        # Info panel
        lines = [
            ("Shape",        shape_name),
            ("PC points",    f"{PC_N}"),
            ("Grasp quality",f"{grasp['quality']:.3f}"),
            ("Gripper width",f"{grasp['width']*100:.1f} cm"),
            ("Phase",        phase_name),
            ("Frame",        f"{frame}/{TOTAL_FRAMES}"),
        ]
        y = 0.90
        ax_info.text(0.05, y, "Grasp Pipeline", color="white",
                     fontsize=13, fontweight="bold",
                     transform=ax_info.transAxes)
        y -= 0.10
        for key, val in lines:
            ax_info.text(0.05, y, key, color="#aaaaaa", fontsize=9,
                         transform=ax_info.transAxes)
            ax_info.text(0.55, y, val, color="white", fontsize=9,
                         transform=ax_info.transAxes)
            y -= 0.09

        # Progress bar (xmin/xmax are in axes fraction natively for axhline)
        prog = frame / TOTAL_FRAMES
        ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
        ax_info.barh(0.04, 0.90, left=0.05, height=0.025, color="#333355")
        ax_info.barh(0.04, 0.90 * prog, left=0.05, height=0.025, color="#00ff88")

        fig.patch.set_facecolor(BG)
        return []

    return fig, update


def render_mp4(shape_name: str, seed: int, out_path: str):
    rng = random.Random(seed)
    np.random.seed(seed)

    maker = MAKERS.get(shape_name)
    if maker is None:
        print(f"Unknown shape '{shape_name}'. Options: {list(MAKERS)}")
        return

    print(f"  Generating {shape_name}...")
    mesh = None
    for _ in range(20):
        try:
            mesh = maker(rng)
        except Exception:
            mesh = None
        if mesh is not None:
            break
    if mesh is None:
        print("  Failed to generate mesh"); return

    if not mesh.is_volume:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
        trimesh.repair.fill_holes(mesh)

    pc, _ = trimesh.sample.sample_surface(mesh, PC_N)

    print("  Computing grasp...")
    grasp = compute_best_grasp(mesh)
    if grasp is None:
        print("  No valid grasp found"); return
    print(f"  Best grasp quality={grasp['quality']:.3f}  width={grasp['width']*100:.1f}cm")

    print(f"  Rendering {TOTAL_FRAMES} frames → {out_path}")
    fig, update_fn = make_frames(mesh, pc, grasp, shape_name)

    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    ani = animation.FuncAnimation(fig, update_fn, frames=TOTAL_FRAMES,
                                   interval=1000//FPS, blit=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, writer=writer, dpi=120,
             savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="dumbbell",
                   choices=list(MAKERS), help="Shape family to demo")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--out",   default="data/viz/grasp_demo.mp4")
    p.add_argument("--all",   action="store_true",
                   help="Render one video per shape family")
    args = p.parse_args()

    if args.all:
        for name in MAKERS:
            out = f"data/viz/grasp_demo_{name}.mp4"
            print(f"\n[{name}]")
            render_mp4(name, args.seed, out)
    else:
        render_mp4(args.shape, args.seed, args.out)


if __name__ == "__main__":
    main()
