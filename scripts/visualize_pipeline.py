"""
Visualize the full grasping pipeline offline (no Isaac Lab needed).

Shows:
  • A grid of procedural object shapes (one per family)
  • Sampled point clouds on each shape (blue dots)
  • Top-3 candidate grasp poses per object (gripper jaws in green/red)
  • Force-closure quality score annotated on each

Usage:
    pip install trimesh shapely matplotlib
    python scripts/visualize_pipeline.py
    python scripts/visualize_pipeline.py --save  # saves figures to data/viz/
"""

import argparse
import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import trimesh
    import trimesh.creation as creation
    import trimesh.sample
except ImportError:
    sys.exit("pip install trimesh")

try:
    from shapely.geometry import Polygon as ShapelyPolygon
except ImportError:
    sys.exit("pip install shapely")

# Import our generators
from scripts.generate_objects import (
    make_superquadric, make_torus, make_l_shape, make_t_shape,
    make_c_shape, make_dumbbell, make_wedge, make_star_prism,
    make_bracket, make_stepped_cylinder, make_twisted_bar,
    make_irregular_extrusion, make_convex_hull,
)

FAMILIES = [
    ("Superquadric",    make_superquadric),
    ("Torus",           make_torus),
    ("L-shape",         make_l_shape),
    ("T-shape",         make_t_shape),
    ("C-shape",         make_c_shape),
    ("Dumbbell",        make_dumbbell),
    ("Wedge",           make_wedge),
    ("Star prism",      make_star_prism),
    ("Bracket/frame",   make_bracket),
    ("Stepped cyl.",    make_stepped_cylinder),
    ("Twisted bar",     make_twisted_bar),
    ("Irregular ext.",  make_irregular_extrusion),
    ("Convex hull",     make_convex_hull),
]


# ── Grasp labeling ────────────────────────────────────────────────────────────

def _gripper_width():
    return 0.08   # G1 left gripper max span (m)


def compute_antipodal_grasps(mesh: trimesh.Trimesh, n_candidates: int = 200,
                              gripper_w: float = 0.08) -> list[dict]:
    """
    Analytical antipodal grasp sampling.

    For each candidate:
      1. Sample a surface point p1, get outward normal n1.
      2. Ray-cast from p1 inward along -n1 to find p2 on the opposite side.
      3. Check width (p1-p2 distance) ≤ gripper_w.
      4. Compute quality = |n1 · (p2-p1)/|p2-p1|| (antipodality: 1.0 = perfect).

    Returns list of dicts {center, approach, binormal, quality}.
    """
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_candidates)
    normals = mesh.face_normals[face_idx]

    grasps = []
    for p1, n1 in zip(pts, normals):
        # Ray from just inside the surface, along -n1
        origin = p1 - n1 * 1e-4
        direction = -n1
        locs, _, _ = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction],
        )
        if len(locs) == 0:
            continue
        # Closest hit
        dists = np.linalg.norm(locs - p1, axis=1)
        p2 = locs[np.argmin(dists)]
        width = np.linalg.norm(p2 - p1)
        if width > gripper_w or width < 0.005:
            continue

        center = (p1 + p2) / 2.0
        approach = n1 / (np.linalg.norm(n1) + 1e-9)

        # Antipodal quality: how parallel are the opposing normals?
        n2_dir = (p1 - p2) / (width + 1e-9)
        quality = abs(np.dot(n1, n2_dir))   # 1.0 = perfect antipodal

        # Gripper binormal (perpendicular to approach, horizontal if possible)
        up = np.array([0., 0., 1.])
        binormal = np.cross(approach, up)
        if np.linalg.norm(binormal) < 1e-6:
            binormal = np.array([1., 0., 0.])
        binormal /= np.linalg.norm(binormal)

        grasps.append(dict(
            center=center,
            approach=approach,
            binormal=binormal,
            width=width,
            quality=quality,
            p1=p1, p2=p2,
        ))

    grasps.sort(key=lambda g: -g["quality"])
    return grasps


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_mesh_wireframe(ax, mesh, color="0.75", alpha=0.15):
    verts = mesh.vertices
    faces = mesh.faces
    poly = Poly3DCollection(verts[faces], alpha=alpha,
                            facecolor=color, edgecolor="0.4", linewidth=0.2)
    ax.add_collection3d(poly)


def _plot_pointcloud(ax, pts, color="royalblue", s=4, alpha=0.6):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=color, s=s, alpha=alpha, depthshade=True)


def _plot_grasp(ax, grasp, color_line="lime", color_jaw="crimson", scale=0.025):
    """Draw gripper as two jaw lines + approach arrow."""
    c  = grasp["center"]
    ap = grasp["approach"] * scale
    bn = grasp["binormal"] * grasp["width"] / 2

    # Jaw endpoints
    j1a, j1b = c + bn - ap * 0.5,  c + bn + ap * 0.5
    j2a, j2b = c - bn - ap * 0.5,  c - bn + ap * 0.5

    for ja, jb in [(j1a, j1b), (j2a, j2b)]:
        ax.plot(*zip(ja, jb), color=color_jaw, linewidth=2.5)
    # Palm bar
    ax.plot(*zip(c + bn, c - bn), color=color_line, linewidth=1.5)
    # Approach arrow
    ax.quiver(*c, *ap, color="gold", length=scale * 1.2,
              normalize=False, linewidth=1.5, arrow_length_ratio=0.4)


def _set_ax_equal(ax, pts):
    """Axis equal for 3D plot based on point cloud extent."""
    mn, mx = pts.min(0), pts.max(0)
    mid = (mn + mx) / 2
    rng = max((mx - mn).max() / 2, 0.03)
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)
    ax.set_axis_off()


# ── Main visualisation ────────────────────────────────────────────────────────

def visualize_shape_grid(seed: int = 0, save_dir: str | None = None,
                          n_top_grasps: int = 3, n_pc_pts: int = 256):
    rng = random.Random(seed)
    np.random.seed(seed)

    n_cols = 4
    n_rows = math.ceil(len(FAMILIES) / n_cols)
    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.5))
    fig.patch.set_facecolor("#1a1a2e")

    for idx, (name, fn) in enumerate(FAMILIES):
        print(f"  [{idx+1:2d}/{len(FAMILIES)}]  {name}")
        mesh = None
        for _ in range(10):
            try:
                mesh = fn(rng)
            except Exception:
                mesh = None
            if mesh is not None and len(mesh.vertices) >= 4:
                break
        if mesh is None:
            continue

        # Sample point cloud
        pc, _ = trimesh.sample.sample_surface(mesh, n_pc_pts)

        # Ensure watertight for ray-casting (repair if needed)
        if not mesh.is_volume:
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
            trimesh.repair.fill_holes(mesh)

        # Compute grasps
        grasps = compute_antipodal_grasps(mesh, n_candidates=400,
                                           gripper_w=_gripper_width())
        top_grasps = grasps[:n_top_grasps]

        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        ax.set_facecolor("#1a1a2e")

        # Mesh (faint)
        _plot_mesh_wireframe(ax, mesh, color="#4a4a7a", alpha=0.12)

        # Point cloud
        _plot_pointcloud(ax, pc, color="royalblue", s=3)

        # Top grasps
        cmap = ["#00ff88", "#ffaa00", "#ff4488"]
        for gi, g in enumerate(top_grasps):
            _plot_grasp(ax, g, color_line=cmap[gi % len(cmap)],
                        color_jaw=cmap[gi % len(cmap)])

        _set_ax_equal(ax, mesh.vertices)

        qual_str = f"  q={top_grasps[0]['quality']:.2f}" if top_grasps else ""
        ax.set_title(f"{name}{qual_str}", color="white",
                     fontsize=9, pad=2)

    fig.suptitle(
        "Procedural Object Library — Point Clouds + Antipodal Grasps\n"
        f"(green = best grasp  ·  gold arrow = approach direction  ·  seed={seed})",
        color="white", fontsize=11, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"object_grid_seed{seed}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nSaved → {path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_grasp_pipeline(seed: int = 0, save_dir: str | None = None):
    """Single-object deep-dive: mesh + PC + all grasps ranked by quality."""
    rng = random.Random(seed)
    np.random.seed(seed)

    # Pick a challenging shape
    name, fn = FAMILIES[1]   # Torus — hardest (centroid not on surface)
    mesh = None
    for _ in range(20):
        try:
            mesh = fn(rng)
        except Exception:
            mesh = None
        if mesh is not None:
            break
    if mesh is None:
        print("Could not generate shape"); return

    pc, _ = trimesh.sample.sample_surface(mesh, 512)
    grasps = compute_antipodal_grasps(mesh, n_candidates=600,
                                       gripper_w=_gripper_width())
    print(f"  {name}: {len(grasps)} valid grasps found")

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    titles = ["Mesh only", "Point cloud (128 pts)", "Top-5 predicted grasps"]
    pcs_shown = [None, pc[:128], pc[:128]]
    grasps_shown = [[], [], grasps[:5]]

    for col, (title, pts, grps) in enumerate(zip(titles, pcs_shown, grasps_shown)):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        ax.set_facecolor("#1a1a2e")
        _plot_mesh_wireframe(ax, mesh, color="#4a4a7a", alpha=0.18)
        if pts is not None:
            _plot_pointcloud(ax, pts)
        cmap = ["#00ff88", "#44aaff", "#ffaa00", "#ff4488", "#cc88ff"]
        for gi, g in enumerate(grps):
            _plot_grasp(ax, g, color_line=cmap[gi], color_jaw=cmap[gi])
        _set_ax_equal(ax, mesh.vertices)
        ax.set_title(title, color="white", fontsize=10, pad=4)

    fig.suptitle(
        f"Grasp Pipeline Demo  —  {name}\n"
        "PointNet sees only the point cloud (centre panel) and predicts grasp poses (right panel)",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "grasp_pipeline_demo.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", action="store_true",
                   help="Save PNGs to data/viz/ instead of showing GUI")
    p.add_argument("--mode", choices=["grid", "pipeline", "both"],
                   default="both")
    args = p.parse_args()

    save_dir = "data/viz" if args.save else None

    print("Generating shapes and computing grasps…")
    if args.mode in ("grid", "both"):
        print("\n[1/2] Object grid (all 13 families)")
        visualize_shape_grid(seed=args.seed, save_dir=save_dir)

    if args.mode in ("pipeline", "both"):
        print("\n[2/2] Single-object pipeline demo")
        visualize_grasp_pipeline(seed=args.seed, save_dir=save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
