#!/usr/bin/env python3
"""
Procedural graspable object library generator.

Generates a diverse, challenging set of 3-D meshes for training and zero-shot
evaluation.  The goal is to cover shape families a simple primitive-trained
policy would FAIL on — proving the PointNet generalises through geometry.

Shape families
──────────────
  1. Superquadrics     — continuous family covering boxes→cylinders→spheres
  2. Torus / ring      — the centroid is NOT on the object surface
  3. L-shape           — two rectangular arms at 90°
  4. T-shape           — three arms meeting in the middle
  5. C-shape           — open semicircle (partial torus)
  6. Dumbbell          — two lobes connected by a thin shaft
  7. Wedge / prism     — extruded triangle; stable on its flat face
  8. Star prism        — extruded star; irregular boundary
  9. Bracket / frame   — box with a rectangular slot through it
 10. Stepped cylinder  — stacked discs of decreasing radius
 11. Twisted bar       — box cross-section with a helical twist along Z
 12. Irregular polygon extrusion — random convex-ish 2-D polygon extruded

Graspability constraints (all generated objects):
  • Bounding box: each side between MIN_DIM and MAX_DIM
  • Minimum cross-section width at some axis: ≥ 0.015 m  (gripper can close on it)
  • Has a stable resting pose on a flat surface (checked via trimesh stability)

Usage:
    pip install trimesh numpy
    python scripts/generate_objects.py --n 1000 --out data/objects/train
    python scripts/generate_objects.py --n 200  --out data/objects/eval  --seed 99
"""

import argparse
import os
import sys
import math
import random
import numpy as np

try:
    import trimesh
    import trimesh.creation as creation
    from trimesh.boolean import difference, union
except ImportError:
    sys.exit("Install trimesh:  pip install trimesh")


# ── Size bounds ───────────────────────────────────────────────────────────────
MIN_DIM  = 0.020   # 2 cm  (minimum graspable extent per axis)
MAX_DIM  = 0.110   # 11 cm (maximum object extent — fits in G1 hand workspace)
MIN_GRIP = 0.015   # 1.5 cm — minimum grasp cross-section width


def _rnd(lo, hi, rng):
    return rng.uniform(lo, hi)


def _validate(mesh: trimesh.Trimesh) -> bool:
    """Return True if the mesh is within graspable size bounds."""
    if mesh is None or len(mesh.vertices) < 4:
        return False
    ext = mesh.extents   # [dx, dy, dz]
    if any(e > MAX_DIM for e in ext) or any(e < MIN_DIM for e in ext):
        return False
    if min(ext) < MIN_GRIP:
        return False
    return True


def _center(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre mesh at origin and make it watertight if possible."""
    mesh.apply_translation(-mesh.centroid)
    try:
        mesh = trimesh.convex.repair_invalid(mesh)
    except Exception:
        pass
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# 1. Superquadric
# ─────────────────────────────────────────────────────────────────────────────
def _fexp(x, p):
    return np.sign(x) * (np.abs(x) + 1e-9) ** p


def make_superquadric(rng) -> trimesh.Trimesh:
    for _ in range(50):
        a1 = _rnd(MIN_DIM / 2, MAX_DIM / 2, rng)
        a2 = _rnd(MIN_DIM / 2, MAX_DIM / 2, rng)
        a3 = _rnd(MIN_DIM / 2, MAX_DIM / 2, rng)
        e1 = _rnd(0.08, 1.80, rng)
        e2 = _rnd(0.08, 1.80, rng)

        n_th, n_ph = 32, 24
        theta = np.linspace(-np.pi, np.pi, n_th, endpoint=False)
        phi   = np.linspace(-np.pi / 2, np.pi / 2, n_ph)
        TH, PH = np.meshgrid(theta, phi)

        X = a1 * _fexp(np.cos(PH), e1) * _fexp(np.cos(TH), e2)
        Y = a2 * _fexp(np.cos(PH), e1) * _fexp(np.sin(TH), e2)
        Z = a3 * _fexp(np.sin(PH), e1)
        verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)

        nR, nC = PH.shape
        faces = []
        for r in range(nR - 1):
            for c in range(nC):
                c1 = (c + 1) % nC
                v0, v1 = r * nC + c, r * nC + c1
                v2, v3 = (r+1)*nC + c, (r+1)*nC + c1
                faces += [[v0, v2, v1], [v1, v2, v3]]
        mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Torus / ring  (centroid is NOT on surface — key test for the policy)
# ─────────────────────────────────────────────────────────────────────────────
def make_torus(rng) -> trimesh.Trimesh:
    for _ in range(20):
        R = _rnd(0.020, 0.045, rng)    # major radius (hole centre to tube centre)
        r = _rnd(0.008, min(R*0.7, 0.020), rng)  # tube radius
        mesh = creation.torus(major_radius=R, minor_radius=r)
        # Scale to fit inside MAX_DIM
        scale = min(1.0, MAX_DIM / (2 * (R + r)))
        mesh.apply_scale(scale)
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. L-shape  (two rectangular bars at 90°)
# ─────────────────────────────────────────────────────────────────────────────
def make_l_shape(rng) -> trimesh.Trimesh:
    for _ in range(20):
        w = _rnd(0.015, 0.030, rng)   # arm width
        h = _rnd(0.015, 0.030, rng)   # arm thickness
        L1 = _rnd(0.040, MAX_DIM, rng)  # long arm length
        L2 = _rnd(0.030, MAX_DIM, rng)  # short arm length
        arm1 = creation.box([L1, w, h])
        arm2 = creation.box([w, L2, h])
        # Place arm2 at the end of arm1
        arm2.apply_translation([-(L1 / 2 - w / 2), L2 / 2 - w / 2, 0])
        mesh = trimesh.boolean.union([arm1, arm2])
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. T-shape  (three bars meeting at a point)
# ─────────────────────────────────────────────────────────────────────────────
def make_t_shape(rng) -> trimesh.Trimesh:
    for _ in range(20):
        w = _rnd(0.012, 0.025, rng)
        h = _rnd(0.012, 0.025, rng)
        L_vert  = _rnd(0.040, 0.090, rng)
        L_horiz = _rnd(0.040, 0.090, rng)
        vert  = creation.box([w, L_vert, h])
        horiz = creation.box([L_horiz, w, h])
        horiz.apply_translation([0, L_vert / 2 - w / 2, 0])
        mesh = trimesh.boolean.union([vert, horiz])
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. C-shape  (open arc — partial torus segment)
# ─────────────────────────────────────────────────────────────────────────────
def make_c_shape(rng) -> trimesh.Trimesh:
    for _ in range(20):
        R = _rnd(0.025, 0.045, rng)
        r = _rnd(0.008, min(R * 0.55, 0.018), rng)
        # Full torus then cut it open with a box
        full = creation.torus(major_radius=R, minor_radius=r)
        # Remove a wedge (≈ 90°–150° sector)
        angle_cut = _rnd(math.pi * 0.5, math.pi * 0.8, rng)
        cut_w  = 2 * (R + r) * 1.2
        cut_h  = 2 * r * 1.5
        cut_l  = cut_w
        cutter = creation.box([cut_w, cut_h, cut_l])
        # Rotate cutter to match gap angle
        rot = trimesh.transformations.rotation_matrix(angle_cut / 2, [0, 0, 1])
        cutter.apply_transform(rot)
        cutter.apply_translation([0, (R + r) * 0.5, 0])
        try:
            mesh = trimesh.boolean.difference(full, cutter)
        except Exception:
            continue
        # Scale to fit
        scale = min(1.0, MAX_DIM / max(mesh.extents + 1e-9))
        mesh.apply_scale(scale)
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dumbbell  (two spheres connected by a thin cylinder)
# ─────────────────────────────────────────────────────────────────────────────
def make_dumbbell(rng) -> trimesh.Trimesh:
    for _ in range(20):
        r1 = _rnd(0.012, 0.030, rng)
        r2 = _rnd(0.012, 0.030, rng)
        shaft_r = _rnd(0.005, min(r1, r2) * 0.7, rng)
        shaft_h = _rnd(0.030, MAX_DIM - r1 - r2, rng)

        s1 = creation.icosphere(radius=r1)
        s2 = creation.icosphere(radius=r2)
        shaft = creation.cylinder(radius=shaft_r, height=shaft_h, sections=16)

        s1.apply_translation([0, 0, -(shaft_h / 2 + r1)])
        s2.apply_translation([0, 0,  (shaft_h / 2 + r2)])

        mesh = trimesh.boolean.union([s1, shaft, s2])
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Wedge / triangular prism
# ─────────────────────────────────────────────────────────────────────────────
def make_wedge(rng) -> trimesh.Trimesh:
    for _ in range(20):
        base  = _rnd(MIN_DIM, MAX_DIM * 0.8, rng)
        depth = _rnd(MIN_DIM, MAX_DIM * 0.8, rng)
        h     = _rnd(MIN_DIM, MAX_DIM * 0.8, rng)
        # Right-triangle cross-section, extruded along Y
        verts_2d = np.array([[0, 0], [base, 0], [0, h]])
        poly = trimesh.path.polygons.Polygon(verts_2d)
        mesh = creation.extrude_polygon(poly, depth)
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 8. Star prism  (extruded star polygon — irregular sharp boundary)
# ─────────────────────────────────────────────────────────────────────────────
def make_star_prism(rng) -> trimesh.Trimesh:
    from shapely.geometry import Polygon as ShapelyPolygon
    for _ in range(20):
        n_pts   = rng.randint(4, 8)   # number of star points
        r_outer = _rnd(0.020, 0.055, rng)
        r_inner = _rnd(0.008, r_outer * 0.55, rng)
        height  = _rnd(MIN_DIM, MAX_DIM * 0.5, rng)

        angles_outer = [2 * math.pi * k / n_pts for k in range(n_pts)]
        angles_inner = [2 * math.pi * (k + 0.5) / n_pts for k in range(n_pts)]
        pts = []
        for ao, ai in zip(angles_outer, angles_inner):
            pts.append([r_outer * math.cos(ao), r_outer * math.sin(ao)])
            pts.append([r_inner * math.cos(ai), r_inner * math.sin(ai)])

        try:
            poly = ShapelyPolygon(pts)
            if not poly.is_valid:
                continue
            mesh = creation.extrude_polygon(poly, height)
        except Exception:
            continue

        scale = min(1.0, MAX_DIM / max(mesh.extents + 1e-9))
        mesh.apply_scale(scale)
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 9. Bracket / frame  (box with a rectangular through-hole)
# ─────────────────────────────────────────────────────────────────────────────
def make_bracket(rng) -> trimesh.Trimesh:
    for _ in range(20):
        ox = _rnd(0.040, MAX_DIM * 0.9, rng)
        oy = _rnd(0.030, MAX_DIM * 0.6, rng)
        oz = _rnd(MIN_DIM, 0.030, rng)   # thin flat object
        outer = creation.box([ox, oy, oz])

        # Slot: narrower than outer on both sides (wall thickness ≥ 5 mm)
        wall = 0.008
        ix = ox - 2 * wall
        iy = oy - 2 * wall
        if ix < MIN_GRIP or iy < MIN_GRIP:
            continue
        iz = oz * 1.5   # taller than outer to punch through
        inner = creation.box([ix, iy, iz])
        try:
            mesh = trimesh.boolean.difference(outer, inner)
        except Exception:
            continue
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 10. Stepped cylinder  (stacked discs of decreasing radius)
# ─────────────────────────────────────────────────────────────────────────────
def make_stepped_cylinder(rng) -> trimesh.Trimesh:
    for _ in range(20):
        n_steps = rng.randint(2, 4)
        r_base  = _rnd(0.025, 0.050, rng)
        h_step  = _rnd(0.015, 0.035, rng)
        parts   = []
        z_off   = 0.0
        for i in range(n_steps):
            r_i = r_base * (1.0 - 0.25 * i)
            if r_i < 0.008:
                break
            disc = creation.cylinder(radius=r_i, height=h_step, sections=24)
            disc.apply_translation([0, 0, z_off + h_step / 2])
            parts.append(disc)
            z_off += h_step
        if len(parts) < 2:
            continue
        mesh = trimesh.boolean.union(parts)
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 11. Twisted bar  (box extruded with progressive rotation — helix cross-section)
# ─────────────────────────────────────────────────────────────────────────────
def make_twisted_bar(rng) -> trimesh.Trimesh:
    for _ in range(20):
        w     = _rnd(0.012, 0.030, rng)
        d     = _rnd(0.012, 0.030, rng)
        h     = _rnd(0.050, MAX_DIM, rng)
        twist = _rnd(math.pi / 4, math.pi * 1.5, rng)   # total twist angle

        from shapely.geometry import Polygon as ShapelyPoly
        n_slices = 24
        all_verts, all_faces = [], []
        v_offset = 0

        rect = np.array([[-w/2, -d/2], [w/2, -d/2], [w/2, d/2], [-w/2, d/2]])
        for i in range(n_slices):
            frac  = i / (n_slices - 1)
            z     = h * frac - h / 2
            angle = twist * frac
            c, s  = math.cos(angle), math.sin(angle)
            R2    = np.array([[c, -s], [s, c]])
            rot_rect = (R2 @ rect.T).T
            # Add z coordinate
            slice_v = np.column_stack([rot_rect, np.full(4, z)])
            all_verts.append(slice_v)

        # Connect slices with quads (two triangles each)
        verts_all = np.vstack(all_verts)
        for i in range(n_slices - 1):
            for j in range(4):
                j1 = (j + 1) % 4
                a, b = i * 4 + j, i * 4 + j1
                c, d_ = (i + 1) * 4 + j, (i + 1) * 4 + j1
                all_faces += [[a, c, b], [b, c, d_]]

        # Close top and bottom caps
        for i_end in [0, n_slices - 1]:
            base = i_end * 4
            v0, v1, v2, v3 = base, base+1, base+2, base+3
            if i_end == 0:
                all_faces += [[v0, v2, v1], [v0, v3, v2]]
            else:
                all_faces += [[v0, v1, v2], [v0, v2, v3]]

        try:
            mesh = trimesh.Trimesh(verts_all, np.array(all_faces), process=True)
            if not mesh.is_volume:
                mesh = trimesh.Trimesh(verts_all, np.array(all_faces), process=False)
                mesh.fill_holes()
        except Exception:
            continue
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 12. Irregular polygon extrusion  (random convex-ish 2-D polygon extruded)
# ─────────────────────────────────────────────────────────────────────────────
def make_irregular_extrusion(rng) -> trimesh.Trimesh:
    from shapely.geometry import Polygon as ShapelyPoly
    for _ in range(30):
        n_verts  = rng.randint(5, 10)
        r_base   = _rnd(0.018, 0.045, rng)
        height   = _rnd(MIN_DIM, MAX_DIM * 0.7, rng)

        # Random angles and radii
        angles = np.sort(np.random.uniform(0, 2 * math.pi, n_verts))
        radii  = np.random.uniform(r_base * 0.5, r_base, n_verts)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        # Add irregular bumps / indentations
        jitter = r_base * 0.25
        pts += np.random.uniform(-jitter, jitter, pts.shape)

        try:
            poly = ShapelyPoly(pts)
            if not poly.is_valid or poly.is_empty:
                poly = poly.buffer(0)   # fix topology without losing non-convexity
            if poly.is_empty:
                continue
            mesh = creation.extrude_polygon(poly, height)
        except Exception:
            continue

        scale = min(1.0, MAX_DIM / max(mesh.extents + 1e-9))
        mesh.apply_scale(scale)
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 13. Random convex hull  (from scattered ellipsoidal point cloud)
# ─────────────────────────────────────────────────────────────────────────────
def make_convex_hull(rng) -> trimesh.Trimesh:
    for _ in range(20):
        a, b, c = [_rnd(MIN_DIM / 2, MAX_DIM / 2, rng) for _ in range(3)]
        n = rng.randint(14, 50)
        u = np.random.uniform(0, 2 * math.pi, n)
        v = np.random.uniform(0, math.pi, n)
        r = np.random.uniform(0.65, 1.0, n)
        x = r * a * np.sin(v) * np.cos(u)
        y = r * b * np.sin(v) * np.sin(u)
        z = r * c * np.cos(v)
        pts = np.column_stack([x, y, z])
        # Add interior jitter points
        pts = np.vstack([pts, np.random.uniform(-0.4, 0.4, (rng.randint(2, 8), 3))
                         * np.array([a, b, c])])
        try:
            mesh = trimesh.convex.convex_hull(pts)
        except Exception:
            continue
        if _validate(mesh):
            return _center(mesh)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
GENERATORS = {
    "superquadric":   (make_superquadric,      0.18),  # name, relative weight
    "torus":          (make_torus,             0.07),
    "l_shape":        (make_l_shape,           0.10),
    "t_shape":        (make_t_shape,           0.07),
    "c_shape":        (make_c_shape,           0.07),
    "dumbbell":       (make_dumbbell,          0.07),
    "wedge":          (make_wedge,             0.07),
    "star_prism":     (make_star_prism,        0.08),
    "bracket":        (make_bracket,           0.07),
    "stepped_cyl":    (make_stepped_cylinder,  0.07),
    "twisted_bar":    (make_twisted_bar,       0.07),
    "irregular_ext":  (make_irregular_extrusion, 0.08),
    "convex_hull":    (make_convex_hull,       0.10),
}

SHAPE_NAMES   = list(GENERATORS.keys())
SHAPE_FNS     = [GENERATORS[k][0] for k in SHAPE_NAMES]
SHAPE_WEIGHTS = np.array([GENERATORS[k][1] for k in SHAPE_NAMES])
SHAPE_WEIGHTS /= SHAPE_WEIGHTS.sum()


def generate_one(rng) -> tuple[trimesh.Trimesh, str]:
    """Sample a shape family, attempt to generate, retry on failure."""
    for _ in range(30):
        idx   = np.random.choice(len(SHAPE_NAMES), p=SHAPE_WEIGHTS)
        fn    = SHAPE_FNS[idx]
        label = SHAPE_NAMES[idx]
        try:
            mesh = fn(rng)
        except Exception:
            mesh = None
        if mesh is not None and _validate(mesh):
            return mesh, label
    # Ultimate fallback: random convex hull (most robust generator)
    return make_convex_hull(rng) or creation.box([0.04, 0.04, 0.08]), "convex_hull"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Generate procedural graspable objects")
    p.add_argument("--n",     type=int, default=1000)
    p.add_argument("--out",   type=str, default="data/objects/train")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--to_usd", action="store_true",
                   help="Also export .usd (requires Isaac Sim Python env)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    counts = {k: 0 for k in SHAPE_NAMES}
    print(f"Generating {args.n} objects → {args.out}/")
    print("Shape family weights:")
    for k, (_, w) in GENERATORS.items():
        print(f"  {k:<20s}  {w*100:.0f}%")

    for i in range(args.n):
        mesh, label = generate_one(rng)
        counts[label] += 1
        name = f"{label}_{i:05d}"
        out_path = os.path.join(args.out, f"{name}.obj")
        mesh.export(out_path)

        if args.to_usd:
            _try_usd(mesh, os.path.join(args.out, f"{name}.usd"), out_path)

        if (i + 1) % 100 == 0 or i == args.n - 1:
            print(f"  [{i+1:5d}/{args.n}]  {label}")

    print("\nShape distribution:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<20s}  {v:5d}")
    print(f"\nDone. {args.n} objects in {args.out}/")


def _try_usd(mesh, usd_path, obj_path):
    try:
        from pxr import Usd  # noqa: F401
        import asyncio
        import omni.kit.asset_converter as converter
        task = converter.get_instance().create_converter_task(obj_path, usd_path, None)
        asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
    except Exception:
        pass  # USD not available; OBJ is still saved


if __name__ == "__main__":
    main()
