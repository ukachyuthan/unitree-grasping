#!/usr/bin/env python3
"""
Standalone script — no Isaac Lab needed.

Treats each fetched real YCB object (scripts/fetch_ycb.py) as a generative
shape family, mirroring generate_meshes.py's --n_per_family convention:

    instance 0      = vanilla real mesh, centered + size-normalized only
    instance 1..N-1 = spectral (Laplace-Beltrami) shape variants of that mesh

Spectral augmentation displaces vertices along their normals by a random
band-limited combination of the mesh's own low-frequency Laplacian
eigenmodes. Unlike isotropic per-vertex noise this varies GLOBAL shape
smoothly (no self-intersecting spikes — displacement is a linear combination
of smooth modes) while leaving HIGH-frequency local curvature — the
contact-patch geometry antipodal grasp quality depends on — close to the
real object's true surface.

Every instance (vanilla + deformed) is re-filtered through the same
antipodal-grasp computation used for procedural objects
(scripts/generate_meshes.py:compute_antipodal_grasps) — this is the
graspability gate, not a hand-tuned size heuristic. Output layout, file
naming, and manifest schema exactly match generate_meshes.py so
convert_to_usd.py and pretrain_grasp.py need no changes.

Usage:
    ./rl_unitree/bin/python scripts/fetch_ycb.py --out data/ycb_raw
    ./rl_unitree/bin/python scripts/generate_ycb_meshes.py \
        --raw data/ycb_raw --out data/objects/train --eval_out data/objects/eval \
        --n_per_family 8 --eval_frac 0.3 --seed 0
"""

from __future__ import annotations

import argparse, json, os, random, sys, time
import numpy as np
import trimesh
import trimesh.sample

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.generate_meshes import compute_antipodal_grasps, GRIPPER_MAX_W, PC_N
from scripts.generate_objects import MIN_DIM, MAX_DIM

# Must match wrappers/isaaclab/envs/grasp_pose_env_cfg.py OBJECT_SCALE — real
# meshes are pre-shrunk by this factor so the env's later uniform spawn scale
# brings them back into the same graspable size envelope as procedural objects.
OBJECT_SCALE = 1.5

MIN_GRASPS         = 5      # graspability gate: fewer valid antipodal pairs => discard instance
DECIMATE_MAX_FACES = 8000   # mesh complexity budget: eigendecomposition cost + USD collision
N_EIGENMODES        = 20    # low-frequency Laplacian eigenmodes used for deformation
DEFORM_AMPLITUDE    = 0.10  # max vertex displacement as a fraction of object extent


def find_mesh_file(root: str) -> str | None:
    """Locate the best mesh file under a fetched YCB object's raw directory."""
    preferred = ["textured.obj", "nontextured.ply", "poisson.ply"]
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            candidates.append(os.path.join(dirpath, fn))
    for pref in preferred:
        for c in candidates:
            if os.path.basename(c) == pref:
                return c
    for ext in (".obj", ".ply", ".stl"):
        for c in candidates:
            if c.lower().endswith(ext):
                return c
    return None


def load_and_prepare(mesh_path: str) -> trimesh.Trimesh | None:
    """Load a raw real-world mesh, repair it into a usable rigid-body volume."""
    try:
        mesh = trimesh.load(mesh_path, process=True, force="mesh")
    except Exception:
        return None
    if mesh is None or len(mesh.vertices) < 4:
        return None

    mesh.apply_translation(-mesh.centroid)

    if len(mesh.faces) > DECIMATE_MAX_FACES:
        try:
            mesh = mesh.simplify_quadric_decimation(DECIMATE_MAX_FACES)
        except Exception:
            pass

    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass
    if not mesh.is_watertight:
        # Real scans are often non-manifold; the convex hull is still a valid
        # graspable rigid-body volume, just loses concavities.
        try:
            mesh = trimesh.convex.convex_hull(mesh)
        except Exception:
            return None

    if len(mesh.vertices) < 4 or not mesh.is_volume:
        return None
    return mesh


def size_normalize(mesh: trimesh.Trimesh, rng: random.Random) -> trimesh.Trimesh | None:
    """Uniformly scale so the largest extent lands in [MIN_DIM, MAX_DIM] (procedural
    objects' own envelope), pre-compensating the env's later OBJECT_SCALE multiplier."""
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    largest = max(mesh.extents)
    if largest < 1e-9:
        return None
    target = rng.uniform(MIN_DIM, MAX_DIM) / OBJECT_SCALE
    mesh.apply_scale(target / largest)
    return mesh


# ── Spectral (Laplace-Beltrami) shape augmentation ─────────────────────────

def _cotangent_laplacian(mesh: trimesh.Trimesh):
    """Symmetric cotangent-weight graph Laplacian of a triangle mesh."""
    import scipy.sparse as sp

    V, F = mesh.vertices, mesh.faces
    n = len(V)

    def cot(a, b, c):
        u, v = b - a, c - a
        cos_ = (u * v).sum(-1)
        cross = np.linalg.norm(np.cross(u, v), axis=-1)
        return cos_ / np.clip(cross, 1e-9, None)

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cot0 = cot(v0, v1, v2)   # angle at vertex 0, opposite edge (1,2)
    cot1 = cot(v1, v2, v0)   # angle at vertex 1, opposite edge (2,0)
    cot2 = cot(v2, v0, v1)   # angle at vertex 2, opposite edge (0,1)

    rows, cols, vals = [], [], []
    for i_idx, j_idx, w in [(F[:, 1], F[:, 2], cot0), (F[:, 2], F[:, 0], cot1), (F[:, 0], F[:, 1], cot2)]:
        w = 0.5 * w
        rows += [i_idx, j_idx]
        cols += [j_idx, i_idx]
        vals += [w, w]

    W = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    D = sp.diags(np.asarray(W.sum(axis=1)).ravel())
    return D - W


def spectral_deform(
    mesh: trimesh.Trimesh, np_rng: np.random.Generator,
    n_modes: int = N_EIGENMODES, amplitude_frac: float = DEFORM_AMPLITUDE,
) -> trimesh.Trimesh | None:
    """Displace vertices along their normals by a random band-limited combination
    of the mesh's own low-frequency Laplacian eigenmodes (see module docstring)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = len(mesh.vertices)
    k = min(n_modes, n - 2)
    if k < 2:
        return None

    try:
        L = _cotangent_laplacian(mesh) + sp.eye(n) * 1e-8
        eigvals, eigvecs = spla.eigsh(L, k=k, sigma=0, which="LM")
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
    except Exception:
        return None

    weights = np_rng.normal(size=k) / np.sqrt(np.arange(1, k + 1))  # bias toward low modes
    field = eigvecs @ weights
    field = field / (np.abs(field).max() + 1e-9)

    scale = amplitude_frac * max(mesh.extents)
    new_verts = mesh.vertices + mesh.vertex_normals * (field[:, None] * scale)

    deformed = trimesh.Trimesh(new_verts, mesh.faces, process=False)
    if not deformed.is_watertight:
        trimesh.repair.fill_holes(deformed)
    if len(deformed.vertices) < 4 or not deformed.is_volume:
        return None
    return deformed


def anisotropic_jitter(mesh: trimesh.Trimesh, rng: random.Random, frac: float = 0.15) -> trimesh.Trimesh:
    """Coarse secondary variety: independent per-axis rescale."""
    mesh = mesh.copy()
    scale = [1.0 + rng.uniform(-frac, frac) for _ in range(3)]
    mesh.apply_scale(scale)
    return mesh


def make_family_instances(
    base_mesh: trimesh.Trimesh, n_per_family: int, rng: random.Random, np_rng: np.random.Generator,
) -> list[trimesh.Trimesh]:
    instance0 = size_normalize(base_mesh, rng)
    if instance0 is None:
        return []
    instances = [instance0]
    for _ in range(n_per_family - 1):
        deformed = spectral_deform(instance0, np_rng)
        if deformed is None:
            continue
        if rng.random() < 0.5:
            deformed = anisotropic_jitter(deformed, rng)
        instances.append(deformed)
    return instances


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw",          type=str, default="data/ycb_raw")
    p.add_argument("--out",          type=str, default="data/objects/train")
    p.add_argument("--eval_out",     type=str, default="data/objects/eval")
    p.add_argument("--n_per_family", type=int, default=8,
                   help="Instances per real object (1 vanilla + n-1 deformed variants)")
    p.add_argument("--eval_frac",    type=float, default=0.3,
                   help="Fraction of real object FAMILIES held out to eval (not instances)")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--grasp_candidates", type=int, default=600)
    p.add_argument("--min_grasps",   type=int, default=MIN_GRASPS)
    args = p.parse_args()

    rng    = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    raw_names = sorted(
        d for d in os.listdir(args.raw)
        if os.path.isdir(os.path.join(args.raw, d))
    )
    if not raw_names:
        sys.exit(f"No raw YCB meshes found in {args.raw}/ — run scripts/fetch_ycb.py first.")

    manifests = {}
    for out_dir in (args.out, args.eval_out):
        os.makedirs(out_dir, exist_ok=True)
        mpath = os.path.join(out_dir, "manifest.json")
        manifests[out_dir] = json.load(open(mpath)) if os.path.exists(mpath) else []

    t0 = time.time()
    n_families_kept = 0

    for fi, raw_name in enumerate(raw_names):
        family_name = f"ycb_{raw_name}"
        mesh_file = find_mesh_file(os.path.join(args.raw, raw_name))
        if mesh_file is None:
            print(f"  [{fi+1:3d}/{len(raw_names)}] {raw_name}: no mesh file found, skipping")
            continue

        base_mesh = load_and_prepare(mesh_file)
        if base_mesh is None:
            print(f"  [{fi+1:3d}/{len(raw_names)}] {raw_name}: failed to load/repair, skipping")
            continue

        instances = make_family_instances(base_mesh, args.n_per_family, rng, np_rng)
        if not instances:
            print(f"  [{fi+1:3d}/{len(raw_names)}] {raw_name}: normalization failed, skipping")
            continue

        split_out = args.eval_out if rng.random() < args.eval_frac else args.out
        family_dir = os.path.join(split_out, family_name)

        kept = 0
        for inst_idx, mesh in enumerate(instances):
            grasps = compute_antipodal_grasps(mesh, n_candidates=args.grasp_candidates, gripper_w=GRIPPER_MAX_W)
            if len(grasps) < args.min_grasps:
                continue

            os.makedirs(family_dir, exist_ok=True)
            obj_path   = os.path.join(family_dir, f"{inst_idx:03d}.obj")
            label_path = os.path.join(family_dir, f"{inst_idx:03d}_grasps.json")
            pc_path    = os.path.join(family_dir, f"{inst_idx:03d}_pc.npy")

            trimesh.exchange.export.export_mesh(mesh, obj_path)
            pc, _ = trimesh.sample.sample_surface(mesh, PC_N)
            np.save(pc_path, pc.astype(np.float32))

            label = dict(
                family=family_name,
                instance=inst_idx,
                source="ycb_real" if inst_idx == 0 else "ycb_spectral_deform",
                n_vertices=len(mesh.vertices),
                extents=mesh.extents.tolist(),
                n_grasps=len(grasps),
                grasps=grasps[:20],
            )
            with open(label_path, "w") as f:
                json.dump(label, f, indent=2)

            manifests[split_out].append(dict(
                family=family_name,
                instance=inst_idx,
                obj_path=os.path.relpath(obj_path, split_out),
                label_path=os.path.relpath(label_path, split_out),
                pc_path=os.path.relpath(pc_path, split_out),
                n_grasps=len(grasps),
                best_quality=grasps[0]["quality"] if grasps else 0.0,
            ))
            kept += 1

        elapsed = time.time() - t0
        split_label = "eval" if split_out == args.eval_out else "train"
        if kept > 0:
            n_families_kept += 1
            print(f"  [{fi+1:3d}/{len(raw_names)}] {family_name:35s} "
                  f"{kept}/{len(instances)} instances kept  ({split_label})  ({elapsed:.0f}s elapsed)")
        else:
            print(f"  [{fi+1:3d}/{len(raw_names)}] {family_name:35s} "
                  f"ungraspable, discarded  ({elapsed:.0f}s elapsed)")

    for out_dir, manifest in manifests.items():
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nDone. {n_families_kept}/{len(raw_names)} YCB families kept "
          f"in {time.time()-t0:.0f}s.")
    print(f"  train manifest: {os.path.join(args.out, 'manifest.json')} "
          f"({len(manifests[args.out])} total instances)")
    print(f"  eval  manifest: {os.path.join(args.eval_out, 'manifest.json')} "
          f"({len(manifests[args.eval_out])} total instances)")


if __name__ == "__main__":
    main()
