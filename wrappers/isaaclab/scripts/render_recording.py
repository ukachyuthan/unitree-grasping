#!/usr/bin/env python3
"""
Render a play_grasp_pose.py --record file to MP4 — no Isaac Sim / RTX needed.

Draws the real recorded rollout: Franka visual meshes at their simulated link
poses, the object mesh at its simulated pose, the policy's contact targets
(orange / cyan), and the lift target height (translucent green plate). The
overlay shows episode, object, phase, and lift progress; each episode ends on a
SUCCESS / FAILED card (success = object reached lift_target_m, as in training).

Usage (from wrappers/isaaclab):
    python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 10 \\
        --video_episodes 10 --checkpoint <ckpt.pt> --record ../../data/viz/rollout.npz
    python scripts/render_recording.py ../../data/viz/rollout.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("recording", help=".npz written by play_grasp_pose.py --record")
parser.add_argument("--out", default=None, help="MP4 path (default: recording path with .mp4)")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--stride", type=int, default=2,
                    help="Physics steps per video frame (2 at 60 Hz physics, 30 fps = real time).")
parser.add_argument("--cam_offset", type=float, nargs=3, default=[0.40, -0.34, 0.26],
                    help="Camera position relative to the object's spawn point (follows each episode).")
parser.add_argument("--cam_eye", type=float, nargs=3, default=None,
                    help="Fixed camera position instead of following the object (needs --cam_target).")
parser.add_argument("--cam_target", type=float, nargs=3, default=None)
parser.add_argument("--fov", type=float, default=45.0)
parser.add_argument("--no_grasp_overlay", dest="show_grasp", action="store_false",
                    help="Hide the on-image markers for the predicted grasp contacts.")
parser.add_argument("--hold_s", type=float, default=1.2, help="Seconds to hold the result card.")
args = parser.parse_args()


def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_mat(pos, quat_wxyz) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(quat_wxyz)
    m[:3, 3] = pos
    return m


def tri_mesh(v, f) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)),
                                  o3d.utility.Vector3iVector(f.astype(np.int32)))
    m.compute_vertex_normals()
    return m


def material(rgba, shader="defaultLit", roughness=0.6):
    mat = rendering.MaterialRecord()
    mat.shader = shader
    mat.base_color = list(rgba)
    mat.base_roughness = roughness
    return mat


def project(r, pts_w: np.ndarray, width: int, height: int) -> np.ndarray:
    """World points (n, 3) → pixel coords (n, 2) with the renderer's current camera."""
    cam = r.scene.camera
    pv = np.asarray(cam.get_projection_matrix()) @ np.asarray(cam.get_view_matrix())
    clip = np.c_[pts_w, np.ones(len(pts_w))] @ pv.T
    ndc = clip[:, :3] / clip[:, 3:4]
    return np.stack([(ndc[:, 0] + 1) / 2 * width, (1 - ndc[:, 1]) / 2 * height], axis=-1)


def draw_grasp(d: ImageDraw.ImageDraw, px: np.ndarray, colors, fnt):
    """Always-on-top markers for the predicted grasp: a ring + dot per contact,
    the jaw axis between two contacts, and the grasp midpoint."""
    if len(px) == 2:
        d.line([tuple(px[0]), tuple(px[1])], fill=(255, 255, 255), width=5)
        d.line([tuple(px[0]), tuple(px[1])], fill=(230, 40, 160), width=2)
        mid = px.mean(axis=0)
        d.line([mid[0] - 7, mid[1], mid[0] + 7, mid[1]], fill=(230, 40, 160), width=2)
        d.line([mid[0], mid[1] - 7, mid[0], mid[1] + 7], fill=(230, 40, 160), width=2)
    labels = ["c1", "c2"] if len(px) == 2 else ["grasp"]
    for k, (x, y) in enumerate(px):
        rgb = tuple(int(255 * v) for v in colors[k % 2][:3])
        d.ellipse([x - 13, y - 13, x + 13, y + 13], outline=(255, 255, 255), width=5)
        d.ellipse([x - 13, y - 13, x + 13, y + 13], outline=rgb, width=3)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=rgb)
        d.text((x + 16, y - 30), labels[k], font=fnt, fill=rgb, stroke_width=3, stroke_fill=(255, 255, 255))


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    rec = np.load(args.recording)
    meta = json.loads(str(rec["meta"]))
    body_pos, body_quat = rec["body_pos"], rec["body_quat"]
    obj_pose, step_ep, step_exec = rec["obj_pose"], rec["step_episode"], rec["step_exec"]
    names = meta["body_names"]
    episodes = meta["episodes"]
    lift_target = meta["lift_target_m"]

    # Phase lookup: physics step within episode → phase name.
    bounds, acc = [], 0
    for name, n in meta["phases"]:
        acc += n
        bounds.append((acc, name))

    def phase_of(s):
        return next((name for end, name in bounds if s < end), bounds[-1][1])

    r = rendering.OffscreenRenderer(args.width, args.height)
    scene = r.scene
    scene.set_background([0.93, 0.93, 0.95, 1.0])
    scene.set_lighting(scene.LightingProfile.SOFT_SHADOWS, np.array([0.4, 0.6, -1.0]))

    # Static: floor and table.
    t = meta["table"]
    tc, ts = np.array(t["center"]), np.array(t["size"])
    table = o3d.geometry.TriangleMesh.create_box(*ts)
    table.translate(tc - ts / 2)
    table.compute_vertex_normals()
    scene.add_geometry("table", table, material([0.62, 0.52, 0.42, 1.0]))
    floor_z = min(0.0, float(tc[2] - ts[2] / 2))
    floor = o3d.geometry.TriangleMesh.create_box(4.0, 4.0, 0.01)
    floor.translate([-1.5, -2.0, floor_z - 0.01])
    floor.compute_vertex_normals()
    scene.add_geometry("floor", floor, material([0.82, 0.82, 0.84, 1.0]))

    # Robot links (meshes stored in each link's frame).
    links = []
    for i, name in enumerate(names):
        if f"link_{i}_v" not in rec:
            continue
        finger = "finger" in name or "hand" in name
        color = [0.18, 0.18, 0.2, 1.0] if finger else [0.92, 0.92, 0.94, 1.0]
        scene.add_geometry(f"link{i}", tri_mesh(rec[f"link_{i}_v"], rec[f"link_{i}_f"]),
                           material(color, roughness=0.35))
        links.append(i)

    shape_meshes = {j: tri_mesh(rec[f"shape_{j}_v"], rec[f"shape_{j}_f"])
                    for j in range(len(meta["shapes"]))}
    marker_colors = [[1.0, 0.55, 0.0, 1.0], [0.0, 0.8, 0.9, 1.0]]

    f_big, f_small = font(54), font(26)
    fixed_cam = args.cam_eye is not None and args.cam_target is not None
    if fixed_cam:
        r.setup_camera(args.fov, np.array(args.cam_target), np.array(args.cam_eye), [0, 0, 1])

    out = args.out or os.path.splitext(args.recording)[0] + ".mp4"
    import imageio.v2 as imageio
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", pixelformat="yuv420p", quality=8)

    n_frames = 0
    for ep_idx in range(len(episodes)):
        ep = episodes[ep_idx]
        steps = np.nonzero(step_ep == ep_idx)[0]
        if len(steps) == 0:
            continue
        # Swap in this episode's object, contact markers, and lift-target plate.
        for name in ("object", "c0", "c1", "goal"):
            if scene.has_geometry(name):
                scene.remove_geometry(name)
        j = meta["shapes"].index(ep["shape"])
        scene.add_geometry("object", shape_meshes[j], material([0.95, 0.45, 0.12, 1.0], roughness=0.5))
        for k, _ in enumerate(ep["contacts_local"]):
            s = o3d.geometry.TriangleMesh.create_sphere(0.007)
            s.compute_vertex_normals()
            scene.add_geometry(f"c{k}", s, material(marker_colors[k % 2], roughness=0.3))
        x0, y0 = obj_pose[steps[0], :2]
        goal_z = ep["spawn_z"] + lift_target
        plate = o3d.geometry.TriangleMesh.create_box(0.12, 0.12, 0.001)
        plate.translate([x0 - 0.06, y0 - 0.06, goal_z - 0.0005])
        plate.compute_vertex_normals()
        scene.add_geometry("goal", plate, material([0.1, 0.8, 0.3, 0.15], shader="defaultLitTransparency"))
        if not fixed_cam:
            target = np.array([x0, y0, ep["spawn_z"] + 0.03])
            r.setup_camera(args.fov, target, target + np.array(args.cam_offset), [0, 0, 1])

        peak = 0.0
        frame_steps = list(steps[::args.stride]) + ([steps[-1]] if (len(steps) - 1) % args.stride else [])
        for si in frame_steps:
            for i in links:
                scene.set_geometry_transform(f"link{i}", pose_mat(body_pos[si, i], body_quat[si, i]))
            T = pose_mat(obj_pose[si, :3], obj_pose[si, 3:7])
            scene.set_geometry_transform("object", T)
            for k, c in enumerate(ep["contacts_local"]):
                scene.set_geometry_transform(f"c{k}", T @ pose_mat(c, [1, 0, 0, 0]))
            img = Image.fromarray(np.asarray(r.render_to_image()))
            d = ImageDraw.Draw(img)
            if args.show_grasp:
                pts = np.array([(T @ np.r_[c, 1.0])[:3] for c in ep["contacts_local"]])
                draw_grasp(d, project(r, pts, args.width, args.height), marker_colors, f_small)
            gain = max(0.0, float(obj_pose[si, 2]) - ep["spawn_z"])
            peak = max(peak, gain)
            lines = [f"Episode {ep_idx + 1}/{len(episodes)}   object: {ep['shape']}",
                     f"Phase: {phase_of(int(step_exec[si]))}",
                     f"Object lift: {100 * gain:4.1f} cm   peak {100 * peak:4.1f}   (target {100 * lift_target:.0f} cm)"]
            if args.show_grasp:
                lines.append("Predicted grasp: c1 / c2 contacts, jaw axis + midpoint")
            d.rectangle([12, 12, 840, 22 + 34 * len(lines)], fill=(255, 255, 255, 200))
            for li, text in enumerate(lines):
                d.text((24, 20 + 34 * li), text, font=f_small, fill=(20, 20, 30))
            writer.append_data(np.asarray(img))
            n_frames += 1

        # Result card: success = reached lift target (contact-gated), same as training.
        ok = ep.get("success", False)
        card = img.copy()
        d = ImageDraw.Draw(card)
        label = "SUCCESS" if ok else "FAILED"
        sub = f"lift {100 * ep.get('lift_frac', 0.0) * lift_target:.1f} / {100 * lift_target:.0f} cm"
        color = (20, 150, 60) if ok else (200, 40, 40)
        w = d.textlength(label, font=f_big)
        cx = args.width / 2
        d.rectangle([cx - w / 2 - 30, args.height - 170, cx + w / 2 + 30, args.height - 40], fill=(255, 255, 255))
        d.text((cx - w / 2, args.height - 160), label, font=f_big, fill=color)
        sw = d.textlength(sub, font=f_small)
        d.text((cx - sw / 2, args.height - 88), sub, font=f_small, fill=(40, 40, 40))
        for _ in range(int(args.hold_s * args.fps)):
            writer.append_data(np.asarray(card))
            n_frames += 1
        print(f"  episode {ep_idx + 1}: {ep['shape']:28s} {label:8s} lift_frac={ep.get('lift_frac', 0):.2f}")

    writer.close()
    n_ok = sum(e.get("success", False) for e in episodes)
    print(f"[render] {n_frames} frames → {out}   ({n_ok}/{len(episodes)} episodes reached the lift target)")


if __name__ == "__main__":
    main()
