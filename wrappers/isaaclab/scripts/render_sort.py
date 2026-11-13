#!/usr/bin/env python3
"""
Render a sort_demo.py --record file to MP4 with open3d (no Isaac Sim / RTX needed).

Draws the recorded robot links and objects at their simulated poses, the table,
tray and bins, the text overlay (scene, current object, phase, tally), and on
planning holds what the policy saw: the object's observed cloud (grey), the 128
points it was given (blue), its two contacts and the approach (green arrow).

Usage (from wrappers/isaaclab):
    python scripts/render_sort.py ../../data/sort/easy.npz [--out easy.mp4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw

_argv, sys.argv = sys.argv, sys.argv[:2]       # render_recording parses its own CLI on import
from render_recording import font, material, pose_mat, project, render_frame, tri_mesh
sys.argv = _argv

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("recording")
parser.add_argument("--out", default=None)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--cam_eye", type=float, nargs=3, default=[1.40, -0.62, 1.05])
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.30, 0.0, 0.18])
parser.add_argument("--fov", type=float, default=52.0)
parser.add_argument("--only_success", action="store_true",
                    help="Cut failed grasp attempts and drop try counts from the overlay (demo cut)")
parser.add_argument("--preview", type=int, nargs="*", default=None, help="Only save these frames as PNGs")
parser.add_argument("--clean", action="store_true",
                    help="No text overlays (title, tally, legend, bin names, cards); keeps the plan drawing (paper figures)")
args, _ = parser.parse_known_args()


def box(size, center):
    m = o3d.geometry.TriangleMesh.create_box(*size)
    m.translate(np.asarray(center) - np.asarray(size) / 2)
    m.compute_vertex_normals()
    return m


def main():
    rec = np.load(args.recording)
    meta = json.loads(str(rec["meta"]))
    bp, bq, obj = rec["body_pos"], rec["body_quat"], rec["obj"]
    W, H = args.width, args.height

    r = rendering.OffscreenRenderer(W, H)
    sc = r.scene
    sc.set_background([0.95, 0.95, 0.96, 1.0])
    sc.set_lighting(sc.LightingProfile.SOFT_SHADOWS, np.array([0.3, 0.5, -1.0]))
    sc.add_geometry("floor", box((4, 4, 0.01), (0.5, 0, -0.755)), material([0.84, 0.84, 0.86, 1]))
    sc.add_geometry("table", box((1.0, 1.4, 0.75), (0.45, 0, -0.375)), material([0.66, 0.54, 0.42, 1]))
    xy = obj[0, :, :2]
    lo, hi = xy.min(0) - [0.12, 0.13], xy.max(0) + [0.12, 0.13]
    sc.add_geometry("tray", box((*(hi - lo), 0.002), (*((hi + lo) / 2), 0.001)), material([0.95, 0.94, 0.90, 1]))
    for k, b in enumerate(meta["bins"]):
        (cx, cy), (sx, sy), h, t = b["center"], b["size"], b["wall_h"], b["wall_t"]
        col = material([*b["color"], 1.0], roughness=0.6)
        sc.add_geometry(f"bin{k}f", box((sx, sy, 0.002), (cx, cy, 0.001)), material([*b["color"], 1.0], roughness=0.9))
        for j, (s, c) in enumerate([((sx + 2 * t, t, h), (cx, cy + sy / 2 + t / 2, h / 2)),
                                     ((sx + 2 * t, t, h), (cx, cy - sy / 2 - t / 2, h / 2)),
                                     ((t, sy, h), (cx + sx / 2 + t / 2, cy, h / 2)),
                                     ((t, sy, h), (cx - sx / 2 - t / 2, cy, h / 2))]):
            sc.add_geometry(f"bin{k}w{j}", box(s, c), col)
    links = []
    for i, name in enumerate(meta["body_names"]):
        if f"link_{i}_v" in rec:
            dark = "finger" in name or "hand" in name
            sc.add_geometry(f"link{i}", tri_mesh(rec[f"link_{i}_v"], rec[f"link_{i}_f"]),
                            material([0.2, 0.2, 0.22, 1] if dark else [0.93, 0.93, 0.95, 1], roughness=0.35))
            links.append(i)
    for j, o in enumerate(meta["objects"]):
        sc.add_geometry(f"obj{j}", tri_mesh(rec[f"obj_{j}_v"], rec[f"obj_{j}_f"]),
                        material([*o["color"], 1.0], roughness=0.5))
    r.setup_camera(args.fov, np.array(args.cam_target), np.array(args.cam_eye), [0, 0, 1])

    f_big, f, f_card = font(30), font(23), font(52)
    out = args.out or os.path.splitext(args.recording)[0] + ".mp4"
    import imageio.v2 as imageio
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", pixelformat="yuv420p", quality=8)
    plans, plan = meta["plans"], None
    last, last_key = None, None
    frames = args.preview if args.preview is not None else range(len(bp))
    if args.only_success:
        cut = {k for a, b, ok in meta.get("attempt_spans", []) if not ok for k in range(a, b)}
        frames = [k for k in frames if k not in cut]
    if args.preview is not None:   # plans are keyed by frame; resolve "same" from earlier keys
        plans = {str(k): next((plans[str(j)] for j in range(k, -1, -1) if plans.get(str(j)) not in (None, "same")), None)
                 if str(k) in plans else None for k in frames}
    for k in frames:
        p = plans.get(str(k))
        plan = None if p is None else (plan if p == "same" else p)
        key = (bp[k].tobytes(), obj[k].tobytes())
        if key != last_key or last is None:      # holds repeat the same state: render once
            for i in links:
                sc.set_geometry_transform(f"link{i}", pose_mat(bp[k, i], bq[k, i]))
            for j in range(len(meta["objects"])):
                sc.set_geometry_transform(f"obj{j}", pose_mat(obj[k, j, :3], obj[k, j, 3:7]))
            last = render_frame(r, last)
            last_key = key
        img = last.copy()
        d = ImageDraw.Draw(img, "RGBA")
        if plan:
            for x, y in project(r, np.asarray(plan["cloud"]), W, H):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(120, 120, 130, 150))
            for x, y in project(r, np.asarray(plan["obs_pts"]), W, H):
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(30, 110, 255, 235))
            mid, a = np.asarray(plan["mid"]), np.asarray(plan["approach"])
            (x1, y1), (x2, y2), (ax, ay), (mx, my) = project(
                r, np.stack([plan["c1"], plan["c2"], mid - a * 0.12, mid]), W, H)
            d.line([(ax, ay), (mx, my)], fill=(255, 255, 255, 255), width=8)
            d.line([(ax, ay), (mx, my)], fill=(30, 170, 80, 255), width=5)
            d.line([(x1, y1), (x2, y2)], fill=(230, 40, 160, 255), width=3)
            for (x, y), col in (((x1, y1), (255, 140, 0)), ((x2, y2), (0, 190, 230))):
                d.ellipse([x - 11, y - 11, x + 11, y + 11], outline=(255, 255, 255), width=6)
                d.ellipse([x - 11, y - 11, x + 11, y + 11], outline=col, width=3)
        if args.clean:
            if args.preview is not None:
                img.save(os.path.splitext(out)[0] + f"_{k}.png")
                continue
            writer.append_data(np.asarray(img))
            continue
        ov = json.loads(meta["overlays"][k])
        if args.only_success:
            import re
            if ov.get("status"):
                ov["status"] = re.sub(r"\s+try \d+.*$", "", ov["status"])
            ov["tally"] = [(re.sub(r"\s+try \d+$", "", t), c if "done" in t or n == 0 else [90, 90, 100])
                           for n, (t, c) in enumerate(ov.get("tally") or [])]
        lines = [ov.get("title", "")] + [x for x in (ov.get("status"), ov.get("phase")) if x]
        d.rectangle([14, 14, 900, 26 + 36 * len(lines)], fill=(255, 255, 255, 220))
        d.text((26, 20), lines[0], font=f_big, fill=(20, 20, 30))
        for n, t in enumerate(lines[1:]):
            d.text((26, 24 + 36 * (n + 1)), t, font=f, fill=(45, 45, 55))
        tally = ov.get("tally") or []
        if tally:
            w = 300
            d.rectangle([W - w - 30, 14, W - 14, 24 + 32 * len(tally)], fill=(255, 255, 255, 220))
            for n, (t, col) in enumerate(tally):
                d.text((W - w - 16, 20 + 32 * n), t, font=f, fill=tuple(col))
        if plan:
            legend = "blue: 128 points the policy sees   orange/cyan: its contacts   green: approach"
            d.rectangle([14, H - 50, W - 14, H - 14], fill=(255, 255, 255, 220))
            d.text((26, H - 45), legend, font=font(20), fill=(45, 45, 55))
        for b in meta["bins"]:
            (bx, by), = project(r, np.array([[b["center"][0], b["center"][1], b["wall_h"] + 0.05]]), W, H)
            tw = d.textlength(b["name"], font=f)
            d.rectangle([bx - tw / 2 - 8, by - 18, bx + tw / 2 + 8, by + 14], fill=(255, 255, 255, 210))
            d.text((bx - tw / 2, by - 15), b["name"], font=f, fill=(30, 30, 40))
        if ov.get("card"):
            text, col = ov["card"]
            w = d.textlength(text, font=f_card)
            d.rectangle([W / 2 - w / 2 - 28, H - 150, W / 2 + w / 2 + 28, H - 64], fill=(255, 255, 255, 235))
            d.text((W / 2 - w / 2, H - 140), text, font=f_card, fill=tuple(col))
        if args.preview is not None:
            img.save(os.path.splitext(out)[0] + f"_{k}.png")
            continue
        writer.append_data(np.asarray(img))
    writer.close()
    print(f"[render] {len(bp)} frames → {out}")


if __name__ == "__main__":
    main()
