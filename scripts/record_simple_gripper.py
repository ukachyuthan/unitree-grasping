"""Record a demo episode of `environments.simple_gripper_env.SimpleGripperEnv` to MP4.

Usage:
    python scripts/record_simple_gripper.py --out demo.mp4 --steps 300 --gui

The script runs a scripted episode (move from A→B, close, lift) and records
frames using PyBullet's camera. Requires `imageio` and ffmpeg available on PATH.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Tuple

import imageio
import numpy as np

from environments.simple_gripper_env import SimpleGripperEnv

try:
    import pybullet as p
except Exception:
    p = None


def get_camera_matrices(eye: Tuple[float, float, float], target: Tuple[float, float, float],
                        up=(0, 0, 1), fov=60.0, width=640, height=480, near=0.01, far=10.0):
    view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up)
    aspect = float(width) / float(height)
    proj = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    return view, proj


def capture_frame(client, view, proj, width, height):
    # p.getCameraImage sometimes accepts physicsClientId or not depending on version
    try:
        img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=client)
    except TypeError:
        img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)

    # img -> (w, h, rgbPixels, depth, seg)
    if len(img) >= 3:
        rgb = img[2]
    else:
        # fallback shape
        rgb = img[0]
    arr = np.reshape(rgb, (height, width, 4))[:, :, :3]
    return arr.astype(np.uint8)


def scripted_episode(env: SimpleGripperEnv, steps: int):
    """Produce a sequence of actions that moves gripper from A->B, closes, and lifts."""
    # Get initial position
    obs = env.reset()
    start = obs["gripper_pos"].copy()
    target = start + np.array([0.12, 0.0, 0.0])

    actions = []
    for t in range(steps):
        frac = t / max(1, steps - 1)
        # phase 1: move toward target in first 40% steps
        if frac < 0.4:
            alpha = frac / 0.4
            pos = start * (1 - alpha) + target * alpha
            dx = pos - env._gripper_pos
            gr_cmd = 1.0
        # phase 2: close gripper for next 20%
        elif frac < 0.6:
            dx = np.zeros(3)
            gr_cmd = 0.0
        # phase 3: lift for next 20%
        elif frac < 0.8:
            dx = np.array([0.0, 0.0, 0.002])
            gr_cmd = 0.0
        # phase 4: move back toward origin in last 20%
        else:
            alpha2 = (frac - 0.8) / 0.2
            pos = target * (1 - alpha2) + start * alpha2
            dx = pos - env._gripper_pos
            gr_cmd = 0.0

        # clip per-step
        dx = np.clip(dx, -0.02, 0.02)
        actions.append(np.concatenate([dx, [float(gr_cmd)]]))
    return actions


def main(out: str, steps: int = 300, gui: bool = True, width: int = 640, height: int = 480):
    env = SimpleGripperEnv(gui=gui)
    # position a camera slightly above and behind the gripper
    obs = env.reset()
    gr_pos = obs["gripper_pos"].copy()
    cam_eye = (gr_pos[0] - 0.05, gr_pos[1] + 0.25, gr_pos[2] + 0.25)
    cam_target = (gr_pos[0] + 0.05, gr_pos[1], gr_pos[2])

    view, proj = get_camera_matrices(cam_eye, cam_target, width=width, height=height)

    writer = imageio.get_writer(out, fps=30, codec="libx264", quality=8)

    actions = scripted_episode(env, steps=steps)
    for t, a in enumerate(actions):
        obs, r, d, info = env.step(a)
        frame = capture_frame(env._client, view, proj, width, height)
        writer.append_data(frame)
        if (t + 1) % 50 == 0:
            print(f"Recorded {t+1}/{steps} frames")
    writer.close()
    env.close()
    print(f"Saved demo video to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="simple_gripper_demo.mp4")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-gui", dest="gui", action="store_false")
    args = p.parse_args()
    main(out=args.out, steps=args.steps, gui=args.gui, width=args.width, height=args.height)
