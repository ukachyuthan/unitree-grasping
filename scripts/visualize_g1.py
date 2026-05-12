#!/usr/bin/env python3
"""
Replay a saved G1 trajectory in MuJoCo (no Isaac Sim / no RTX renderer needed).

Usage:
    # Interactive viewer (requires a display)
    ./rl_unitree/bin/python scripts/visualize_g1.py --traj data/isaaclab_logs/trajectory.npz

    # Render to mp4 without a display
    ./rl_unitree/bin/python scripts/visualize_g1.py --traj data/isaaclab_logs/trajectory.npz --video

    # Control render resolution / fps
    ./rl_unitree/bin/python scripts/visualize_g1.py --traj data/isaaclab_logs/trajectory.npz --video --width 1280 --height 720 --fps 50
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco
from robot_descriptions import g1_mj_description

parser = argparse.ArgumentParser(description="Replay G1 trajectory in MuJoCo")
parser.add_argument("--traj", type=str, default="data/isaaclab_logs/trajectory.npz",
                    help="Trajectory npz produced by play_g1.py")
parser.add_argument("--video", action="store_true",
                    help="Render frames to disk instead of opening the viewer")
parser.add_argument("--out", type=str, default="g1_policy.mp4",
                    help="Output video file (requires ffmpeg)")
parser.add_argument("--width",  type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps",    type=int, default=50,
                    help="Playback fps (Isaac Lab default sim dt 0.02 s → 50 Hz)")
parser.add_argument("--cam_distance", type=float, default=3.0)
parser.add_argument("--cam_elevation", type=float, default=-20.0)
args = parser.parse_args()

# ── Load trajectory ───────────────────────────────────────────────────────────
if not os.path.exists(args.traj):
    sys.exit(f"Trajectory file not found: {args.traj}\n"
             "Run play_g1.py first: ./rl_unitree/bin/python scripts/play_g1.py --headless")

traj = np.load(args.traj, allow_pickle=True)
isaac_joint_names = list(traj["joint_names"])   # list[str]
joint_pos_traj    = traj["joint_pos"]            # (T, n_joints)
root_pos_traj     = traj["root_pos"]             # (T, 3)  xyz world
root_quat_traj    = traj["root_quat"]            # (T, 4)  wxyz (Isaac Lab / MuJoCo convention)
T = joint_pos_traj.shape[0]
print(f"[viz] Loaded {T} steps, {len(isaac_joint_names)} Isaac Lab joints")

# ── Load MuJoCo model ─────────────────────────────────────────────────────────
# Patch the MJCF to set the offscreen framebuffer to the requested render size.
# MuJoCo's default is 640×480; anything larger raises ValueError unless we set it.
# We write a patched copy next to the original so relative mesh paths still work.
import tempfile

mjcf_dir  = os.path.dirname(g1_mj_description.MJCF_PATH)
mjcf_name = os.path.basename(g1_mj_description.MJCF_PATH)

with open(g1_mj_description.MJCF_PATH, "r") as f:
    xml_str = f.read()

visual_insert = f'<global offwidth="{args.width}" offheight="{args.height}"/>'
if "<visual>" in xml_str:
    xml_str = xml_str.replace("<visual>", f"<visual>{visual_insert}", 1)
else:
    xml_str = xml_str.replace("<worldbody>",
                               f"<visual>{visual_insert}</visual>\n  <worldbody>", 1)

# Add floor, lights, and skybox — the menagerie MJCF ships with none of these.
env_inject = """
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.4 0.4 0.4" rgb2="0.6 0.6 0.6"/>
    <material name="floor_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.53 0.80 0.98" rgb2="0.15 0.45 0.78" width="512" height="512"/>
  </asset>
"""
world_inject = """
    <geom name="floor" type="plane" size="20 20 0.1" material="floor_mat" condim="3"/>
    <light name="sun" directional="true" pos="0 0 8" dir="0.2 0.2 -1"
           diffuse="0.9 0.9 0.9" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="fill" directional="true" pos="0 0 4" dir="-0.2 -0.2 -0.8"
           diffuse="0.3 0.3 0.3" specular="0 0 0" castshadow="false"/>
"""
# Insert asset block before </mujoco> and inject world elements after <worldbody>
xml_str = xml_str.replace("</mujoco>", f"{env_inject}\n</mujoco>", 1)
xml_str = xml_str.replace("<worldbody>", f"<worldbody>{world_inject}", 1)

patched_path = os.path.join(mjcf_dir, f"_patched_{mjcf_name}")
with open(patched_path, "w") as f:
    f.write(xml_str)

try:
    model = mujoco.MjModel.from_xml_path(patched_path)
finally:
    os.remove(patched_path)

data  = mujoco.MjData(model)
print(f"[viz] MuJoCo model: {model.njnt} joints, nq={model.nq}")

# Build name → (isaac_idx, mujoco_qpos_addr) mapping
isaac_name_to_idx = {n: i for i, n in enumerate(isaac_joint_names)}
joint_map = []  # list of (isaac_idx, mujoco_qpos_addr)
for j in range(model.njnt):
    mj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if mj_name == "floating_base_joint":
        continue
    if mj_name in isaac_name_to_idx:
        joint_map.append((isaac_name_to_idx[mj_name], model.jnt_qposadr[j]))

print(f"[viz] Mapped {len(joint_map)} joints between Isaac Lab and MuJoCo")
unmapped_mj = []
for j in range(model.njnt):
    mj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if mj_name != "floating_base_joint" and mj_name not in isaac_name_to_idx:
        unmapped_mj.append(mj_name)
if unmapped_mj:
    print(f"[viz] MuJoCo joints held at default pose (no Isaac Lab match): {unmapped_mj}")


def apply_frame(step: int):
    """Write Isaac Lab trajectory frame into MuJoCo qpos/qvel."""
    # Root pose via freejoint: qpos[0:3]=xyz, qpos[3:7]=wxyz
    data.qpos[0:3] = root_pos_traj[step]
    data.qpos[3:7] = root_quat_traj[step]
    # Named joints
    for isaac_idx, qpos_adr in joint_map:
        data.qpos[qpos_adr] = joint_pos_traj[step, isaac_idx]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# ── Render video ──────────────────────────────────────────────────────────────
if args.video:
    import subprocess, tempfile, shutil

    if not shutil.which("ffmpeg"):
        sys.exit("[viz] ffmpeg not found — install it with: sudo apt install ffmpeg")

    frames_dir = tempfile.mkdtemp(prefix="g1_frames_")
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Position camera to track robot
    renderer.update_scene(data, camera=-1)  # use free camera
    scene_opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.type   = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([0.0, 0.0, 0.8])
    cam.distance    = args.cam_distance
    cam.elevation   = args.cam_elevation
    cam.azimuth     = 135.0

    print(f"[viz] Rendering {T} frames to {frames_dir} ...")
    for step in range(T):
        apply_frame(step)
        # Track robot position
        cam.lookat[0] = root_pos_traj[step, 0]
        cam.lookat[1] = root_pos_traj[step, 1]
        cam.lookat[2] = root_pos_traj[step, 2] + 0.5
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        rgb = renderer.render()
        frame_path = os.path.join(frames_dir, f"frame_{step:05d}.png")
        import PIL.Image
        PIL.Image.fromarray(rgb).save(frame_path)
        if step % 100 == 0:
            print(f"[viz] Frame {step}/{T}")

    renderer.close()
    print(f"[viz] Encoding video with ffmpeg ...")
    cmd = [
        "ffmpeg", "-y",
        "-r", str(args.fps),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        args.out,
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(frames_dir)
    print(f"[viz] Video saved to {args.out}")

# ── Interactive viewer ────────────────────────────────────────────────────────
else:
    try:
        import mujoco.viewer
    except ImportError:
        sys.exit("[viz] mujoco.viewer not available. Try --video instead.")

    step_duration = 1.0 / args.fps
    print(f"[viz] Opening MuJoCo viewer — press ESC or close window to exit")
    print(f"[viz] Playing back {T} steps at {args.fps} fps ({T/args.fps:.1f} s)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial camera position
        viewer.cam.lookat = np.array([0.0, 0.0, 0.8])
        viewer.cam.distance = args.cam_distance
        viewer.cam.elevation = args.cam_elevation
        viewer.cam.azimuth = 135.0

        for step in range(T):
            if not viewer.is_running():
                break
            t0 = time.perf_counter()
            apply_frame(step)
            # Track robot
            viewer.cam.lookat[0] = root_pos_traj[step, 0]
            viewer.cam.lookat[1] = root_pos_traj[step, 1]
            viewer.cam.lookat[2] = root_pos_traj[step, 2] + 0.5
            viewer.sync()
            elapsed = time.perf_counter() - t0
            remaining = step_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print("[viz] Playback finished")
