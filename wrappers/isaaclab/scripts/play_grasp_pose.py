#!/usr/bin/env python3
"""
Evaluate a trained grasp-pose policy and optionally record MP4 rollouts.

Usage:
    # Live viewport (recommended on 8 GB GPU — use 1 env)
    python wrappers/isaaclab/scripts/play_grasp_pose.py \\
        --checkpoint data/grasp_logs/grasp_pose_envs32_*/grasp_pose_final.pt \\
        --num_envs 1

    # Headless MP4 (approach → close → lift; grasp markers auto-enabled)
    python wrappers/isaaclab/scripts/play_grasp_pose.py --headless --enable_cameras \\
        --checkpoint data/grasp_logs/.../grasp_pose_final.pt \\
        --video --video_episodes 5 --num_envs 1 \\
        --out data/viz/grasp_pose_rollout.mp4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play grasp-pose RL policy")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Trained .pt (omit with --debug_action)")
parser.add_argument("--debug_action", type=str, choices=["zero", "random", "antipodal", "center"],
                    default=None,
                    help="Skip checkpoint; use fixed/oracle action for IK/exec debug. "
                         "'antipodal' loads best horizontal antipodal (c1,c2) per shape. "
                         "'center' = Path A single point at PC centroid (control smoke test). "
                         "'random' samples random contact pairs in grasp bounds.")
parser.add_argument("--two_point", action="store_true",
                    help="Force 6D two-point contact mode (c1,c2).")
parser.add_argument("--path_a", action="store_true",
                    help="Path A: center + tilt + roll + width (default if neither flag set).")
parser.add_argument("--no_project_pc", action="store_true",
                    help="Disable project_grasp_to_pc (exact action contacts for IK debug)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=20,
                    help="Episodes to evaluate (stats printed)")
parser.add_argument("--video", action="store_true",
                    help="Record dense MP4 of grasp execution")
parser.add_argument("--video_episodes", type=int, default=3,
                    help="Episodes included in the MP4 when --video is set")
parser.add_argument("--out", type=str, default="data/viz/grasp_pose_rollout.mp4")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_pc_points", type=int, default=128)
parser.add_argument("--pc_embed_dim", type=int, default=128)
parser.add_argument("--cam_eye", type=float, nargs=3, default=[0.85, -0.35, 0.45],
                    help="Video camera position (world x y z), near the object")
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.50, 0.0, 0.12],
                    help="Video camera look-at point (world x y z) = object")
parser.add_argument("--visualize_grasp", action="store_true",
                    help="Show grasp point (yellow) and palm IK target (blue) markers")
parser.add_argument("--cycle_shapes", action="store_true",
                    help="Cycle objects 0..9 each episode (for multi-object demo videos)")
parser.add_argument("--fixed_spawn", action="store_true",
                    help="Disable spawn xy/yaw randomization (default: keep training randomization)")
parser.add_argument("--honest", action="store_true",
                    help="PD-only reach (ik_write_joint_state=False). Matches training/eval.")
parser.add_argument("--use_real_objects", type=lambda s: s.lower() != "false", default=True,
                    help="Include YCB objects (match training distribution)")
parser.add_argument("--variants_per_family", type=int, default=1,
                    help="Instances per family to evaluate on. 1 (default) = original objects "
                         "only, so --cycle_shapes stays one object per family; 0 = all variants.")
parser.add_argument("--record", type=str, default=None,
                    help="Headless alternative to --video: save env 0's state every physics step "
                         "for the first --video_episodes episodes to this .npz, then render it "
                         "with scripts/render_recording.py (no RTX renderer needed).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from envs.grasp_pose_env_cfg import GraspPoseEnvCfg, OBS_DIM, NUM_ACTIONS, EXEC_STEPS
from envs.grasp_pose_env import GraspPoseEnv
from envs._object_registry import shape_asset
from models.grasp_pose_actor_critic import GraspPoseActorCritic
from state_recorder import StateRecorder


def _norm_axis(x: float, lo: float, hi: float) -> float:
    return float(max(-1.0, min(1.0, 2.0 * (x - lo) / (hi - lo) - 1.0)))


def random_contact_pair_action(cfg: GraspPoseEnvCfg, device: str) -> torch.Tensor:
    """Random (c1,c2) in grasp bounds with valid pinch width + near-horizontal jaw."""
    import math
    import random

    x_lo, x_hi = cfg.grasp_x_bounds
    y_lo, y_hi = cfg.grasp_y_bounds
    z_lo, z_hi = cfg.grasp_z_bounds
    w_lo, w_hi = cfg.grasp_width_bounds

    for _ in range(64):
        # Midpoint anywhere in the box; jaw yaw random in xy; width in bounds.
        mid = [
            random.uniform(x_lo, x_hi),
            random.uniform(y_lo, y_hi),
            random.uniform(z_lo, z_hi),
        ]
        yaw = random.uniform(0.0, math.pi)
        width = random.uniform(w_lo, w_hi)
        # Small random tilt of the closing axis (±25°) — still mostly horizontal.
        pitch = random.uniform(-0.45, 0.45)
        ax = math.cos(pitch) * math.cos(yaw)
        ay = math.cos(pitch) * math.sin(yaw)
        az = math.sin(pitch)
        c1 = [mid[i] - 0.5 * width * v for i, v in enumerate((ax, ay, az))]
        c2 = [mid[i] + 0.5 * width * v for i, v in enumerate((ax, ay, az))]
        if (
            x_lo <= c1[0] <= x_hi and x_lo <= c2[0] <= x_hi
            and y_lo <= c1[1] <= y_hi and y_lo <= c2[1] <= y_hi
            and z_lo <= c1[2] <= z_hi and z_lo <= c2[2] <= z_hi
        ):
            if c1[1] > c2[1]:
                c1, c2 = c2, c1
            vals = [
                _norm_axis(c1[0], x_lo, x_hi), _norm_axis(c1[1], y_lo, y_hi), _norm_axis(c1[2], z_lo, z_hi),
                _norm_axis(c2[0], x_lo, x_hi), _norm_axis(c2[1], y_lo, y_hi), _norm_axis(c2[2], z_lo, z_hi),
            ]
            print(
                f"[play] random: c1=({c1[0]:+.3f},{c1[1]:+.3f},{c1[2]:+.3f})  "
                f"c2=({c2[0]:+.3f},{c2[1]:+.3f},{c2[2]:+.3f})  width={width:.3f}m"
            )
            return torch.tensor([vals], dtype=torch.float32, device=device)

    # Fallback: horizontal along +y through origin.
    width = 0.5 * (w_lo + w_hi)
    c1 = [0.0, -0.5 * width, 0.0]
    c2 = [0.0, 0.5 * width, 0.0]
    vals = [
        _norm_axis(c1[0], x_lo, x_hi), _norm_axis(c1[1], y_lo, y_hi), _norm_axis(c1[2], z_lo, z_hi),
        _norm_axis(c2[0], x_lo, x_hi), _norm_axis(c2[1], y_lo, y_hi), _norm_axis(c2[2], z_lo, z_hi),
    ]
    return torch.tensor([vals], dtype=torch.float32, device=device)


def antipodal_action_for_shape(shape: str, cfg: GraspPoseEnvCfg, device: str) -> torch.Tensor:
    """Best in-bounds horizontal antipodal (c1,c2) → tanh action for IK diagnostics."""
    import json
    import math

    g_path = shape_asset(shape, "_grasps.json")
    pc_path = shape_asset(shape, "_pc.npy")

    x_lo, x_hi = cfg.grasp_x_bounds
    y_lo, y_hi = cfg.grasp_y_bounds
    z_lo, z_hi = cfg.grasp_z_bounds
    w_lo, w_hi = cfg.grasp_width_bounds
    scale = float(cfg.object_scale)

    def _in_bounds(pt):
        return (
            x_lo <= pt[0] <= x_hi
            and y_lo <= pt[1] <= y_hi
            and z_lo <= pt[2] <= z_hi
        )

    def _pack(c1, c2, width, tag):
        if c1[1] > c2[1]:
            c1, c2 = c2, c1
        vals = [
            _norm_axis(c1[0], x_lo, x_hi), _norm_axis(c1[1], y_lo, y_hi), _norm_axis(c1[2], z_lo, z_hi),
            _norm_axis(c2[0], x_lo, x_hi), _norm_axis(c2[1], y_lo, y_hi), _norm_axis(c2[2], z_lo, z_hi),
        ]
        print(
            f"[play] {tag} {shape}: c1=({c1[0]:+.3f},{c1[1]:+.3f},{c1[2]:+.3f})  "
            f"c2=({c2[0]:+.3f},{c2[1]:+.3f},{c2[2]:+.3f})  width={width:.3f}m"
        )
        return torch.tensor([vals], dtype=torch.float32, device=device)

    grasps = []
    if g_path.exists():
        with open(g_path) as f:
            data = json.load(f)
        grasps = data["grasps"] if isinstance(data, dict) else data
        grasps = sorted(grasps, key=lambda g: g["quality"], reverse=True)

    for g in grasps:
        approach = list(g["approach"])
        n = math.sqrt(sum(v * v for v in approach)) + 1e-9
        a = [v / n for v in approach]
        # Prefer closing axis nearly horizontal; else project onto xy.
        if abs(a[2]) > 0.85:
            continue
        xy = math.sqrt(a[0] * a[0] + a[1] * a[1])
        if xy < 1e-4:
            continue
        a = [a[0] / xy, a[1] / xy, 0.0]
        width = float(g["width"]) * scale
        width = max(w_lo, min(w_hi, width))
        center = [c * scale for c in g["center"]]
        # Keep mid near table mid-height of object.
        center[2] = max(z_lo + 0.005, min(z_hi - 0.005, center[2]))
        c1 = [center[i] + 0.5 * width * a[i] for i in range(3)]
        c2 = [center[i] - 0.5 * width * a[i] for i in range(3)]
        if _in_bounds(c1) and _in_bounds(c2):
            return _pack(c1, c2, width, "antipodal")

    # Fallback: horizontal pinch through scaled PC centroid along +y.
    import numpy as np
    if pc_path.exists():
        pc = np.load(str(pc_path)).astype(np.float32) * scale
        center = pc.mean(axis=0).tolist()
    else:
        center = [0.0, 0.0, 0.0]
    center[2] = max(z_lo + 0.005, min(z_hi - 0.005, center[2]))
    width = 0.5 * (w_lo + w_hi)
    c1 = [center[0], center[1] - 0.5 * width, center[2]]
    c2 = [center[0], center[1] + 0.5 * width, center[2]]
    return _pack(c1, c2, width, "fallback-y")


def center_action_for_shape(shape: str, cfg: GraspPoseEnvCfg, device: str) -> torch.Tensor:
    """Path A: single grasp point at scaled PC centroid → tanh action (control smoke test)."""
    import numpy as np

    pc_path = shape_asset(shape, "_pc.npy")
    x_lo, x_hi = cfg.grasp_x_bounds
    y_lo, y_hi = cfg.grasp_y_bounds
    z_lo, z_hi = cfg.grasp_z_bounds
    t_lo, t_hi = cfg.grasp_tilt_bounds
    r_lo, r_hi = cfg.grasp_roll_bounds
    w_lo, w_hi = cfg.grasp_width_bounds
    scale = float(cfg.object_scale)

    if pc_path.exists():
        pc = np.load(str(pc_path)).astype(np.float32) * scale
        center = pc.mean(axis=0)
    else:
        center = np.zeros(3, dtype=np.float32)
    center[0] = np.clip(center[0], x_lo, x_hi)
    center[1] = np.clip(center[1], y_lo, y_hi)
    center[2] = np.clip(center[2], z_lo, z_hi)

    tilt = 0.0
    roll = 0.741  # Franka panda_joint7 home
    width = 0.5 * (w_lo + w_hi)

    vals = [
        _norm_axis(float(center[0]), x_lo, x_hi),
        _norm_axis(float(center[1]), y_lo, y_hi),
        _norm_axis(float(center[2]), z_lo, z_hi),
        _norm_axis(tilt, t_lo, t_hi),
        _norm_axis(roll, r_lo, r_hi),
        _norm_axis(width, w_lo, w_hi),
    ]
    print(
        f"[play] center {shape}: pt=({center[0]:+.3f},{center[1]:+.3f},{center[2]:+.3f})  "
        f"tilt={tilt:.2f}  roll={roll:.2f}  width={width:.3f}m  (Path A, PC snap on)"
    )
    return torch.tensor([vals], dtype=torch.float32, device=device)


def load_policy(ckpt_path: str, device: str) -> GraspPoseActorCritic:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model = GraspPoseActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=NUM_ACTIONS,
        num_pc_points=args.num_pc_points,
        pc_embed_dim=args.pc_embed_dim,
    ).to(device)
    own = model.state_dict()
    try:
        model.load_state_dict(state, strict=True)
        print("[play] checkpoint: full load")
    except RuntimeError:
        loaded = {}
        for k, v in state.items():
            if k not in own or own[k].shape == v.shape:
                loaded[k] = v
            elif k == "actor_head.4.weight" and v.shape[0] <= own[k].shape[0]:
                w = own[k].clone()
                w[: v.shape[0]] = v
                loaded[k] = w
            elif k == "actor_head.4.bias" and v.shape[0] <= own[k].shape[0]:
                b = own[k].clone()
                b[: v.shape[0]] = v
                loaded[k] = b
            elif k == "std" and v.numel() <= own[k].numel():
                s = own[k].clone()
                s[: v.numel()] = v
                loaded[k] = s
        model.load_state_dict(loaded, strict=False)
        print(f"[play] checkpoint: partial load ({len(loaded)}/{len(own)} keys)")
    model.eval()
    return model


def _obs_tensor(obs_td, device):
    return obs_td["policy"].to(device)


def _decode_action(action: torch.Tensor, cfg: GraspPoseEnvCfg) -> dict:
    """Mirror env de-normalization for debug prints."""
    a = action[0].clamp(-1, 1)
    x_lo, x_hi = cfg.grasp_x_bounds
    y_lo, y_hi = cfg.grasp_y_bounds
    z_lo, z_hi = cfg.grasp_z_bounds

    def denorm(i0):
        return (
            ((a[i0] + 1) / 2 * (x_hi - x_lo) + x_lo).item(),
            ((a[i0 + 1] + 1) / 2 * (y_hi - y_lo) + y_lo).item(),
            ((a[i0 + 2] + 1) / 2 * (z_hi - z_lo) + z_lo).item(),
        )

    if cfg.two_point_grasp:
        c1, c2 = denorm(0), denorm(3)
        import math
        width = math.sqrt(sum((c2[i] - c1[i]) ** 2 for i in range(3)))
        mid = tuple(0.5 * (c1[i] + c2[i]) for i in range(3))
        return {"c1": c1, "c2": c2, "grasp_local": mid, "width_m": width, "two_point": True}

    t_lo, t_hi = cfg.grasp_tilt_bounds
    r_lo, r_hi = cfg.grasp_roll_bounds
    w_lo, w_hi = cfg.grasp_width_bounds
    g = denorm(0)
    tilt = ((a[3] + 1) / 2 * (t_hi - t_lo) + t_lo).item()
    roll = ((a[4] + 1) / 2 * (r_hi - r_lo) + r_lo).item()
    if a.numel() >= 6:
        width = ((a[5] + 1) / 2 * (w_hi - w_lo) + w_lo).item()
    else:
        width = 0.5 * (w_lo + w_hi)
    return {
        "c1": g, "c2": g, "grasp_local": g,
        "tilt_rad": tilt, "roll_rad": roll, "width_m": width, "two_point": False,
    }


def rollout_dense_frames(env: GraspPoseEnv, action: torch.Tensor, recorder=None, shape=None) -> list:
    """Step physics manually and capture an rgb frame (and/or recorder state) each sub-step."""
    u = env.unwrapped
    u._pre_physics_step(action)
    grasp_used = u._grasp_target[0].detach().cpu().clone()
    if recorder is not None:
        recorder.begin_episode(shape)
    capture = u.render_mode == "rgb_array"
    is_rendering = capture or u.sim.has_gui() or u.sim.has_rtx_sensors()
    frames = []

    for _ in range(u.cfg.decimation):
        u._apply_action()
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        if is_rendering:
            u.sim.render()
            if capture:
                frame = u.render(recompute=True)
                if frame is not None and frame.size > 0 and frame.any():
                    frames.append(frame)
        u.scene.update(dt=u.physics_dt)
        if recorder is not None:
            recorder.capture()

    rew = u._get_rewards()
    done_ids = torch.arange(u.num_envs, device=u.device)
    u._reset_idx(done_ids)
    return frames, rew, grasp_used


def write_mp4(frames: list, path: str, fps: int = 30):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    writer = imageio.get_writer(
        path, fps=fps, codec="libx264",
        pixelformat="yuv420p", quality=8,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[play] saved video → {path}  ({len(frames)} frames)")


def main():
    device = args.device
    if not args.debug_action and not args.checkpoint:
        print("[play] provide --checkpoint or --debug_action zero|random|antipodal")
        sys.exit(1)

    render_mode = "rgb_array" if args.video else None
    env_cfg = GraspPoseEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    env_cfg.use_real_objects = args.use_real_objects
    env_cfg.variants_per_family = args.variants_per_family
    env_cfg.use_camera_pc = False
    env_cfg.visualize_grasp_point = args.visualize_grasp or args.video or not args.headless
    if args.cycle_shapes:
        env_cfg.eval_cycle_shapes = True
        env_cfg.eval_num_shapes = 10
    if args.fixed_spawn:
        env_cfg.randomize_object_spawn = False
    if args.no_project_pc or args.debug_action in ("antipodal", "random"):
        env_cfg.project_grasp_to_pc = False
        print("[play] project_grasp_to_pc=False (exact contacts for IK debug)")
    # Path A single-point smoke test (center on object PC).
    if args.path_a or args.debug_action == "center":
        env_cfg.two_point_grasp = False
        env_cfg.action_space = 6
        print("[play] path_a=True  action=center+tilt+roll+width")
    elif args.two_point or args.debug_action in ("antipodal", "random"):
        env_cfg.two_point_grasp = True
        env_cfg.action_space = 6
        # Kinematic joint writes for oracle IK debug; policy eval uses cfg default (False).
        env_cfg.ik_write_joint_state = bool(
            args.debug_action in ("antipodal", "random")
        )
        print("[play] two_point_grasp=True  action_space=6")
        print(f"[play] ik_write_joint_state={env_cfg.ik_write_joint_state}")
    if args.honest:
        env_cfg.ik_write_joint_state = False
        print("[play] honest=True  ik_write_joint_state=False (PD-only reach)")
    # Always print grasp honesty flags (fake lifts came from contact_carry weld).
    print(
        f"[play] contact_carry={env_cfg.contact_carry}  "
        f"kinematic_grasp={env_cfg.kinematic_grasp}  "
        f"require_contact_for_lift={env_cfg.require_contact_for_lift_reward}  "
        f"contact_force_thresh={env_cfg.contact_force_thresh}N"
    )
    if env_cfg.visualize_grasp_point:
        env_cfg.visualize_object_pc = True
        print("[play] grasp markers: yellow=midpoint  orange/cyan=contact1/2  "
              "magenta=IK palm  blue=actual palm  green=finger midpoint")
        print("[play] grey dots = object point cloud (contacts should sit on them)")
        print(f"[play] two_point_grasp={env_cfg.two_point_grasp}  "
              f"project_grasp_to_pc={env_cfg.project_grasp_to_pc}")
        print(f"[play] spawn_randomize={env_cfg.randomize_object_spawn}  "
              f"(use --fixed_spawn for locked pose)")
    if args.video:
        env_cfg.sim.render_interval = 1

    env = GraspPoseEnv(cfg=env_cfg, render_mode=render_mode)
    if args.seed is not None:
        env.seed(args.seed)
        import random
        random.seed(args.seed)

    if args.video:
        # Move the render camera close to the object for a near view.
        env.unwrapped.sim.set_camera_view(eye=args.cam_eye, target=args.cam_target)
        print(f"[play] camera eye={args.cam_eye} target={args.cam_target}")

    policy = None
    if args.debug_action:
        print(f"[play] debug mode: action={args.debug_action} (no checkpoint)")
    else:
        ckpt = os.path.abspath(args.checkpoint)
        if not os.path.isfile(ckpt):
            print(f"[play] checkpoint not found: {ckpt}")
            sys.exit(1)
        policy = load_policy(ckpt, device)
        print(f"[play] loaded {ckpt}")

    n_act = int(env_cfg.action_space)
    print(f"[play] device={device}  envs={args.num_envs}  episodes={args.num_episodes}  "
          f"use_real_objects={args.use_real_objects}  action_dim={n_act}")

    obs_dict, _ = env.reset()
    recorder = StateRecorder(env) if args.record else None
    successes, total_reward = 0, 0.0
    video_frames: list = []
    finger_errs, palm_errs, jaw_aligns = [], [], []
    left_errs, right_errs, close_errs = [], [], []

    for ep in range(1, args.num_episodes + 1):
        shape = env.unwrapped._shape_names[env.unwrapped._env_shape[0].item()]
        if args.debug_action == "zero":
            action = torch.zeros(args.num_envs, n_act, device=device)
        elif args.debug_action == "random":
            action = random_contact_pair_action(env_cfg, device)
            if args.num_envs > 1:
                action = action.expand(args.num_envs, -1).clone()
        elif args.debug_action == "antipodal":
            action = antipodal_action_for_shape(shape, env_cfg, device)
            if args.num_envs > 1:
                action = action.expand(args.num_envs, -1).clone()
        elif args.debug_action == "center":
            action = center_action_for_shape(shape, env_cfg, device)
            if args.num_envs > 1:
                action = action.expand(args.num_envs, -1).clone()
        else:
            obs = _obs_tensor(obs_dict, device)
            with torch.no_grad():
                action = policy.act_inference(obs)

        grasp_used = None
        if (args.video or recorder is not None) and ep <= args.video_episodes:
            frames, rew, grasp_used = rollout_dense_frames(env, action, recorder, shape)
            if recorder is not None:
                recorder.end_episode()
            video_frames.extend(frames)
            obs_dict = env._get_observations()
        else:
            obs_dict, rew, _, _, _ = env.step(action)
            grasp_used = env.unwrapped._grasp_target[0].detach().cpu().clone()

        u = env.unwrapped
        fe = float(u._ik_finger_err[0].item())
        pe = float(u._ik_palm_err[0].item())
        ja = float(u._ik_jaw_align[0].item())
        le = float(u._ik_left_contact_err[0].item())
        re_ = float(u._ik_right_contact_err[0].item())
        lec = float(u._ik_left_contact_err_closed[0].item())
        rec = float(u._ik_right_contact_err_closed[0].item())
        ce = float(u._ik_contact_err_after_close[0].item())
        finger_errs.append(fe)
        palm_errs.append(pe)
        jaw_aligns.append(ja)
        left_errs.append(le)
        right_errs.append(re_)
        close_errs.append(ce)

        r = rew.mean().item()
        total_reward += r
        # Success = object lifted to lift_target_m; shaping reward never counts.
        lift_ok = bool(u._last_lift_success[0].item())
        if lift_ok:
            successes += 1
        # Keep the shape used for this episode (reset inside step advances the cycle).
        dec = _decode_action(action, env_cfg)
        if dec.get("two_point"):
            # Log post-projection contacts the controller actually uses (not raw policy decode).
            c1_t = u._contact_left_local[0].detach().cpu().tolist()
            c2_t = u._contact_right_local[0].detach().cpu().tolist()
            width_m = float(u._grasp_width[0].item())
            c1, c2 = c1_t, c2_t
            print(
                f"  ep {ep:3d}/{args.num_episodes}  shape={shape:14s}  reward={r:.3f}  "
                f"lift_ok={lift_ok}  "
                f"c1=({c1[0]:+.3f},{c1[1]:+.3f},{c1[2]:+.3f})  "
                f"c2=({c2[0]:+.3f},{c2[1]:+.3f},{c2[2]:+.3f})  "
                f"width={width_m:.3f}m  "
                f"mid_err={fe*100:.1f}cm  palm_err={pe*100:.1f}cm  jaw={ja:.2f}  "
                f"open L/R→c={le*100:.1f}/{re_*100:.1f}cm  "
                f"closed L/R→c={lec*100:.1f}/{rec*100:.1f}cm"
            )
            if ep <= 2:
                pw = u._ik_palm_w[0].detach().cpu().tolist()
                pt = u._ik_palm_tgt_w[0].detach().cpu().tolist()
                gw = u._ik_grasp_w[0].detach().cpu().tolist()
                fw = u._ik_finger_w[0].detach().cpu().tolist()
                print(
                    f"         palm=({pw[0]:+.3f},{pw[1]:+.3f},{pw[2]:+.3f})  "
                    f"palm_tgt=({pt[0]:+.3f},{pt[1]:+.3f},{pt[2]:+.3f})  "
                    f"grasp=({gw[0]:+.3f},{gw[1]:+.3f},{gw[2]:+.3f})  "
                    f"finger=({fw[0]:+.3f},{fw[1]:+.3f},{fw[2]:+.3f})"
                )
        else:
            g = dec["grasp_local"]
            print(
                f"  ep {ep:3d}/{args.num_episodes}  shape={shape:14s}  reward={r:.3f}  "
                f"lift_ok={lift_ok}  "
                f"raw=({g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f})  "
                f"tilt={dec['tilt_rad']:+.2f}rad  width={dec['width_m']:.3f}m  "
                f"finger_err={fe*100:.1f}cm  palm_err={pe*100:.1f}cm"
            )

    n = args.num_episodes
    print(f"\n[play] mean_reward={total_reward/n:.3f}  "
          f"success_rate={100*successes/n:.1f}%  "
          f"(object lifted ≥ lift_target_m={env_cfg.lift_target_m*100:.0f}cm)")
    if finger_errs:
        import statistics as _stats
        print(
            f"[play] IK reach @ approach end:  "
            f"mid_err={100*_stats.mean(finger_errs):.1f}±{100*_stats.pstdev(finger_errs):.1f}cm  "
            f"palm_err={100*_stats.mean(palm_errs):.1f}±{100*_stats.pstdev(palm_errs):.1f}cm  "
            f"jaw_align={_stats.mean(jaw_aligns):.2f}±{_stats.pstdev(jaw_aligns):.2f}  "
            f"L→contact={100*_stats.mean(left_errs):.1f}±{100*_stats.pstdev(left_errs):.1f}cm  "
            f"R→contact={100*_stats.mean(right_errs):.1f}±{100*_stats.pstdev(right_errs):.1f}cm"
        )
        print(
            f"[play] after gripper close:  "
            f"mean_finger→contact={100*_stats.mean(close_errs):.1f}±"
            f"{100*_stats.pstdev(close_errs):.1f}cm"
        )

    if recorder is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        recorder.save(os.path.abspath(args.record))

    if args.video and video_frames:
        write_mp4(video_frames, os.path.abspath(args.out))
    elif args.video:
        print("[play] WARNING: no frames captured — try without --headless or check --enable_cameras")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
