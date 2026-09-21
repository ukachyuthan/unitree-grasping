#!/usr/bin/env python3
"""
RL training for grasp-pose prediction (Path A).

Policy: PointNet(PC) → 3D grasp position
Reward: purely physical — how far did the object actually lift in simulation

Pre-training (optional warm start — skippable):
    # Legacy: 3D grasp-center regression (encoder only)
    ./rl_unitree/bin/python scripts/pretrain_grasp.py

    # Recommended for two_point_grasp: BC on (c1,c2) antipodal contacts
    ./rl_unitree/bin/python scripts/pretrain_grasp_2pt.py
    This initialises encoder + actor_head so exploration starts near valid pinches.

RL training:
    # Cold start (random init)
    ./rl_unitree/bin/python scripts/train_grasp_pose.py --headless

    # Warm start (faster convergence, same end-point)
    ./rl_unitree/bin/python scripts/train_grasp_pose.py --headless \
        --pretrain data/grasp_weights/grasp_2pt_pretrain_best.pt

The RL reward is only lift_height — no regression loss, no geometric supervision.
Over training the policy discovers which regions of the point cloud predict
physically successful grasps, which may differ from antipodal geometry labels.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",   type=int,   default=64,
                    help="Parallel envs (use 32–64 on 8 GB GPU)")
parser.add_argument("--max_iters",  type=int,   default=5000)
parser.add_argument("--log_dir",    type=str,   default="data/grasp_logs")
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--pretrain",   type=str,   default=None,
                    help="Optional: pretrained .pt warm start (encoder, or "
                         "encoder+actor_head when mode=two_point from "
                         "scripts/pretrain_grasp_2pt.py)")
parser.add_argument("--resume",     type=str,   default=None,
                    help="Optional: full grasp_pose_*.pt checkpoint to resume policy weights")
parser.add_argument("--num_pc_points", type=int, default=128)
parser.add_argument("--pc_embed_dim",  type=int, default=128)
parser.add_argument("--use_real_objects", type=lambda s: s.lower() != "false", default=True,
                    help="Include real (YCB-derived) objects alongside procedural shapes. "
                         "Pass --use_real_objects false to reproduce the original RNG-only run.")
parser.add_argument("--path_a", action="store_true",
                    help="Path A action space: center + tilt + roll + width (not c1/c2).")
parser.add_argument("--two_point", action="store_true",
                    help="Two-point action space: c1 + c2 (overrides --path_a).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys
from datetime import datetime
from pathlib import Path

import json

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from envs.grasp_pose_env_cfg import (
    GraspPoseEnvCfg, OBS_DIM, NUM_ACTIONS, N_APPROACH, N_CLOSE, N_DESCEND,
)
from envs.grasp_pose_env import GraspPoseEnv
from models.grasp_pose_actor_critic import GraspPoseActorCritic


def _obs_tensor(obs_td, device):
    return obs_td["policy"].to(device)


def ppo_update(ac, optimizer, obs, actions, old_logp, returns, advantages,
               clip_param, value_coef, entropy_coef, max_grad_norm):
    ac.act(obs)
    logp = ac.get_actions_log_prob(actions)
    entropy = ac.entropy.mean()
    values = ac.evaluate(obs).squeeze(-1)

    ratio = torch.exp(logp - old_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = 0.5 * (returns - values).pow(2).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
    optimizer.step()
    return policy_loss.item(), value_loss.item(), entropy.item(), loss.item(), grad_norm.item()


def train_ppo(env, ac, device, log_dir, max_iters, writer=None, metrics_path=None,
              success_threshold=0.25, lr=3e-4,
              num_steps_per_env=16, num_epochs=5, num_mini_batches=4,
              clip_param=0.2, value_coef=1.0, entropy_coef=0.02,
              max_grad_norm=1.0, save_interval=25):
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr)
    num_envs = env.num_envs
    batch_size = num_envs * num_steps_per_env
    mini_batch_size = max(1, batch_size // num_mini_batches)

    for it in range(1, max_iters + 1):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []

        obs_td = env.get_observations()
        for _ in range(num_steps_per_env):
            obs = _obs_tensor(obs_td, device)
            with torch.no_grad():
                ac.act(obs)
                actions = ac.distribution.sample()
                logp = ac.get_actions_log_prob(actions)
                values = ac.evaluate(obs).squeeze(-1)

            obs_td, rew, _, _ = env.step(actions)

            obs_buf.append(obs)
            act_buf.append(actions)
            logp_buf.append(logp)
            rew_buf.append(rew.to(device))
            val_buf.append(values)

        obs_all = torch.cat(obs_buf, dim=0)
        act_all = torch.cat(act_buf, dim=0)
        logp_all = torch.cat(logp_buf, dim=0)
        rew_all = torch.cat(rew_buf, dim=0)
        val_all = torch.cat(val_buf, dim=0)

        # One-step episodes: return = reward, advantage = return - value
        returns = rew_all
        advantages = (returns - val_all).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        last_policy_loss = last_value_loss = last_entropy = 0.0
        last_total_loss = last_grad_norm = 0.0
        for _ in range(num_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                (last_policy_loss, last_value_loss, last_entropy,
                 last_total_loss, last_grad_norm) = ppo_update(
                    ac, optimizer,
                    obs_all[idx], act_all[idx], logp_all[idx],
                    returns[idx], advantages[idx],
                    clip_param, value_coef, entropy_coef, max_grad_norm,
                )

        mean_rew = rew_all.mean().item()
        std_rew = rew_all.std(unbiased=False).item()
        min_rew = rew_all.min().item()
        max_rew = rew_all.max().item()
        success_rate = (rew_all >= success_threshold).float().mean().item()
        mean_value = val_all.mean().item()
        mean_return = returns.mean().item()
        mean_adv = advantages.mean().item()
        action_abs = act_all.abs().mean().item()

        u = env.unwrapped
        mean_lift    = u._last_lift_reward.mean().item()    if hasattr(u, "_last_lift_reward")    else float("nan")
        mean_leg_still = u._last_leg_still.mean().item()   if hasattr(u, "_last_leg_still")      else float("nan")
        mean_contact = u._last_contact_reward.mean().item() if hasattr(u, "_last_contact_reward") else float("nan")
        mean_place   = u._last_place_reward.mean().item()   if hasattr(u, "_last_place_reward")   else float("nan")
        mean_graspnet = u._last_graspnet_reward.mean().item() if hasattr(u, "_last_graspnet_reward") else float("nan")
        mean_surface = u._last_surface_contact_reward.mean().item() if hasattr(u, "_last_surface_contact_reward") else float("nan")

        stats = {
            "iter": it,
            "mean_reward": mean_rew,
            "std_reward": std_rew,
            "min_reward": min_rew,
            "max_reward": max_rew,
            "success_rate": success_rate,
            "mean_lift_reward": mean_lift,
            "mean_leg_still": mean_leg_still,
            "mean_contact_reward": mean_contact,
            "mean_place_reward": mean_place,
            "mean_graspnet_reward": mean_graspnet,
            "mean_surface_contact_reward": mean_surface,
            "mean_value": mean_value,
            "mean_return": mean_return,
            "mean_advantage": mean_adv,
            "action_abs_mean": action_abs,
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "total_loss": last_total_loss,
            "grad_norm": last_grad_norm,
        }

        if metrics_path is not None:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")

        if it == 1 or it % 10 == 0:
            print(
                f"  iter {it:5d}/{max_iters}  "
                f"reward={mean_rew:.4f}±{std_rew:.4f}  "
                f"lift={mean_lift:.4f}  contact={mean_contact:.4f}  surface={mean_surface:.4f}  "
                f"place={mean_place:.4f}  graspnet={mean_graspnet:.4f}  "
                f"success={success_rate*100:.1f}%  "
                f"v_loss={last_value_loss:.4f}  "
                f"pi_loss={last_policy_loss:.4f}  "
                f"entropy={last_entropy:.3f}  "
                f"grad={last_grad_norm:.2f}"
            )

        if writer is not None:
            writer.add_scalar("train/mean_reward", mean_rew, it)
            writer.add_scalar("train/std_reward", std_rew, it)
            writer.add_scalar("train/min_reward", min_rew, it)
            writer.add_scalar("train/max_reward", max_rew, it)
            writer.add_scalar("train/success_rate", success_rate, it)
            writer.add_scalar("train/mean_lift_reward", mean_lift, it)
            writer.add_scalar("train/mean_leg_still", mean_leg_still, it)
            writer.add_scalar("train/mean_contact_reward", mean_contact, it)
            writer.add_scalar("train/mean_place_reward", mean_place, it)
            writer.add_scalar("train/mean_graspnet_reward", mean_graspnet, it)
            writer.add_scalar("train/mean_surface_contact_reward", mean_surface, it)
            writer.add_scalar("train/mean_value", mean_value, it)
            writer.add_scalar("train/mean_return", mean_return, it)
            writer.add_scalar("train/mean_advantage", mean_adv, it)
            writer.add_scalar("train/action_abs_mean", action_abs, it)
            writer.add_scalar("train/policy_loss", last_policy_loss, it)
            writer.add_scalar("train/value_loss", last_value_loss, it)
            writer.add_scalar("train/entropy", last_entropy, it)
            writer.add_scalar("train/total_loss", last_total_loss, it)
            writer.add_scalar("train/grad_norm", last_grad_norm, it)
            writer.flush()

        if it % save_interval == 0:
            ckpt = os.path.join(log_dir, f"grasp_pose_{it}.pt")
            torch.save({"model": ac.state_dict(), "iteration": it}, ckpt)

    return ac


def _load_checkpoint(ac, path: str, device, strict_full: bool = True,
                     encoder_only: bool = False) -> str:
    """Load grasp_pose checkpoint; partial load when expanding action head (e.g. 5-D → 6-D)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    own = ac.state_dict()
    if encoder_only:
        loaded = {
            k: v for k, v in state.items()
            if ("encoder" in k) and k in own and own[k].shape == v.shape
        }
        ac.load_state_dict(loaded, strict=False)
        return f"encoder-only ({len(loaded)}/{len(own)} keys)"
    if strict_full:
        try:
            ac.load_state_dict(state, strict=True)
            return "full"
        except RuntimeError:
            pass
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
    ac.load_state_dict(loaded, strict=False)
    n = len(loaded)
    return f"partial ({n}/{len(own)} keys, xyz warm-start)"


def main():
    device = args.device  # from AppLauncher: cuda:0 (default) or cpu
    print(f"[grasp-pose-train] policy device={device}  envs={args.num_envs}")

    env_cfg = GraspPoseEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    env_cfg.use_real_objects = args.use_real_objects
    if args.two_point:
        env_cfg.two_point_grasp = True
    elif args.path_a:
        env_cfg.two_point_grasp = False
    if not args.enable_cameras:
        env_cfg.use_camera_pc = False
    # Honest physics for two-point: no kinematic approach cheat (train == eval).
    if env_cfg.two_point_grasp:
        env_cfg.ik_write_joint_state = False
        env_cfg.ik_write_approach_only = True
        env_cfg.contact_carry = False
        env_cfg.require_contact_for_lift_reward = True
    print(f"[grasp-pose-train] use_real_objects={args.use_real_objects}")
    print(f"[grasp-pose-train] use_camera_pc={env_cfg.use_camera_pc}")
    print(f"[grasp-pose-train] simulation device={env_cfg.sim.device}")
    print(
        f"[grasp-pose-train] two_point_grasp={env_cfg.two_point_grasp}  "
        f"NUM_ACTIONS={NUM_ACTIONS}  "
        f"ik_write={env_cfg.ik_write_joint_state}/approach_only={env_cfg.ik_write_approach_only}  "
        f"contact_carry={env_cfg.contact_carry}  "
        f"require_contact_lift={env_cfg.require_contact_for_lift_reward}  "
        f"project_pc={env_cfg.project_grasp_to_pc}  "
        f"surface_w={env_cfg.surface_contact_reward_weight}  "
        f"phases A/C/D={N_APPROACH}/{N_CLOSE}/{N_DESCEND}"
    )

    env = GraspPoseEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    run_name = f"grasp_pose_envs{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[grasp-pose-train] logging to {log_dir}")

    ac = GraspPoseActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=NUM_ACTIONS,
        num_pc_points=args.num_pc_points,
        pc_embed_dim=args.pc_embed_dim,
        init_noise_std=0.5,
    ).to(device)

    if args.resume and os.path.exists(args.resume):
        mode = _load_checkpoint(ac, args.resume, device, encoder_only=False)
        print(f"[grasp-pose-train] resume: {mode} from {args.resume}")
        print(f"[grasp-pose-train] two_point_grasp={env_cfg.two_point_grasp}  "
              f"NUM_ACTIONS={NUM_ACTIONS}  project_grasp_to_pc={env_cfg.project_grasp_to_pc}")
    elif args.pretrain and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location=device, weights_only=False)
        ac.actor_encoder.load_state_dict(ckpt["encoder"])
        ac.critic_encoder.load_state_dict(ckpt["encoder"])
        loaded_head = False
        # 2-point BC ckpt: also warm-start actor_head (critic head stays random).
        if (
            ckpt.get("mode") == "two_point"
            and "head" in ckpt
            and getattr(env_cfg, "two_point_grasp", False)
        ):
            try:
                ac.actor_head.load_state_dict(ckpt["head"])
                loaded_head = True
            except RuntimeError as e:
                print(f"[grasp-pose-train] WARNING: actor_head warm-start failed: {e}")
        if loaded_head:
            print(
                f"[grasp-pose-train] warm start: encoder+actor_head (2pt BC) "
                f"from {args.pretrain}"
            )
        else:
            print(f"[grasp-pose-train] warm start: encoder loaded from {args.pretrain}")
    elif args.pretrain:
        print(f"[grasp-pose-train] WARNING: pretrain checkpoint not found: {args.pretrain}")
    else:
        print("[grasp-pose-train] cold start: random initialization")

    n_params = sum(p.numel() for p in ac.parameters())
    print(f"[grasp-pose-train] GraspPoseActorCritic ({n_params:,} params)")

    tb_dir = os.path.join(log_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    print(f"[grasp-pose-train] tensorboard → {tb_dir}")
    print(f"[grasp-pose-train] metrics log  → {metrics_path}")

    train_ppo(
        env, ac, device, log_dir,
        max_iters=args.max_iters,
        writer=writer,
        metrics_path=metrics_path,
    )

    writer.close()

    final_path = os.path.join(log_dir, "grasp_pose_final.pt")
    torch.save({"model": ac.state_dict(), "iteration": args.max_iters}, final_path)
    print(f"[grasp-pose-train] saved → {final_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
