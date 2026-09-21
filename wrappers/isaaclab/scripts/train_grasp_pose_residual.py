#!/usr/bin/env python3
"""PPO training for residual predictive grasp control (Path A + closed-loop corrections)."""

import argparse

from isaaclab.app import AppLauncher

_DEFAULT_CKPT = (
    "data/grasp_logs/grasp_pose_envs64_20260729_000049/grasp_pose_400.pt"
)

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iters", type=int, default=3000)
parser.add_argument("--log_dir", type=str, default="data/grasp_logs")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--pretrain",
    type=str,
    default=_DEFAULT_CKPT,
    help="Path A checkpoint for encoder + grasp-head warm start",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Full residual checkpoint to resume training",
)
parser.add_argument(
    "--freeze_grasp_iters",
    type=int,
    default=500,
    help="Freeze grasp head for first N iters while residual learns "
         "(raise when use_antipodal_base + freeze_grasp_to_antipodal)",
)
parser.add_argument("--num_pc_points", type=int, default=128)
parser.add_argument("--pc_embed_dim", type=int, default=128)
parser.add_argument("--num_steps_per_env", type=int, default=48,
                    help="Policy steps collected per env per iteration")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import bootstrap

bootstrap()

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from envs.grasp_pose_residual_env_cfg import OBS_DIM, NUM_ACTIONS
from envs.grasp_pose_residual_env import GraspPoseResidualEnv, GraspPoseResidualEnvCfg
from models.grasp_pose_residual_actor_critic import (
    GraspPoseResidualActorCritic,
    load_grasp_pose_warm_start,
)


def _obs_tensor(obs_td, device):
    return obs_td["policy"].to(device)


def _resolve_path(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return str(p.resolve())
    # Relative to wrappers/isaaclab (script cwd).
    alt = Path.cwd() / path
    if alt.is_file():
        return str(alt.resolve())
    return path


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """rewards, values, dones: (T, N). Returns returns, advantages (T, N)."""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else torch.zeros(N, device=rewards.device)
        nonterminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return returns, advantages


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


def _set_grasp_trainable(ac: GraspPoseResidualActorCritic, trainable: bool):
    for p in ac.actor_pc_encoder.parameters():
        p.requires_grad = trainable
    for p in ac.grasp_head.parameters():
        p.requires_grad = trainable
    ac.grasp_std.requires_grad = trainable


def train_ppo(env, ac, device, log_dir, max_iters, writer=None, metrics_path=None,
              success_threshold=0.5, lr=3e-4,
              num_steps_per_env=48, num_epochs=5, num_mini_batches=4,
              clip_param=0.2, value_coef=0.5, entropy_coef=0.005,
              max_grad_norm=1.0, save_interval=25, gamma=0.99, lam=0.95,
              freeze_grasp_iters=0):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ac.parameters()), lr=lr)
    num_envs = env.num_envs
    batch_size = num_envs * num_steps_per_env
    mini_batch_size = max(1, batch_size // num_mini_batches)

    for it in range(1, max_iters + 1):
        if freeze_grasp_iters > 0:
            _set_grasp_trainable(ac, it > freeze_grasp_iters)
            if it == freeze_grasp_iters + 1:
                optimizer = torch.optim.Adam(ac.parameters(), lr=lr)
                print(f"[residual-train] unfreezing grasp head at iter {it}")

        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        obs_td = env.get_observations()
        for _ in range(num_steps_per_env):
            obs = _obs_tensor(obs_td, device)
            with torch.no_grad():
                ac.act(obs)
                actions = ac.distribution.sample()
                logp = ac.get_actions_log_prob(actions)
                values = ac.evaluate(obs).squeeze(-1)

            obs_td, rew, terminated, _ = env.step(actions)
            done = terminated

            obs_buf.append(obs)
            act_buf.append(actions)
            logp_buf.append(logp)
            rew_buf.append(rew.to(device))
            val_buf.append(values)
            done_buf.append(done.to(device).float())

        obs_all = torch.cat(obs_buf, dim=0)
        act_all = torch.cat(act_buf, dim=0)
        logp_all = torch.cat(logp_buf, dim=0)
        rew_stacked = torch.stack(rew_buf, dim=0)
        val_stacked = torch.stack(val_buf, dim=0)
        done_stacked = torch.stack(done_buf, dim=0)

        returns_stacked, adv_stacked = compute_gae(rew_stacked, val_stacked, done_stacked, gamma, lam)

        rew_all = rew_stacked.reshape(-1)
        returns_all = returns_stacked.reshape(-1)
        adv_all = adv_stacked.reshape(-1).detach()
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

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
                    returns_all[idx], adv_all[idx],
                    clip_param, value_coef, entropy_coef, max_grad_norm,
                )

        mean_rew = rew_all.mean().item()
        std_rew = rew_all.std(unbiased=False).item()

        # Success = terminal sparse lift reward (dense steps are ~0.02–0.15).
        done_flat = done_stacked.reshape(-1) > 0.5
        terminal_rew = rew_all[done_flat]
        if terminal_rew.numel() > 0:
            success_rate = (terminal_rew >= success_threshold).float().mean().item()
            mean_terminal = terminal_rew.mean().item()
        else:
            success_rate = 0.0
            mean_terminal = float("nan")

        u = env.unwrapped
        mean_lift = u.terminal_lift_reward.mean().item() if hasattr(u, "terminal_lift_reward") else float("nan")

        stats = {
            "iter": it,
            "mean_reward": mean_rew,
            "std_reward": std_rew,
            "mean_terminal_reward": mean_terminal,
            "mean_lift": mean_lift,
            "success_rate": success_rate,
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "grad_norm": last_grad_norm,
        }
        if metrics_path:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")

        if it == 1 or it % 10 == 0:
            print(
                f"  iter {it:5d}/{max_iters}  dense={mean_rew:.4f}±{std_rew:.4f}  "
                f"terminal={mean_terminal:.3f}  lift={mean_lift:.3f}  "
                f"success={success_rate*100:.1f}%  entropy={last_entropy:.3f}  "
                f"grad={last_grad_norm:.2f}"
            )

        if writer:
            writer.add_scalar("train/mean_reward", mean_rew, it)
            writer.add_scalar("train/success_rate", success_rate, it)
            writer.add_scalar("train/mean_lift", mean_lift, it)
            writer.add_scalar("train/entropy", last_entropy, it)
            writer.flush()

        if it % save_interval == 0:
            torch.save({"model": ac.state_dict(), "iteration": it},
                       os.path.join(log_dir, f"grasp_residual_{it}.pt"))

    return ac


def main():
    device = args.device
    print(f"[residual-train] device={device}  envs={args.num_envs}  "
          f"steps/env={args.num_steps_per_env}")

    env_cfg = GraspPoseResidualEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    # Method 2 defaults are already True on the cfg; keep Path A action semantics.
    env_cfg.two_point_grasp = False
    # Match Path A train: skip camera sensors unless --enable_cameras (mesh PC only).
    if not args.enable_cameras:
        env_cfg.use_camera_pc = False
    print(
        f"[residual-train] antipodal_base={env_cfg.use_antipodal_base}  "
        f"freeze_grasp_to_antipodal={env_cfg.freeze_grasp_to_antipodal}  "
        f"place_prob={env_cfg.task_mode_prob_place}  "
        f"use_camera_pc={env_cfg.use_camera_pc}"
    )

    env = GraspPoseResidualEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    run_name = f"grasp_residual_envs{args.num_envs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[residual-train] logging to {log_dir}")

    ac = GraspPoseResidualActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=NUM_ACTIONS,
        num_pc_points=args.num_pc_points,
        pc_embed_dim=args.pc_embed_dim,
    ).to(device)

    resume_path = _resolve_path(args.resume)
    pretrain_path = _resolve_path(args.pretrain)

    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        ac.load_state_dict(ckpt["model"], strict=True)
        print(f"[residual-train] resumed full checkpoint from {resume_path}")
    elif pretrain_path and os.path.isfile(pretrain_path):
        n = load_grasp_pose_warm_start(
            ac, pretrain_path, device, freeze_grasp=args.freeze_grasp_iters > 0
        )
        print(f"[residual-train] warm-started {n} params from {pretrain_path}")
    else:
        print("[residual-train] cold start (no checkpoint found)")

    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
    metrics_path = os.path.join(log_dir, "metrics.jsonl")

    train_ppo(
        env, ac, device, log_dir,
        max_iters=args.max_iters,
        writer=writer,
        metrics_path=metrics_path,
        num_steps_per_env=args.num_steps_per_env,
        freeze_grasp_iters=args.freeze_grasp_iters,
    )

    final_path = os.path.join(log_dir, "grasp_residual_final.pt")
    torch.save({"model": ac.state_dict(), "iteration": args.max_iters}, final_path)
    print(f"[residual-train] saved → {final_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
