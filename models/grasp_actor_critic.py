"""
Custom Actor-Critic with a PointNet encoder backbone.

Replaces the standard RSL-RL ActorCritic (flat MLP) with:

    obs (OBS_DIM) ──┬── [0 : PC_FLAT]   → PointNetEncoder  → embed_dim
                    └── [PC_FLAT : ]     → identity
                         concat (embed_dim + PROPRIO_DIM)
                         → actor MLP  → action_dim
                         → critic MLP → 1

Both actor and critic have their OWN PointNet encoder (independent weights).
This mirrors how RSL-RL's ActorCritic works with separate actor / critic MLPs
and avoids value-function gradient interference on the feature extractor.

Compatible with rsl_rl.runners.OnPolicyRunner:
  runner = OnPolicyRunner(env, runner_cfg.to_dict(),
                          actor_critic_class=PointNetActorCritic,
                          log_dir=..., device=...)
Pass extra kwargs via runner_cfg.policy or directly in runner kwargs:
  {"num_pc_points": 128, "pc_embed_dim": 128,
   "actor_hidden_dims": [256, 128], "critic_hidden_dims": [256, 128]}
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from models.pointnet_encoder import PointNetEncoder, build_mlp


class PointNetActorCritic(ActorCritic):
    """
    Drop-in replacement for rsl_rl.modules.ActorCritic that prepends a
    PointNet encoder before the MLP policy / value heads.

    Extra __init__ kwargs (forwarded via **kwargs from the runner):
        num_pc_points   : int  — must match grasping.g1_grasp_env_cfg.NUM_PC_POINTS
        pc_embed_dim    : int  — PointNet output width  (default 128)
        actor_hidden_dims  : list[int]  (default [256, 128])
        critic_hidden_dims : list[int]  (default [256, 128])
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_pc_points: int = 128,
        pc_embed_dim: int = 128,
        actor_hidden_dims: list = None,
        critic_hidden_dims: list = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # Bypass parent __init__ to avoid building its MLP; call nn.Module directly
        nn.Module.__init__(self)

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128]

        self.num_pc_points = num_pc_points
        pc_flat = num_pc_points * 3
        proprio_dim = num_actor_obs - pc_flat   # PROPRIO_DIM

        # ── Actor ────────────────────────────────────────────────────────────
        self.actor_pc_encoder = PointNetEncoder(embed_dim=pc_embed_dim)
        actor_in = pc_embed_dim + proprio_dim
        self.actor = build_mlp(actor_in, actor_hidden_dims, num_actions)

        # ── Critic ───────────────────────────────────────────────────────────
        # num_critic_obs may differ from num_actor_obs (privileged critic info)
        # If it's the same layout, reuse the same split.
        critic_pc_flat = num_pc_points * 3
        critic_proprio  = num_critic_obs - critic_pc_flat
        self.critic_pc_encoder = PointNetEncoder(embed_dim=pc_embed_dim)
        critic_in = pc_embed_dim + critic_proprio
        self.critic = build_mlp(critic_in, critic_hidden_dims, 1)

        # ── Noise ────────────────────────────────────────────────────────────
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _encode_obs(self, obs: torch.Tensor, pc_encoder: PointNetEncoder) -> torch.Tensor:
        """Split obs → [pc | proprio], encode pc, concat."""
        pc_flat = self.num_pc_points * 3
        pc   = obs[:, :pc_flat].view(-1, self.num_pc_points, 3)   # (B, N, 3)
        prop = obs[:, pc_flat:]                                    # (B, PROPRIO_DIM)
        emb  = pc_encoder(pc)                                      # (B, embed_dim)
        return torch.cat([emb, prop], dim=-1)                      # (B, embed+proprio)

    # ── Required ActorCritic interface ───────────────────────────────────────
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        features = self._encode_obs(observations, self.actor_pc_encoder)
        mean = self.actor(features)
        std = (
            self.std.expand_as(mean)
            if self.noise_std_type == "scalar"
            else torch.exp(self.log_std).expand_as(mean)
        )
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        features = self._encode_obs(observations, self.actor_pc_encoder)
        return self.actor(features)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        features = self._encode_obs(critic_observations, self.critic_pc_encoder)
        return self.critic(features)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
