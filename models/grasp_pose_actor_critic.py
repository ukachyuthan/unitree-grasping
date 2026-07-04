"""
Actor-Critic for single-shot grasp pose prediction.

Policy interface:
    input  : point cloud  (B, N_PTS*3)  — object surface in robot-base frame
    output : grasp position  (B, 3)     — [x, y, z] scaled to workspace

The output is in tanh space [-1, 1]^3; the env scales it to the actual
workspace bounds before executing the IK.

Critic has the same PointNet encoder but with an independent copy of weights,
predicting scalar value for the current object/situation.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from models.pointnet_encoder import PointNetEncoder


class GraspPoseActorCritic(nn.Module):
    """RSL-RL-compatible actor-critic for grasp position prediction."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int = 3,
        num_pc_points: int = 128,
        pc_embed_dim: int = 128,
        init_noise_std: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        nn.Module.__init__(self)

        self.num_pc_points = num_pc_points
        self.num_actions = num_actions

        # Actor: PC → embed → position (tanh-bounded)
        self.actor_encoder = PointNetEncoder(embed_dim=pc_embed_dim)
        self.actor_head = nn.Sequential(
            nn.Linear(pc_embed_dim, 128), nn.ELU(),
            nn.Linear(128, 64),           nn.ELU(),
            nn.Linear(64, num_actions),
            nn.Tanh(),
        )

        # Critic: independent PC encoder → scalar value
        self.critic_encoder = PointNetEncoder(embed_dim=pc_embed_dim)
        self.critic_head = nn.Sequential(
            nn.Linear(pc_embed_dim, 128), nn.ELU(),
            nn.Linear(128, 64),           nn.ELU(),
            nn.Linear(64, 1),
        )

        # Learnable per-dimension noise std (starts at init_noise_std)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _encode_pc(self, obs: torch.Tensor, encoder: PointNetEncoder) -> torch.Tensor:
        """obs is flat (B, N*3); reshape to (B, N, 3) before encoding."""
        B = obs.shape[0]
        pc = obs[:, : self.num_pc_points * 3].view(B, self.num_pc_points, 3)
        return encoder(pc)

    # ── RSL-RL interface ──────────────────────────────────────────────────────

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
        feat = self._encode_pc(observations, self.actor_encoder)
        mean = self.actor_head(feat)
        std  = self.std.clamp(min=1e-4).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        feat = self._encode_pc(observations, self.actor_encoder)
        return self.actor_head(feat)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        feat = self._encode_pc(critic_observations, self.critic_encoder)
        return self.critic_head(feat)
