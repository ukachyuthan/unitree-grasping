"""Actor-critic for residual grasp control: PointNet(PC) + proprio → grasp + residuals."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from models.pointnet_encoder import PointNetEncoder, build_mlp

NUM_GRASP_ACTIONS = 5
NUM_RESIDUAL_ACTIONS = 7
NUM_ACTIONS = NUM_GRASP_ACTIONS + NUM_RESIDUAL_ACTIONS


class GraspPoseResidualActorCritic(nn.Module):
    """RSL-RL-compatible policy with separate grasp (PC-only) and residual heads."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int = NUM_ACTIONS,
        num_pc_points: int = 128,
        pc_embed_dim: int = 128,
        actor_hidden_dims: list | None = None,
        critic_hidden_dims: list | None = None,
        init_grasp_noise_std: float = 0.15,
        init_residual_noise_std: float = 0.08,
        **kwargs,
    ):
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [128, 64]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128]

        self.num_pc_points = num_pc_points
        self.num_actions = num_actions
        pc_flat = num_pc_points * 3
        proprio_dim = num_actor_obs - pc_flat
        critic_proprio = num_critic_obs - pc_flat

        self.actor_pc_encoder = PointNetEncoder(embed_dim=pc_embed_dim)
        # Matches Path A grasp_pose_actor_critic architecture for warm-start.
        self.grasp_head = nn.Sequential(
            nn.Linear(pc_embed_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, NUM_GRASP_ACTIONS),
            nn.Tanh(),
        )
        self.residual_head = build_mlp(
            pc_embed_dim + proprio_dim, actor_hidden_dims, NUM_RESIDUAL_ACTIONS
        )

        self.critic_pc_encoder = PointNetEncoder(embed_dim=pc_embed_dim)
        self.critic = build_mlp(pc_embed_dim + critic_proprio, critic_hidden_dims, 1)

        self.grasp_std = nn.Parameter(init_grasp_noise_std * torch.ones(NUM_GRASP_ACTIONS))
        self.residual_std = nn.Parameter(init_residual_noise_std * torch.ones(NUM_RESIDUAL_ACTIONS))
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pc_flat = self.num_pc_points * 3
        pc = obs[:, :pc_flat].view(-1, self.num_pc_points, 3)
        prop = obs[:, pc_flat:]
        return pc, prop

    def _action_std(self) -> torch.Tensor:
        g = self.grasp_std.clamp(1e-4, 0.5)
        r = self.residual_std.clamp(1e-4, 0.25)
        return torch.cat([g, r])

    def _action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        pc, prop = self._split_obs(obs)
        embed = self.actor_pc_encoder(pc)
        grasp = self.grasp_head(embed)
        residual = torch.tanh(self.residual_head(torch.cat([embed, prop], dim=-1)))
        return torch.cat([grasp, residual], dim=-1)

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
        mean = self._action_mean(observations)
        std = self._action_std().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        return self._action_mean(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        pc, prop = self._split_obs(critic_observations)
        embed = self.critic_pc_encoder(pc)
        return self.critic(torch.cat([embed, prop], dim=-1))


def load_grasp_pose_warm_start(
    ac: GraspPoseResidualActorCritic,
    path: str,
    device: torch.device | str,
    freeze_grasp: bool = False,
) -> int:
    """Load PointNet encoder + grasp head from a Path A grasp_pose_*.pt checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    loaded = 0

    mappings = [
        ("actor_encoder.", "actor_pc_encoder"),
        ("critic_encoder.", "critic_pc_encoder"),
    ]
    for src_prefix, dst_attr in mappings:
        sub = {k[len(src_prefix) :]: v for k, v in state.items() if k.startswith(src_prefix)}
        if sub:
            getattr(ac, dst_attr).load_state_dict(sub, strict=True)
            loaded += len(sub)

    head_sub = {k[len("actor_head.") :]: v for k, v in state.items() if k.startswith("actor_head.")}
    if head_sub:
        own = ac.grasp_head.state_dict()
        for k, v in head_sub.items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                own[k] = v
            elif k in ("4.weight", "4.bias"):
                n = min(v.shape[0], own[k].shape[0])
                own[k][:n] = v[:n]
            else:
                continue
            loaded += v.numel()
        ac.grasp_head.load_state_dict(own)

    if "std" in state:
        if state["std"].shape == ac.grasp_std.shape:
            ac.grasp_std.data.copy_(state["std"].clamp(1e-4, 0.5))
        elif state["std"].numel() <= ac.grasp_std.numel():
            n = state["std"].numel()
            ac.grasp_std.data[:n].copy_(state["std"].clamp(1e-4, 0.5))
            # Unloaded tilt/roll dims keep their init (not zero).
            loaded += n

    if freeze_grasp:
        for p in ac.actor_pc_encoder.parameters():
            p.requires_grad = False
        for p in ac.grasp_head.parameters():
            p.requires_grad = False
        ac.grasp_std.requires_grad = False

    return loaded
