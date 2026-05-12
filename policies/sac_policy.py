"""
Soft Actor-Critic (SAC) policy implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class GaussianActor(nn.Module):
    """Actor network for SAC with Gaussian policy"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs: Observation tensor
            
        Returns:
            mu: Mean of Gaussian
            log_std: Log standard deviation
        """
        x = self.activation(self.fc1(obs))
        x = self.activation(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std


class QNetwork(nn.Module):
    """Q-network for SAC"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


class SACPolicy:
    """
    Soft Actor-Critic policy
    
    Hyperparameters:
    - learning_rate: Learning rate for optimizer
    - gamma: Discount factor
    - tau: Target network update rate
    - alpha_lr: Learning rate for entropy coefficient
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 device: str = "cuda:0",
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha_lr: float = 3e-4,
                 auto_entropy_tuning: bool = True):
        """
        Initialize SAC policy
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            device: Device to use
            learning_rate: Actor/Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha_lr: Entropy coefficient learning rate
            auto_entropy_tuning: Whether to auto-tune entropy coefficient
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Networks
        self.actor = GaussianActor(obs_dim, action_dim).to(device)
        self.q1 = QNetwork(obs_dim, action_dim).to(device)
        self.q2 = QNetwork(obs_dim, action_dim).to(device)
        self.q1_target = QNetwork(obs_dim, action_dim).to(device)
        self.q2_target = QNetwork(obs_dim, action_dim).to(device)
        
        # Copy weights
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=learning_rate)
        
        # Entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Select action from observation
        
        Args:
            obs: Observation tensor
            deterministic: If True, use mean action; else sample
            
        Returns:
            action: Selected action
        """
        mu, log_std = self.actor(obs)
        
        if deterministic:
            return torch.tanh(mu)
        
        std = log_std.exp()
        z = torch.randn_like(std)
        action = torch.tanh(mu + std * z)
        return action
    
    def save(self, path: str):
        """Save policy networks to file"""
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load policy networks from file"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
