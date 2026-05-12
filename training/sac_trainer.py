"""
SAC training loop
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from tqdm import tqdm


class SACTrainer:
    """
    Soft Actor-Critic trainer
    
    Implements the SAC algorithm for continuous control
    """
    
    def __init__(self,
                 policy,
                 env,
                 replay_buffer,
                 device: str = "cuda:0"):
        """
        Initialize SAC trainer
        
        Args:
            policy: SACPolicy instance
            env: Training environment
            replay_buffer: ReplayBuffer instance
            device: Device to train on
        """
        self.policy = policy
        self.env = env
        self.buffer = replay_buffer
        self.device = device
    
    def train_step(self,
                   batch_size: int = 256,
                   gamma: float = 0.99) -> Dict[str, float]:
        """
        Perform one training step (update on batch)
        
        Args:
            batch_size: Batch size for training
            gamma: Discount factor
            
        Returns:
            Dictionary with training metrics
        """
        if self.buffer.size < batch_size:
            return {}
        
        # Sample batch
        obs, actions, rewards, next_obs, dones = self.buffer.sample(batch_size)
        
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update Q-networks
        with torch.no_grad():
            next_mu, next_log_std = self.policy.actor(next_obs)
            next_std = next_log_std.exp()
            next_z = torch.randn_like(next_std)
            next_action = torch.tanh(next_mu + next_std * next_z)
            
            next_q1 = self.policy.q1_target(next_obs, next_action)
            next_q2 = self.policy.q2_target(next_obs, next_action)
            next_q = torch.min(next_q1, next_q2)
            
            target_q = rewards + (1 - dones) * gamma * next_q
        
        q1_loss = F.mse_loss(self.policy.q1(obs, actions), target_q)
        q2_loss = F.mse_loss(self.policy.q2(obs, actions), target_q)
        
        self.policy.q1_opt.zero_grad()
        q1_loss.backward()
        self.policy.q1_opt.step()
        
        self.policy.q2_opt.zero_grad()
        q2_loss.backward()
        self.policy.q2_opt.step()
        
        # Update actor
        mu, log_std = self.policy.actor(obs)
        std = log_std.exp()
        z = torch.randn_like(std)
        action = torch.tanh(mu + std * z)
        
        q1_pi = self.policy.q1(obs, action)
        q2_pi = self.policy.q2(obs, action)
        q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.policy.alpha * (-torch.log(std + 1e-6).sum(dim=1, keepdim=True)) - q_pi).mean()
        
        self.policy.actor_opt.zero_grad()
        actor_loss.backward()
        self.policy.actor_opt.step()
        
        # Update entropy coefficient
        if self.policy.auto_entropy_tuning:
            entropy = -torch.log(std + 1e-6).sum(dim=1, keepdim=True)
            alpha_loss = -(self.policy.log_alpha * (entropy + self.policy.target_entropy).detach()).mean()
            
            self.policy.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.policy.alpha_opt.step()
        
        # Soft update target networks
        for param, target_param in zip(self.policy.q1.parameters(), self.policy.q1_target.parameters()):
            target_param.data.copy_(self.policy.tau * param.data + (1 - self.policy.tau) * target_param.data)
        
        for param, target_param in zip(self.policy.q2.parameters(), self.policy.q2_target.parameters()):
            target_param.data.copy_(self.policy.tau * param.data + (1 - self.policy.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.policy.alpha.item(),
        }
    
    def collect_experience(self, num_steps: int) -> Dict[str, float]:
        """
        Collect experience in environment
        
        Args:
            num_steps: Number of environment steps to collect
            
        Returns:
            Dictionary with collection metrics
        """
        obs = self.env.reset()
        episode_reward = 0.0
        num_episodes = 0
        
        for _ in range(num_steps):
            # Select action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.policy.select_action(obs_tensor, deterministic=False)
            action = action.cpu().numpy()[0]
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                num_episodes += 1
                obs = self.env.reset()
                episode_reward = 0.0
        
        return {
            'episodes_collected': num_episodes,
            'steps_collected': num_steps,
        }
