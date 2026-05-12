"""
Replay buffer for SAC training
"""
import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms
    """
    
    def __init__(self,
                 buffer_size: int = 1000000,
                 obs_dim: int = None,
                 action_dim: int = None):
        """
        Initialize replay buffer
        
        Args:
            buffer_size: Maximum number of transitions to store
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.idx = 0
        self.size = 0
        
        # Pre-allocate arrays
        if obs_dim is not None and action_dim is not None:
            self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
            self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
            self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
            self.rewards = np.zeros(buffer_size, dtype=np.float32)
            self.dones = np.zeros(buffer_size, dtype=np.bool_)
    
    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_obs: np.ndarray,
            done: bool):
        """
        Add transition to buffer
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample random batch from buffer
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            obs, actions, rewards, next_obs, dones
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.size == self.buffer_size
