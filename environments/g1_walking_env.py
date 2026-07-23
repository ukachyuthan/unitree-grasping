"""
Base environment for G1 walking in Isaac Gym
"""
from typing import Dict, Tuple
import numpy as np
from abc import ABC, abstractmethod


class G1WalkingEnv(ABC):
    """
    Abstract base class for G1 walking environments.
    
    Handles:
    - Robot setup and control
    - Observation/action spaces
    - Reward calculation
    - Episode management
    """
    
    def __init__(self, 
                 num_envs: int = 1,
                 device: str = "cuda:0",
                 terrain_config: Dict = None):
        """
        Initialize G1 walking environment
        
        Args:
            num_envs: Number of parallel environments
            device: Device to run on (cuda:0, cpu, etc.)
            terrain_config: Terrain configuration dictionary
        """
        self.num_envs = num_envs
        self.device = device
        self.terrain_config = terrain_config or {}
        
        # Placeholder dimensions - will be set by subclass
        self.obs_dim = None
        self.action_dim = None
        
    @abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Execute one step of environment dynamics
        
        Args:
            actions: Action tensor of shape (num_envs, action_dim)
            
        Returns:
            observations: Next observations
            rewards: Reward for each env
            dones: Done flags for each env
            infos: Additional info dictionaries
        """
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment and return initial observations
        
        Returns:
            observations: Initial observations
        """
        pass
    
    @abstractmethod
    def compute_reward(self, actions: np.ndarray) -> np.ndarray:
        """
        Compute reward based on current state and actions
        
        Args:
            actions: Action tensor
            
        Returns:
            rewards: Reward for each environment
        """
        pass
    
    def close(self):
        """Clean up resources"""
        pass
