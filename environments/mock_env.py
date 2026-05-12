"""
Mock Isaac Gym environment for testing without hardware requirements
"""
import numpy as np
from typing import Dict, Tuple
from .g1_walking_env import G1WalkingEnv


class MockG1Environment(G1WalkingEnv):
    """
    Mock G1 environment for testing without Isaac Gym
    
    Simulates robot dynamics with simple physics
    """
    
    def __init__(self,
                 num_envs: int = 4,
                 device: str = "cuda:0",
                 terrain_config: Dict = None):
        """
        Initialize mock environment
        
        Args:
            num_envs: Number of parallel environments
            device: Device to run on (ignored for mock)
            terrain_config: Terrain configuration
        """
        super().__init__(num_envs, device, terrain_config)
        
        self.num_dofs = 19  # G1 has 19 actuated joints
        self.obs_dim = 48
        self.action_dim = 19
        
        # State
        self.joint_pos = np.zeros((num_envs, self.num_dofs))
        self.joint_vel = np.zeros((num_envs, self.num_dofs))
        self.root_pos = np.zeros((num_envs, 3))
        self.root_vel = np.zeros((num_envs, 3))
        self.root_ang = np.zeros((num_envs, 3))  # Roll, pitch, yaw
        
        # Simulation params
        self.dt = 0.005
        self.max_steps = 1000
        self.step_count = np.zeros(num_envs, dtype=int)
        
        # Terrain-specific damping
        self.friction = terrain_config.get('friction', 1.0) if terrain_config else 1.0
        self.damping = terrain_config.get('damping', 0.0) if terrain_config else 0.0
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        self.joint_pos[:] = 0.0
        self.joint_vel[:] = 0.0
        self.root_pos[:] = 0.0
        self.root_vel[:] = 0.0
        self.root_ang[:] = 0.0
        self.root_pos[:, 2] = 0.5  # Height above ground
        self.step_count[:] = 0
        
        return self._get_observations()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Execute one step
        
        Args:
            actions: Joint torques or targets
            
        Returns:
            obs, rewards, dones, infos
        """
        # Simple dynamics: acceleration = action - damping * velocity
        acceleration = actions - self.damping * self.joint_vel
        
        # Integrate velocity
        self.joint_vel += acceleration * self.dt
        
        # Integrate position
        self.joint_pos += self.joint_vel * self.dt
        
        # Clamp joint positions to reasonable ranges
        self.joint_pos = np.clip(self.joint_pos, -1.57, 1.57)
        
        # Simple vertical dynamics
        gravity = -9.81
        self.root_vel[:, 2] += gravity * self.dt
        self.root_pos[:, 2] += self.root_vel[:, 2] * self.dt
        
        # Ground collision
        self.root_pos[:, 2] = np.maximum(self.root_pos[:, 2], 0.0)
        if np.any(self.root_pos[:, 2] <= 0.0):
            self.root_vel[:, 2] = 0.0
        
        # Get observations
        obs = self._get_observations()
        
        # Compute rewards
        rewards = self.compute_reward(actions)
        
        # Check termination
        self.step_count += 1
        dones = self.step_count >= self.max_steps
        
        # Reset done environments
        for i in np.where(dones)[0]:
            self._reset_env(i)
        
        infos = [{"step": self.step_count[i]} for i in range(self.num_envs)]
        
        return obs, rewards, dones, infos
    
    def _reset_env(self, env_idx: int):
        """Reset a single environment"""
        self.joint_pos[env_idx] = 0.0
        self.joint_vel[env_idx] = 0.0
        self.root_pos[env_idx] = [0.0, 0.0, 0.5]
        self.root_vel[env_idx] = 0.0
        self.root_ang[env_idx] = 0.0
        self.step_count[env_idx] = 0
    
    def _get_observations(self) -> np.ndarray:
        """Extract observations"""
        obs = np.zeros((self.num_envs, self.obs_dim))
        
        # Pack observations: [root_pos, root_vel, root_ang, joint_pos, joint_vel]
        idx = 0
        
        # Root position (3)
        obs[:, idx:idx+3] = self.root_pos
        idx += 3
        
        # Root velocity (3)
        obs[:, idx:idx+3] = self.root_vel
        idx += 3
        
        # Root orientation (3)
        obs[:, idx:idx+3] = self.root_ang
        idx += 3
        
        # Joint positions (19)
        obs[:, idx:idx+self.num_dofs] = self.joint_pos
        idx += self.num_dofs
        
        # Joint velocities (19)
        obs[:, idx:idx+self.num_dofs] = self.joint_vel
        idx += self.num_dofs
        
        return obs
    
    def compute_reward(self, actions: np.ndarray) -> np.ndarray:
        """
        Compute rewards
        
        Reward components:
        - Height reward: Penalize falling
        - Smoothness: Penalize jerky actions
        - Efficiency: Penalize large actions
        """
        rewards = np.zeros(self.num_envs)
        
        # Height reward: encourage standing upright
        height_reward = np.where(
            self.root_pos[:, 2] > 0.3,
            0.5,  # Good height
            -1.0   # Fallen
        )
        
        # Smoothness penalty: penalize large joint velocities
        velocity_penalty = -0.01 * np.sum(np.abs(self.joint_vel), axis=1)
        
        # Action penalty: penalize large actions
        action_penalty = -0.001 * np.sum(np.abs(actions), axis=1)
        
        rewards = height_reward + velocity_penalty + action_penalty
        
        return rewards
    
    def close(self):
        """Close environment"""
        pass
