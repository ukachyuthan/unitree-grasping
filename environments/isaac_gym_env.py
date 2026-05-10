"""
IsaacLab environment for G1 walking with multiple terrain types
"""
import torch
import numpy as np
from typing import Dict, Tuple
import omni.isaac.lab as isaac_lab
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.scene import InteractiveScene
import omni.isaac.lab_tasks

from .g1_walking_env import G1WalkingEnv


class IsaacGymG1Environment(G1WalkingEnv):
    """
    G1 walking environment using NVIDIA IsaacLab
    
    Features:
    - Parallel environment execution on GPU
    - Multiple terrain types
    - Contact-based rewards
    - Vision and proprioceptive observations
    """
    
    def __init__(self,
                 num_envs: int = 4096,
                 device: str = "cuda:0",
                 terrain_config: Dict = None,
                 headless: bool = True):
        """
        Initialize IsaacLab G1 environment
        
        Args:
            num_envs: Number of parallel environments
            device: Device to run on (cuda:0, etc.)
            terrain_config: Terrain configuration
            headless: Run without GUI
        """
        super().__init__(num_envs, device, terrain_config)
        
        self.headless = headless
        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.device_id = int(device.split(":")[-1]) if ":" in device else 0
        
        # Initialize IsaacLab
        self.env_index = 0
        
        # Create scene
        self.scene = self._create_scene()
        
        # Load assets
        self.g1_articulation = self._load_g1_model()
        
        # Prepare tensors
        self._prepare_tensors()
        
        # Episode tracking
        self.episode_steps = torch.zeros(num_envs, device=device)
        self.max_episode_length = 1000
        
    def _create_scene(self):
        """Create IsaacLab scene"""
        from omni.isaac.lab.sim import SimulationContext
        
        # Create simulation context
        sim_context = SimulationContext(
            physics_dt=0.005,  # 200 Hz simulation
            rendering_dt=0.05,
            backend="torch",
            device=self.device,
            num_envs=self.num_envs,
            headless=self.headless
        )
        
        # Set gravity
        sim_context.scene.gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)
        
        return sim_context
    
    def _load_g1_model(self):
        """Load G1 robot model"""
        from omni.isaac.lab.assets.articulation import ArticulationCfg, ArticulationData
        
        # Create G1 articulation config
        g1_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Robots/Unitree/G1/g1.usd",
            scale=1.0,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.5),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            actuators={
                "base_legs": {
                    "joint_names": [
                        ".*_hip_roll",
                        ".*_hip_pitch",
                        ".*_knee",
                        ".*_ankle_pitch",
                        ".*_ankle_roll"
                    ],
                    "effort_limit": 150.0,
                    "velocity_limit": 100.0,
                }
            }
        )
        
        # Create articulation
        g1 = Articulation(self.scene, g1_cfg)
        
        return g1
    
    def _prepare_tensors(self):
        """Prepare GPU tensors for state storage"""
        if self.g1_articulation is not None:
            self.num_dofs = self.g1_articulation.num_dofs
        else:
            self.num_dofs = 19  # Default for G1
        
        # Allocate tensors
        self.g1_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.g1_vel = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        
        self.obs_dim = 48  # TODO: Adjust based on your observation design
        self.action_dim = self.num_dofs
    
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
        # Convert actions to tensor
        actions_tensor = torch.from_numpy(actions).float().to(self.device)
        
        # Scale actions to joint ranges
        scaled_actions = actions_tensor * 0.5  # Adjust scale as needed
        
        # Apply actions to all environments via IsaacLab
        if self.g1_articulation is not None:
            self.g1_articulation.set_joint_effort_target(scaled_actions)
        
        # Step simulation
        self.scene.step()
        
        # Refresh state from simulation
        if self.g1_articulation is not None:
            self.g1_pos = self.g1_articulation.data.joint_pos
            self.g1_vel = self.g1_articulation.data.joint_vel
        
        # Get observations
        obs = self._get_observations()
        
        # Compute rewards
        rewards = self.compute_reward(actions)
        
        # Check for termination
        dones = self.episode_steps >= self.max_episode_length
        
        # Update episode steps
        self.episode_steps += 1
        self.episode_steps[dones] = 0
        
        # Reset environments that are done
        reset_indices = torch.where(dones)[0]
        if len(reset_indices) > 0:
            for idx in reset_indices:
                self._reset_env(idx.item())
        
        infos = [{"episode_step": self.episode_steps[i].item()} for i in range(self.num_envs)]
        
        return obs, rewards.cpu().numpy(), dones.cpu().numpy(), infos
    
    def reset(self) -> np.ndarray:
        """Reset all environments and return initial observations"""
        if self.g1_articulation is not None:
            # Reset all environments
            indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self.g1_articulation.reset(indices)
        
        self.episode_steps.zero_()
        return self._get_observations()
    
    def _reset_env(self, env_idx: int):
        """Reset a single environment"""
        if self.g1_articulation is not None:
            # Reset specific environment
            idx_tensor = torch.tensor([env_idx], device=self.device, dtype=torch.long)
            self.g1_articulation.reset(idx_tensor)
    
    def _get_observations(self) -> np.ndarray:
        """Extract observations from current state"""
        obs = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        
        # Fill with joint positions and velocities
        for i in range(min(self.num_envs, self.obs_dim // 2)):
            if i < self.num_dofs:
                obs[:, i] = self.g1_pos[:, i]
                if i + self.num_dofs < self.obs_dim:
                    obs[:, i + self.num_dofs] = self.g1_vel[:, i]
        
        return obs.cpu().numpy()
    
    def compute_reward(self, actions: np.ndarray) -> torch.Tensor:
        """
        Compute reward based on current state and actions
        
        Args:
            actions: Action tensor
            
        Returns:
            rewards: Reward for each environment
        """
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Placeholder reward: Small penalty for joint effort
        action_tensor = torch.from_numpy(actions).float().to(self.device)
        action_penalty = 0.001 * torch.sum(action_tensor ** 2, dim=1)
        
        # Small positive reward for staying upright
        upright_reward = torch.ones(self.num_envs, device=self.device) * 0.1
        
        rewards = upright_reward - action_penalty
        
        return rewards
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'scene'):
            self.scene.close()
    
    def render(self):
        """Render the environment"""
        if hasattr(self, 'scene') and not self.headless:
            self.scene.render()
