"""
Isaac Gym environment for G1 walking with multiple terrain types
"""
import torch
import numpy as np
from typing import Dict, Tuple
from isaacgym import gymapi, gymutil

from .g1_walking_env import G1WalkingEnv


class IsaacGymG1Environment(G1WalkingEnv):
    """
    G1 walking environment using NVIDIA Isaac Gym
    
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
        Initialize Isaac Gym G1 environment
        
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
        
        # Initialize Isaac Gym
        self.gym = gymapi.Gym()
        
        # Create simulation
        self.sim = self._create_simulation()
        
        # Load assets
        self.g1_asset = self._load_g1_model()
        self.terrain_asset = self._load_terrain_assets()
        
        # Create environments and actors
        self.envs = []
        self.g1_handles = []
        self.terrain_handles = []
        self._create_environments()
        
        # Prepare tensors
        self._prepare_tensors()
        
        # Episode tracking
        self.episode_steps = torch.zeros(num_envs, device=device)
        self.max_episode_length = 1000
        
    def _create_simulation(self):
        """Create Isaac Gym simulation"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.005  # 200 Hz simulation
        sim_params.substeps = 2
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # PhysX parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1
        sim_params.physx.num_pos_iterations = 4
        sim_params.physx.num_vel_iterations = 1
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        
        # Create simulation
        sim = self.gym.create_sim(
            compute_device_id=self.device_id,
            graphics_device_id=self.device_id,
            type=gymapi.SIM_PHYSX,
            params=sim_params
        )
        
        if not self.headless:
            self.gym.subscribe_viewer_keyboard_event(sim, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(sim, gymapi.KEY_V, "toggle_viewer_sync")
        
        return sim
    
    def _load_g1_model(self):
        """Load G1 robot URDF model"""
        # TODO: Update this path to your G1 URDF
        asset_root = "path/to/unitree/assets"
        asset_file = "g1/g1.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.001
        asset_options.thickness = 0.002
        asset_options.linear_damping = 0.0
        asset_options.angular_damping = 0.0
        asset_options.max_linear_velocity = 1000.0
        asset_options.max_angular_velocity = 1000.0
        asset_options.disable_gravity = False
        
        try:
            asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        except Exception as e:
            print(f"Warning: Could not load G1 asset: {e}")
            print("Using placeholder. Update path to actual G1 URDF.")
            asset = None
        
        return asset
    
    def _load_terrain_assets(self):
        """Load terrain meshes"""
        # Placeholder for terrain assets
        return {}
    
    def _create_environments(self):
        """Create parallel environments with robots and terrains"""
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, 8)
            self.envs.append(env)
            
            # Add G1 robot
            if self.g1_asset is not None:
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
                pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                
                g1_handle = self.gym.create_actor(env, self.g1_asset, pose, "g1", i, 0)
                self.g1_handles.append(g1_handle)
            
            # Add ground plane
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            plane_params.distance = 0
            plane_params.static_friction = 1.0
            plane_params.dynamic_friction = 1.0
            plane_params.restitution = 0.0
            self.gym.add_ground(self.sim, plane_params)
    
    def _prepare_tensors(self):
        """Prepare GPU tensors for state storage"""
        # Get DOF info
        if self.g1_handles:
            dof_info = self.gym.get_actor_dof_properties(self.envs[0], self.g1_handles[0])
            self.num_dofs = len(dof_info)
        else:
            self.num_dofs = 19  # Default for G1
        
        # Allocate tensors
        self.g1_states = torch.zeros((self.num_envs, self.num_dofs * 2), device=self.device)
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
        
        # Apply actions to all environments
        for i, env in enumerate(self.envs):
            if i < len(self.g1_handles):
                # Scale actions to joint ranges
                scaled_actions = actions_tensor[i] * 0.5  # Adjust scale as needed
                self.gym.set_dof_actuation(env, self.g1_handles[i], scaled_actions)
        
        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Refresh state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        
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
        for i, done in enumerate(dones):
            if done:
                self._reset_env(i)
        
        infos = [{"episode_step": self.episode_steps[i].item()} for i in range(self.num_envs)]
        
        return obs, rewards.cpu().numpy(), dones.cpu().numpy(), infos
    
    def reset(self) -> np.ndarray:
        """Reset all environments and return initial observations"""
        for i in range(self.num_envs):
            self._reset_env(i)
        
        self.episode_steps.zero_()
        return self._get_observations()
    
    def _reset_env(self, env_idx: int):
        """Reset a single environment"""
        if env_idx < len(self.g1_handles):
            # Reset to default pose
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            self.gym.set_actor_transform(
                self.envs[env_idx],
                self.g1_handles[env_idx],
                pose
            )
            
            # Zero velocities
            zero_vel = gymapi.Vec3(0, 0, 0)
            self.gym.set_actor_linear_velocity(self.envs[env_idx], self.g1_handles[env_idx], zero_vel)
            self.gym.set_actor_angular_velocity(self.envs[env_idx], self.g1_handles[env_idx], zero_vel)
            
            # Reset DOF states
            dof_states = torch.zeros((self.num_dofs, 2), dtype=torch.float32)
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymapi.unwrap_tensor(dof_states),
                gymapi.unwrap_tensor(torch.tensor([env_idx], dtype=torch.long)),
                1
            )
    
    def _get_observations(self) -> np.ndarray:
        """Extract observations from current state"""
        obs = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        
        # Placeholder: Fill with joint positions and velocities
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
        if hasattr(self, 'gym'):
            self.gym.destroy_sim(self.sim)
            if not self.headless:
                self.gym.destroy_viewer(self.viewer)
    
    def render(self):
        """Render the environment"""
        if not self.headless and hasattr(self, 'viewer'):
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
