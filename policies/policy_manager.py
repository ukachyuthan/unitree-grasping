"""
Manager for handling multiple terrain-specific policies
"""
import os
from typing import Dict, Optional
from .sac_policy import SACPolicy


class PolicyManager:
    """
    Manages multiple SAC policies for different terrains
    
    Handles:
    - Creating and storing policies per terrain
    - Saving/loading policies
    - Policy selection logic
    """
    
    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.policies: Dict[str, SACPolicy] = {}
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def create_policy(self,
                     terrain_name: str,
                     obs_dim: int,
                     action_dim: int,
                     **kwargs) -> SACPolicy:
        """
        Create a new policy for a terrain
        
        Args:
            terrain_name: Name of the terrain
            obs_dim: Observation dimension
            action_dim: Action dimension
            **kwargs: Additional arguments for SACPolicy
            
        Returns:
            policy: Created SAC policy
        """
        policy = SACPolicy(obs_dim, action_dim, **kwargs)
        self.policies[terrain_name] = policy
        return policy
    
    def get_policy(self, terrain_name: str) -> Optional[SACPolicy]:
        """Get policy for a terrain"""
        return self.policies.get(terrain_name)
    
    def save_policy(self, terrain_name: str, identifier: str = ""):
        """
        Save policy checkpoint
        
        Args:
            terrain_name: Terrain name
            identifier: Optional identifier for checkpoint (e.g., step number)
        """
        if terrain_name not in self.policies:
            raise ValueError(f"No policy found for terrain: {terrain_name}")
        
        filename = f"{terrain_name}_policy"
        if identifier:
            filename += f"_{identifier}"
        filename += ".pt"
        
        path = os.path.join(self.checkpoint_dir, filename)
        self.policies[terrain_name].save(path)
    
    def load_policy(self, terrain_name: str, identifier: str = ""):
        """
        Load policy checkpoint
        
        Args:
            terrain_name: Terrain name
            identifier: Optional identifier for checkpoint
        """
        if terrain_name not in self.policies:
            raise ValueError(f"No policy found for terrain: {terrain_name}")
        
        filename = f"{terrain_name}_policy"
        if identifier:
            filename += f"_{identifier}"
        filename += ".pt"
        
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Policy checkpoint not found: {path}")
        
        self.policies[terrain_name].load(path)
    
    def list_terrains(self):
        """List all terrain policies"""
        return list(self.policies.keys())
