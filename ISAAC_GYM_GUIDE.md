# Isaac Gym Integration Guide

This document provides guidance on integrating Isaac Gym for the G1 robot environment.

## Prerequisites

- NVIDIA GPU (RTX 3090 or better recommended)
- NVIDIA CUDA Toolkit 11.8+
- NVIDIA cuDNN 8.0+
- Isaac Gym installed (see NVIDIA documentation)

## Implementation Steps

### 1. Create Isaac Gym Environment

The `environments/isaac_gym_env.py` should extend the base `G1WalkingEnv` class and implement:

```python
class IsaacGymG1Environment(G1WalkingEnv):
    def __init__(self, ...):
        # Initialize Isaac Gym simulation
        # Load G1 URDF model
        # Setup cameras and sensors
        pass
    
    def step(self, actions):
        # Apply actions to robot
        # Step simulation
        # Extract observations
        # Compute rewards
        pass
    
    def reset(self):
        # Reset robot to starting pose
        # Reset terrain
        pass
```

### 2. Observation Space

Recommended observations for G1 walking:
- **Position**: Root position (x, y, z)
- **Orientation**: Root rotation (quaternion or euler)
- **Joint State**: Position and velocity for all joints (23 DOF for G1)
- **Foot Contact**: Binary contact flags for each foot
- **Proprioception**: IMU reading, gyroscope
- **Command**: Desired velocity/direction

Total obs_dim: ~48-64 dimensions

### 3. Action Space

- **Action dim**: 19 (one per leg joint, excluding fixed joints)
- **Action range**: [-1, 1] (normalized, then scaled to joint limits)
- **Framerate**: 20 Hz (control_dt = 0.05s)

### 4. Reward Function

The reward should encourage:
1. **Velocity tracking**: Follow desired velocity command
2. **Smoothness**: Penalize jerky motion
3. **Energy efficiency**: Penalize large joint torques
4. **Stability**: Penalize falling

Example:
```python
def compute_reward(self):
    # Velocity reward
    v_reward = max_velocity_reward * exp(-velocity_error^2)
    
    # Smoothness penalty
    smooth_penalty = action_regularization * ||action||^2
    
    # Energy penalty
    energy_penalty = torque_limit * ||torques||^2
    
    # Fall penalty
    fall_penalty = -1.0 if robot_fallen else 0
    
    return v_reward + smooth_penalty + energy_penalty + fall_penalty
```

### 5. Terrain Implementation

For each terrain, modify:
- **Friction coefficient**: Affects slip behavior
- **Terrain geometry**: Use mesh colliders for roughness
- **Damping**: Joint damping for rocky terrains
- **Material properties**: Stiffness, restitution

### 6. Isaac Gym Setup Example

```python
from isaacgym import gymapi, gymutil
import numpy as np

class IsaacGymG1Environment(G1WalkingEnv):
    def __init__(self, num_envs=4096, device="cuda:0", terrain_config=None):
        super().__init__(num_envs, device, terrain_config)
        
        # Initialize gym
        self.gym = gymapi.Gym()
        
        # Create simulation
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.005
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
        self.sim = self.gym.create_sim(
            compute_device=device.split(':')[0],
            graphics_device=0,
            type=gymapi.SIM_PHYSX,
            params=sim_params
        )
        
        # Load assets
        asset_root = "path/to/assets"
        asset_file = "unitree_g1/g1.urdf"
        
        # Create environments
        self.envs = []
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, ...)
            # Add robot, terrain, camera actors
            self.envs.append(env)
        
        # Buffer for states and observations
        self.obs_buf = torch.zeros((num_envs, self.obs_dim), device=device)
        self.reward_buf = torch.zeros(num_envs, device=device)
```

## Testing

To test the environment:

```bash
python -c "
from environments.isaac_gym_env import IsaacGymG1Environment
env = IsaacGymG1Environment(num_envs=4, device='cuda:0')
obs = env.reset()
print(f'Obs shape: {obs.shape}')
for _ in range(10):
    obs, reward, done, info = env.step(env.sample_actions())
env.close()
"
```

## Common Issues

### Out of Memory
- Reduce `num_envs` (start with 1, then scale up)
- Reduce visual quality in simulation params

### Unstable Physics
- Reduce `sim_dt`
- Adjust joint damping and stiffness
- Enable contact filtering

### Low FPS
- Disable rendering for training
- Use headless mode
- Reduce contact buffer size

## References

- [Isaac Gym User Guide](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
- [Isaac Gym Examples](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- [NVIDIA Physics Engine Documentation](https://docs.nvidia.com/isaac/archive/2020.11/common/physics.html)
