#!/usr/bin/env python3
"""
Main training script for G1 walking

This script trains SAC policies for different terrains and can eventually
combine them into a meta-policy that selects the best policy per terrain.
"""
import os
import sys
import argparse
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from utils import Logger, ConfigLoader

# Try to import Isaac Gym, fallback to mock if not available
try:
    from environments.isaac_gym_env import IsaacGymG1Environment
    HAS_ISAAC_GYM = True
except ImportError:
    HAS_ISAAC_GYM = False
    print("WARNING: Isaac Gym not found. Using mock environment for testing.")
    print("For real training, install Isaac Gym from: https://developer.nvidia.com/isaac-gym")

from environments.mock_env import MockG1Environment
from policies import SACPolicy, PolicyManager
from training import ReplayBuffer, SACTrainer


def train_single_terrain(terrain_name: str, 
                        config_path: str = "configs/base_config.yaml",
                        num_steps: int = 1000000,
                        device: str = "cuda:0",
                        headless: bool = True):
    """
    Train a single terrain policy
    
    Args:
        terrain_name: Name of terrain to train on
        config_path: Path to training config
        num_steps: Number of training steps
        device: Device to train on
        headless: Run without GUI
    """
    logger = Logger(name=f"train_{terrain_name}")
    
    # Load configs
    base_config = ConfigLoader.load(config_path)
    terrain_config = ConfigLoader.load("configs/terrain_configs.yaml")
    
    if terrain_name not in terrain_config['terrains']:
        raise ValueError(f"Unknown terrain: {terrain_name}")
    
    logger.info(f"Starting training for terrain: {terrain_name}")
    logger.info(f"Total steps: {num_steps}")
    logger.info(f"Device: {device}")
    
    # Create environment
    logger.info("Creating environment...")
    try:
        if HAS_ISAAC_GYM:
            logger.info("Using Isaac Gym (GPU-accelerated)")
            env = IsaacGymG1Environment(
                num_envs=base_config['environment']['num_envs'],
                device=device,
                terrain_config=terrain_config['terrains'][terrain_name],
                headless=headless
            )
        else:
            logger.warning("Isaac Gym not available. Using mock environment (slow, for testing only)")
            env = MockG1Environment(
                num_envs=4,  # Use fewer envs for mock
                device=device,
                terrain_config=terrain_config['terrains'][terrain_name]
            )
        logger.info(f"Environment created: {env.num_envs} parallel envs")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        logger.info("Please install Isaac Gym: https://developer.nvidia.com/isaac-gym")
        return
    
    # Get observation and action dimensions
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    
    logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    policy = SACPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        learning_rate=base_config['training']['learning_rate'],
        gamma=base_config['training']['gamma'],
        tau=base_config['training']['tau'],
        alpha_lr=base_config['training']['alpha_lr'],
        auto_entropy_tuning=base_config['training']['auto_entropy_tuning']
    )
    logger.info("SAC policy created")
    
    # Create replay buffer
    buffer = ReplayBuffer(
        buffer_size=base_config['training']['buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim
    )
    logger.info(f"Replay buffer created (size: {base_config['training']['buffer_size']})")
    
    # Create trainer
    trainer = SACTrainer(policy, env, buffer, device=device)
    
    # Policy manager for saving
    policy_manager = PolicyManager()
    policy_manager.create_policy(terrain_name, obs_dim, action_dim)
    policy_manager.policies[terrain_name] = policy
    
    # Training loop
    logger.info("Starting training loop...")
    step = 0
    episode_rewards = []
    episode_reward = 0.0
    
    obs = env.reset()
    
    pbar = tqdm(total=num_steps, desc=f"Training {terrain_name}")
    
    while step < num_steps:
        # Collect experience
        action = policy.select_action(torch.FloatTensor(obs).to(device), deterministic=False)
        action = action.cpu().numpy()
        
        next_obs, rewards, dones, infos = env.step(action)
        
        # Store in replay buffer (per-environment)
        for i in range(env.num_envs):
            buffer.add(
                obs[i],
                action[i] if len(action.shape) > 1 else action,
                rewards[i],
                next_obs[i],
                dones[i]
            )
            
            episode_reward += rewards[i]
        
        obs = next_obs
        step += env.num_envs
        
        # Train on batch
        if buffer.size > base_config['training']['batch_size']:
            train_metrics = trainer.train_step(
                batch_size=base_config['training']['batch_size'],
                gamma=base_config['training']['gamma']
            )
        
        # Track episodes
        for done in dones:
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
        
        # Logging
        if step % base_config['training']['log_freq'] == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            pbar.set_postfix({"avg_reward": f"{avg_reward:.2f}"})
        
        # Evaluation and saving
        if step % base_config['training']['save_freq'] == 0:
            logger.info(f"Saving checkpoint at step {step}")
            policy_manager.save_policy(terrain_name, identifier=str(step))
            logger.info(f"Checkpoint saved: {step} steps")
        
        pbar.update(env.num_envs)
    
    pbar.close()
    
    # Final save
    logger.info("Training complete! Saving final policy...")
    policy_manager.save_policy(terrain_name, identifier="final")
    
    # Save config
    ConfigLoader.save(
        base_config,
        os.path.join("data/logs", f"config_{terrain_name}.yaml")
    )
    
    logger.info(f"Training finished for {terrain_name}")
    logger.info(f"Total episodes: {len(episode_rewards)}")
    logger.info(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train G1 walking policy")
    parser.add_argument("--terrain", type=str, default="flat",
                       help="Terrain to train on (flat, rocky, slopes, stairs)")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to config file")
    parser.add_argument("--num-steps", type=int, default=1000000,
                       help="Number of training steps")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to train on")
    parser.add_argument("--no-gui", action="store_true", default=True,
                       help="Run headless (no GUI)")
    
    args = parser.parse_args()
    
    train_single_terrain(
        args.terrain,
        args.config,
        args.num_steps,
        args.device,
        headless=args.no_gui
    )


if __name__ == "__main__":
    main()
