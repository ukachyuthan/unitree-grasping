#!/usr/bin/env python3
"""
Evaluation script for trained policies
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_policy(terrain_name: str, checkpoint_path: str, num_episodes: int = 5):
    """
    Evaluate a trained policy
    
    Args:
        terrain_name: Terrain to evaluate on
        checkpoint_path: Path to policy checkpoint
        num_episodes: Number of episodes to evaluate
    """
    print(f"Evaluating policy for {terrain_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    
    # TODO: Implement evaluation loop


def main():
    parser = argparse.ArgumentParser(description="Evaluate G1 walking policy")
    parser.add_argument("--terrain", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    
    args = parser.parse_args()
    evaluate_policy(args.terrain, args.checkpoint, args.episodes)


if __name__ == "__main__":
    main()
