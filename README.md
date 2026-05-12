# Unitree G1 Multi-Terrain Walking with Reinforcement Learning

A project for training the Unitree G1 humanoid robot to walk across multiple terrain types using Soft Actor-Critic (SAC) reinforcement learning, with eventual support for terrain-adaptive meta-policies.

## Project Overview

The goal of this project is to:

1. **Train specialized policies** for different terrain types (flat, rocky, slopes, stairs)
2. **Develop a terrain detection system** that can identify terrain in real-time
3. **Learn a meta-policy** that selects the appropriate specialized policy based on detected terrain

This enables the G1 robot to adapt its walking strategy based on environmental conditions.

## Architecture

```
unitree-walking/
├── environments/           # G1 walking environments for different terrains
│   ├── g1_walking_env.py   # Base environment class
│   └── isaac_gym_env.py    # Isaac Gym integration (to implement)
├── policies/               # RL policy implementations
│   ├── sac_policy.py       # SAC policy networks
│   └── policy_manager.py   # Multi-policy management
├── training/               # Training utilities
│   ├── sac_trainer.py      # SAC training loop
│   └── replay_buffer.py    # Experience replay buffer
├── meta_learning/          # Meta-policy and terrain detection
│   └── __init__.py         # Terrain detector and meta-policy
├── utils/                  # Utilities
│   ├── logger.py           # Logging utilities
│   └── config_loader.py    # Configuration management
├── configs/                # Configuration files
│   ├── base_config.yaml    # Training hyperparameters
│   └── terrain_configs.yaml # Terrain definitions
├── data/
│   ├── checkpoints/        # Saved policies
│   └── logs/               # Training logs
└── scripts/                # Standalone scripts
    ├── train.py            # Main training script
    └── evaluate.py         # Policy evaluation
```

## Configuration

### Terrains

Currently supported terrains (defined in `configs/terrain_configs.yaml`):
- **flat**: Smooth, flat ground (baseline)
- **rocky**: Uneven, rocky ground with obstacles
- **slopes**: Inclined/declined terrain
- **stairs**: Step-like terrain

### Training Hyperparameters

Key parameters in `configs/base_config.yaml`:
- **algorithm**: SAC (Soft Actor-Critic)
- **num_envs**: Number of parallel environments (4096 for Isaac Gym)
- **learning_rate**: 3e-4
- **buffer_size**: 1,000,000 transitions
- **gamma**: 0.99 (discount factor)
- **tau**: 0.005 (soft update coefficient)
- **auto_entropy_tuning**: True

## Getting Started

### Installation

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install Isaac Gym (required for real training)
# See ISAAC_GYM_INSTALL.md for detailed instructions
# Download from: https://developer.nvidia.com/isaac-gym
# Then: pip install -e /path/to/IsaacGym_Preview_4_Package
```

**Note**: Isaac Gym is **NOT** available on PyPI. See [ISAAC_GYM_INSTALL.md](ISAAC_GYM_INSTALL.md) for complete setup.

### Quick Start (Without Isaac Gym)

Test the training pipeline using the mock environment:

```bash
python scripts/train.py --terrain flat --num-steps 10000
```

This will:
- Use 4 CPU-based environments (slow, for testing)
- Save checkpoints to `data/checkpoints/`
- Log to `data/logs/`

### Training a Single Policy (With Isaac Gym)

Once Isaac Gym is installed:

```bash
python scripts/train.py --terrain flat --num-steps 1000000 --device cuda:0
```

### Supported Terrains

```bash
python scripts/train.py --terrain rocky
python scripts/train.py --terrain slopes
python scripts/train.py --terrain stairs
```

### Evaluating a Policy

```bash
python scripts/evaluate.py --terrain flat --checkpoint data/checkpoints/flat_policy_1000000.pt --episodes 5
```

## Project Timeline

### Phase 1: Foundation (Current)
- [x] Project structure and configuration
- [x] SAC policy implementation
- [x] Replay buffer
- [ ] Isaac Gym G1 environment implementation
- [ ] Single-terrain training loop

### Phase 2: Multi-Terrain Training
- [ ] Environment variations for each terrain
- [ ] Train specialized policies per terrain
- [ ] Policy checkpointing and management

### Phase 3: Terrain Detection
- [ ] Implement terrain detector (heuristic or neural)
- [ ] Collect terrain classification data

### Phase 4: Meta-Policy Learning
- [ ] Implement meta-policy selection mechanism
- [ ] Train meta-policy for terrain-adaptive control
- [ ] Evaluate on mixed terrains

### Phase 5: Optimization & Testing
- [ ] Deploy to real robot
- [ ] Fine-tune on real-world data
- [ ] Benchmark performance

## Algorithm Details

### Soft Actor-Critic (SAC)

SAC is an off-policy deep RL algorithm that:
- Maximizes both expected return and entropy (for exploration)
- Uses separate Q-networks for value estimation
- Employs an entropy-regularized Gaussian policy

Key components:
- **Actor**: Maps observations to action distributions
- **Critic**: Two Q-networks estimate state-action values
- **Replay Buffer**: Stores and samples transitions
- **Entropy Coefficient**: Auto-tuned for exploration-exploitation trade-off

### Multi-Terrain Approach

1. Train separate SAC policies for each terrain
2. Each policy specializes in its terrain's characteristics
3. Terrain detector classifies current environment
4. Meta-policy routes to appropriate specialized policy

## Next Steps

1. **Integrate Isaac Gym**: Implement `IsaacGymG1Environment`
2. **Implement training loop**: Full training with data collection and updates
3. **Add terrain variation**: Procedural terrain generation
4. **Develop terrain detector**: Heuristic or learned approach
5. **Train policies**: Conduct multi-terrain training experiments

## References

- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)
- [Isaac Gym Documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
- [Unitree G1 Robot](https://www.unitree.com/)

## License

To be determined

## Contact

For questions or contributions, please open an issue or discussion.