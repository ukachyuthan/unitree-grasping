# Development Guide

## Project Structure

```
unitree-walking/
├── environments/        # G1 environment implementations
├── policies/           # RL policies
├── training/           # Training loops and utilities
├── meta_learning/      # Meta-policy and terrain detection
├── utils/              # Shared utilities
├── configs/            # Configuration files
├── scripts/            # Executable scripts
├── data/              # Logs, checkpoints
└── tests/             # Unit tests (to be added)
```

## Code Organization Principles

1. **Modularity**: Each component should be independent and testable
2. **Configuration-driven**: Use YAML configs for hyperparameters
3. **Type hints**: Use Python type hints for clarity
4. **Documentation**: Docstrings for all public functions
5. **Error handling**: Meaningful error messages

## Adding a New Component

### Adding a New RL Algorithm

1. Create `policies/your_algorithm.py`
2. Implement required methods: `select_action()`, `save()`, `load()`
3. Create trainer in `training/your_trainer.py`
4. Add to `__init__.py` files for imports

### Adding a New Terrain

1. Add terrain config to `configs/terrain_configs.yaml`
2. Modify environment to handle new terrain type
3. Test environment with new terrain
4. Train policy on new terrain

### Adding a New Environment

1. Create `environments/your_env.py`
2. Extend `G1WalkingEnv` base class
3. Implement: `step()`, `reset()`, `compute_reward()`
4. Add to `environments/__init__.py`

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_policies.py::test_sac_policy -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Code Style

Follow PEP 8 with black formatter:

```bash
# Format code
black . --line-length 100

# Sort imports
isort .

# Lint
flake8 .
```

## Logging

Use the centralized logger:

```python
from utils import Logger

logger = Logger(log_dir="data/logs", name="my_experiment")
logger.info("Training started")
logger.warning("High loss detected")
logger.error("Training failed")
```

## Configuration

Load configs with ConfigLoader:

```python
from utils import ConfigLoader

config = ConfigLoader.load("configs/base_config.yaml")
terrain_config = ConfigLoader.load("configs/terrain_configs.yaml")

# Merge configs
merged = ConfigLoader.merge(config, override)

# Save modified config
ConfigLoader.save(config, "data/logs/config_used.yaml")
```

## Performance Optimization

1. **Parallel environments**: Use Isaac Gym's built-in parallelization
2. **GPU computation**: Keep everything on GPU when possible
3. **Batch training**: Use larger batch sizes for efficiency
4. **Mixed precision**: Use float16 where applicable

## Debugging

### Enable verbose logging
```bash
python scripts/train.py --terrain flat --log-level DEBUG
```

### Save training state
The trainer saves checkpoints at regular intervals. To resume:
```bash
python scripts/train.py --terrain flat --checkpoint data/checkpoints/flat_policy_500000.pt
```

### Analyze logs with TensorBoard
```bash
tensorboard --logdir data/logs
```

## Common Tasks

### Modify reward function
Edit `compute_reward()` in the environment class

### Change hyperparameters
Modify `configs/base_config.yaml` or pass via command line

### Switch algorithms
Change `algorithm` in config and use different trainer

### Add curriculum learning
Enable and configure in `configs/terrain_configs.yaml`
