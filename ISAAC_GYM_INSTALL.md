# Isaac Gym Installation Guide

## Quick Summary

`isaacgym` is **NOT** available on PyPI. It must be downloaded and installed from NVIDIA directly.

## Prerequisites

- **NVIDIA GPU**: RTX 3090 or better recommended
- **CUDA 11.8+**: Check with `nvcc --version`
- **cuDNN 8.0+**
- **Python 3.9+**
- **pip with development tools**

## Step 1: Verify Your Setup

```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check Python version
python --version

# Ensure pip is up to date
pip install --upgrade pip setuptools wheel
```

## Step 2: Download Isaac Gym

1. Go to [NVIDIA Developer - Isaac Gym](https://developer.nvidia.com/isaac-gym)
2. Register/login to your NVIDIA account
3. Download the latest **Isaac Gym Preview** package
4. Extract the downloaded file:

```bash
# Example (adjust version as needed)
tar -xzf IsaacGym_Preview_4_Package.tar.gz
cd IsaacGym_Preview_4_Package
```

## Step 3: Install Isaac Gym

```bash
# Navigate to Isaac Gym directory
cd IsaacGym_Preview_4_Package

# Install in editable mode
pip install -e .
```

Or with verbose output for debugging:

```bash
pip install -e . -v
```

## Step 4: Verify Installation

```bash
# Test basic import
python -c "from isaacgym import gymapi; print('✓ Isaac Gym installed successfully')"

# Run Isaac Gym example (optional)
python examples/01_acquire_data.py
```

## Common Installation Issues

### Issue 1: "No module named 'isaacgym'"

**Solution**: Make sure Isaac Gym directory is in your Python path:

```bash
# Check installed packages
pip show isaacgym

# If not found, reinstall from extracted directory
cd /path/to/IsaacGym_Preview_4_Package
pip install -e .
```

### Issue 2: CUDA mismatch

**Solution**: Install PyTorch for your CUDA version:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: "libcuda.so.1 not found"

**Solution**: Set CUDA library path:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

Add this to your `~/.bashrc` for persistence:

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
source ~/.bashrc
```

### Issue 4: Out of Memory during install

**Solution**: Isaac Gym is large. Ensure you have:

```bash
# Check available disk space
df -h

# Minimum 10GB free recommended
```

### Issue 5: Permission denied

**Solution**: Use user install:

```bash
pip install -e . --user
```

## Testing Isaac Gym Works With This Project

After successful installation:

```bash
# Navigate to project root
cd /path/to/unitree-walking

# Try training with Isaac Gym (will auto-detect)
python scripts/train.py --terrain flat --num-steps 10000 --device cuda:0
```

If successful, you'll see:
```
Using Isaac Gym (GPU-accelerated)
Environment created: 4096 parallel envs
```

## Alternative: Use Mock Environment (No GPU Required)

If you can't install Isaac Gym yet, you can still test the training pipeline:

```bash
# Training will automatically use MockG1Environment
python scripts/train.py --terrain flat --num-steps 10000

# Output will show:
# WARNING: Isaac Gym not found. Using mock environment for testing.
```

The mock environment is ~10x slower but allows you to:
- Test the training loop
- Verify configuration
- Debug policy code
- Get initial results

## Environment Variables for Isaac Gym

```bash
# Disable rendering (faster training)
export ISAAC_DISABLE_RENDERING=1

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Enable Isaac Gym logging
export ISAAC_GYM_LOGLEVEL=2
```

## Performance Tuning

For optimal training speed:

```bash
# Use high-end GPU (RTX 3090 or A100)
# Set num_envs high (4096-8192)
# Run headless: python scripts/train.py --no-gui
```

Typical performance:
- **Mock environment**: 50-100 FPS
- **Isaac Gym (single env)**: 500+ FPS
- **Isaac Gym (4096 envs)**: 5000+ FPS (total steps per second)

## Troubleshooting Checklist

- [ ] CUDA installed and working (`nvidia-smi` shows GPU)
- [ ] PyTorch installed for your CUDA version
- [ ] Isaac Gym extracted to accessible location
- [ ] Installed with `pip install -e .` from extracted directory
- [ ] Can import: `python -c "from isaacgym import gymapi"`
- [ ] LD_LIBRARY_PATH set if needed
- [ ] Enough GPU memory for batch size
- [ ] Enough disk space (10GB+ free)

## Getting Help

If you're still stuck:

1. Check NVIDIA Isaac Gym documentation: https://docs.omniverse.nvidia.com/
2. Try Isaac Gym examples first: `python examples/01_acquire_data.py`
3. Check GPU VRAM: `nvidia-smi` (should be <50% used at startup)
4. Look for error in detailed output: `pip install -e . -v`

## Next Steps

Once Isaac Gym is installed:

```bash
# Start training on flat terrain
python scripts/train.py --terrain flat --num-steps 1000000 --device cuda:0

# Monitor with TensorBoard
tensorboard --logdir data/logs
```

---

**Need help?** Check the main README.md or DEVELOPMENT.md for more info.
