from setuptools import setup, find_packages

setup(
    name="unitree-walking",
    version="0.1.0",
    description="Multi-terrain walking policy training for Unitree G1 humanoid robot",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "isaacgym",
        "stable-baselines3>=2.0.0",
        "numpy>=1.23.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
    ],
)
