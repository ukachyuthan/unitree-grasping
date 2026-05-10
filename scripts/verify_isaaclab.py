#!/usr/bin/env python3
"""
Verify Isaac Lab install before training.
Run this after completing all pip installs.

Usage:
    ./rl_unitree/bin/python scripts/verify_isaaclab.py
"""
import sys

errors = []

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK]  {label}" + (f": {result}" if result else ""))
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        errors.append(label)

print("\n=== Isaac Lab environment check ===\n")

check("torch + CUDA", lambda: __import__("torch").cuda.get_device_name(0))
check("isaacsim importable", lambda: __import__("isaacsim"))
check("isaaclab_rl importable", lambda: __import__("isaaclab_rl"))
check("rsl_rl importable", lambda: __import__("rsl_rl"))

# isaaclab_tasks / isaaclab_assets / isaaclab.envs require the Isaac Sim app
# (AppLauncher) to be running — they cannot be imported standalone.
print("  [NOTE] isaaclab_tasks + isaaclab_assets require AppLauncher; verified at runtime.")

print()
if errors:
    print(f"[!] {len(errors)} issue(s) found: {errors}")
    sys.exit(1)
else:
    print("[✓] All checks passed. Ready to train.\n")
    print("Run training with:")
    print("  ./rl_unitree/bin/python scripts/train_isaaclab_g1.py --headless --num_envs 64\n")
