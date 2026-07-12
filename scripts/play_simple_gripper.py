"""Demo runner for the simplified gripper env.

Run `python scripts/play_simple_gripper.py` to see a short random policy
interact with the environment. Use --gui to open a PyBullet GUI.
"""

import argparse
import numpy as np

from environments.simple_gripper_env import SimpleGripperEnv


def main(gui: bool = False):
    env = SimpleGripperEnv(gui=gui)
    obs = env.reset()
    for t in range(120):
        # random small motion and occasionally close gripper
        dx = (np.random.rand(3) - 0.5) * 0.02
        gr_cmd = 0.2 if (t % 40) < 20 else 1.0
        action = np.concatenate([dx, [gr_cmd]])
        obs, r, d, info = env.step(action)
        print(f"t={t} reward={r:.3f}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()
    main(gui=args.gui)
