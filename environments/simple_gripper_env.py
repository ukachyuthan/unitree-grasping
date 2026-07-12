"""Simple PyBullet-based gripper environment for fast prototyping.

- Minimal two-finger gripper (two box links) and a single box object.
- No robot kinematics: gripper base is teleported to commanded poses.
- Provides point-cloud sampling of the object surface for PointNet input.
- Reward hook: a user-supplied `grasp_scorer(pointcloud, grasp_point)` may be
  provided to compute the initial PointNet-style grasp score. If not
  supplied, a simple heuristic score is used.

This environment is intentionally tiny and dependency-light so you can run
grasp experiments without Isaac/Unitree.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except Exception as e:  # pragma: no cover - optional dependency
    raise ImportError("pybullet is required for simple_gripper_env: pip install pybullet")


class SimpleGripperEnv:
    """A very small gym-like environment for grasp experiments.

    Observation: dict with keys:
      - 'pointcloud' : (N, 3) numpy array in gripper base frame
      - 'gripper_pos': (3,) xyz of gripper base
      - 'gripper_open': float in [0, 1]

    Action: np.array([dx, dy, dz, gripper_cmd]) where dx,dy,dz are desired
    position deltas for the gripper base (applied as a velocity step) and
    gripper_cmd in [0,1] is the desired open fraction (1.0 open, 0.0 closed).
    """

    def __init__(
        self,
        gui: bool = False,
        time_step: float = 1.0 / 240.0,
        num_points: int = 128,
        grasp_scorer: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        self.gui = gui
        self.time_step = time_step
        self.num_points = int(num_points)
        self.grasp_scorer = grasp_scorer

        self._client = None
        self._gripper_uid = None
        self._object_uid = None

        self._reset_sim()

    def _reset_sim(self):
        if self._client is not None:
            try:
                p.disconnect(self._client)
            except Exception:
                pass
        if self.gui:
            self._client = p.connect(p.GUI)
        else:
            self._client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(self.time_step, physicsClientId=self._client)

    def _build_gripper(self, base_pos=(0, 0, 0.2)) -> int:
        """Create a simple two-finger gripper as a single multibody.

        The gripper base is a fixed body; fingers are two small boxes that can
        be moved by setting their relative positions.
        """
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.02])
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.02])

        finger_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.02, 0.06])
        finger_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.02, 0.06])

        mass = 0.0
        base_uid = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=base_pos,
        )

        # Create two finger bodies and attach with fixed constraints for simplicity
        f1_pos = (base_pos[0], base_pos[1] + 0.035, base_pos[2])
        f2_pos = (base_pos[0], base_pos[1] - 0.035, base_pos[2])
        f1 = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=finger_collision,
            baseVisualShapeIndex=finger_visual,
            basePosition=f1_pos,
        )
        f2 = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=finger_collision,
            baseVisualShapeIndex=finger_visual,
            basePosition=f2_pos,
        )

        # Keep track of finger uids in an attribute (not a single multibody)
        self._finger_uids = (f1, f2)
        return base_uid

    def _spawn_box_object(self, half_extents=(0.03, 0.03, 0.03), base_pos=(0.05, 0, 0.03)) -> int:
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents)
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        uid = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=base_pos,
        )
        return uid

    def reset(self, object_pose: Optional[Tuple[float, float, float]] = None) -> Dict:
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(self.time_step, physicsClientId=self._client)
        p.loadURDF("plane.urdf")

        gripper_base_pos = [0.0, 0.0, 0.18]
        self._gripper_uid = self._build_gripper(base_pos=gripper_base_pos)

        if object_pose is None:
            # random small offsets on table
            ox = 0.05 + 0.06 * (np.random.rand() - 0.5)
            oy = 0.02 * (np.random.rand() - 0.5)
            oz = 0.03
            object_pose = (ox, oy, oz)
        self._object_uid = self._spawn_box_object(base_pos=object_pose)

        # gripper state
        self._gripper_pos = np.array(gripper_base_pos, dtype=np.float32)
        self._gripper_open = 1.0

        obs = self._get_obs()
        return obs

    def _get_object_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self._object_uid)
        return np.array(pos), np.array(orn)

    def _get_obs(self) -> Dict:
        pc = self.sample_object_pointcloud(self.num_points)
        return {
            "pointcloud": pc.astype(np.float32),
            "gripper_pos": self._gripper_pos.copy(),
            "gripper_open": float(self._gripper_open),
        }

    def sample_object_pointcloud(self, n: int) -> np.ndarray:
        """Uniformly sample points on the object's box surface in gripper base frame.

        Returns (n, 3) points.
        """
        pos, _ = self._get_object_pose()
        # Use the collision shape extents we created earlier (approx 0.03)
        he = np.array([0.03, 0.03, 0.03])

        samples = []
        while len(samples) < n:
            # sample a face then sample u,v in [-1,1]
            face = np.random.randint(0, 6)
            u = (np.random.rand() - 0.5) * 2.0
            v = (np.random.rand() - 0.5) * 2.0
            if face == 0:
                pt = pos + np.array([he[0], u * he[1], v * he[2]])
            elif face == 1:
                pt = pos + np.array([-he[0], u * he[1], v * he[2]])
            elif face == 2:
                pt = pos + np.array([u * he[0], he[1], v * he[2]])
            elif face == 3:
                pt = pos + np.array([u * he[0], -he[1], v * he[2]])
            elif face == 4:
                pt = pos + np.array([u * he[0], v * he[1], he[2]])
            else:
                pt = pos + np.array([u * he[0], v * he[1], -he[2]])
            samples.append(pt)
        pts = np.stack(samples, axis=0)
        # transform into gripper base frame (gripper base at self._gripper_pos)
        pts = pts - self._gripper_pos
        return pts

    def step(self, action: np.ndarray, n_substeps: int = 8) -> Tuple[Dict, float, bool, Dict]:
        """Apply action and step simulation.

        Action: [dx, dy, dz, gripper_cmd]
        """
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (4,)
        # clip small steps for stability
        delta = np.clip(action[:3], -0.02, 0.02)
        self._gripper_pos += delta
        self._gripper_open = float(np.clip(action[3], 0.0, 1.0))

        # teleport gripper base and fingers
        p.resetBasePositionAndOrientation(self._gripper_uid, self._gripper_pos.tolist(), [0, 0, 0, 1])
        # place fingers relative to base along y-axis by open fraction
        spread = 0.02 + 0.03 * self._gripper_open
        f1_pos = (self._gripper_pos[0], self._gripper_pos[1] + spread, self._gripper_pos[2])
        f2_pos = (self._gripper_pos[0], self._gripper_pos[1] - spread, self._gripper_pos[2])
        p.resetBasePositionAndOrientation(self._finger_uids[0], f1_pos, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self._finger_uids[1], f2_pos, [0, 0, 0, 1])

        for _ in range(n_substeps):
            p.stepSimulation(physicsClientId=self._client)
            if self.gui:
                time.sleep(self.time_step)

        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = False
        info = {}
        return obs, float(reward), done, info

    def compute_reward(self, obs: Dict) -> float:
        """Compute reward using the optional grasp_scorer or a heuristic.

        Reward components:
         - initial_pointnet_score: returned by grasp_scorer if provided
         - lift_reward: positive if object's z increases after a close action
         - move_reward: positive for translating object towards target region
         - contact_area_approx: estimated by closeness of finger locations to
           sampled object points (proxy for contact area)
        """
        pc = obs["pointcloud"]
        # guess grasp point: center of point cloud in gripper frame
        grasp_point = pc.mean(axis=0)

        score = 0.0
        if self.grasp_scorer is not None:
            try:
                score = float(self.grasp_scorer(pc, grasp_point))
            except Exception:
                score = 0.0

        # heuristic contact proxy: fraction of points close to finger positions
        f1_pos = np.array(p.getBasePositionAndOrientation(self._finger_uids[0])[0]) - self._gripper_pos
        f2_pos = np.array(p.getBasePositionAndOrientation(self._finger_uids[1])[0]) - self._gripper_pos
        d1 = np.linalg.norm(pc - f1_pos[None, :], axis=-1)
        d2 = np.linalg.norm(pc - f2_pos[None, :], axis=-1)
        contact_proxy = (np.mean(d1 < 0.015) + np.mean(d2 < 0.015)) * 0.5

        # lift detection: simple check whether object Z is above table by margin
        obj_pos, _ = self._get_object_pose()
        lift_bonus = float(obj_pos[2] > 0.06)

        # move reward: how far object moved along +x from initial spawn roughly
        # (we don't track initial explicitly here; use x position as proxy)
        move_bonus = float(obj_pos[0] > 0.08)

        total = 1.0 * score + 1.5 * lift_bonus + 0.8 * move_bonus + 1.2 * contact_proxy
        return total

    def close(self):
        p.disconnect(self._client)
        self._client = None


if __name__ == "__main__":
    # quick manual smoke test
    env = SimpleGripperEnv(gui=False)
    obs = env.reset()
    print("Initial obs keys:", list(obs.keys()))
    for i in range(20):
        action = np.array([0.01, 0.0, -0.002, 0.0]) if i < 10 else np.array([-0.005, 0.0, 0.0, 0.0])
        obs, r, d, _ = env.step(action)
        print(f"step {i} reward={r:.3f}")
    env.close()
