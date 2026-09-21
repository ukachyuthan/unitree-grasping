"""Config for residual predictive control on top of scripted grasp execution."""

from __future__ import annotations

from isaaclab.utils import configclass

from envs.grasp_pose_env_cfg import (
    GraspPoseEnvCfg,
    EXEC_STEPS,
    NUM_PC_POINTS,
)

# Grasp (5) + residual EE delta (7): Δpos(3) Δrot(3) Δgrip(1)
NUM_GRASP_ACTIONS: int = 5
NUM_RESIDUAL_ACTIONS: int = 7
NUM_ACTIONS: int = NUM_GRASP_ACTIONS + NUM_RESIDUAL_ACTIONS

# Proprio appended to flat point cloud.
PROPRIO_DIM: int = 12
OBS_DIM: int = NUM_PC_POINTS * 3 + PROPRIO_DIM

# Policy acts every N physics steps (~15 Hz at 60 Hz sim).
POLICY_DECIMATION: int = 4
POLICY_STEPS_PER_EPISODE: int = (EXEC_STEPS + POLICY_DECIMATION - 1) // POLICY_DECIMATION


@configclass
class GraspPoseResidualEnvCfg(GraspPoseEnvCfg):
    decimation: int = POLICY_DECIMATION
    episode_length_s: float = (EXEC_STEPS / 60.0) * 1.5
    action_space: int = NUM_ACTIONS
    observation_space: int = OBS_DIM

    # Residual scales (action ∈ [-1, 1], applied over one policy interval).
    residual_pos_scale: float = 0.012       # metres total per policy step
    residual_rot_scale: float = 0.06        # rad axis-angle total per policy step
    residual_grip_scale: float = 0.008      # metres finger opening delta per policy step

    # Dense shaping (added each policy step; terminal step uses sparse lift reward).
    reach_reward_weight: float = 0.25
    reach_sigma_m: float = 0.05
    step_contact_reward_weight: float = 0.15
    step_lift_reward_weight: float = 0.35

    # Lift-only for v1 (avoid place-phase desync complexity).
    task_mode_prob_place: float = 0.0

    # Method 2: scripted antipodal grasp as the base; policy residual corrects IK.
    # If True, first-step grasp comes from labels (optionally + small policy Δ).
    use_antipodal_base: bool = True
    # Ignore policy grasp head entirely (pure residual on fixed antipodal).
    freeze_grasp_to_antipodal: bool = True
    # When freeze_grasp_to_antipodal=False, policy grasp is a tanh delta scaled by this.
    antipodal_grasp_delta_scale: float = 0.25
    # Reject near-vertical closing axes (bad for top-down Path A).
    antipodal_max_abs_approach_z: float = 0.85
