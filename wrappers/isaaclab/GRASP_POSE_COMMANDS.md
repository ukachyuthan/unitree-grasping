# Grasp-pose train & eval commands

Franka top-down grasping with **gated pose IK** (wrist yaw toward object once the palm is within 4 cm; lift is position-only).

Always run from the **repo root** with `conda activate unitree_isaaclab`.

```bash
conda activate unitree_isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
cd ~/Documents/Collab_Research/unitree-grasping
```

## Train

```bash
# 96 envs (RTX 5050 8 GB — do not run video/eval at the same time)
python wrappers/isaaclab/scripts/train_grasp_pose.py \
  --headless --num_envs 96 --max_iters 2000

# Warm-start from 3-D checkpoint into 5-D tilt/roll policy (stronger orient IK enabled)
python wrappers/isaaclab/scripts/train_grasp_pose.py \
  --headless --num_envs 96 --max_iters 2000 \
  --resume wrappers/isaaclab/data/grasp_logs/grasp_pose_envs96_20260718_205748/grasp_pose_200.pt

tail -f /tmp/grasp_train_5d.log

# Optional encoder warm start (pretrain only loads PointNet weights)
python wrappers/isaaclab/scripts/train_grasp_pose.py \
  --headless --num_envs 96 --max_iters 2000 \
  --pretrain data/grasp_weights/grasp_pretrain_best.pt
```

Logs and checkpoints: `data/grasp_logs/grasp_pose_envs<N>_<timestamp>/`

Monitor:

```bash
tail -f /tmp/grasp_train_ik.log          # if you tee training output
tensorboard --logdir data/grasp_logs/<run>/tb
```

## Evaluate

```bash
# 10-object deterministic cycle (seed 42)
python wrappers/isaaclab/scripts/play_grasp_pose.py \
  --headless --num_envs 1 --num_episodes 10 --cycle_shapes --seed 42 \
  --checkpoint wrappers/isaaclab/data/grasp_logs/grasp_pose_envs96_20260718_205748/grasp_pose_200.pt

# Broader stats (random objects)
python wrappers/isaaclab/scripts/play_grasp_pose.py \
  --headless --num_envs 1 --num_episodes 30 --seed 42 \
  --checkpoint wrappers/isaaclab/data/grasp_logs/grasp_pose_envs96_20260718_205748/grasp_pose_200.pt
```

Success = episode reward ≥ 0.5 (object lifted ≥ 5 cm). Printed as `lift_ok=True`.

## Record video

```bash
python wrappers/isaaclab/scripts/play_grasp_pose.py \
  --headless --enable_cameras --video --video_episodes 10 \
  --cycle_shapes --num_envs 1 --seed 42 \
  --checkpoint wrappers/isaaclab/data/grasp_logs/grasp_pose_envs96_20260718_205748/grasp_pose_200.pt \
  --out data/viz/grasp_pose_10objects.mp4
```

Grasp markers (yellow = policy point, blue = palm, green = finger midpoint) auto-enable with `--video`.

## Debug (no checkpoint)

```bash
# Scripted execution only — zero action at object centre
python wrappers/isaaclab/scripts/play_grasp_pose.py \
  --headless --num_envs 1 --debug_action zero

# Single-episode contact / lift stats
python wrappers/isaaclab/scripts/debug_franka_grasp.py --headless --num_envs 1
```

## Best checkpoint (current run, stopped at iter 740)

| Checkpoint | Train success | 10-shape eval |
|------------|---------------|---------------|
| `grasp_pose_200.pt` | 84% @ iter 200 | **90%** mean reward 0.65 |

Path: `wrappers/isaaclab/data/grasp_logs/grasp_pose_envs96_20260718_205748/grasp_pose_200.pt`

## Residual predictive control (Path A+)

Closed-loop policy refines scripted IK every 4 physics steps (~15 Hz). Warm-starts grasp head from Path A.

```bash
# Train (default: warm-start grasp_pose_200.pt, freeze grasp 200 iters)
python wrappers/isaaclab/scripts/train_grasp_pose_residual.py \
  --headless --num_envs 64 --max_iters 2000 --num_steps_per_env 48 \
  --pretrain wrappers/isaaclab/data/grasp_logs/grasp_pose_envs96_20260718_205748/grasp_pose_200.pt \
  --freeze_grasp_iters 200 2>&1 | tee /tmp/grasp_residual_train.log

# Eval
python wrappers/isaaclab/scripts/play_grasp_pose_residual.py \
  --headless --num_envs 1 --num_episodes 10 --cycle_shapes --seed 42 \
  --checkpoint wrappers/isaaclab/data/grasp_logs/grasp_residual_envs64_<timestamp>/grasp_residual_final.pt

# Video
python wrappers/isaaclab/scripts/play_grasp_pose_residual.py \
  --headless --enable_cameras --video --video_episodes 5 --cycle_shapes \
  --checkpoint wrappers/isaaclab/data/grasp_logs/grasp_residual_envs64_<timestamp>/grasp_residual_final.pt \
  --out data/viz/grasp_residual_rollout.mp4
```

**Architecture:** obs = PC (384) + proprio (12) → grasp(5) PC-only head + residual(7) from PC+proprio.  
Success metric = terminal lift reward ≥ 0.5 (not dense step rewards).
