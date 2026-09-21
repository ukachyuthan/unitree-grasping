# Grasp-pose train & eval commands

Franka grasping (Path A): PointNet → 5D grasp pose (xyz + wrist tilt + roll); approach/close/lift scripted with gated orient IK, safe-grip smoothstep, and optional pick-and-place.

Always run from the **repo root** with `conda activate unitree_isaaclab`.

```bash
conda activate unitree_isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
cd ~/Documents/Collab_Research/unitree-grasping
```

**8 GB GPU (RTX 5050):** do not run training and video/eval at the same time. Kill stale GPU Python if eval OOMs.

---

## Train

```bash
cd wrappers/isaaclab

# Fresh run (96 envs)
python scripts/train_grasp_pose.py \
  --headless --num_envs 96 --max_iters 2000 --seed 42

# Resume from latest merged run (peak ~84.8% @ iter 496)
python scripts/train_grasp_pose.py \
  --headless --num_envs 96 --max_iters 2000 --seed 42 \
  --resume data/grasp_logs/grasp_pose_envs96_20260726_191922/grasp_pose_500.pt

# Procedural-only ablation (no YCB objects yet)
python scripts/train_grasp_pose.py \
  --headless --num_envs 96 --max_iters 2000 \
  --use_real_objects false \
  --resume data/grasp_logs/grasp_pose_envs96_20260726_191922/grasp_pose_825.pt

# With real YCB + camera PC (after data pipeline below; lower num_envs if OOM)
python scripts/train_grasp_pose.py \
  --headless --enable_cameras --num_envs 64 --max_iters 2000 \
  --use_real_objects true
```

Logs and checkpoints: `wrappers/isaaclab/data/grasp_logs/grasp_pose_envs<N>_<timestamp>/`

Monitor:

```bash
tail -f wrappers/isaaclab/data/grasp_logs/grasp_pose_envs96_20260726_191922/metrics.jsonl
tensorboard --logdir wrappers/isaaclab/data/grasp_logs/<run>/tb
```

### Failure replay queue → VR demo dataset

Failed grasps are replayed during training, and the ones that keep failing are
set aside so you can demonstrate them by hand in VR (`../vr-grasping-project`).
This is on by default.

1. Every failed episode (`lift_reward < 0.5`) joins a **replay queue**, storing
   its exact shape and settled pose.
2. On each step, every env has a 10% chance that its next episode replays a
   random queued failure instead of a fresh random spawn
   (`GraspPoseEnv.queue_replays`). Replays train the policy like any other
   episode.
3. How a replay resolves its entry:
   - **Succeeds:** the policy has learned it, and it leaves the queue.
   - **Fails:** it counts one strike.
   - **4 failed replays:** it's exported as a VR case and leaves the queue.
     The original failure doesn't count toward the 4.
4. When the queue is full (512), new failures are turned away instead of
   evicting queued ones. Otherwise entries would churn out before any reached 4
   replays.
5. Nothing is queued for the first 50 iterations, since an untrained policy
   fails everything.

At the defaults, an entry is replayed about once every
`512 / (0.1 × num_envs)` steps. With 64 envs that's about 80 steps (5 iters of
16 steps), so a case needs roughly 20 iterations of steady failure to export.
To export faster, raise `--vr_replay_prob` or shrink `--vr_queue_capacity`.

Output goes to `<repo>/data/vr_failures/<run_name>/`, whatever directory you
launch from:

```
manifest.json          exported cases
monitor.json           live queue: size, entries by replay strikes, per-shape counts
cases/<case_id>.json   object pose, point cloud, mesh ref, original + 4 failed replays
objects/<shape>.obj    mesh copy, so the run folder is self-contained
demos/<case_id>/       written by the VR app
```

```bash
watch -n5 'python -m json.tool data/vr_failures/<run>/monitor.json | head -40'
```

TensorBoard also gets `vr/cases_exported`, `vr/queue_size` and
`vr/replay_success_rate`. The last is the fraction of replays that now succeed,
a direct read on whether the policy is fixing its hard cases. Replays are part
of the training batch, so `train/success_rate` includes them.

Flags: `--vr_fail_threshold 4`, `--vr_replay_prob 0.1`, `--vr_queue_capacity 512`,
`--vr_success_lift 0.5`, `--vr_warmup_iters 50`, `--vr_max_cases 200`,
`--vr_failures_dir <dir>` (`--vr_failures_dir ''` disables it, and replays
with it).

Collect the demos you've recorded as `(c1, c2)` contact pairs for BC:

```bash
python scripts/load_vr_demos.py --out data/vr_failures/demos.npz
```

The curator's tests don't need Isaac: `python3 -m pytest tests/test_vr_failures.py`.

---

## Evaluate (stats only)

Success = episode reward ≥ 0.5 (object lifted ≥ 3 cm). Printed as `lift_ok=True`.

```bash
cd wrappers/isaaclab

# 10-object deterministic cycle (seed 42)
python scripts/play_grasp_pose.py \
  --headless --num_envs 1 --num_episodes 10 --cycle_shapes --seed 42 \
  --checkpoint data/grasp_logs/grasp_pose_envs96_20260726_191922/grasp_pose_500.pt

# Broader stats (random objects)
python scripts/play_grasp_pose.py \
  --headless --num_envs 1 --num_episodes 30 --seed 42 \
  --checkpoint data/grasp_logs/grasp_pose_envs96_20260726_191922/grasp_pose_500.pt
```

---

## Record video

Grasp markers (yellow = policy point, blue = palm, green = finger midpoint) auto-enable with `--video`.

```bash
cd wrappers/isaaclab

# Current best checkpoint (peak train iter ~496)
python scripts/play_grasp_pose.py \
  --headless --enable_cameras --video --video_episodes 10 \
  --cycle_shapes --num_envs 1 --seed 42 \
  --checkpoint data/grasp_logs/grasp_pose_envs96_20260726_191922/grasp_pose_500.pt \
  --out ../../data/viz/grasp_iter500_10objects.mp4

# Prior merged run (iter 550, ~80% eval)
python scripts/play_grasp_pose.py \
  --headless --enable_cameras --video --video_episodes 10 \
  --cycle_shapes --num_envs 1 --seed 42 \
  --checkpoint data/grasp_logs/grasp_pose_envs96_20260725_163218/grasp_pose_550.pt \
  --out ../../data/viz/grasp_merged_iter550_10objects.mp4
```

Output videos: `data/viz/`

---

## Real-object data pipeline (YCB + GraspNet)

Run once before training with `use_real_objects=true` or GraspNet reward.

```bash
# From repo root
python scripts/fetch_ycb.py
python scripts/generate_ycb_meshes.py \
  --raw data/ycb_raw --out data/objects/train --eval_out data/objects/eval \
  --n_per_family 8 --eval_frac 0.3 --seed 0

# Requires contact-graspnet-pytorch + a pretrained checkpoint path
python scripts/generate_graspnet_labels.py \
  --data data/objects/train --checkpoint /path/to/contact_graspnet_ckpt.pt
python scripts/generate_graspnet_labels.py \
  --data data/objects/eval --checkpoint /path/to/contact_graspnet_ckpt.pt
```

Install mesh deps if needed: `pip install trimesh manifold3d shapely`

---

## Debug (no checkpoint)

```bash
cd wrappers/isaaclab

python scripts/play_grasp_pose.py \
  --headless --num_envs 1 --debug_action zero

python scripts/debug_franka_grasp.py --headless --num_envs 1
```

---

## Checkpoints reference

| Run | Best checkpoint | Peak train success | Notes |
|-----|-----------------|-------------------|-------|
| `grasp_pose_envs96_20260726_191922` | `grasp_pose_500.pt` | **84.8%** @ iter 496 | Current merged stack + orient IK |
| `grasp_pose_envs96_20260725_163218` | `grasp_pose_550.pt` | **86.8%** @ iter 543 | Pre-real-PC merge; video recorded |
| `grasp_pose_envs96_20260718_205748` | `grasp_pose_200.pt` | 84% @ iter 200 | 3D→5D warm-start source |

Paths relative to `wrappers/isaaclab/data/grasp_logs/`.

---

## Residual predictive control (Path A+, experimental)

```bash
cd wrappers/isaaclab

python scripts/train_grasp_pose_residual.py \
  --headless --num_envs 64 --max_iters 2000 --num_steps_per_env 48 \
  --pretrain data/grasp_logs/grasp_pose_envs96_20260718_205748/grasp_pose_200.pt \
  --freeze_grasp_iters 200

python scripts/play_grasp_pose_residual.py \
  --headless --enable_cameras --video --video_episodes 5 --cycle_shapes \
  --checkpoint data/grasp_logs/grasp_residual_envs64_<timestamp>/grasp_residual_final.pt \
  --out ../../data/viz/grasp_residual_rollout.mp4
```
