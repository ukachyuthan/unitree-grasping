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

## 6-DoF grasping (default)

`--grasp_mode six_dof` (default in train and play) learns **where and from which
direction** to grasp, from the point cloud alone:

- **Observation:** 128-point cloud, world-aligned axes, origin at the centroid of
  the observed (noisy) points. No object pose is used.
- **Action (7):** contact points c1, c2 (in that same frame) + approach angle ψ about
  the jaw axis c1→c2. ψ = 0 is the most top-down approach, ±90° is horizontal (from
  the side). The approach can never come from below the table.
- **Executor (`--executor floating`, default):** a floating Franka hand, not the arm,
  so the reward measures the grasp and not this arm's reachability. It is teleported
  to a pre-grasp 10 cm back along the approach, inserts in a straight line, closes,
  and lifts. On a real robot a motion planner executes the chosen grasp.
  `--executor franka_arm` runs the full arm with 7-joint pose IK instead.
- **Execution noise:** ±1 cm / ±5° (std) on the executed pose, so learned grasps
  tolerate real-arm placement error. `play --no_exec_noise` turns it off.
- **Success:** the object reaches `lift_target_m` (5 cm) with finger force on the
  object, not the table. Episodes are lift-only (no pick-and-place).

**Warm start from Contact-GraspNet** (so RL does not start cold). One-time setup:
a separate `graspnet` conda env (torch cu128) with
[contact_graspnet_pytorch](https://github.com/elchun/contact_graspnet_pytorch) cloned to
`~/contact_graspnet_pytorch` (`pip install --no-build-isolation -e .`, plus `pyrender embreex`).

```bash
# 1. Label every object: ray-cast depth views of the object on a table, run the
#    pretrained Panda Contact-GraspNet, keep the best grasps → <instance>_graspnet.json
#    (~12 s per object; these also switch on the env's graspnet reward term)
conda activate graspnet
PYOPENGL_PLATFORM=egl python scripts/generate_graspnet_labels.py --data data/objects/train
PYOPENGL_PLATFORM=egl python scripts/generate_graspnet_labels.py --data data/objects/eval

# 2. Distil them into the RL actor (BC, env-identical observations; sim-free)
conda activate unitree_isaaclab
python scripts/pretrain_grasp_6dof.py          # → data/grasp_weights/grasp_6dof_pretrain_best.pt
```

```bash
cd wrappers/isaaclab

# Train (6-DoF, floating hand), warm-started from the GraspNet BC policy
python scripts/train_grasp_pose.py --headless --num_envs 256 --max_iters 2000 \
  --pretrain ../../data/grasp_weights/grasp_6dof_pretrain_best.pt

# Scripted antipodal grasps from a chosen approach angle, recorded + rendered
python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 5 --video_episodes 5 \
  --debug_action antipodal --debug_psi 90 --no_exec_noise --record ../../data/viz/side.npz
python scripts/render_recording.py ../../data/viz/side.npz      # → data/viz/side.mp4

# Evaluate / record a trained 6-DoF checkpoint
python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 20 --video_episodes 10 \
  --checkpoint data/grasp_logs/<run>/grasp_pose_<iter>.pt --record ../../data/viz/eval.npz
```

### Closing the failure loop (demos → training)

Training exports the cases the policy keeps failing (see "Failure replay queue"
below). Someone demonstrates them, and the demos come back into training: each
demo case is replayed at its exact pose, with an imitation loss toward the
demonstrated grasp on top of PPO (`training/demo_replay.py`).

```bash
RUN=../../data/vr_failures/<run>

# 1. Hold out half the cases (per family) as a test of generalization; the VR app
#    hides them and the teacher skips them. Do this once, before any demos.
python ../../scripts/split_failure_cases.py $RUN

# 2a. Human demos: the VR app (../vr-grasping-project). Press IMPOSSIBLE on a case
#     no two-finger gripper could lift from that pose; it is then left out of the
#     metrics and the same (shape, yaw bin) is never exported again.
# 2b. Or the simulated teacher: best-ranked Contact-GraspNet grasp that lifts twice
python scripts/teach_failure_cases.py --headless --cases $RUN --out ../../data/teacher_demos
#     control: the same kind of demos on random objects and poses
python scripts/teach_failure_cases.py --headless --random_cases 300 --out ../../data/teacher_demos_random

# 3. Check the demos: does the Franka hand reproduce each one? → <root>/verify.json
python scripts/teach_failure_cases.py --headless --verify ../../data/vr_failures

# 4. Train with them (any mix of roots; --demo_sources webxr-hands|sim-teacher filters)
python scripts/train_grasp_pose.py --headless --num_envs 256 --max_iters 150 \
  --resume <checkpoint> --demos_dir ../../data/vr_failures ../../data/teacher_demos

# 5. Did it work? Policy success on the demonstrated and the held-out cases
python scripts/teach_failure_cases.py --headless --eval_cases $RUN --checkpoint <new checkpoint>
```

Training logs `demo/success_rate`: the policy's own success on demo episodes
(not the demo's), the loop's headline number. Demo and verdict files:

```
<run>/demos/<case_id>/<timestamp>.json   VR demo (contacts + hand frames; ψ = wrist → contacts)
<run>/verdicts/<case_id>.json            latest verdict: impossible | feasible
<run>/split.json                         demo / held-out case ids
```

The top-down contact-pair mode below is `--grasp_mode top_down`; its checkpoints
(6-D actions, object-frame observations) are not compatible with 6-DoF.

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
