# Unitree Grasping — Agent Handoff

> Last updated: 2026-07-06  
> Use this to kick-start a new agent chat. **Open the repo folder in Cursor first:**
> `/home/swekar/Documents/Collab_Research/unitree-grasping`

---

## Project & paths

| Item | Value |
|------|--------|
| **Repo** | `/home/swekar/Documents/Collab_Research/unitree-grasping` |
| **Branch** | `unitree-grasp-isaaclab` (commit `89a4183`, may need push) |
| **Remote** | `https://github.com/ukachyuthan/unitree-grasping.git` |
| **Conda env** | `unitree_isaaclab` (Isaac Lab 5.1 + Isaac Sim 5.1) |
| **GPU** | RTX 5050 8GB |

---

## What we're building (Path A)

- **Policy:** PointNet(PC) → one 3D grasp point per episode (object-local frame)
- **Execution:** Scripted IK approach → gripper close → lift (135 physics steps)
- **Reward:** `0.75 × lift_reward + 0.25 × leg_stillness`
- **Architecture:** Single-decision bandit; custom PPO in `wrappers/isaaclab/scripts/train_grasp_pose.py`

---

## What was broken (root cause of `mean_reward = 0`)

1. PhysX Jacobian was wrong for G1 palm — arm didn't reach the object
2. Objects drifted during approach
3. Grasp lock never triggered (`t >= 0.99` but max `t = 0.95`)
4. Binary reward + broken execution → always 0

---

## What was fixed (commit `89a4183`)

### `wrappers/isaaclab/envs/grasp_pose_env.py`

- **Numeric IK** (finite-diff Jacobian on palm) — approach error ~0.8 cm
- **Object pinning** during approach/close
- **Simulated grasp weld** after gripper close (object follows palm)
- **Leg/torso/right-arm held** at home every step (`_hold_idle_joints`)
- **Leg stillness reward** (25% of total)
- Continuous lift reward restored

### `wrappers/isaaclab/envs/grasp_pose_env_cfg.py`

- Fixed spawn at `(0.30, 0.0)`, `spawn_z_offset=0.045`
- `elbow_pitch=1.05` for lower EE
- `leg_joint_names`, reward weights

### `wrappers/isaaclab/scripts/train_grasp_pose.py`

- TensorBoard → `<run>/tb/`
- `metrics.jsonl` every iteration
- Logs: `lift`, `leg`, `policy_loss`, `value_loss`, `entropy`, `grad_norm`

### Other

- `debug_ik.py`, `play_grasp_pose.py` (`--debug_action zero|random`)
- `scripts/run_next_grasp_train.sh` (auto-launcher; watcher was killed manually)

---

## Training history

### Smoke run (stopped at ~iter 460/500)

| | |
|--|--|
| **Log dir** | `data/grasp_logs/grasp_pose_envs8_20260705_172354/` |
| **Checkpoints** | `grasp_pose_100.pt`, `_200.pt`, `_300.pt`, `_400.pt` |
| **Config** | 8 envs, cold start, no pretrain |
| **Reward** | Plateaued ~0.99–1.0 by iter 20 |

### Eval on `grasp_pose_400.pt` (old policy, new env code)

| Mode | Mean reward | Success (≥3 cm) |
|------|-------------|-----------------|
| Policy | 0.10 | 0% |
| Zero action `[0,0,0]` | 0.30 | 0% |
| Random action | 0.29 | 10% |

**Policy collapsed** to corner action `[-1, -1, ~0.9]` — not learning shape-specific grasps.

### Eval video

`data/viz/grasp_pose_eval_iter100.mp4` (iter 100, old env — 80% success in that clip)

---

## Known caveats

1. **Grasp is simulated** (welded to palm after close), not real contact physics
2. **Numeric IK is slow** — 5 `sim.forward()` per approach step; 32 envs may be heavy on 8GB GPU
3. **Old checkpoints invalid** for new env (IK, reward, spawn, leg penalty all changed)
4. **Pretrain still useful** — `data/grasp_weights/grasp_pretrain_best.pt`
5. **Friend's branch** — Isaac Sim work in `wrappers/isaacsim/` only (`COLLAB.md`)

---

## Immediate next steps

### 1. Push code

```bash
cd ~/Documents/Collab_Research/unitree-grasping
git push -u origin unitree-grasp-isaaclab
```

### 2. Start improved training

```bash
conda activate unitree_isaaclab
cd ~/Documents/Collab_Research/unitree-grasping
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes

python wrappers/isaaclab/scripts/train_grasp_pose.py \
  --headless --device cuda:0 \
  --num_envs 32 --max_iters 2000 --seed 42 \
  --pretrain data/grasp_weights/grasp_pretrain_best.pt
```

Use `--num_envs 16` if GPU OOMs.

### 3. Monitor

```bash
# TensorBoard
tensorboard --logdir data/grasp_logs
# → http://127.0.0.1:6006/

# Live metrics
tail -f data/grasp_logs/<run_name>/metrics.jsonl
```

### 4. Eval after ~100 iters

```bash
python wrappers/isaaclab/scripts/play_grasp_pose.py \
  --headless --device cpu \
  --checkpoint data/grasp_logs/<run>/grasp_pose_100.pt \
  --num_episodes 10

# Baselines
python wrappers/isaaclab/scripts/play_grasp_pose.py --debug_action zero ...
python wrappers/isaaclab/scripts/play_grasp_pose.py --debug_action random ...

# Video
python wrappers/isaaclab/scripts/play_grasp_pose.py --headless --enable_cameras \
  --checkpoint data/grasp_logs/<run>/grasp_pose_100.pt \
  --video --num_envs 1 --out data/viz/eval.mp4
```

**Success criteria for next run:**

- Policy actions vary by object shape (not constant corner)
- `mean_leg_still` stays high in TensorBoard
- Lift reward beats zero-action baseline

---

## Key files

```
wrappers/isaaclab/
  envs/grasp_pose_env.py        # env + IK + rewards
  envs/grasp_pose_env_cfg.py    # spawn, phases, leg reward weights
  scripts/train_grasp_pose.py   # PPO + TensorBoard + metrics.jsonl
  scripts/play_grasp_pose.py    # eval + video
  scripts/debug_ik.py           # IK diagnostic PNG
models/grasp_pose_actor_critic.py
data/grasp_weights/grasp_pretrain_best.pt
COLLAB.md                       # collaboration rules
```

---

## CLI gotchas

| Wrong | Right |
|-------|-------|
| `--max_iterations 500` | `--max_iters 500` |

---

## Open problems (future work)

1. Fix center-grasp execution so `[0,0,0]` reliably works
2. Replace numeric IK with proper G1 IK (faster at scale)
3. Real contact grasp physics (replace weld hack)
4. Per-shape eval to verify policy uses point cloud
5. Friend continues on `wrappers/isaacsim/` separately

---

## Prompt for new agent

```
Read AGENT_HANDOFF.md, push unitree-grasp-isaaclab if needed,
start pretrain-backed training (32 envs, 2000 iters), and eval at iter 100.
```
