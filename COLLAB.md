# Collaboration: Isaac Lab vs Isaac Sim

Two people, two sim stacks, **one shared core**. Sim-specific code lives under `wrappers/` and never needs to work on the other person's machine.

## What is shared (merge via PR — both care)

These have **no Isaac Sim / Isaac Lab imports** (or should not):

| Path | Purpose |
|------|---------|
| `models/` | PointNet, `GraspPoseActorCritic`, checkpoints format |
| `scripts/generate_meshes.py` | Procedural OBJ + point clouds + grasp labels |
| `scripts/generate_objects.py` | Shape generators |
| `scripts/pretrain_grasp.py` | Supervised encoder warm-start |
| `scripts/visualize_pipeline.py` | Offline matplotlib viz |
| `scripts/render_demo.py` | Offline MP4 demo (no sim) |
| `data/objects/` | Train/eval meshes, `.npy` PCs, `_grasps.json` |
| `requirements.txt` | Core deps (torch, trimesh, …) |

**Contract both sides must honor:**
- Checkpoint keys: `{"model": state_dict, "iteration": int}`
- Obs dim: `128 points × 3 = 384` (Path A grasp-pose)
- Action dim: `3` (tanh grasp position in robot-base frame)
- Object layout: `data/objects/train/<shape_family>/000.{obj,usd,_pc.npy,_grasps.json}`

## What is sim-specific (stay in `wrappers/`)

| Your wrapper (`wrappers/isaaclab/`) | Friend's wrapper (`wrappers/isaacsim/`) |
|-------------------------------------|----------------------------------------|
| Isaac Lab 5.x + Isaac Sim 5.1 | Isaac Sim standalone (older Ubuntu) |
| `AppLauncher`, `DirectRLEnv` | omni.isaac scene scripts |
| USD via `MeshConverter` | Their USD/OBJ loading |
| Custom PPO in `train_grasp_pose.py` | Their training loop / eval |

**Rule:** Do not edit the other person's `wrappers/<stack>/` folder. Do not put sim imports in `models/` or mesh scripts.

## Current repo state (honest)

Today, Isaac Lab code still lives at the repo root (`grasping/grasp_pose_env.py`, `scripts/train_grasp_pose.py`). That works for you but is not ideal for collaboration.

**Target:** move Isaac Lab glue into `wrappers/isaaclab/` over time. Friend adds only under `wrappers/isaacsim/`. Root stays sim-agnostic.

## Folder layout

```
unitree-grasping/
├── models/                    # SHARED
├── data/objects/              # SHARED (meshes too large — use git LFS or regenerate)
├── scripts/                   # SHARED offline tools only (see list above)
├── grasping/                  # SHARED logic where possible (see note on pointcloud_utils)
│
├── wrappers/
│   ├── isaaclab/              # YOU — Isaac Lab 5.1, Ubuntu 24
│   │   ├── README.md
│   │   ├── requirements.txt   # pins isaaclab, isaacsim versions
│   │   ├── envs/              # grasp_pose_env.py, *_cfg.py (Isaac Lab)
│   │   └── scripts/           # train, play, convert_to_usd, add_physics
│   │
│   └── isaacsim/              # FRIEND — Isaac Sim standalone
│       ├── README.md
│       ├── requirements.txt
│       ├── envs/              # their scene + robot setup
│       └── scripts/           # their train / play entry points
│
└── COLLAB.md                  # this file
```

## Git workflow

1. Branch off `feat-pivot-to-grasping` (or `main` once merged).
2. **Shared changes** → PR touching only `models/`, `scripts/{generate,pretrain,visualize}*`, `data/` manifests.
3. **Your sim glue** → PR only under `wrappers/isaaclab/`.
4. **Friend's sim glue** → PR only under `wrappers/isaacsim/`.
5. Never commit: checkpoints, `data/grasp_logs/`, `.env`, `rl_unitree/`, conda paths.

## Environment matrix (fill in friend's column)

| | You (Isaac Lab) | Friend (Isaac Sim) |
|---|-----------------|---------------------|
| OS | Ubuntu 24.04 | Ubuntu 22.04 (?) |
| Python | 3.11 conda `unitree_isaaclab` | their venv |
| Sim | Isaac Sim 5.1 + Isaac Lab 5.1 | Isaac Sim 4.x standalone (?) |
| Train entry | `wrappers/isaaclab/scripts/train_grasp_pose.py` | `wrappers/isaacsim/scripts/...` |
| Smoke test | `play_grasp_pose.py --num_envs 1` | their play script |

## Changes made on Isaac Lab side (your wrapper)

These are **your** stack adaptations — friend does not need them:

- Custom PPO loop in `train_grasp_pose.py` (rsl_rl 5.x `OnPolicyRunner` API mismatch)
- `AppLauncher.add_app_launcher_args` for `--headless`, `--device`, `--enable_cameras`
- `GraspPoseEnvCfg.sim.device = cuda:0` (was cpu)
- USD pipeline: `convert_to_usd.py`, `add_physics_to_usd.py`
- Path A env: `grasp_pose_env.py` (bandit-style single grasp decision)
- `play_grasp_pose.py` for eval + MP4

Friend's Isaac Sim changes (IK, scene, etc.) belong in **`wrappers/isaacsim/`** only.

## How wrappers import the core

From any wrapper script:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # unitree-grasping/
sys.path.insert(0, str(REPO_ROOT))

from models.grasp_pose_actor_critic import GraspPoseActorCritic
# wrapper-local:
from envs.grasp_pose_env import GraspPoseEnv  # relative to wrappers/isaaclab/
```

Run from repo root:

```bash
cd ~/Documents/Collab_Research/unitree-grasping
conda activate unitree_isaaclab
python wrappers/isaaclab/scripts/train_grasp_pose.py --headless ...
```

## One small shared fix still needed

`grasping/pointcloud_utils.py` imports `quat_rotate` from Isaac Lab. Replace with a local torch implementation so the **core** is truly sim-free. Until then, treat `grasping/` as shared-but-Isaac-Lab-tainted for real-camera inference only.

## Migration checklist

- [x] Move Isaac Lab envs → `wrappers/isaaclab/envs/`
- [x] Move Isaac Lab scripts → `wrappers/isaaclab/scripts/`
- [x] Root `scripts/train_grasp_pose.py` etc. are thin shims (backward compatible)
- [x] Remove `isaaclab` import from `grasping/pointcloud_utils.py`
- [ ] Friend scaffolds `wrappers/isaacsim/` with README + play script
- [ ] Add `tests/test_pointnet.py` (no sim) — runs in CI for both
