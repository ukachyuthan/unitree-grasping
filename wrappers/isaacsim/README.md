# Isaac Sim wrapper (friend's machine)

Isaac Sim **standalone** on older Ubuntu. This folder is **friend-owned** — do not require Isaac Lab here.

## Add here

```
wrappers/isaacsim/
├── README.md           ← fill in exact Isaac Sim version + Ubuntu
├── requirements.txt    ← pin omni/isaacsim packages only
├── envs/
│   └── g1_grasp_scene.py   # scene, robot spawn, table, objects
└── scripts/
    ├── train_grasp.py      # loads checkpoint from shared models/
    └── play_grasp.py       # rollout / video
```

## Import shared core

```python
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.grasp_pose_actor_critic import GraspPoseActorCritic
```

## Must match shared contract

- Load checkpoints: `torch.load(path)["model"]`
- Policy input: point cloud flat `(B, 384)` — 128 points in **robot-base frame**
- Policy output: `(B, 3)` tanh → scale to workspace (see `grasp_pose_env_cfg.py` bounds)
- Objects: read from `data/objects/train/<family>/000.usd` or `.obj`

## Fill in (friend)

| Field | Value |
|-------|-------|
| Ubuntu version | |
| Isaac Sim version | |
| Python version | |
| Train command | |
| Play command | |

## Do not commit

- Local Isaac Sim install
- Nucleus cache paths
- Machine-specific kit overrides
