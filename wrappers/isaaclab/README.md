# Isaac Lab wrapper (your machine)

Isaac Lab **5.1** + Isaac Sim **5.1**, Ubuntu 24, conda env `unitree_isaaclab`.

## Owns

Everything under this folder:

```
wrappers/isaaclab/
├── bootstrap.py          # sys.path setup (call after AppLauncher)
├── envs/                 # DirectRLEnv implementations
│   ├── grasp_pose_env.py
│   ├── grasp_pose_env_cfg.py
│   ├── g1_grasp_env.py
│   └── g1_grasp_env_cfg.py
└── scripts/              # train, play, USD tools
    ├── train_grasp_pose.py
    ├── play_grasp_pose.py
    ├── convert_to_usd.py
    └── ...
```

## Does not own (shared core at repo root)

- `models/` — PointNet, actor-critic checkpoints
- `grasping/pointcloud_utils.py` — sim-free camera math
- `scripts/pretrain_grasp.py`, `generate_meshes.py`, `visualize_pipeline.py`

## Run (from repo root)

```bash
conda activate unitree_isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
cd ~/Documents/Collab_Research/unitree-grasping

# Option A — wrapper path (preferred)
python wrappers/isaaclab/scripts/train_grasp_pose.py --headless --device cuda:0 --num_envs 32 \
  --pretrain data/grasp_weights/grasp_pretrain_best.pt

# Option B — shell helper
./wrappers/isaaclab/run_train.sh --headless --device cuda:0 --num_envs 32 \
  --pretrain data/grasp_weights/grasp_pretrain_best.pt

# Option C — legacy shim (still works)
python scripts/train_grasp_pose.py --headless ...
```

## Local-only (do not commit)

- `data/grasp_logs/`, large checkpoints
- conda env / Isaac install paths
