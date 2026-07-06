#!/usr/bin/env bash
# Wait for the smoke-test run to finish, then launch the improved training run.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WAIT_FOR="${REPO}/data/grasp_logs/grasp_pose_envs8_20260705_172354/grasp_pose_final.pt"

echo "[run_next] waiting for current training to finish..."
echo "[run_next] watching: $WAIT_FOR"
while [[ ! -f "$WAIT_FOR" ]]; do
  sleep 30
done
echo "[run_next] previous run complete at $(date)"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_isaaclab
cd "$REPO"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1

exec python wrappers/isaaclab/scripts/train_grasp_pose.py \
  --headless \
  --device cuda:0 \
  --num_envs 32 \
  --max_iters 2000 \
  --seed 42 \
  --pretrain data/grasp_weights/grasp_pretrain_best.pt
