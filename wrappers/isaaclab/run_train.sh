#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
exec python wrappers/isaaclab/scripts/train_grasp_pose.py "$@"
