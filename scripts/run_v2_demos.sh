#!/bin/bash
# RL + human demos on the real-scale object set: run_v2_rl.sh's recipe and seeds, plus the
# robust VR round-3 demos (data/vr_round3_train) replayed at their exact failure cases. Half
# of the demo episodes execute the demo's grasp, only demo grasps that lifted are imitated,
# and the imitation weight fades out over 40 iterations. Fixed before any test result.
#
#   scripts/run_v2_demos.sh            (from the repo root; compare with scripts/eval_v2_curve.sh)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate unitree_isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
source "$(dirname "$0")/v2_env.sh"
cd ~/unitree-grasping/wrappers/isaaclab
OUT=$PAPER_OUT; mkdir -p $OUT/logs
ITERS=${ITERS:-150}
PRETRAIN=../../data/grasp_weights_v2/grasp_6dof_pretrain_best.pt
RECIPE="--headless --num_envs 256 --max_iters $ITERS --pretrain $PRETRAIN --reset_std 0.3 --entropy_coef 0.01 --min_std 0.05 --tilted_spawn_prob 0.1"
DEMOS="--demos_dir ../../data/vr_round3_train --demo_replay_prob 0.1 --demo_exec_prob 0.5 --bc_filter --bc_only_lifted --bc_decay_iters 40"

run() {        # run <seed>
  local name=D_s$1 dir=data/grasp_logs/paper_v2/D_s$1
  echo "$(date +%H:%M) start $name" >> $OUT/progress.txt
  python scripts/train_grasp_pose.py $RECIPE $DEMOS --seed $1 --log_dir $dir \
    --vr_failures_dir ../../data/vr_failures_v2_D > $OUT/logs/$name.log 2>&1
  echo "$(date +%H:%M) trained $name → $(ls $dir/*/grasp_pose_final.pt 2>/dev/null | head -n 1)" >> $OUT/progress.txt
}

run 1 & sleep 120; run 2 & wait
run 3
echo "$(date +%H:%M) v2 RL + demos done" >> $OUT/progress.txt
