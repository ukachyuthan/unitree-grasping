#!/bin/bash
# RL-only on the real-scale object set (scripts/v2_env.sh), from the v2 GraspNet warm start:
# seeds 1 and 2 side by side, then seed 3; each tested on the held-out flat / tilted draws
# (seeds 31 / 32, no execution noise, as eval_graspnet_baselines.sh). Repeated failures are
# exported to data/vr_failures_v2 for the next VR round.
#
#   scripts/run_v2_rl.sh            (from the repo root; ~5 h per pair of seeds)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate unitree_isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
source "$(dirname "$0")/v2_env.sh"
cd ~/unitree-grasping/wrappers/isaaclab
OUT=$PAPER_OUT; mkdir -p $OUT/logs $OUT/eval
ITERS=${ITERS:-150}
PRETRAIN=../../data/grasp_weights_v2/grasp_6dof_pretrain_best.pt
RECIPE="--headless --num_envs 256 --max_iters $ITERS --pretrain $PRETRAIN --reset_std 0.3 --entropy_coef 0.01 --min_std 0.05 --tilted_spawn_prob 0.1"

evaluate() {   # evaluate <name> <checkpoint>
  local n=$1 c=$2
  python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 200 --eval_objects --seed 31 \
    --checkpoint $c --no_exec_noise > $OUT/eval/${n}_flat.log 2>&1
  python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 200 --eval_objects --seed 32 \
    --checkpoint $c --no_exec_noise --tilted_spawn_prob 1.0 > $OUT/eval/${n}_tilted.log 2>&1
  echo "$(date +%H:%M) eval $n: flat $(grep -c 'lift_ok=True' $OUT/eval/${n}_flat.log)/$(grep -c 'lift_ok=' $OUT/eval/${n}_flat.log)" \
       "tilted $(grep -c 'lift_ok=True' $OUT/eval/${n}_tilted.log)/$(grep -c 'lift_ok=' $OUT/eval/${n}_tilted.log)" >> $OUT/results.txt
}

run() {        # run <seed>
  local name=A_s$1 dir=data/grasp_logs/paper_v2/A_s$1
  echo "$(date +%H:%M) start $name" >> $OUT/progress.txt
  python scripts/train_grasp_pose.py $RECIPE --seed $1 --log_dir $dir \
    --vr_failures_dir ../../data/vr_failures_v2 > $OUT/logs/$name.log 2>&1
  local ckpt=$(ls $dir/*/grasp_pose_final.pt 2>/dev/null | head -n 1)
  echo "$(date +%H:%M) trained $name → ${ckpt:-NO CHECKPOINT}" >> $OUT/progress.txt
  [ -n "$ckpt" ] && evaluate $name $ckpt
}

run 1 & sleep 120; run 2 & wait
run 3
echo "$(date +%H:%M) v2 RL done" >> $OUT/progress.txt
