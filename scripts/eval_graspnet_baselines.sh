#!/bin/bash
# GraspNet baselines on the paper's held-out test draws (same seeds as run_paper_experiment.sh):
# its top-ranked grasp, and the best-ranked of its top 10 passing geometric checks.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate unitree_isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
cd ~/unitree-grasping/wrappers/isaaclab
OUT=../../data/paper; mkdir -p $OUT/eval
for a in graspnet graspnet_filtered; do
  python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 200 --eval_objects --seed 31 \
    --debug_action $a --no_exec_noise > $OUT/eval/${a}_flat.log 2>&1
  python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 200 --eval_objects --seed 32 \
    --debug_action $a --no_exec_noise --tilted_spawn_prob 1.0 > $OUT/eval/${a}_tilted.log 2>&1
  echo "$(date +%H:%M) eval $a: flat $(grep -c 'lift_ok=True' $OUT/eval/${a}_flat.log)/$(grep -c 'lift_ok=' $OUT/eval/${a}_flat.log)" \
       "tilted $(grep -c 'lift_ok=True' $OUT/eval/${a}_tilted.log)/$(grep -c 'lift_ok=' $OUT/eval/${a}_tilted.log)" >> $OUT/results.txt
done
