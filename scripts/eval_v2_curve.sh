#!/bin/bash
# Learning curves: test every saved checkpoint of the RL-only (A) and RL + demos (D) runs on
# the same draws as run_v2_rl.sh (held-out objects, flat seed 31 / tilted seed 32, no execution
# noise) and on the round-3 hard cases. Newest D checkpoints first; one test at a time.
# Appends "<name> <iter> flat k/n tilted k/n demo_half% heldout_half%" to data/paper_v2/curve.txt.
#
#   scripts/eval_v2_curve.sh           (from the repo root; stops when every D seed has its final)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate unitree_isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
source "$(dirname "$0")/v2_env.sh"
cd ~/unitree-grasping/wrappers/isaaclab
OUT=$PAPER_OUT; mkdir -p $OUT/curve; CASES=../../data/vr_round3/round3

test_ckpt() {  # test_ckpt <name> <iter> <checkpoint>
  local tag=$1_$2 c=$3
  python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 200 --eval_objects --seed 31 \
    --checkpoint $c --no_exec_noise > $OUT/curve/${tag}_flat.log 2>&1
  python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 200 --eval_objects --seed 32 \
    --checkpoint $c --no_exec_noise --tilted_spawn_prob 1.0 > $OUT/curve/${tag}_tilted.log 2>&1
  python scripts/teach_failure_cases.py --headless --num_envs 64 --eval_cases $CASES --checkpoint $c \
    --repeats 2 > $OUT/curve/${tag}_hard.log 2>&1
  echo "$1 $2 flat $(grep -c 'lift_ok=True' $OUT/curve/${tag}_flat.log)/$(grep -c 'lift_ok=' $OUT/curve/${tag}_flat.log)" \
       "tilted $(grep -c 'lift_ok=True' $OUT/curve/${tag}_tilted.log)/$(grep -c 'lift_ok=' $OUT/curve/${tag}_tilted.log)" \
       "hard: $(grep -a 'policy success' $OUT/curve/${tag}_hard.log | sed 's/.*success on //' | tr '\n' ' ')" >> $OUT/curve.txt
  touch $OUT/curve/$tag.done
}

next() {       # the next untested checkpoint: name iter path
  for n in D_s1 D_s2 D_s3 A_s1 A_s2 A_s3; do
    for i in 25 50 75 100 125 150; do
      local c=$(ls data/grasp_logs/paper_v2/$n/*/grasp_pose_$i.pt 2>/dev/null | head -n 1)
      [ -n "$c" ] && [ ! -f $OUT/curve/${n}_$i.done ] && { echo "$n $i $c"; return; }
    done
  done
}

mem_ok() { [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 5 ]; }  # one test adds ~4 GB to the two trainers

while :; do
  mem_ok || { sleep 60; continue; }
  set -- $(next)
  if [ -n "$1" ]; then test_ckpt "$@"; continue; fi
  [ -z "$(ls data/grasp_logs/paper_v2/D_s[123]/*/grasp_pose_final.pt 2>/dev/null | wc -l | grep -x 3)" ] || break
  sleep 60
done
echo "$(date +%H:%M) curve done" >> $OUT/progress.txt
