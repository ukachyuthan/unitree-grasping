#!/bin/bash
# Run paper-experiment jobs one after another, each as soon as fewer than two
# trainings are on the GPU (counting runs started elsewhere).
#
#   scripts/run_queue.sh "run A 2" "run A 3"      (from the repo root)
source "$(dirname "$0")/run_paper_experiment.sh"
EXEC_ARGS="--demo_exec_prob 0.5"
busy() { pgrep -fc '[t]rain_grasp_pose.py'; }
for j in "$@"; do
  while [ "$(busy)" -ge 2 ]; do sleep 60; done
  ( eval "$j" ) &
  sleep 120   # let the job's trainer start before counting again
done
wait
echo "$(date +%H:%M) queue done: $*" >> $OUT/progress.txt
