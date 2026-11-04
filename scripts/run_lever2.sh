#!/bin/bash
# Lever 2: condition D = RL + human VR demos, with the demo grasps themselves
# executed on part of the demo episodes (--demo_exec_prob), so the policy learns
# from them succeeding instead of only imitating them.
#
# Same start policy, recipe, demos, iterations and test as run_paper_experiment.sh.
# demo_exec_prob 0.5 is fixed before any test result (half the demo episodes,
# 10% of all episodes) and not tuned on the test.
#
# D runs first (the answer the paper needs), then the remaining A seeds as its
# baseline, then B and C. A seed 1 is already running from the first launch.
#
#   scripts/run_lever2.sh            (from the repo root; logs in data/paper/)
source "$(dirname "$0")/run_paper_experiment.sh"
EXEC_ARGS="--demo_exec_prob 0.5"

JOBS=(
  "run D 1 --demos_dir $HUMAN $DEMO_ARGS $EXEC_ARGS"
  "run D 2 --demos_dir $HUMAN $DEMO_ARGS $EXEC_ARGS"
  "run A 2"
  "run D 3 --demos_dir $HUMAN $DEMO_ARGS $EXEC_ARGS"
  "run A 3"
  "run B 1 --demos_dir $HUMAN $DEMO_ARGS"
  "run C 1 --demos_dir $TEACHER $DEMO_ARGS"
  "run B 2 --demos_dir $HUMAN $DEMO_ARGS"
  "run C 2 --demos_dir $TEACHER $DEMO_ARGS"
  "run B 3 --demos_dir $HUMAN $DEMO_ARGS"
  "run C 3 --demos_dir $TEACHER $DEMO_ARGS"
)
# A seed 1 (from the first launch) holds one of the two GPU slots until it finishes.
busy() { echo $(( $(jobs -rp | wc -l) + $(pgrep -fc '[l]og_dir data/grasp_logs/paper/A_s1') )); }
for j in "${JOBS[@]}"; do
  while [ "$(busy)" -ge 2 ]; do sleep 60; done
  ( eval "$j" ) &
  sleep 5
done
wait
echo "$(date +%H:%M) ALL DONE" >> $OUT/progress.txt
