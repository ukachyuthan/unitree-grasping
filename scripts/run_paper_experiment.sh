#!/bin/bash
# The paper's experiment: does human-in-the-loop (VR demos of the policy's failures)
# improve grasping of unseen objects over RL alone, and over an automatic teacher?
#
#   A  RL only                                   (continue training the start policy)
#   B  RL + human VR demos, transferred           (pose / variant / flat transfer)
#   C  RL + automatic teacher demos, transferred  (Contact-GraspNet, same cases, same pipeline)
#
# Same start policy, recipe, iterations and compute; 3 seeds each. Every finished
# run is evaluated at its final iteration on fresh held-out draws (flat, tilted,
# failure cases), and so is the start policy, as the baseline to beat.
#
#   scripts/run_paper_experiment.sh            (from the repo root; logs in data/paper/)
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate unitree_isaaclab
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
cd ~/unitree-grasping/wrappers/isaaclab

OUT=../../data/paper; mkdir -p $OUT/logs $OUT/eval
START=data/grasp_logs/grasp_pose_envs256_20261026_123416/grasp_pose_125.pt
CASES=../../data/vr_failures/grasp_pose_envs256_20261027_111345
ITERS=150
RECIPE="--headless --num_envs 256 --max_iters $ITERS --resume $START --lr 1e-4 --entropy_coef 0.01 --min_std 0.05 --tilted_spawn_prob 0.1"
HUMAN="../../data/pose_transfer ../../data/pose_transfer_variants ../../data/pt_human_flat"
TEACHER="../../data/pt_teacher_tilted ../../data/pt_teacher_variants ../../data/pt_teacher_flat"
DEMO_ARGS="--demo_replay_prob 0.2 --bc_coef 1"
TRANSFER="--headless --psi_grid -60 -30 0 30 60 --num_envs 256 --seed 1"

evaluate() {   # evaluate <name> <checkpoint>
  local n=$1 c=$2
  python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 200 --eval_objects --seed 31 \
    --checkpoint $c --no_exec_noise > $OUT/eval/${n}_flat.log 2>&1
  python scripts/play_grasp_pose.py --headless --num_envs 1 --num_episodes 200 --eval_objects --seed 32 \
    --checkpoint $c --no_exec_noise --tilted_spawn_prob 1.0 > $OUT/eval/${n}_tilted.log 2>&1
  timeout 2400 python scripts/teach_failure_cases.py --headless --eval_cases $CASES --checkpoint $c \
    --repeats 3 --num_envs 64 > $OUT/eval/${n}_cases.log 2>&1
  echo "$(date +%H:%M) eval $n: flat $(grep -c 'lift_ok=True' $OUT/eval/${n}_flat.log)/$(grep -c 'lift_ok=' $OUT/eval/${n}_flat.log)" \
       "tilted $(grep -c 'lift_ok=True' $OUT/eval/${n}_tilted.log)/$(grep -c 'lift_ok=' $OUT/eval/${n}_tilted.log)" \
       "$(grep '^\[teach\] policy success' $OUT/eval/${n}_cases.log | tr '\n' ' ')" >> $OUT/results.txt
}

run() {        # run <cond> <seed> [extra args...]
  local cond=$1 seed=$2; shift 2
  local name=${cond}_s${seed} dir=data/grasp_logs/paper/${cond}_s${seed}
  echo "$(date +%H:%M) start $name" >> $OUT/progress.txt
  python scripts/train_grasp_pose.py $RECIPE --vr_failures_dir "" --seed $seed --log_dir $dir "$@" \
    > $OUT/logs/$name.log 2>&1
  local ckpt=$(ls $dir/*/grasp_pose_final.pt 2>/dev/null | head -n 1)
  echo "$(date +%H:%M) trained $name → ${ckpt:-NO CHECKPOINT}" >> $OUT/progress.txt
  [ -n "$ckpt" ] && evaluate $name $ckpt
}

transfers() {
  python scripts/teach_failure_cases.py --transfer ../../data/vr_failures --variants --flat_poses \
    --poses 8 --out ../../data/pt_human_flat $TRANSFER > $OUT/logs/transfer_human_flat.log 2>&1
  python scripts/teach_failure_cases.py --transfer ../../data/teacher_demos_top40 --poses 16 \
    --out ../../data/pt_teacher_tilted $TRANSFER > $OUT/logs/transfer_teacher_tilted.log 2>&1
  python scripts/teach_failure_cases.py --transfer ../../data/teacher_demos_top40 --variants \
    --poses 8 --out ../../data/pt_teacher_variants $TRANSFER > $OUT/logs/transfer_teacher_variants.log 2>&1
  python scripts/teach_failure_cases.py --transfer ../../data/teacher_demos_top40 --variants --flat_poses \
    --poses 8 --out ../../data/pt_teacher_flat $TRANSFER > $OUT/logs/transfer_teacher_flat.log 2>&1
  # Every transfer must have produced demos, or B / C would silently train on
  # fewer (or no) demos than intended.
  for f in human_flat teacher_tilted teacher_variants teacher_flat; do
    if ! grep -q '^\[teach\] transfer: [0-9]* lifting' $OUT/logs/transfer_$f.log; then
      echo "$(date +%H:%M) TRANSFER FAILED: $f — B / C not started" >> $OUT/progress.txt
      touch $OUT/transfers.failed
      return
    fi
  done
  grep -h '^\[teach\] transfer: [0-9]* lifting' $OUT/logs/transfer_*.log >> $OUT/progress.txt
  touch $OUT/transfers.done
}

wait_transfers() {
  until [ -f $OUT/transfers.done ] || [ -f $OUT/transfers.failed ]; do sleep 60; done
  [ -f $OUT/transfers.done ]
}

# Baseline and transfers first, alongside A seed 1.
# (the start policy's evaluation is kept if a previous launch already measured it)
( grep -q "eval start_policy:" $OUT/results.txt 2>/dev/null || evaluate start_policy $START; transfers ) &
( run A 1 ) &
wait -n
# Then the rest, two trainings at a time.
JOBS=(
  "wait_transfers && run B 1 --demos_dir $HUMAN $DEMO_ARGS"
  "wait_transfers && run C 1 --demos_dir $TEACHER $DEMO_ARGS"
  "run A 2"
  "wait_transfers && run B 2 --demos_dir $HUMAN $DEMO_ARGS"
  "wait_transfers && run C 2 --demos_dir $TEACHER $DEMO_ARGS"
  "run A 3"
  "wait_transfers && run B 3 --demos_dir $HUMAN $DEMO_ARGS"
  "wait_transfers && run C 3 --demos_dir $TEACHER $DEMO_ARGS"
)
for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
  ( eval "$j" ) &
done
wait
echo "$(date +%H:%M) ALL DONE" >> $OUT/progress.txt
