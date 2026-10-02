#!/usr/bin/env bash
# Reproduce every run reported in the README.
#
# Runs are grouped into phases of two or three concurrent jobs. Concurrency
# matters: this workload is memory-bandwidth bound, so five concurrent jobs run
# each episode about 4x slower than one job alone (1.15 s/episode solo, 1.6 s
# with two jobs, 5.2 s with five). Two at a time keeps every individual run
# inside the 45-minute budget.
#
# Measured on an Apple M3 Pro (11 cores, 18 GB): about 100 minutes of wall time
# end to end. Checkpoints, logs and metrics land in runs/, which is gitignored.
#
# Usage:  scripts/run_benchmarks.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

COMMON="--device cpu --log-every 50 --start C --max-steps 40 --max-atoms 38 --seed 0"

train_bg () {
  local name=$1; shift
  mkdir -p "runs/$name"
  echo "launching $name"
  # shellcheck disable=SC2086
  $PY -m droptimus.cli train --out "runs/$name" "$@" > "runs/$name/train.log" 2>&1 &
}

$PY -m droptimus.cli download

# --- Phase 1: reward-shaping ablation ---------------------------------------
# Which reward shape to use is an empirical question, so it is measured rather
# than assumed. 250 episodes each, three jobs concurrently.
for mode in terminal paper dense; do
  # shellcheck disable=SC2086
  train_bg "ablate-qed-$mode" --objective qed --episodes 250 --reward-mode "$mode" $COMMON
done
wait
echo "phase 1 done"

# --- Phase 2: headline single-objective runs --------------------------------
# shellcheck disable=SC2086
train_bg qed-fp-1200  --objective qed            --episodes 1200 $COMMON
# shellcheck disable=SC2086
train_bg logp-fp-1200 --objective penalized_logp --episodes 1200 $COMMON
wait
echo "phase 2 done"

# --- Phase 3: encoder comparison at a matched budget -----------------------
# The GNN is ~3.7x slower per episode than the fingerprint MLP, so both get 400
# episodes to make the comparison fair rather than giving the GNN fewer.
# shellcheck disable=SC2086
train_bg qed-gnn-400 --objective qed --episodes 400 --encoder gnn $COMMON
# shellcheck disable=SC2086
train_bg qed-fp-400  --objective qed --episodes 400 $COMMON
wait
echo "phase 3 done"

# --- Phase 4: similarity-constrained improvement ---------------------------
# ZINC800-logP: the 800 ZINC250k molecules with the lowest penalized logP, one
# episode per start molecule, 20 steps to keep edits close to the original.
mkdir -p runs/constrained-logp
$PY -m droptimus.cli train --device cpu --out runs/constrained-logp --log-every 50 \
  --objective constrained --base-objective penalized_logp --delta 0.4 \
  --start-set zinc800-logp --start-set-size 800 --episodes 800 \
  --max-steps 20 --max-atoms 40 --seed 0 > runs/constrained-logp/train.log 2>&1
echo "phase 4 done"

# --- Evaluation -------------------------------------------------------------
# Each evaluation samples 100 epsilon-greedy episodes, reports the deterministic
# greedy rollout separately, and runs the same number of random-edit baseline
# episodes from the same start molecules.
evaluate_run () {
  local name=$1; shift
  echo "evaluating $name"
  $PY -m droptimus.cli evaluate --checkpoint "runs/$name/checkpoint.pt" \
    --device cpu --out "runs/$name/metrics.json" "$@" > "runs/$name/evaluate.log" 2>&1
}

for name in ablate-qed-terminal ablate-qed-paper ablate-qed-dense \
            qed-fp-1200 logp-fp-1200 qed-gnn-400 qed-fp-400; do
  evaluate_run "$name" --start-set single --start C --episodes 100
done
# One greedy episode per ZINC800-logP start molecule; no exploration needed,
# because the 800 different start molecules already give a distribution.
evaluate_run constrained-logp --start-set zinc800-logp --episodes 800 --delta 0.4 --epsilon 0.0

$PY scripts/make_report.py runs > runs/RESULTS.md
echo "results table written to runs/RESULTS.md"
