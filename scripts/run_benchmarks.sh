#!/usr/bin/env bash
# Reproduce every run reported in the README.
#
# Five training runs and their evaluations. On an Apple M3 Pro (11 cores, 18 GB)
# all five train concurrently in roughly 40 minutes of wall time; each
# individual run is under 45 minutes. Checkpoints and logs land in runs/ (which
# is gitignored).
#
# Usage:  scripts/run_benchmarks.sh [--sequential]
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
# Thread count per job. This workload is memory-bandwidth bound, so 2 threads
# per job and 5 concurrent jobs is within a few percent of running one job on
# all 11 cores.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
SEQUENTIAL=${1:-}

$PY -m droptimus.cli download

train_run () {
  local name=$1; shift
  mkdir -p "runs/$name"
  echo "launching $name"
  if [[ "$SEQUENTIAL" == "--sequential" ]]; then
    $PY -m droptimus.cli train --device cpu --out "runs/$name" --log-every 50 "$@" \
      2>&1 | tee "runs/$name/train.log"
  else
    $PY -m droptimus.cli train --device cpu --out "runs/$name" --log-every 50 "$@" \
      > "runs/$name/train.log" 2>&1 &
  fi
}

# Headline runs: 1500 episodes x 40 steps from a single carbon atom.
train_run qed-fp-1500 \
  --objective qed --start C --episodes 1500 --max-steps 40 --max-atoms 38 --seed 0
train_run logp-fp-1500 \
  --objective penalized_logp --start C --episodes 1500 --max-steps 40 --max-atoms 38 --seed 0

# Encoder comparison at a matched episode budget (the GNN is ~3.7x slower per
# episode, so 500 episodes is what fits the 45-minute cap).
train_run qed-gnn-500 \
  --objective qed --start C --episodes 500 --max-steps 40 --max-atoms 38 --seed 0 --encoder gnn
train_run qed-fp-500 \
  --objective qed --start C --episodes 500 --max-steps 40 --max-atoms 38 --seed 0

# Similarity-constrained improvement on ZINC800-logP (the 800 ZINC250k
# molecules with the lowest penalized logP), one episode per start molecule.
train_run constrained-logp \
  --objective constrained --base-objective penalized_logp --delta 0.4 \
  --start-set zinc800-logp --start-set-size 800 --episodes 800 \
  --max-steps 20 --max-atoms 40 --seed 0

wait
echo "all training runs finished"

evaluate_run () {
  local name=$1; shift
  echo "evaluating $name"
  $PY -m droptimus.cli evaluate --checkpoint "runs/$name/checkpoint.pt" \
    --device cpu --out "runs/$name/metrics.json" "$@" 2>&1 \
    | tee "runs/$name/evaluate.log"
}

# Single-start objectives: 100 greedy episodes from the same start molecule,
# plus 100 random-edit baseline episodes.
evaluate_run qed-fp-1500  --start-set single --start C --episodes 100
evaluate_run logp-fp-1500 --start-set single --start C --episodes 100
evaluate_run qed-gnn-500  --start-set single --start C --episodes 100
evaluate_run qed-fp-500   --start-set single --start C --episodes 100
# Constrained task: one greedy episode per ZINC800-logP start molecule.
evaluate_run constrained-logp --start-set zinc800-logp --episodes 800 --delta 0.4

$PY scripts/make_report.py runs > runs/RESULTS.md
echo "results table written to runs/RESULTS.md"
