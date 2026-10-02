#!/usr/bin/env bash
# Reproduce every number in the README.
#
# Concurrency matters here. This workload is memory-bandwidth bound, so running
# five jobs at once makes each episode about 4x slower than running one alone
# (measured: 1.15 s/episode solo, 1.6 s with two jobs, 5.2 s with five). Runs are
# therefore grouped into phases of two or three, which keeps every individual run
# inside the 45-minute budget.
#
# Measured end to end on an Apple M3 Pro (11 cores, 18 GB): about 80 minutes.
# Checkpoints, logs and metrics land in runs/, which is gitignored.
#
# Usage:  scripts/run_benchmarks.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Everything not named here comes from droptimus/config.py.
COMMON="--device cpu --log-every 100 --start C --max-steps 40 --max-atoms 38 --seed 0"

train_bg () {
  local name=$1; shift
  mkdir -p "runs/$name"
  echo "launching $name"
  # shellcheck disable=SC2086
  $PY -m droptimus.cli train --out "runs/$name" "$@" > "runs/$name/train.log" 2>&1 &
}

$PY -m droptimus.cli download

# --- Phase 1: configuration probes ------------------------------------------
# Reward shaping and learning rate are empirical questions, so they are measured
# rather than assumed. These are exploratory runs, reported separately from the
# headline results so a probe cannot be mistaken for a result.
# shellcheck disable=SC2086
for mode in terminal paper dense; do
  train_bg "ablate-qed-$mode" --objective qed --episodes 250 --reward-mode "$mode" \
    --bootstrap-actions 48 --learning-rate 1e-4 $COMMON
done
wait

# The discount and learning-rate grid, at the cheaper bootstrap setting.
# shellcheck disable=SC2086
train_bg probe-lr5e-4      --objective qed --episodes 600 --learning-rate 5e-4 --discount 0.9 $COMMON
# shellcheck disable=SC2086
train_bg probe-g0.9-lr1e-4 --objective qed --episodes 600 --learning-rate 1e-4 --discount 0.9 $COMMON
wait
# shellcheck disable=SC2086
train_bg probe-g1.0-lr5e-4 --objective qed --episodes 600 --learning-rate 5e-4 --discount 1.0 $COMMON
# shellcheck disable=SC2086
train_bg probe-g1.0-lr1e-4 --objective qed --episodes 600 --learning-rate 1e-4 --discount 1.0 $COMMON
wait
echo "phase 1 done"

# --- Phase 2: headline single-objective runs --------------------------------
# shellcheck disable=SC2086
train_bg qed-fp-2000  --objective qed            --episodes 2000 $COMMON
# shellcheck disable=SC2086
train_bg logp-fp-2000 --objective penalized_logp --episodes 2000 $COMMON
wait
echo "phase 2 done"

# --- Phase 3: encoder comparison at a matched budget -----------------------
# The GNN is several times slower per episode than the fingerprint MLP, so both
# get 500 episodes -- a matched budget rather than matched wall-clock, so the
# comparison is about the representation and not about who got more training.
# shellcheck disable=SC2086
train_bg qed-gnn-500 --objective qed --episodes 500 --encoder gnn $COMMON
# shellcheck disable=SC2086
train_bg qed-fp-500  --objective qed --episodes 500 $COMMON

# --- Phase 4: similarity-constrained improvement ---------------------------
# ZINC800-logP: the 800 ZINC250k molecules with the lowest penalized logP, 20
# steps per episode to keep edits close to the original.
#
# Episodes must exceed the number of start molecules. The environment cycles
# through the start set, so training for 800 episodes over 800 molecules shows
# each one exactly once -- that is not training, it is a single evaluation pass,
# and it is what the first attempt here did (runs/constrained-logp, kept in the
# tables as the negative result). 2400 episodes is three passes.
mkdir -p runs/constrained-logp-3pass
$PY -m droptimus.cli train --out runs/constrained-logp-3pass --device cpu --log-every 200 \
  --objective constrained --base-objective penalized_logp --delta 0.4 \
  --start-set zinc800-logp --start-set-size 800 --episodes 2400 \
  --max-steps 20 --max-atoms 40 --seed 0 > runs/constrained-logp-3pass/train.log 2>&1 &
wait
echo "phases 3 and 4 done"

# --- Evaluation -------------------------------------------------------------
# Each evaluation reports the deterministic greedy rollout, a distribution from
# epsilon-greedy episodes, and the same number of random-edit baseline episodes
# from the same start molecules.
evaluate_run () {
  local name=$1; shift
  echo "evaluating $name"
  $PY -m droptimus.cli evaluate --checkpoint "runs/$name/checkpoint.pt" \
    --device cpu --out "runs/$name/metrics.json" "$@" > "runs/$name/evaluate.log" 2>&1
}

for name in qed-fp-2000 logp-fp-2000 qed-gnn-500 qed-fp-500; do
  evaluate_run "$name" --start-set single --start C --episodes 100
done
# Probes use 60 episodes, enough to rank configurations and cheaper to run.
for name in ablate-qed-terminal ablate-qed-paper ablate-qed-dense \
            probe-lr5e-4 probe-g0.9-lr1e-4 probe-g1.0-lr5e-4 probe-g1.0-lr1e-4; do
  evaluate_run "$name" --start-set single --start C --episodes 60 --novelty-reference-size 0
done
# No exploration needed for the constrained task: 800 different start molecules
# already give a distribution.
evaluate_run constrained-logp-3pass \
  --start-set zinc800-logp --episodes 800 --delta 0.4 --epsilon 0.0

# --- Phase 5: reward shaping for penalized logP ----------------------------
# Penalized logP is maximized by a long carbon chain, so the task is really "add
# ~37 carbons". With terminal-only reward that signal has to propagate 40 steps
# back; dense reward pays for each carbon immediately. Run solo, since it is the
# last job.
mkdir -p runs/logp-fp-dense-1200
$PY -m droptimus.cli train --out runs/logp-fp-dense-1200 --device cpu --log-every 100 \
  --objective penalized_logp --episodes 1200 --reward-mode dense $COMMON \
  > runs/logp-fp-dense-1200/train.log 2>&1
evaluate_run logp-fp-dense-1200 --start-set single --start C --episodes 100
echo "phase 5 done"

$PY scripts/make_report.py runs > runs/RESULTS.md
$PY scripts/update_readme.py runs/RESULTS.md
echo "results written to runs/RESULTS.md and spliced into README.md"
