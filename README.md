# dr.optimus

A reproduction of **MolDQN** --- Zhou, Kearnes, Li, Zare & Riley (2019),
*"Optimization of Molecules via Deep Reinforcement Learning"*,
[Scientific Reports 9:10752](https://www.nature.com/articles/s41598-019-47148-x).

A double-DQN agent edits a molecular graph one bond at a time and is scored on
the standard open benchmark objectives: QED drug-likeness, penalized logP, and
similarity-constrained improvement of penalized logP. Every edit is validated by
RDKit, so every molecule the agent produces is chemically well-formed.

This is a **benchmark reproduction**, not a drug-discovery tool. See
[Limitations](#limitations).

---

## What is actually implemented

**Environment** (`droptimus/env/`) --- the molecule-editing MDP of the paper's
section 3.1. State is `(molecule, steps taken)`; an action *is* a candidate
successor molecule, enumerated as:

- **atom addition** --- attach C, N or O (configurable) with a single, double or
  triple bond to any atom with the free valence for it;
- **bond addition** --- bond two atoms that both have free valence, or raise an
  existing bond's order; new rings are filtered to allowed sizes (3--6);
- **bond removal** --- lower a bond's order or delete it, keeping the result
  connected (a detached single atom is dropped, as in the reference
  implementation).

Every candidate is run through `Chem.SanitizeMol` and discarded if it fails, so
an impossible valence is unreachable. Episodes run for a fixed step budget (40
by default) from a configurable start molecule.

**Agent** (`droptimus/agent/`) --- double DQN with a replay buffer and a
hard-synced target network. Because the action set is state-dependent, the
network scores one candidate molecule at a time and the agent takes the argmax:

- `fingerprint` (default, the MolDQN architecture): MLP `2048 -> 1024 -> 512 ->
  128 -> 32 -> 1` over a Morgan fingerprint concatenated with the remaining-step
  horizon;
- `gnn`: a relational graph convolution with one weight matrix per bond type
  (single/double/triple/aromatic), masked mean+max pooling, then the same head.

Target: `y = r + gamma * Q_target(argmax_a' Q_online(a'))`. Huber loss, Adam,
gradient clipping. PyTorch, CPU or MPS (see
[Why CPU and not MPS](#why-cpu-and-not-mps)).

**Objectives** (`droptimus/objectives/`) --- a registry; `make_objective(name)`
builds one:

| name | definition |
| --- | --- |
| `qed` | RDKit QED (Bickerton et al. 2012), in [0, 1] |
| `penalized_logp` | `logP - SA - large-ring penalty`, each term z-normalized over ZINC250k (Kusner et al. 2017) |
| `similarity` | Tanimoto similarity of Morgan fingerprints to a reference molecule |
| `constrained` | a base objective minus a steep penalty for falling below a Tanimoto similarity threshold `delta` |

Adding an objective is one decorated factory function; nothing else changes.

---

## Install and run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/python -m pytest            # 254 tests, ~6 s
```

Fetch the benchmark set (ZINC250k, 22 MB, gitignored):

```bash
.venv/bin/python -m droptimus.cli download
```

Train, optimize a single molecule, evaluate:

```bash
# QED from a single carbon atom, 1500 episodes x 40 steps (~30 min on an M3 Pro)
droptimus train --objective qed --start C --episodes 1500 --out runs/qed

# Edit one molecule with a trained agent
droptimus optimize "CCOc1ccccc1C(=O)O" --checkpoint runs/qed/checkpoint.pt --show-trajectory

# Measure the agent and the random-edit baseline on the same start molecules
droptimus evaluate --checkpoint runs/qed/checkpoint.pt --start-set single --start C \
  --episodes 100 --out runs/qed/metrics.json
```

Reproduce every number in this README:

```bash
scripts/run_benchmarks.sh      # 5 training runs + evaluations, ~40 min wall time
```

Invalid input fails fast and plainly:

```
$ droptimus optimize "C(((" --checkpoint runs/qed/checkpoint.pt
error: Invalid SMILES 'C((((': RDKit could not parse the structure.
```

---

## Measured results

Everything in this section was measured by `droptimus evaluate` on the
checkpoints named in the tables, on an Apple M3 Pro (11 cores, 18 GB), and
spliced in by `scripts/update_readme.py` --- the numbers are not typed by hand.
The published MolDQN column is quoted from Zhou et al. (2019) for comparison and
was **not** reproduced by this code. Regenerate everything with
`scripts/run_benchmarks.sh`.

### The short version

**The agent learns, and it does not reach the published numbers.** On QED from a
single carbon atom it clearly improves over training, and it beats a random-edit
baseline on the *mean* objective. It does not reliably beat the random baseline's
*best* molecule, and it is far from MolDQN's reported 0.948. The gap is
dominated by training budget: these runs are a small fraction of the paper's 5000
episodes per objective, and the learning curves in `runs/*/history.json` had not
flattened when the budget ran out. The honest summary is "a working
implementation measured at a laptop-scale budget", not "a reproduction of the
published result".

The random-edit baseline is a strong opponent here, and that is worth
understanding rather than hiding: QED has a broad optimum, so 40 random
valence-valid edits from a carbon atom often land on a mid-0.5 QED molecule, and
the best of 100 such tries is around 0.79. Beating that reliably is the whole
difficulty of the benchmark.

<!-- RESULTS:START -->
<!-- RESULTS:END -->

---

## Limitations

**This is a benchmark reproduction, not a drug-discovery claim.** QED and
penalized logP are cheap scalar proxies that the field uses because they are
fast and public, not because they predict whether a molecule is a useful drug.
Penalized logP in particular is famously gameable: its maximum over this action
space is a long greasy carbon chain, which is exactly what an agent optimizing it
produces. Nothing here was synthesized, assayed, or checked by a chemist. A high
number in the table below means the agent found the optimum of a formula.

**Where this falls short of the published work.** See the comparison table
above for the measured gap. The main reasons, in order of how much they
probably matter:

1. **Training budget.** The paper trains for 5000 episodes per objective. These
   runs are much shorter, to keep a single training run on a laptop inside about
   45 minutes. The measured learning curves in `runs/*/history.json` were still
   improving when the budget ran out, so these numbers are a lower bound on what
   this code would reach, not its ceiling.
2. **No bootstrapped-DQN ensemble.** The paper uses bootstrapped DQN (multiple
   Q-heads, each trained on a resampled subset of experience) for exploration.
   This implementation uses a single Q-head with epsilon-greedy exploration,
   which explores less efficiently.
3. **Subsampled bootstrap targets.** The Q-learning target maxes over the
   successor molecules, and a drug-sized molecule has several hundred of them.
   Scoring all of them for every transition in a batch dominates wall-clock
   time, so `AgentConfig.bootstrap_actions` (default 48) subsamples the set. A
   max over a subset is biased low, so the learned values are slightly
   pessimistic.
4. **Almost no hyperparameter search.** Network width, replay capacity, discount
   and target-sync interval are the paper's values or obvious defaults. Two
   choices were made by measurement, both because they trade directly against
   the wall-clock budget rather than because they flattered a result: the
   learning rate (one probe over two values, 600 episodes each --- see the
   ablation table) and `bootstrap_actions`. Nothing else was tuned.

**Objective-specific caveats.**

- The similarity constraint is enforced with a steep linear penalty
  (`PENALTY_WEIGHT = 20.0` objective units per unit of similarity shortfall)
  rather than a hard rejection, because a hard wall gives the agent no gradient
  back over the boundary. The reported fraction of molecules that actually meet
  the threshold is measured, so the softness cannot hide a failure --- but a
  "constrained" result with a satisfied fraction below 1.0 is not a constrained
  result for those molecules.
- Novelty is measured against the first 50,000 ZINC250k molecules plus the start
  molecules, not against all 249,456. A molecule counted as novel may still
  appear in the part of ZINC the reference set did not cover.
- Validity is 1.000 by construction, not by training: the environment only
  proposes candidates that pass `Chem.SanitizeMol`. It appears in the table to
  confirm the invariant holds, and it is not evidence about the model.

**Reward shaping differs from the reference implementation by default.**
`--reward-mode terminal` (the default here) pays the objective once, at the
horizon. The reference implementation pays `objective * discount^(steps left)` at
*every* step, which is available as `--reward-mode paper`. Returns are not
comparable between modes; the measured comparison is in the table above.

**Engineering limits worth knowing.**

- Molecules are capped at `--max-atoms` (38 by default, the paper's cap).
  Enumerating the action space is quadratic in atom count, so this cap is a
  wall-clock budget as much as a chemical choice: a 36-atom molecule takes about
  34 ms per step to enumerate on an M3 Pro, against 2 ms for a 13-atom one.
- `--device auto` resolves to CPU, not MPS. This workload is many small forward
  passes, where kernel-launch overhead dominates; MPS measured 1.51 s/episode
  against 1.20 s/episode on CPU. `--device mps` is still available and would be
  the right choice with a larger batch or network.
- Only C, N and O can be added by default (`--atom-types`). Start molecules may
  contain any element RDKit knows; the featurizer covers twelve.
