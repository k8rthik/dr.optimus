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

<!-- RESULTS -->

---

## Limitations

<!-- LIMITATIONS -->
