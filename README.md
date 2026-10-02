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
gradient clipping. PyTorch on CPU or MPS; `--device auto` picks CPU, and
[Limitations](#limitations) says why.

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
.venv/bin/python -m pytest                              # 413 tests, ~15 s
.venv/bin/python -m pytest --cov --cov-report=term      # 95% statement coverage
```

Fetch the benchmark set (ZINC250k, 22 MB, gitignored):

```bash
.venv/bin/python -m droptimus.cli download
```

Train, optimize a single molecule, evaluate:

```bash
# QED from a single carbon atom, 2000 episodes x 40 steps (~30 min on an M3 Pro
# with nothing else running; see the throughput table below)
droptimus train --objective qed --start C --episodes 2000 --out runs/qed

# Edit one molecule with a trained agent
droptimus optimize "CCOc1ccccc1C(=O)O" --checkpoint runs/qed/checkpoint.pt --show-trajectory

# Measure the agent and the random-edit baseline on the same start molecules
droptimus evaluate --checkpoint runs/qed/checkpoint.pt --start-set single --start C \
  --episodes 100 --out runs/qed/metrics.json
```

Reproduce every number in this README:

```bash
scripts/run_benchmarks.sh                          # training runs + evaluations
python scripts/make_report.py runs > runs/RESULTS.md
python scripts/update_readme.py runs/RESULTS.md    # splices the tables in
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

Three results matter, and one of them is a failure.

**1. QED: the agent works, the graph encoder works better, and neither reaches
the published number.** The best QED result here is the *GNN* encoder at 500
episodes --- final-episode top three 0.822 / 0.811 / 0.805, mean 0.588 --- against
MolDQN-naive's published 0.934 and MolDQN-bootstrap's 0.948. It beats the
random-edit baseline clearly (final-episode mean 0.211, best 0.606).

**2. The graph encoder beat the fingerprint MLP with a quarter of the episodes.**
At a matched 500 episodes, GNN final-episode mean 0.588 against the fingerprint
MLP's 0.437; and the GNN at 500 episodes still beat the fingerprint MLP at
*2000* episodes (0.502). Per episode the GNN costs about 3.5x more wall-clock
(3.45 s against 0.98 s), so at matched wall-clock the gap narrows --- but per
episode it is not close. **This is one seed per configuration.** The direction is
suggestive, not established; a real claim would need several seeds, and the
fingerprint MLP is what the paper used.

**3. Penalized logP failed, and failed in an informative way.** On the
best-molecule-visited protocol the trained agent is *worse than its own random
baseline*: agent best 0.430 against the random walk's +1.719 over the same 100
episodes from the same start molecule. The agent only wins on the terminal-state
protocol (mean -4.71 against -6.86), i.e. it ends on less-bad molecules than a
random walk does while never finding better ones. Published MolDQN-naive gets
11.51. Plainly: on this objective, at this budget, random editing finds better
molecules than the learned policy does.

The reason looks like reward scale, and switching the reward shape confirms about
half of it. Penalized logP runs from about -6 (a single carbon) to +11, and its
optimum is a ~38-carbon chain, so the task is really "add 37 carbons in a row".
With terminal-only reward and gamma = 0.9 over 40 steps, the value of an early
state is 0.9^39 ~ 0.015 times a number around -6, and the agent has almost no
gradient to climb. Paying the per-step change in objective instead
(`--reward-mode dense`, 1200 episodes) moves every number:

| penalized logP | terminal, 2000 ep | dense, 1200 ep | random baseline | MolDQN-naive |
| --- | --- | --- | --- | --- |
| best visited, max | -0.430 | **+1.432** | +1.719 | n/a |
| best visited, mean | -1.752 | **-0.237** | -1.010 | n/a |
| final episode, max | -0.695 | **-0.424** | -3.405 | 11.51 |
| final episode, mean | -4.705 | **-2.215** | -6.855 | n/a |

The dense agent also visibly learns the right *idea*: its best molecule over
training is hexane (`CCCCCC`, +2.472), i.e. it has worked out that the answer is
a carbon chain and simply does not extend one far enough. It still does not beat
the random baseline's single best molecule (+1.432 against +1.719), and it is
nowhere near 11.51. So reward shaping explains a large part of the failure but not
all of it, and the configuration probe having been run on QED only was a real
methodological cost.

### Two protocols, and the gap between them is itself a measurement

Much of this literature reports the best molecule *visited* during an episode.
Zhou et al. Table 1 does something stricter: it scores the last 100 *terminal*
states, the molecule each episode actually ended on. Both are reported below, and
the published comparison uses the terminal-state numbers.

The best-visited protocol flatters random search badly. The random-edit baseline
here scores 0.493 mean on best-visited and 0.211 on terminal states --- 40 random
edits pass through a decent molecule and then wander off it, and best-visited
credits the pass-through. That is exactly why the paper scores terminal states,
and it is why the paper's own random-action baseline reads 0.640 rather than
something higher.

The gap is diagnostic for the agent too. The environment always offers a "no
modification" action, so under a terminal reward the optimal policy is to reach a
good molecule and then sit on it --- a converged agent would show no gap at all.
Ours still shows one, which measures how far from converged these runs are.

### What the reward-shaping ablation did and did not show

Three reward shapes at 250 QED episodes (terminal 0.525, dense 0.479, the
reference implementation's per-step discounted reward 0.516, all best-visited
means) all sit within noise of the random baseline's 0.493 best-visited mean. At
that budget the ablation does not separate them, so it is reported as "no
separation found" rather than as a ranking. The default stays `terminal` because
it is the unambiguous episodic formulation, not because it won.

<!-- RESULTS:START -->
<!-- RESULTS:END -->

---

## Limitations

**This is a benchmark reproduction, not a drug-discovery claim.** QED and
penalized logP are cheap scalar proxies that the field uses because they are fast
and public, not because they predict whether a molecule is a useful drug.
Penalized logP in particular is famously gameable: its maximum over this action
space is a ~38-carbon chain, and the published 11.84 is the score of essentially
that molecule. Nothing here was synthesized, assayed, or checked by a chemist. A
high number in the tables above means an agent found the optimum of a formula,
and nothing more.

**Where this falls short of the published work.** See the comparison table
above for the measured gap. The main reasons, in order of how much they
probably matter:

1. **Training budget.** The paper trains for 5000 episodes per objective. These
   runs are much shorter, to keep a single training run on a laptop inside about
   45 minutes (the two 2000-episode runs took 47 minutes each, running two at a
   time). The measured learning curves in `runs/*/history.json` were still
   improving when the budget ran out, so the QED numbers are a lower bound on
   what this code would reach, not its ceiling. For penalized logP the budget is
   probably *not* the main problem --- see the reward-scale point below.
2. **No bootstrapped-DQN ensemble.** The paper's headline numbers come from
   bootstrapped DQN (multiple Q-heads, each trained on a resampled subset of
   experience) for exploration. This implementation uses a single Q-head with
   epsilon-greedy exploration. The paper reports that variant separately as
   MolDQN-naive --- QED 0.934, penalized logP 11.51 --- so that, not 0.948 /
   11.84, is the number this code should be measured against.
3. **Subsampled bootstrap targets.** The Q-learning target maxes over the
   successor molecules, and a drug-sized molecule has several hundred of them.
   Scoring all of them for every transition in a batch dominates wall-clock
   time, so `AgentConfig.bootstrap_actions` (default 16) subsamples the set. A
   max over a subset is biased low, so the learned values are slightly
   pessimistic. One gradient step costs 27.5 ms at 48 and 13.5 ms at 16, so the
   bias buys roughly twice as many gradient steps per minute.
4. **A small hyperparameter probe, not a search.** Network width, replay
   capacity, batch size and target-sync interval are the paper's values or
   obvious defaults, untouched. What was probed, and is reported in the
   configuration-probe table above rather than hidden: reward shaping (three
   modes, 250 episodes each), the discount and learning rate as a 2x2 grid (600
   episodes each), and `bootstrap_actions`. The discount probe argued *against*
   my prior and the paper's 0.9 was kept. No probe used more than 600 episodes,
   and none was run on the penalized-logP or constrained objectives, so those
   inherit settings chosen on QED.

5. **Reward scale on penalized logP.** The configuration probe ran on QED only,
   and penalized logP has a very different reward scale (about -6 to +11, against
   QED's [0, 1]). Combined with terminal-only reward and gamma = 0.9 over 40
   steps, the learning signal at the start of an episode is on the order of 0.015
   x -6, and the logP agent does not learn. This is a settings failure inherited
   from an objective it was not probed on, not evidence that the implementation
   cannot optimize penalized logP.

**Objective-specific caveats.**

- The similarity constraint is enforced with a steep linear penalty
  (`PENALTY_WEIGHT = 100.0` objective units per unit of similarity shortfall,
  which is the lambda of the paper's section 3.2) rather than a hard rejection,
  because a hard wall gives the agent no gradient back over the boundary. The
  reported fraction of molecules that actually meet the threshold is measured, so
  the softness cannot hide a failure --- but a
  "constrained" result with a satisfied fraction below 1.0 is not a constrained
  result for those molecules.
- Novelty is measured against the first 50,000 ZINC250k molecules plus the start
  molecules, not against all 249,456. A molecule counted as novel may still
  appear in the part of ZINC the reference set did not cover.
- Validity is 1.000 by construction, not by training: the environment only
  proposes candidates that pass `Chem.SanitizeMol`. It appears in the table to
  confirm the invariant holds, and it is not evidence about the model.

**One configuration conclusion I got wrong, for the record.** With a terminal
reward over a fixed 40-step horizon, an undiscounted return (gamma = 1) looks
better motivated than the paper's gamma = 0.9, which makes an early state's value
0.9^39 ~ 0.015 times the final objective --- a badly conditioned regression
target. The mid-training logs appeared to confirm it. They were not comparable
(different exploration rates, and a running best-so-far rather than a fixed
evaluation), and the post-hoc evaluation reversed the result: gamma 1.0 was worse
on every metric. The probe table above has both. The paper's value is the
default.

**Reward shaping differs from the reference implementation by default.**
`--reward-mode terminal` (the default here) pays the objective once, at the
horizon. The reference implementation pays `objective * discount^(steps left)` at
*every* step, which is available as `--reward-mode paper`. Returns are not
comparable between modes; the measured comparison is in the table above.

**An agent trained from one start molecule barely transfers.** The
single-objective runs start every episode from a single carbon atom, so they
learn to *build* a molecule from nothing and never see a drug-sized input.
Pointed at 40 ZINC molecules instead (`--start-set fixture`, 600-episode QED
checkpoint), the agent reached mean QED 0.752 against a start mean of 0.703 and
improved 57% of them --- but the random-edit baseline on the same 40 molecules
reached 0.746 and improved 48%. The transfer is real and it is almost entirely
what random editing would have given you. Individual cases vary a lot: aspirin
(`CCOc1ccccc1C(=O)O`, QED 0.744) comes back unchanged, while
`CN(Cc1ccc(OC(F)(F)F)cc1)C(=O)c1csc(-c2cccs2)n1` goes from 0.605 to 0.771.

For editing arbitrary molecules, train over a start *set*
(`--start-set zinc800-logp` or `--start-set zinc-sample`), which is what the
constrained run does.

**Engineering limits worth knowing.**

- Molecules are capped at `--max-atoms`, 38 by default. That number matches the
  paper's 38-step budget for penalized logP, which bounds a molecule grown from
  nothing to 38 heavy atoms; it is a wall-clock budget as much as a chemical
  choice, since enumerating the action space is quadratic in atom count (a
  36-atom molecule takes about 34 ms per step to enumerate on an M3 Pro, against
  2 ms for a 13-atom one).
- These runs use 40 steps per episode for both objectives. The paper used 40 for
  QED and 38 for penalized logP, the latter to match GCPN's budget, so the logP
  runs here are given two extra steps.
- `--device auto` resolves to CPU, not MPS. This workload is many small forward
  passes, where kernel-launch overhead dominates; MPS measured 1.51 s/episode
  against 1.20 s/episode on CPU. `--device mps` is still available and would be
  the right choice with a larger batch or network.
- Only C, N and O can be added by default (`--atom-types`). Start molecules may
  contain any element RDKit knows; the featurizer covers twelve.
