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

Four results matter, and two of them are failures.

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
baseline*: agent best -0.430 against the random walk's +1.719 over the same 100
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

Both halves of that need saying. Terminal-only reward was a **real** cause of the
failure --- dense reward improves every number, with 1200 episodes against 2000 ---
and the dense agent visibly learns the right idea: its best molecule over training
is hexane (`CCCCCC`, +2.472), so it has worked out that the answer is a carbon
chain and simply does not extend one far enough. But it is **still not good**: the
random-edit baseline's single best molecule (+1.719) beats the dense agent's
(+1.432), and the published MolDQN-naive figure is 11.51. Reward shaping explains
a large part of this failure and not all of it, and having run the configuration
probe on QED only was a real methodological cost.

**4. On the similarity-constrained task the trained policy is worse than random
editing.** ZINC800-logP, delta = 0.4, 2400 training episodes (three passes over
the 800 start molecules), scored on the best molecule each episode found:

| delta = 0.4 | penalized logP improvement | improved | constraint satisfied | best single |
| --- | --- | --- | --- | --- |
| trained agent, 2400 episodes | +0.92 +/- 3.20 | 27.8% | 95.5% | +2.77 |
| trained agent, 800 episodes | +0.97 +/- 2.62 | 33.8% | 96.0% | +1.22 |
| random-edit baseline | **+1.38 +/- 4.10** | **36.4%** | 94.4% | +0.79 |
| MolDQN-naive | +3.13 +/- 1.57 | --- | 100% | --- |
| MolDQN-bootstrap | +3.37 +/- 1.62 | --- | 100% | --- |

Tripling the training budget changed nothing here: 800 episodes and 2400 episodes
land in the same place, and the longer run improves *fewer* molecules. So the
single-pass setup error (described in the limitations) was a real bug worth
fixing, and it was not what caused this failure.

The random walk improves more molecules, and by more, at the same constraint
satisfaction rate. The trained policy's one best molecule is better (+2.77
against +0.79), but on the metric the paper reports --- mean improvement across
all 800 --- it loses to random editing. There is no reading of this table on which
the learned policy is doing useful work.

Two details matter for reading that table:

- **Scoring the best molecule visited is not a generous choice here, it is the
  only sensible one.** After a full 20-step budget of forced edits, similarity to
  the start molecule has collapsed to about 0.15 and essentially nothing satisfies
  delta = 0.4 --- the agent's terminal states satisfy it 1.9% of the time, the
  baseline's 0.0%. The terminal state measures the step budget, not the policy.
- **The greedy (epsilon 0) policy makes no edit at all.** Its start, best and
  final molecules are the same string and the objective is unchanged at -62.52.
  Faced with a penalty of 100 per unit of similarity shortfall, it has learned
  that the safest action is to do nothing, forever. That is a degenerate policy
  and naming it as a failure mode is the honest report; quoting its score as a
  result would not be.

So the two failure modes are opposite, which is informative. Under a terminal
reward on QED the agent will not stop --- it finds a good molecule and keeps
editing, which is what the gap between the best-visited and final-episode columns
measures. Under a large constraint penalty it will not start. Neither run learned
to use the "no modification" action as a *choice*; one ignores it and the other
hides behind it.

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

### Objective values

`greedy` is the single molecule the deterministic policy produces. `best of 100` and `mean` come from 100 epsilon-greedy episodes (epsilon 0.1), because a deterministic policy from one start molecule returns the same molecule every time. The random-edit baseline takes the same number of episodes from the same start molecules, choosing uniformly among valid edits.

| run | objective | episodes | greedy | best of 100 | mean +/- sd | random baseline best | random baseline mean | MolDQN-naive best | MolDQN-bootstrap best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constrained-logp | constrained | 800 | -62.517 | 0.050 | -8.379 +/- 4.798 | -1.614 | -8.142 +/- 4.275 | n/a | n/a |
| constrained-logp-3pass | constrained | 2400 | -62.517 | 1.736 | -8.447 +/- 4.574 | -1.614 | -8.142 +/- 4.275 | n/a | n/a |
| logp-fp-2000 | penalized_logp | 2000 | -1.772 | -0.430 | -1.752 +/- 0.301 | 1.719 | -1.010 +/- 0.971 | 11.510 | 11.840 |
| logp-fp-dense-1200 | penalized_logp | 1200 | -0.335 | 1.432 | -0.237 +/- 0.706 | 1.719 | -1.010 +/- 0.971 | 11.510 | 11.840 |
| qed-fp-2000 | qed | 2000 | 0.455 | 0.815 | 0.563 +/- 0.106 | 0.675 | 0.493 +/- 0.073 | 0.934 | 0.948 |
| qed-fp-500 | qed | 500 | 0.530 | 0.802 | 0.537 +/- 0.079 | 0.675 | 0.493 +/- 0.073 | 0.934 | 0.948 |
| qed-gnn-500 | qed | 500 | 0.811 | 0.839 | 0.636 +/- 0.128 | 0.675 | 0.493 +/- 0.073 | 0.934 | 0.948 |

### Final-episode molecules (the paper's protocol)

Zhou et al. Table 1 reports the top three scores among the last 100 *terminal* states, so this is the table to compare against the published column. MolDQN-naive is a single Q-network with epsilon-greedy exploration, which is what this code implements; MolDQN-bootstrap adds an ensemble of Q-heads, which it does not.

| run | objective | best of n (final) | mean +/- sd (final) | baseline best (final) | MolDQN-naive best | MolDQN-bootstrap best |
| --- | --- | --- | --- | --- | --- | --- |
| constrained-logp | constrained | -6.485 | -36.592 +/- 7.916 | -10.239 | n/a | n/a |
| constrained-logp-3pass | constrained | 1.736 | -34.253 +/- 8.392 | -10.239 | n/a | n/a |
| logp-fp-2000 | penalized_logp | -0.695 | -4.705 +/- 1.560 | -3.405 | 11.510 | 11.840 |
| logp-fp-dense-1200 | penalized_logp | -0.424 | -2.215 +/- 1.263 | -3.405 | 11.510 | 11.840 |
| qed-fp-2000 | qed | 0.815 | 0.502 +/- 0.118 | 0.606 | 0.934 | 0.948 |
| qed-fp-500 | qed | 0.757 | 0.437 +/- 0.141 | 0.606 | 0.934 | 0.948 |
| qed-gnn-500 | qed | 0.822 | 0.588 +/- 0.164 | 0.606 | 0.934 | 0.948 |

### Generation quality

| run | n | validity | uniqueness | novelty | mean similarity to start | mean improvement | improved | meets similarity threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constrained-logp | 800 | 1.000 | 1.000 | 0.338 | 0.835 | +0.809 +/- 1.961 | 34% | 0.960 |
| constrained-logp-3pass | 800 | 1.000 | 1.000 | 0.278 | 0.859 | +0.741 +/- 2.271 | 28% | 0.955 |
| logp-fp-2000 | 100 | 1.000 | 0.150 | 1.000 | 0.001 | +4.478 +/- 0.301 | 100% | n/a |
| logp-fp-dense-1200 | 100 | 1.000 | 0.200 | 1.000 | 0.000 | +5.993 +/- 0.706 | 100% | n/a |
| qed-fp-2000 | 100 | 1.000 | 0.710 | 1.000 | 0.000 | +0.203 +/- 0.106 | 100% | n/a |
| qed-fp-500 | 100 | 1.000 | 0.700 | 1.000 | 0.000 | +0.178 +/- 0.079 | 100% | n/a |
| qed-gnn-500 | 100 | 1.000 | 0.910 | 1.000 | 0.000 | +0.276 +/- 0.128 | 100% | n/a |

Validity is 1.000 by construction, not by training: the environment only proposes candidates that pass RDKit sanitization. The column confirms the invariant holds.

### Throughput

| run | episodes | steps/s | s/episode | wall-clock (min) |
| --- | --- | --- | --- | --- |
| constrained-logp | 800 | 18.0 | 1.11 | 14.8 |
| constrained-logp-3pass | 2400 | 23.0 | 0.87 | 34.8 |
| logp-fp-2000 | 2000 | 28.4 | 1.41 | 47.0 |
| logp-fp-dense-1200 | 1200 | 34.4 | 1.16 | 23.3 |
| qed-fp-2000 | 2000 | 28.2 | 1.42 | 47.2 |
| qed-fp-500 | 500 | 30.6 | 1.31 | 10.9 |
| qed-gnn-500 | 500 | 12.8 | 3.12 | 26.0 |

### Configuration probes

| configuration | episodes | greedy | best of n | mean +/- sd |
| --- | --- | --- | --- | --- |
| ablate-qed-dense | 250 | 0.458 | 0.705 | 0.479 +/- 0.071 |
| ablate-qed-paper | 250 | 0.483 | 0.737 | 0.516 +/- 0.076 |
| ablate-qed-terminal | 250 | 0.417 | 0.732 | 0.525 +/- 0.083 |
| probe-g0.9-lr1e-4 | 600 | 0.614 | 0.727 | 0.519 +/- 0.105 |
| probe-g1.0-lr1e-4 | 600 | 0.432 | 0.715 | 0.518 +/- 0.079 |
| probe-g1.0-lr5e-4 | 600 | 0.385 | 0.653 | 0.481 +/- 0.080 |
| probe-lr5e-4 | 600 | 0.557 | 0.763 | 0.541 +/- 0.099 |
| random-edit baseline | n/a | n/a | 0.788 | 0.493 +/- 0.079 |

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

6. **The first constrained run trained for one pass over its start set.** 800
   episodes over 800 start molecules assigns each molecule exactly one episode,
   which is an evaluation pass, not training, and `train()` now warns when the
   start set is cycled fewer than twice. Both runs are reported, because the fix
   turned out not to matter: at delta = 0.4 the one-pass and three-pass runs score
   +0.97 and +0.92 improvement respectively. A real bug, and not the cause of the
   failure --- which is worth separating rather than letting the fix take credit.

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
