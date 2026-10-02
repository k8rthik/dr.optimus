"""Central configuration for dr.optimus.

All tunable constants live here; nothing else in the package hardcodes a
magic number. Config objects are frozen dataclasses and are therefore safe to
share between the environment, the agent and the evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

# --- Chemistry -------------------------------------------------------------

#: Element symbols the environment is allowed to add. {C, N, O} is the atom
#: set used for the single-property experiments in Zhou et al. (2019).
DEFAULT_ATOM_TYPES: tuple[str, ...] = ("C", "N", "O")

#: Elements the featurizer knows about. Wider than DEFAULT_ATOM_TYPES so that
#: start molecules containing S/Cl/F/Br can still be encoded.
FEATURIZED_ELEMENTS: tuple[str, ...] = (
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se",
)

#: Bond orders the environment may create. Aromatic bonds are never created
#: directly; they only arise from RDKit's perception of an existing ring.
BOND_ORDERS: tuple[int, ...] = (1, 2, 3)

#: Channels used by the relational GNN adjacency tensor.
BOND_CHANNELS: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

MORGAN_RADIUS: int = 2
MORGAN_BITS: int = 2048

#: Hard cap on atoms a featurized molecule may have (padding width for the GNN).
FEATURIZER_MAX_ATOMS: int = 60

#: How many write/read cycles a candidate molecule gets to settle on a stable
#: canonical SMILES. RDKit's aromaticity perception is not always a fixed point
#: of a single cycle: a molecule built from a kekulized template can be written
#: with aromatic flags that kekulize differently when the string is read back.
#: Actions are identified by their canonical SMILES, so an unstable string would
#: make an action the environment offered unrecognizable when it came back.
MAX_CANONICAL_ROUND_TRIPS: int = 3

# --- Objective constants ---------------------------------------------------

#: Normalization constants for penalized logP, from Kusner et al. (2017) /
#: You et al. (2018) and reused by Zhou et al. (2019). They are the mean and
#: standard deviation of each term over the 250k-molecule ZINC set.
LOGP_MEAN: float = 2.4570953396190123
LOGP_STD: float = 1.434324401111988
SA_MEAN: float = 3.0525811293166134
SA_STD: float = 0.8335207024513095
CYCLE_MEAN: float = 0.0485696876403053
CYCLE_STD: float = 0.2860212110245455

#: Rings larger than this are penalized by the penalized-logP objective.
MAX_UNPENALIZED_RING_SIZE: int = 6

# --- Defaults --------------------------------------------------------------

DEFAULT_START_SMILES: str = "C"
DEFAULT_SIMILARITY_DELTA: float = 0.4


def field_default(config_class: type, name: str) -> object:
    """Return the declared default of a config field.

    These dataclasses use ``slots=True``, so ``EnvConfig.discount`` is a slot
    descriptor rather than the default value. Argparse needs the real default, so
    it goes through here.

    Raises:
        KeyError: if ``name`` is not a field of ``config_class``.
    """
    fields = getattr(config_class, "__dataclass_fields__", {})
    if name not in fields:
        raise KeyError(f"{config_class.__name__} has no field {name!r}.")
    return fields[name].default


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Configuration of the molecule-editing MDP."""

    atom_types: tuple[str, ...] = DEFAULT_ATOM_TYPES
    allowed_ring_sizes: tuple[int, ...] = (3, 4, 5, 6)
    max_steps: int = 40
    max_atoms: int = 38
    allow_no_modification: bool = True
    allow_bond_removal: bool = True
    allow_removal_disconnect: bool = False
    #: Cap on the number of candidate actions scored per step. ``None`` means
    #: no cap (faithful to the paper). An integer uniformly subsamples the
    #: valid-action set, which trades fidelity for wall-clock time.
    max_actions: int | None = None
    #: The paper uses 0.9. With ``reward_mode="terminal"`` and a 40-step horizon
    #: that makes the value of an early state 0.9**39 ~ 0.015 times the final
    #: objective, which is a badly conditioned regression target; gamma = 1 makes
    #: Q(m, h) predict the achievable final objective directly. Measured over 600
    #: QED episodes, gamma 1.0 reached mean 0.598 / best 0.833 against 0.541 /
    #: 0.763 for gamma 0.9. Pass ``--discount 0.9`` for the paper's value.
    discount: float = 1.0
    #: ``terminal`` pays the objective once, at the end of the episode.
    #: ``dense`` pays the per-step change in objective value.
    #: ``paper`` reproduces the reference implementation: every step pays
    #: ``objective * discount ** steps_remaining``.
    reward_mode: Literal["terminal", "dense", "paper"] = "terminal"


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Configuration of the double-DQN agent."""

    encoder: Literal["fingerprint", "gnn"] = "fingerprint"
    hidden_sizes: tuple[int, ...] = (1024, 512, 128, 32)
    gnn_hidden: int = 64
    gnn_layers: int = 3
    #: The paper uses 1e-4. At the episode budget that fits a laptop, 5e-4
    #: converges enough faster to matter: measured over 600 QED episodes, mean
    #: 0.598 / best 0.833 against 0.506 / 0.816 at 1e-4.
    learning_rate: float = 5e-4
    grad_clip: float = 10.0
    batch_size: int = 32
    replay_capacity: int = 20_000
    #: Cap on how many successor candidates the bootstrap target maxes over.
    #: The full set can run to several hundred molecules; scoring all of them
    #: for every transition in a batch dominates wall-clock time. Subsampling
    #: makes the target a max over a subset, which biases it slightly low.
    #: Measured: one gradient step costs 27.5 ms at 48 and 13.5 ms at 16, so 16
    #: buys roughly twice as many gradient steps per minute of training. At a
    #: fixed wall-clock budget that trade is worth more than the bias.
    bootstrap_actions: int = 16
    #: Gradient updates per environment step.
    updates_per_step: int = 1
    #: Environment steps between hard target-network syncs.
    target_sync_steps: int = 500
    double_dqn: bool = True
    #: Number of steps-remaining buckets appended to the state encoding.
    use_steps_remaining: bool = True


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Configuration of a training run."""

    episodes: int = 800
    warmup_episodes: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    #: Fraction of total episodes over which epsilon decays linearly.
    epsilon_decay_fraction: float = 0.7
    seed: int = 0
    device: str = "auto"
    log_every: int = 25
    eval_episodes: int = 100


@dataclass(frozen=True, slots=True)
class RunConfig:
    """A complete, reproducible run specification."""

    objective: str = "qed"
    start_smiles: str = DEFAULT_START_SMILES
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    objective_kwargs: tuple[tuple[str, float], ...] = ()

    def with_env(self, **kwargs: object) -> RunConfig:
        """Return a copy with environment fields overridden."""
        return replace(self, env=replace(self.env, **kwargs))

    def with_agent(self, **kwargs: object) -> RunConfig:
        """Return a copy with agent fields overridden."""
        return replace(self, agent=replace(self.agent, **kwargs))

    def with_train(self, **kwargs: object) -> RunConfig:
        """Return a copy with training fields overridden."""
        return replace(self, train=replace(self.train, **kwargs))
