"""Tests for valid-action enumeration in the molecule-editing MDP.

The action space follows Zhou et al. (2019), section 3.1: atom addition, bond
addition (including bond-order increase) and bond removal (including
bond-order decrease), all filtered by RDKit valence sanitization.
"""

from __future__ import annotations

import pytest

from droptimus.chem.molecule import canonical_smiles, parse_smiles
from droptimus.config import EnvConfig
from droptimus.env.actions import (
    atom_additions,
    bond_additions,
    bond_removals,
    free_valence_map,
    valid_actions,
)
from droptimus.errors import InvalidSmilesError

CONFIG = EnvConfig(atom_types=("C", "N", "O"))


class TestFreeValence:
    def test_methane_has_four_free_valences(self) -> None:
        mapping = free_valence_map(parse_smiles("C"))
        assert mapping[1] == (0,)
        assert mapping[2] == (0,)
        assert mapping[3] == (0,)

    def test_benzene_has_one_free_valence_per_carbon(self) -> None:
        mapping = free_valence_map(parse_smiles("c1ccccc1"))
        assert len(mapping[1]) == 6
        assert mapping[2] == ()

    def test_fully_substituted_carbon_has_none(self) -> None:
        mapping = free_valence_map(parse_smiles("CC(C)(C)C"))
        assert 1 not in mapping or len(mapping[1]) == 4  # the four methyls


class TestAtomAdditions:
    def test_methane_grows_by_one_heavy_atom(self) -> None:
        results = atom_additions(parse_smiles("C"), CONFIG)
        assert canonical_smiles("CC") in results
        assert canonical_smiles("CN") in results
        assert canonical_smiles("CO") in results
        # Double and triple bonds to the single carbon are reachable too.
        assert canonical_smiles("C=C") in results
        assert canonical_smiles("C#C") in results

    def test_every_result_is_valid_and_larger(self) -> None:
        start = parse_smiles("CCO")
        for smiles in atom_additions(start, CONFIG):
            mol = parse_smiles(smiles)
            assert mol.GetNumAtoms() == start.GetNumAtoms() + 1

    def test_respects_atom_type_restriction(self) -> None:
        carbon_only = EnvConfig(atom_types=("C",))
        results = atom_additions(parse_smiles("C"), carbon_only)
        assert all("N" not in s and "O" not in s for s in results)

    def test_returns_frozenset(self) -> None:
        assert isinstance(atom_additions(parse_smiles("C"), CONFIG), frozenset)

    def test_no_additions_when_no_free_valence(self) -> None:
        # Every carbon in this molecule is fully substituted by heavy atoms.
        results = atom_additions(parse_smiles("C#N"), EnvConfig(atom_types=("C",)))
        # The nitrogen is saturated and the carbon has one free valence only.
        assert all(parse_smiles(s).GetNumAtoms() == 3 for s in results)


class TestBondAdditions:
    def test_creates_ring_in_allowed_size(self) -> None:
        # Hexane can close to cyclohexane: shortest path length 6.
        results = bond_additions(parse_smiles("CCCCCC"), CONFIG)
        assert canonical_smiles("C1CCCCC1") in results

    def test_rejects_ring_outside_allowed_sizes(self) -> None:
        config = EnvConfig(atom_types=("C",), allowed_ring_sizes=(3, 4, 5))
        results = bond_additions(parse_smiles("CCCCCC"), config)
        assert canonical_smiles("C1CCCCC1") not in results

    def test_increases_existing_bond_order(self) -> None:
        results = bond_additions(parse_smiles("CC"), CONFIG)
        assert canonical_smiles("C=C") in results
        assert canonical_smiles("C#C") in results

    def test_never_exceeds_valence(self) -> None:
        for smiles in bond_additions(parse_smiles("CCCCCC"), CONFIG):
            parse_smiles(smiles)  # raises if the valence is impossible

    def test_atom_count_is_unchanged(self) -> None:
        start = parse_smiles("CCCCCC")
        for smiles in bond_additions(start, CONFIG):
            assert parse_smiles(smiles).GetNumAtoms() == start.GetNumAtoms()


class TestBondRemovals:
    def test_lowers_bond_order(self) -> None:
        results = bond_removals(parse_smiles("C=C"), CONFIG)
        assert canonical_smiles("CC") in results

    def test_breaking_a_ring_is_allowed(self) -> None:
        results = bond_removals(parse_smiles("C1CCCCC1"), CONFIG)
        assert canonical_smiles("CCCCCC") in results

    def test_does_not_return_multi_fragment_molecules_by_default(self) -> None:
        for smiles in bond_removals(parse_smiles("CCOCC"), CONFIG):
            assert "." not in smiles

    def test_empty_for_single_atom(self) -> None:
        assert bond_removals(parse_smiles("C"), CONFIG) == frozenset()


class TestValidActions:
    def test_union_of_the_three_action_types(self) -> None:
        mol = parse_smiles("CCO")
        expected = (
            atom_additions(mol, CONFIG)
            | bond_additions(mol, CONFIG)
            | bond_removals(mol, CONFIG)
        )
        actions = valid_actions("CCO", CONFIG)
        assert expected <= actions

    def test_no_modification_included_when_enabled(self) -> None:
        actions = valid_actions("CCO", EnvConfig(allow_no_modification=True))
        assert canonical_smiles("CCO") in actions

    def test_no_modification_excluded_when_disabled(self) -> None:
        config = EnvConfig(allow_no_modification=False, allow_bond_removal=False)
        actions = valid_actions("CCO", config)
        assert canonical_smiles("CCO") not in actions

    def test_atom_cap_stops_growth(self) -> None:
        config = EnvConfig(atom_types=("C",), max_atoms=3)
        actions = valid_actions("CCC", config)
        assert all(parse_smiles(a).GetNumAtoms() <= 3 for a in actions)

    def test_never_empty_for_a_real_molecule(self) -> None:
        assert len(valid_actions("CCO", CONFIG)) > 0

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(InvalidSmilesError):
            valid_actions("bogus", CONFIG)

    def test_results_are_canonical(self) -> None:
        for smiles in valid_actions("CCO", CONFIG):
            assert smiles == canonical_smiles(smiles)

    def test_max_actions_cap_is_respected(self) -> None:
        import random

        config = EnvConfig(atom_types=("C", "N", "O"), max_actions=5)
        actions = valid_actions("CCCCCC", config, rng=random.Random(0))
        assert len(actions) == 5

    def test_cap_is_deterministic_given_a_seed(self) -> None:
        import random

        config = EnvConfig(atom_types=("C", "N", "O"), max_actions=7)
        first = valid_actions("CCCCCC", config, rng=random.Random(42))
        second = valid_actions("CCCCCC", config, rng=random.Random(42))
        assert first == second

    def test_single_atom_start_can_still_act(self) -> None:
        actions = valid_actions("C", CONFIG)
        assert canonical_smiles("CC") in actions


class TestActionsAreAddressable:
    """Every offered action must canonicalize to itself.

    The environment identifies an action by its canonical SMILES, so an action
    whose SMILES reads back as a different string would be offered to the agent
    and then rejected by step(). This crashed a 600-episode training run.
    """

    def test_regression_fused_ring_with_unstable_aromaticity(self) -> None:
        # Built from a kekulized template, this molecule's successor
        # 'NC12c3o[nH][nH]n1c32' sanitizes cleanly but reads back as
        # 'NC12C3=C1N2NNO3'.
        for action in valid_actions("NC1c2cn1[nH][nH]o2", CONFIG):
            assert action == canonical_smiles(action), action

    @pytest.mark.parametrize(
        "smiles",
        [
            "NC1c2cn1[nH][nH]o2",
            "C",
            "CCO",
            "c1ccccc1",
            "c1ccc2ccccc2c1",
            "O=c1[nH]cnc2[nH]cnc12",
            "C1=CC2=NN=C(O2)C1",
            "c1cnn2ccccc12",
            "N1NOC2=C1C=C2",
            "C1OC2NNC2N1",
            "O=C1C=CC2=NOC2=C1",
            "c1ccc2c(c1)oc1ccccc12",
            "CC(=O)Oc1ccccc1C(=O)O",
        ],
    )
    def test_every_action_is_a_canonicalization_fixed_point(self, smiles: str) -> None:
        for action in valid_actions(smiles, CONFIG):
            assert action == canonical_smiles(action), f"{smiles} -> {action}"

    def test_every_action_is_accepted_by_the_environment(self) -> None:
        """The end-to-end invariant: anything offered can be stepped."""
        import random

        from droptimus.env.mdp import MoleculeEnv
        from droptimus.objectives import make_objective

        env = MoleculeEnv(
            objective=make_objective("qed"),
            config=EnvConfig(atom_types=("C", "N", "O"), max_steps=12, max_atoms=14),
            start_smiles="NC1c2cn1[nH][nH]o2",
            rng=random.Random(0),
        )
        rng = random.Random(5)
        for _ in range(25):
            state = env.reset()
            while not state.done:
                offered = sorted(env.valid_actions())
                state = env.step(rng.choice(offered)).state
