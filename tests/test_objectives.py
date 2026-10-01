"""Tests for the objective registry and the individual objectives."""

from __future__ import annotations

import pytest

from droptimus.errors import InvalidSmilesError, ObjectiveError, UnknownObjectiveError
from droptimus.objectives import available_objectives, make_objective
from droptimus.objectives.logp import penalized_logp
from droptimus.objectives.qed import qed_score

BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CELECOXIB = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
LONG_ALKANE = "CCCCCCCCCCCCCCCCCCCC"


class TestRegistry:
    def test_core_objectives_registered(self) -> None:
        names = available_objectives()
        for expected in ("qed", "penalized_logp", "similarity", "constrained"):
            assert expected in names

    def test_unknown_name_raises_with_suggestions(self) -> None:
        with pytest.raises(UnknownObjectiveError) as excinfo:
            make_objective("definitely_not_real")
        message = str(excinfo.value)
        assert "definitely_not_real" in message
        assert "qed" in message

    def test_make_objective_is_case_insensitive(self) -> None:
        assert make_objective("QED").name == "qed"

    def test_registry_listing_is_sorted_tuple(self) -> None:
        names = available_objectives()
        assert isinstance(names, tuple)
        assert list(names) == sorted(names)


class TestQed:
    def test_in_unit_interval(self) -> None:
        for smiles in (BENZENE, ASPIRIN, CELECOXIB, LONG_ALKANE):
            assert 0.0 <= qed_score(smiles) <= 1.0

    def test_drug_like_beats_long_alkane(self) -> None:
        assert qed_score(CELECOXIB) > qed_score(LONG_ALKANE)

    def test_objective_matches_function(self) -> None:
        objective = make_objective("qed")
        assert objective(ASPIRIN) == pytest.approx(qed_score(ASPIRIN))

    def test_components_reported(self) -> None:
        components = make_objective("qed").components(ASPIRIN)
        assert "qed" in components

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(InvalidSmilesError):
            qed_score("not_a_molecule")


class TestPenalizedLogp:
    def test_long_alkane_scores_high(self) -> None:
        # Penalized logP is famously maximized by long carbon chains.
        assert penalized_logp(LONG_ALKANE) > penalized_logp(BENZENE)

    def test_large_ring_is_penalized(self) -> None:
        small_ring = "C1CCCCC1"
        huge_ring = "C1CCCCCCCCCCC1"
        assert penalized_logp(huge_ring) < penalized_logp(small_ring)

    def test_components_sum_to_value(self) -> None:
        objective = make_objective("penalized_logp")
        components = objective.components(ASPIRIN)
        total = (
            components["normalized_logp"]
            + components["normalized_sa"]
            + components["normalized_cycle"]
        )
        assert total == pytest.approx(objective(ASPIRIN))

    def test_known_reference_values(self) -> None:
        # Regression lock on the standard formula (Kusner et al. 2017, reused by
        # Zhou et al. 2019). Values measured with rdkit 2026.03 and its bundled
        # SA scorer; they are stable across RDKit patch releases.
        assert penalized_logp("CCO") == pytest.approx(-0.2577, abs=1e-3)
        assert penalized_logp("c1ccccc1") == pytest.approx(2.0952, abs=1e-3)
        assert penalized_logp("C" * 20) == pytest.approx(6.4546, abs=1e-3)

    def test_ring_excess_term_only_fires_above_six(self) -> None:
        from droptimus.objectives.logp import largest_ring_excess
        from droptimus.chem.molecule import parse_smiles

        assert largest_ring_excess(parse_smiles("C1CCCCC1")) == 0
        assert largest_ring_excess(parse_smiles("C1CCCCCCC1")) == 2
        assert largest_ring_excess(parse_smiles("CCCC")) == 0


class TestSimilarity:
    def test_identity_is_one(self) -> None:
        objective = make_objective("similarity", reference=ASPIRIN)
        assert objective(ASPIRIN) == pytest.approx(1.0)

    def test_requires_reference(self) -> None:
        with pytest.raises(ObjectiveError):
            make_objective("similarity")

    def test_reference_must_be_valid(self) -> None:
        with pytest.raises(InvalidSmilesError):
            make_objective("similarity", reference="@@@")


class TestConstrained:
    def test_satisfying_molecule_scores_like_base(self) -> None:
        objective = make_objective(
            "constrained", reference=ASPIRIN, base="penalized_logp", delta=0.0
        )
        assert objective(ASPIRIN) == pytest.approx(penalized_logp(ASPIRIN))

    def test_dissimilar_molecule_is_penalized(self) -> None:
        objective = make_objective(
            "constrained", reference=ASPIRIN, base="penalized_logp", delta=0.6
        )
        assert objective(LONG_ALKANE) < penalized_logp(LONG_ALKANE)

    def test_components_expose_similarity_and_satisfaction(self) -> None:
        objective = make_objective("constrained", reference=ASPIRIN, delta=0.4)
        components = objective.components(ASPIRIN)
        assert components["similarity"] == pytest.approx(1.0)
        assert components["constraint_satisfied"] == 1.0

    def test_delta_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ObjectiveError):
            make_objective("constrained", reference=ASPIRIN, delta=1.5)

    def test_cannot_nest_constrained_in_itself(self) -> None:
        with pytest.raises(ObjectiveError):
            make_objective("constrained", reference=ASPIRIN, base="constrained")

    def test_name_records_configuration(self) -> None:
        objective = make_objective(
            "constrained", reference=ASPIRIN, base="qed", delta=0.4
        )
        assert "qed" in objective.name
        assert "0.4" in objective.name
