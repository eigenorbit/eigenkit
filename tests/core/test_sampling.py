"""Counts-based measurement core.

No SDK and no hardware: `_sampling` is pure numpy, so these run everywhere.
The load-bearing test is `TestAgainstExactSimulator`, which feeds the reducer
the *exact* probabilities of each rotated circuit as if they were counts. Any
error in the grouping, the basis rotation, the parity signs or the bit order
shows up there as a mismatch against the numpy simulator's analytic
expectation values, with no shot noise to hide behind.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from eigenkit import Circuit, Encoding, Variational, all_z
from eigenkit.core.backends import _sampling
from eigenkit.core.ir import Gate, Observable


def _obs(text: str) -> Observable:
    return Observable.from_string(text)


class TestPlanMeasurements:
    def test_all_z_is_one_group(self):
        groups = _sampling.plan_measurements(list(all_z(4)), 4)
        assert len(groups) == 1
        assert groups[0].basis == ("Z", "Z", "Z", "Z")
        assert len(groups[0].terms) == 4

    def test_disjoint_qubits_share_a_group(self):
        groups = _sampling.plan_measurements([_obs("1.0 X0"), _obs("1.0 Y1")], 2)
        assert len(groups) == 1
        assert groups[0].basis == ("X", "Y")

    def test_conflicting_bases_split(self):
        groups = _sampling.plan_measurements([_obs("1.0 X0"), _obs("1.0 Z0")], 1)
        assert [g.basis for g in groups] == [("X",), ("Z",)]

    def test_unconstrained_qubits_stay_free(self):
        """A qubit no term has claimed is "I", so a later term can still take
        it — that is what keeps `Z1` out of a group of its own."""
        groups = _sampling.plan_measurements([_obs("1.0 X0 Y2"), _obs("1.0 Z1")], 3)
        assert len(groups) == 1
        assert groups[0].basis == ("X", "Z", "Y")

    def test_grouping_can_be_switched_off(self):
        observables = [_obs("1.0 Z0 Z1"), _obs("1.0 Z0")]
        assert len(_sampling.plan_measurements(observables, 2)) == 1
        ungrouped = _sampling.plan_measurements(observables, 2, group_commuting=False)
        assert len(ungrouped) == 2

    def test_terms_point_back_at_their_observable(self):
        groups = _sampling.plan_measurements([_obs("2.0 Z0"), _obs("3.0 Z1")], 2)
        terms = {t.observable_index: t.coeff for t in groups[0].terms}
        assert terms == {0: 2.0, 1: 3.0}

    def test_qubit_outside_the_circuit_is_rejected(self):
        with pytest.raises(ValueError, match="outside the 2-qubit circuit"):
            _sampling.plan_measurements([_obs("1.0 Z5")], 2)

    def test_is_z_basis(self):
        (z_group,) = _sampling.plan_measurements([_obs("1.0 Z0")], 2)
        (x_group,) = _sampling.plan_measurements([_obs("1.0 X0")], 2)
        assert z_group.is_z_basis
        assert not x_group.is_z_basis


class TestRotationGates:
    def test_z_and_identity_need_nothing(self):
        assert _sampling.rotation_gates(("Z", "I")) == ()

    def test_x_is_one_hadamard(self):
        assert _sampling.rotation_gates(("X",)) == (("H", 0),)

    def test_y_is_sdg_then_h(self):
        assert _sampling.rotation_gates(("Y",)) == (("Sdg", 0), ("H", 0))

    def test_gates_are_ordered_by_qubit(self):
        assert _sampling.rotation_gates(("X", "Z", "Y")) == (
            ("H", 0),
            ("Sdg", 2),
            ("H", 2),
        )


class TestNormalizeCounts:
    def test_keys_are_reversed_to_qubit_0_msb(self):
        # Little-endian "001" is qubit 0 excited, which the library writes "100".
        assert _sampling.normalize_counts({"001": 7}, 3) == {"100": 7}

    def test_short_keys_are_padded(self):
        assert _sampling.normalize_counts({"1": 4}, 3) == {"100": 4}

    def test_separate_registers_are_joined(self):
        assert _sampling.normalize_counts({"1 0": 3}, 2) == {"01": 3}

    def test_keys_that_collide_after_padding_are_summed(self):
        assert _sampling.normalize_counts({"1": 2, "01": 3}, 2) == {"10": 5}

    def test_big_endian_input_is_left_alone(self):
        assert _sampling.normalize_counts({"100": 7}, 3, little_endian=False) == {
            "100": 7
        }

    def test_rejects_a_key_wider_than_the_circuit(self):
        with pytest.raises(ValueError, match="more than the 2-qubit circuit"):
            _sampling.normalize_counts({"111": 1}, 2)

    def test_rejects_a_non_binary_key(self):
        with pytest.raises(ValueError, match="unexpected characters"):
            _sampling.normalize_counts({"0x1": 1}, 3)


class TestProbabilitiesFromCounts:
    def test_keys_map_to_basis_state_indices(self):
        # Qubit-0-MSB "100" is basis state 4 of 8.
        probabilities = _sampling.probabilities_from_counts({"100": 10}, 3)
        assert probabilities[4] == 1.0
        assert probabilities.sum() == pytest.approx(1.0)
        assert len(probabilities) == 8

    def test_unmeasured_states_are_exact_zeros(self):
        probabilities = _sampling.probabilities_from_counts({"00": 3, "11": 1}, 2)
        np.testing.assert_array_equal(probabilities, [0.75, 0.0, 0.0, 0.25])

    def test_width_is_capped(self):
        too_wide = _sampling.MAX_DENSE_QUBITS + 1
        with pytest.raises(ValueError, match="sample\\(\\)"):
            _sampling.probabilities_from_counts({"0" * too_wide: 1}, too_wide)

    def test_empty_histogram_is_rejected(self):
        with pytest.raises(ValueError, match="no shots"):
            _sampling.probabilities_from_counts({}, 2)


class TestExpectationsFromCounts:
    def test_parity_sets_the_sign(self):
        (group,) = _sampling.plan_measurements([_obs("1.0 Z0")], 1)
        assert _sampling.expectations_from_counts(
            [group], [{"0": 100}], 1, 1
        ) == pytest.approx([1.0])
        assert _sampling.expectations_from_counts(
            [group], [{"1": 100}], 1, 1
        ) == pytest.approx([-1.0])

    def test_coefficients_are_applied(self):
        (group,) = _sampling.plan_measurements([_obs("2.5 Z0")], 1)
        assert _sampling.expectations_from_counts(
            [group], [{"0": 40}], 1, 1
        ) == pytest.approx([2.5])

    def test_terms_of_one_observable_are_summed(self):
        (group,) = _sampling.plan_measurements([_obs("1.0 Z0 + 1.0 Z1")], 2)
        # |01>: Z0 = +1, Z1 = -1, so the sum is 0.
        assert _sampling.expectations_from_counts(
            [group], [{"01": 10}], 1, 2
        ) == pytest.approx([0.0])

    def test_shots_are_counted_per_group(self):
        """A group that ran with fewer shots still normalizes by its own
        total, so an uneven pair of jobs does not skew its neighbour."""
        groups = _sampling.plan_measurements([_obs("1.0 X0"), _obs("1.0 Z0")], 1)
        values = _sampling.expectations_from_counts(
            groups, [{"0": 10}, {"1": 1000}], 2, 1
        )
        assert values == pytest.approx([1.0, -1.0])

    def test_histogram_count_must_match_the_plan(self):
        groups = _sampling.plan_measurements([_obs("1.0 X0"), _obs("1.0 Z0")], 1)
        with pytest.raises(ValueError, match="one histogram per measurement group"):
            _sampling.expectations_from_counts(groups, [{"0": 5}], 2, 1)

    def test_empty_group_is_rejected(self):
        groups = _sampling.plan_measurements([_obs("1.0 Z0")], 1)
        with pytest.raises(ValueError, match="no shots"):
            _sampling.expectations_from_counts(groups, [{}], 1, 1)

    def test_ragged_keys_are_rejected(self):
        groups = _sampling.plan_measurements([_obs("1.0 Z0")], 2)
        with pytest.raises(ValueError, match="normalize_counts"):
            _sampling.expectations_from_counts(groups, [{"0": 5}], 1, 2)


class TestAgainstExactSimulator:
    """Reduce *exact* probabilities as if they were counts, and the result has
    to be the simulator's analytic expectation values. This is the end-to-end
    correctness gate for the whole sampling path; the hardware adapters add
    shot noise on top of it, nothing else."""

    @staticmethod
    def _exact_counts(circuit, group, weights, inputs, n_qubits):
        """The rotated circuit's exact distribution, scaled into whole-looking
        counts. Uses the numpy simulator, so there is no sampling error."""
        rotation = tuple(
            Gate(name=name, qubits=(qubit,))
            for name, qubit in _sampling.rotation_gates(group.basis)
        )
        rotated = dataclasses.replace(circuit.spec, gates=circuit.spec.gates + rotation)
        probabilities = Circuit(
            blocks=rotated, observables=all_z(n_qubits), provider="numpy"
        ).probabilities(weights, inputs)
        return {
            format(state, f"0{n_qubits}b"): probability * 1e9
            for state, probability in enumerate(probabilities)
            if probability > 0
        }

    @pytest.mark.parametrize("group_commuting", [True, False])
    def test_matches_analytic_expectations(self, group_commuting):
        n_qubits = 3
        observables = [
            _obs("1.0 Z0 Z1"),
            _obs("0.5 X0 Y2 - 2.0 Z1"),
            _obs("1.0 X0 X1 X2"),
            _obs("1.5 Y1"),
        ]
        circuit = Circuit(
            blocks=[
                Encoding(in_features=n_qubits),
                Variational(depth=2, rotation="RY"),
            ],
            observables=observables,
            provider="numpy",
        )
        rng = np.random.default_rng(0)
        weights = rng.normal(size=circuit.spec.n_weights)
        inputs = rng.normal(size=n_qubits)

        groups = _sampling.plan_measurements(
            observables, n_qubits, group_commuting=group_commuting
        )
        counts = [
            self._exact_counts(circuit, group, weights, inputs, n_qubits)
            for group in groups
        ]
        reduced = _sampling.expectations_from_counts(
            groups, counts, len(observables), n_qubits
        )
        np.testing.assert_allclose(reduced, circuit(weights, inputs), rtol=0, atol=1e-12)

    def test_probabilities_round_trip(self):
        circuit = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1, rotation="RY")],
            observables=all_z(2),
            provider="numpy",
        )
        rng = np.random.default_rng(1)
        weights = rng.normal(size=circuit.spec.n_weights)
        inputs = rng.normal(size=2)
        exact = circuit.probabilities(weights, inputs)
        counts = {
            format(state, "02b"): probability * 1e9
            for state, probability in enumerate(exact)
        }
        np.testing.assert_allclose(
            _sampling.probabilities_from_counts(counts, 2), exact, rtol=0, atol=1e-12
        )
