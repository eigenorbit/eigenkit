"""Controlled rotations need the four-term parameter-shift rule.

The two-term rule `[f(t + pi/2) - f(t - pi/2)] / 2` is exact only when the
gate's generator has a single eigenvalue gap. A controlled rotation's
generator `|1><1| (x) P/2` has gaps `{1/2, 1}`, so its expectation carries two
frequencies and the two-term rule is wrong by tens of percent — silently, on
every backend that differentiates by shifting rather than natively.

These tests pin the rule against the numpy backend's exact adjoint gradient,
on both routes that shift: the naive per-weight one and the per-gate
chain-rule one.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import (
    MULTI_FREQUENCY_GATES,
    CircuitSpec,
    Gate,
    InputRef,
    Observable,
    PauliTerm,
    Scale,
    WeightRef,
    inputs_needing_four_term_shift,
    weights_needing_four_term_shift,
)


X0 = Observable(terms=(PauliTerm(1.0, (("X", 0),)),))
ANGLES = [0.7, 2.3, -1.1]

# Backends that differentiate by shifting, so they exercise the rule. Each is
# skipped when its SDK is absent.
SHIFT_BACKENDS = ["qiskit", "qiskit-aer", "qsim", "braket"]


def _cry_spec() -> CircuitSpec:
    """H(0) then CRY(w0, 0->1): the control is in superposition, which is
    where the two rules disagree. With the control in |1> they agree, so a
    test that prepares |1> first cannot see the bug."""
    return CircuitSpec(
        n_qubits=2,
        n_weights=1,
        n_inputs=0,
        gates=(Gate("H", (0,)), Gate("CRY", (0, 1), (WeightRef(0),))),
    )


def _scaled_cry_spec() -> CircuitSpec:
    """A weight wrapped in `Scale` makes the naive path unsafe, so this routes
    through the per-gate chain-rule jacobian instead."""
    return CircuitSpec(
        n_qubits=2,
        n_weights=1,
        n_inputs=1,
        gates=(
            Gate("H", (0,)),
            Gate("CRY", (0, 1), (Scale(WeightRef(0), 2.0),)),
            Gate("RY", (1,), (InputRef(0),)),
        ),
    )


def _backend(name):
    pytest.importorskip(
        {
            "qiskit": "qiskit",
            "qiskit-aer": "qiskit_aer",
            "qsim": "qsimcirq",
            "braket": "braket",
        }[name]
    )
    return name


class TestTheRuleIsSelected:
    def test_cry_is_the_multi_frequency_gate(self):
        assert "CRY" in MULTI_FREQUENCY_GATES

    def test_a_weight_feeding_a_cry_is_flagged(self):
        assert weights_needing_four_term_shift(_cry_spec()) == frozenset({0})

    def test_a_weight_feeding_only_single_frequency_gates_is_not(self):
        spec = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=0,
            gates=(Gate("RY", (0,), (WeightRef(0),)),),
        )
        assert weights_needing_four_term_shift(spec) == frozenset()

    def test_the_flag_reaches_through_an_arithmetic_wrapper(self):
        assert weights_needing_four_term_shift(_scaled_cry_spec()) == frozenset({0})

    def test_inputs_are_analysed_the_same_way(self):
        spec = CircuitSpec(
            n_qubits=2,
            n_weights=0,
            n_inputs=1,
            gates=(Gate("CRY", (0, 1), (InputRef(0),)),),
        )
        assert inputs_needing_four_term_shift(spec) == frozenset({0})


class TestGradientsMatchTheExactAdjoint:
    """numpy differentiates a CRY natively (adjoint), so it is the oracle."""

    @pytest.mark.parametrize("theta", ANGLES)
    @pytest.mark.parametrize("provider", SHIFT_BACKENDS)
    def test_naive_path(self, provider, theta):
        spec = _cry_spec()
        weights = np.array([theta])
        expected = Circuit(spec, X0, "numpy", "numpy").jacobian(weights)
        actual = Circuit(spec, X0, _backend(provider), "numpy").jacobian(weights)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    @pytest.mark.parametrize("theta", ANGLES)
    @pytest.mark.parametrize("provider", SHIFT_BACKENDS)
    def test_chain_rule_path(self, provider, theta):
        spec = _scaled_cry_spec()
        weights, inputs = np.array([theta]), np.array([0.4])
        reference = Circuit(spec, X0, "numpy", "numpy")
        assert reference._has_unsafe_gates, "expected the chain-rule route"
        expected, expected_in = reference._gradient_jacobians(
            weights, inputs, need_input_grad=True
        )
        circuit = Circuit(spec, X0, _backend(provider), "numpy")
        actual, actual_in = circuit._gradient_jacobians(
            weights, inputs, need_input_grad=True
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(actual_in, expected_in, atol=1e-6)

    @pytest.mark.parametrize("theta", ANGLES)
    def test_the_two_term_rule_really_does_disagree(self, theta):
        """Guard the guard: if this ever stops failing, the fixtures above
        stopped exercising the case they were written for."""
        from eigenkit.core.provider import _shift_column

        spec = _cry_spec()
        weights = np.array([theta])
        circuit = Circuit(spec, X0, "numpy", "numpy")
        exact = circuit.jacobian(weights)[0]

        def evaluate(vector):
            return circuit.compiled.run(vector, np.empty(0))

        # Both rules through the same helper the jacobians use, so this
        # compares what the library actually applies rather than a rewritten
        # copy of the rule.
        two_term = _shift_column(evaluate, weights, 0, four_term=False)[0]
        four_term = _shift_column(evaluate, weights, 0, four_term=True)[0]

        assert abs(two_term - exact) > 0.05 * abs(exact), (
            "the two-term rule agrees here, so this fixture no longer "
            "exercises a multi-frequency gate"
        )
        np.testing.assert_allclose(four_term, exact, atol=1e-9)


class TestSingleFrequencyGatesAreUnaffected:
    """The four-term rule is exact for one frequency too, so single-frequency
    gates must keep matching the adjoint.

    Over the same backend list as the four-term tests above, so every shift
    backend is checked, not just one.
    """

    @pytest.mark.parametrize("provider", SHIFT_BACKENDS)
    @pytest.mark.parametrize("gate_name", ["RY", "PhaseShift", "CPHASE"])
    @pytest.mark.parametrize("theta", ANGLES)
    def test_still_matches_the_adjoint(self, gate_name, theta, provider):
        from eigenkit.core.provider import resolve_provider

        backend = _backend(provider)
        supported = resolve_provider(backend).capabilities.supported_gates
        if gate_name not in supported:
            # Braket has no CPHASE, by its own declaration. Asked of the
            # capabilities rather than hard-coded, so a backend that gains the
            # gate starts being covered without an edit here.
            pytest.skip(f"{provider} does not declare {gate_name}")
        qubits = (0, 1) if gate_name == "CPHASE" else (1,)
        spec = CircuitSpec(
            n_qubits=2,
            n_weights=1,
            n_inputs=0,
            gates=(
                Gate("H", (0,)),
                Gate("H", (1,)),
                Gate(gate_name, qubits, (WeightRef(0),)),
            ),
        )
        weights = np.array([theta])
        expected = Circuit(spec, X0, "numpy", "numpy").jacobian(weights)
        actual = Circuit(spec, X0, backend, "numpy").jacobian(weights)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
