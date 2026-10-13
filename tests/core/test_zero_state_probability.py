"""Tests for `Circuit.zero_state_probability` — the non-Pauli measurement
primitive used by quantum kernels and other fidelity-based observables.

Verifies cross-backend numerical agreement and a few sanity properties:
- |<0|psi(x=0)>|² = 1 (no rotation applied keeps |0…0>).
- |<0|psi(x)>|² agrees with the closed-form `prod_i cos²(x_i/2)` for an
  RX angle-encoding circuit.
- A fidelity-inversion circuit U(x)U†(x) returns 1 exactly (state
  evolves back to |0…0>).

These run on the numpy interface (the default): the circuits carry only
data inputs and no trainable weights, so an empty weight vector is passed.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from eigenkit import Circuit, Encoding
from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.ir import Negate


_NO_WEIGHTS = np.empty(0, dtype=np.float64)


def _zsp(blocks, provider, inputs):
    """Zero-state probability of a weight-free circuit on the numpy interface."""
    circuit = Circuit(blocks=blocks, provider=provider, interface="numpy")
    return circuit.zero_state_probability(
        _NO_WEIGHTS, np.asarray(inputs, dtype=np.float64)
    )


def _is_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _all_providers():
    providers = ["numpy", "pennylane"]
    if _is_available("qiskit"):
        providers.append("qiskit")
    if _is_available("braket"):
        providers.append("braket")
    return providers


class TestZeroInputs:
    """No rotations applied → state stays |0…0>; probability is exactly 1."""

    @pytest.mark.parametrize("provider", _all_providers())
    def test_zero_inputs_give_unit_probability(self, provider):
        p = _zsp([Encoding(in_features=3, rotation="RX")], provider, np.zeros(3))
        assert abs(p - 1.0) < 1e-10


class TestRXEncodingClosedForm:
    """For RX(x) on each qubit: |<0|psi>|² = ∏_i cos²(x_i/2)."""

    @pytest.mark.parametrize("provider", _all_providers())
    def test_matches_closed_form(self, provider):
        x = np.array([0.5, 0.3, -0.7], dtype=np.float64)
        expected = float(np.prod([math.cos(xi / 2) ** 2 for xi in x]))
        p = _zsp([Encoding(in_features=3, rotation="RX")], provider, x)
        assert abs(p - expected) < 1e-10


class TestCrossBackendParity:
    @pytest.mark.parametrize("rotation", ["RX", "RY", "RZ"])
    def test_all_backends_agree(self, rotation):
        x = np.array([0.7, -0.3, 0.5, 0.2], dtype=np.float64)
        results = {
            provider: _zsp([Encoding(in_features=4, rotation=rotation)], provider, x)
            for provider in _all_providers()
            if provider != "pennylane" or _is_available("pennylane")
        }
        ref = results["numpy"]
        for provider, p in results.items():
            assert abs(p - ref) < 1e-10, f"{provider} disagrees with numpy: {p} vs {ref}"


class TestInversionCircuit:
    """U(x) U†(x) leaves the state at |0…0>; ZSP must equal 1 exactly,
    independent of x."""

    @pytest.mark.parametrize("provider", _all_providers())
    def test_inversion_returns_to_zero(self, provider):
        n = 3
        x_value = np.array([0.4, -0.2, 0.6], dtype=np.float64)

        # Build circuit: forward(x_inputs) then adjoint with negated angles
        # using x_inputs (same input vector, no second copy needed for
        # this self-inversion test).
        qc = CircuitBuilder(n_qubits=n)
        x = qc.add_inputs(n)
        for q in range(n):
            qc.rx(x[q], qubit=q)
        for q in reversed(range(n)):
            qc.rx(Negate(x[q]), qubit=q)
        ir = qc.build()

        p = _zsp(ir, provider, x_value)
        assert abs(p - 1.0) < 1e-9, f"{provider}: U U† should give 1, got {p}"


class TestRangeAndType:
    @pytest.mark.parametrize("provider", _all_providers())
    def test_returns_float_in_unit_interval(self, provider):
        p = _zsp(
            [Encoding(in_features=2, rotation="RX")], provider, np.array([1.7, -2.3])
        )
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0 + 1e-12
