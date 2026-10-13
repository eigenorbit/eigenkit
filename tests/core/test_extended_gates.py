"""Tests for the X / S / Sdg gates of the IR.

Covers numpy correctness, cross-backend forward parity, and the adjoint
(S <-> Sdg).
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import CircuitSpec, adjoint
from eigenkit.core.observables import X, Y, Z, all_z, from_string


_WEIGHTS = np.array([0.5, -0.3], dtype=np.float64)


def _circuit_with_extended_gates() -> CircuitSpec:
    qc = CircuitBuilder(n_qubits=2)
    w = qc.add_weights(2)
    qc.h(0)
    qc.x(1)
    qc.s(0)
    qc.ry(w[0], 1)
    qc.sdg(1)
    qc.cnot(0, 1)
    qc.rx(w[1], 0)
    qc.x(0)
    return qc.build()


class TestNumpyCorrectness:
    def test_x_flips_to_one(self):
        spec = CircuitBuilder(n_qubits=1).x(0).build()
        out = Circuit(spec, Z(0).to_observable(), "numpy", "numpy")(np.empty(0))
        assert np.isclose(out, -1.0)  # X|0> = |1>, <Z> = -1

    def test_s_adds_phase_i(self):
        # S|+> = (|0> + i|1>)/sqrt(2) = +1 eigenstate of Y.
        spec = CircuitBuilder(n_qubits=1).h(0).s(0).build()
        out = Circuit(
            spec, [X(0).to_observable(), Y(0).to_observable()], "numpy", "numpy"
        )(np.empty(0))
        assert np.allclose(out, [0.0, 1.0], atol=1e-10)

    def test_sdg_adds_phase_minus_i(self):
        spec = CircuitBuilder(n_qubits=1).h(0).sdg(0).build()
        out = Circuit(spec, Y(0).to_observable(), "numpy", "numpy")(np.empty(0))
        assert np.isclose(out, -1.0, atol=1e-10)

    def test_builder_records_extended_gates(self):
        spec = CircuitBuilder(n_qubits=1).x(0).s(0).sdg(0).build()
        assert [g.name for g in spec.gates] == ["X", "S", "Sdg"]


class TestForwardParity:
    @pytest.fixture(params=["pennylane", "qiskit", "braket"])
    def other_provider(self, request):
        pytest.importorskip(request.param)
        return request.param

    def test_extended_gates_match_numpy(self, other_provider):
        spec = _circuit_with_extended_gates()
        observables = [
            X(0).to_observable(),
            Y(0).to_observable(),
            Z(1).to_observable(),
            from_string("X0 Y1"),
        ]
        reference = Circuit(spec, observables, "numpy", "numpy")(_WEIGHTS)
        candidate = Circuit(spec, observables, other_provider, "numpy")(_WEIGHTS)
        assert np.allclose(reference, candidate, atol=1e-8)


class TestAdjoint:
    def test_s_maps_to_sdg(self):
        adj = adjoint(CircuitBuilder(n_qubits=1).s(0).build())
        assert [g.name for g in adj.gates] == ["Sdg"]

    def test_circuit_times_adjoint_is_identity(self):
        spec = _circuit_with_extended_gates()
        adj = adjoint(spec)
        combined = CircuitSpec(
            n_qubits=spec.n_qubits,
            n_weights=spec.n_weights,
            n_inputs=spec.n_inputs,
            gates=spec.gates + adj.gates,
        )
        circuit = Circuit(combined, all_z(spec.n_qubits), "numpy", "numpy")
        assert np.isclose(circuit.zero_state_probability(_WEIGHTS), 1.0, atol=1e-10)
