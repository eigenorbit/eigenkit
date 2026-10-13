"""The `CircuitBuilder.pauli_exponential` decomposition must implement
exp(-iθ·P) exactly (up to a global phase that can't affect expectations)."""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.circuit import Circuit
from eigenkit.core.observables import X, Y, Z


def _expectations(spec, observables, theta):
    circuit = Circuit(spec, observables, "numpy", "numpy")
    return circuit(np.array([theta], dtype=np.float64))


@pytest.mark.parametrize("theta", [0.3, 1.1, -0.7, 2.5])
def test_single_x_equals_rx(theta):
    # exp(-iθ X) == RX(2θ).
    qc = CircuitBuilder(n_qubits=1)
    w = qc.add_weights(1)
    qc.pauli_exponential((("X", 0),), w[0])
    exp_spec = qc.build()

    ref = CircuitBuilder(n_qubits=1)
    wref = ref.add_weights(1)
    from eigenkit.core.ir import Scale

    ref.rx(Scale(wref[0], 2.0), 0)
    ref_spec = ref.build()

    observables = [X(0).to_observable(), Y(0).to_observable(), Z(0).to_observable()]
    assert np.allclose(
        _expectations(exp_spec, observables, theta),
        _expectations(ref_spec, observables, theta),
        atol=1e-10,
    )


@pytest.mark.parametrize("theta", [0.3, 1.1, -0.7, 2.5])
def test_single_y_equals_ry(theta):
    # exp(-iθ Y) == RY(2θ).
    qc = CircuitBuilder(n_qubits=1)
    w = qc.add_weights(1)
    qc.pauli_exponential((("Y", 0),), w[0])
    exp_spec = qc.build()

    ref = CircuitBuilder(n_qubits=1)
    wref = ref.add_weights(1)
    from eigenkit.core.ir import Scale

    ref.ry(Scale(wref[0], 2.0), 0)
    ref_spec = ref.build()

    observables = [X(0).to_observable(), Y(0).to_observable(), Z(0).to_observable()]
    assert np.allclose(
        _expectations(exp_spec, observables, theta),
        _expectations(ref_spec, observables, theta),
        atol=1e-10,
    )


def test_zz_commutes_with_z_measurements():
    # exp(-iθ Z0 Z1) commutes with Z_i, so <Z0>, <Z1>, <Z0 Z1> are unchanged
    # from the initial |+0> state where they are 0, 1, 0.
    qc = CircuitBuilder(n_qubits=2)
    w = qc.add_weights(1)
    qc.h(0)  # |+> on q0 so a Z0Z1 rotation can act non-trivially on phases
    qc.pauli_exponential((("Z", 0), ("Z", 1)), w[0])
    spec = qc.build()
    circuit = Circuit(
        spec,
        [Z(0).to_observable(), Z(1).to_observable(), (Z(0) * Z(1)).to_observable()],
        "numpy",
        "numpy",
    )
    out = circuit(np.array([0.9], dtype=np.float64))
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-10)


def test_empty_pauli_is_noop():
    qc = CircuitBuilder(n_qubits=1)
    w = qc.add_weights(1)
    qc.pauli_exponential((), w[0])
    assert len(qc) == 0
