"""Numpy reference-simulator support for the extended gate set
(Z, PhaseShift, SWAP, MCX, MCZ, CRY), so the default backend can run the
IQAE / HHL circuits built from those gates.

Forward values are checked against analytic expectations and, where PennyLane is
installed, against the PennyLane backend (the cross-backend oracle). Gradients of
the parametric ones (PhaseShift, CRY) are checked against their analytic
derivatives and against parameter-shift, for both the single-sample and the
batched adjoint sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.circuit import Circuit
from eigenkit.core.observables import Z, all_z


pennylane = pytest.importorskip("pennylane", reason="PennyLane oracle not installed")


def _numpy(spec, obs):
    return Circuit(spec, obs, "numpy", "numpy")


def _pennylane(spec, obs):
    return Circuit(spec, obs, "pennylane", "numpy")


# ---- forward: analytic expectations ----


def test_z_gate_equals_x_under_hadamards():
    qc = CircuitBuilder(n_qubits=2)
    qc.h(0).z(0).h(0)  # H Z H == X
    assert _numpy(qc.build(), Z(0).to_observable())(np.empty(0)) == pytest.approx(-1.0)


def test_swap_exchanges_qubits():
    qc = CircuitBuilder(n_qubits=2)
    qc.x(0).swap(0, 1)  # |10> -> |01>
    out = _numpy(qc.build(), all_z(2))(np.empty(0))
    assert out == pytest.approx([1.0, -1.0])


def test_mcx_flips_target_only_when_all_controls_set():
    fired = CircuitBuilder(n_qubits=3)
    fired.x(0).x(1).mcx(0, 1, 2)  # Toffoli: both controls set -> flip target
    assert _numpy(fired.build(), Z(2).to_observable())(np.empty(0)) == pytest.approx(-1.0)

    idle = CircuitBuilder(n_qubits=3)
    idle.x(0).mcx(0, 1, 2)  # one control unset -> no flip
    assert _numpy(idle.build(), Z(2).to_observable())(np.empty(0)) == pytest.approx(1.0)


def test_mcz_phase_flip():
    qc = CircuitBuilder(n_qubits=3)
    qc.x(0).h(1).mcz(0, 1).h(1)  # CZ turns |+> into |-> on qubit 1 -> Z1 = -1
    assert _numpy(qc.build(), Z(1).to_observable())(np.empty(0)) == pytest.approx(-1.0)


def test_phaseshift_forward_matches_cosine():
    qc = CircuitBuilder(n_qubits=1)
    w = qc.add_weights(1)
    qc.h(0).phase_shift(w[0], 0).h(0)  # <Z> = cos(phi)
    c = _numpy(qc.build(), Z(0).to_observable())
    phi = 0.7
    assert c(np.array([phi])) == pytest.approx(np.cos(phi))


def test_cry_forward_and_control_off():
    qc = CircuitBuilder(n_qubits=2)
    w = qc.add_weights(1)
    qc.x(0).cry(w[0], 0, 1)  # control set -> RY(theta) on target -> <Z1> = cos(theta)
    theta = 1.1
    assert _numpy(qc.build(), Z(1).to_observable())(np.array([theta])) == pytest.approx(
        np.cos(theta)
    )

    off = CircuitBuilder(n_qubits=2)
    w2 = off.add_weights(1)
    off.cry(w2[0], 0, 1)  # control |0> -> identity on target
    assert _numpy(off.build(), Z(1).to_observable())(np.array([1.1])) == pytest.approx(
        1.0
    )


# ---- gradients: analytic derivatives ----


def test_phaseshift_gradient_matches_minus_sine():
    qc = CircuitBuilder(n_qubits=1)
    w = qc.add_weights(1)
    qc.h(0).phase_shift(w[0], 0).h(0)
    c = _numpy(qc.build(), Z(0).to_observable())
    phi = 0.9
    assert c.jacobian(np.array([phi])) == pytest.approx(-np.sin(phi))


def test_cry_gradient_matches_minus_sine():
    qc = CircuitBuilder(n_qubits=2)
    w = qc.add_weights(1)
    qc.x(0).cry(w[0], 0, 1)
    c = _numpy(qc.build(), Z(1).to_observable())
    theta = 1.3
    assert c.jacobian(np.array([theta])) == pytest.approx(-np.sin(theta))


# ---- parity vs PennyLane on a mixed circuit (forward + grad, single + batch) ----


def _mixed_spec():
    qc = CircuitBuilder(n_qubits=3)
    w = qc.add_weights(3)
    ins = qc.add_inputs(2)
    qc.rx(ins[0], 0).ry(w[0], 1).cry(w[1], 0, 1)
    qc.phase_shift(w[2], 2).mcx(0, 1, 2).swap(1, 2).z(0).ry(ins[1], 2)
    return qc.build()


def test_mixed_circuit_parity_with_pennylane():
    spec = _mixed_spec()
    npc = _numpy(spec, all_z(3))
    plc = _pennylane(spec, all_z(3))
    rng = np.random.default_rng(3)
    w = rng.standard_normal(3)
    x = rng.standard_normal(2)
    assert np.allclose(npc(w, x), plc(w, x), atol=1e-9)
    assert np.allclose(npc.jacobian(w, x), plc.jacobian(w, x), atol=1e-7)


def test_mixed_circuit_batched_parity_with_pennylane():
    spec = _mixed_spec()
    npc = _numpy(spec, all_z(3))
    plc = _pennylane(spec, all_z(3))
    w = np.array([0.4, -0.6, 0.2])
    x_batch = np.random.default_rng(4).standard_normal((5, 2))
    assert np.allclose(npc(w, x_batch), plc(w, x_batch), atol=1e-9)


def test_batched_gradient_matches_single_sample_loop():
    spec = _mixed_spec()
    c = _numpy(spec, all_z(3))
    w = np.array([0.4, -0.6, 0.2])
    x_batch = np.random.default_rng(5).standard_normal((4, 2))
    batched, _ = c._gradient_jacobians(w, x_batch)
    looped = np.stack([c._gradient_jacobians(w, row)[0] for row in x_batch])
    assert np.allclose(batched, looped, atol=1e-10)
