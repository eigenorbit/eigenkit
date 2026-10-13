"""Tests for HamiltonianQPEBlock — honest Trotter-based QPE for HHL.

Correctness is checked against the trusted dense QPEBlock in the same backend
(so any endianness convention cancels): for a diagonal A the Trotterisation is
exact and the two blocks must produce identical unitaries; for a dense A the
Hamiltonian block converges to QPEBlock as the Trotter step count grows.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from eigenkit.algorithms._hamiltonian_sim import pauli_decompose
from eigenkit.core.blocks import (
    HamiltonianQPEBlock,
    QPEBlock,
    QubitUnitary,
    assemble,
)
from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import CircuitSpec


def _spec(gates, n: int) -> CircuitSpec:
    return CircuitSpec(n_qubits=n, n_weights=0, n_inputs=0, gates=tuple(gates))


def _unitary(spec: CircuitSpec) -> np.ndarray:
    """Full unitary of a spec via the numpy backend (qubit-0-MSB order)."""
    circ = Circuit(spec, provider="numpy")
    dim = 2**spec.n_qubits
    cols = []
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        cols.append(circ.statevector(np.zeros(0), initial_state=e))
    return np.column_stack(cols)


def _qpe_ref_unitary(A: np.ndarray, n_clock: int, t: float) -> np.ndarray:
    n_state = A.shape[0].bit_length() - 1
    U = scipy.linalg.expm(1j * A * t)
    oracle = assemble([QubitUnitary(U, wires=tuple(range(n_state)))])
    frag = QPEBlock(oracle, n_clock=n_clock).build(n_clock + n_state)
    return _unitary(_spec(frag.gates, n_clock + n_state))


def _hqpe_unitary(A, n_clock, t, steps, order=1, inverse=False) -> np.ndarray:
    n_state = A.shape[0].bit_length() - 1
    frag = HamiltonianQPEBlock(
        pauli_decompose(A),
        n_clock=n_clock,
        time=t,
        steps=steps,
        order=order,
        inverse=inverse,
    ).build(n_clock + n_state)
    return _unitary(_spec(frag.gates, n_clock + n_state))


def _diag(vals) -> np.ndarray:
    return np.diag(np.array(vals, dtype=complex))


def _random_hermitian(dim: int, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return scale * (m + m.conj().T)


# ---- equivalence to the trusted QPEBlock ----


@pytest.mark.parametrize(
    "vals,n_clock",
    [([0.5, 1.3], 3), ([0.4, 1.1, 0.7, 1.9], 2)],
)
def test_matches_qpeblock_diagonal_exact(vals, n_clock):
    # Diagonal A -> only I/Z terms -> they commute -> Trotter is exact at
    # steps=1, so the unitary must equal the dense QPEBlock's exactly.
    A = _diag(vals)
    t = 0.5
    assert np.allclose(
        _hqpe_unitary(A, n_clock, t, steps=1), _qpe_ref_unitary(A, n_clock, t), atol=1e-9
    )


def test_matches_qpeblock_dense_converges():
    A = _random_hermitian(2, seed=1, scale=0.3)
    n_clock, t = 2, 0.5
    ref = _qpe_ref_unitary(A, n_clock, t)
    errs = [
        np.linalg.norm(_hqpe_unitary(A, n_clock, t, steps=s) - ref, 2)
        for s in (1, 4, 16, 64)
    ]
    assert all(b < a for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-2, errs


def test_order2_more_accurate_than_order1():
    A = _random_hermitian(2, seed=8, scale=0.3)
    n_clock, t = 2, 0.5
    ref = _qpe_ref_unitary(A, n_clock, t)
    e1 = np.linalg.norm(_hqpe_unitary(A, n_clock, t, steps=8, order=1) - ref, 2)
    e2 = np.linalg.norm(_hqpe_unitary(A, n_clock, t, steps=8, order=2) - ref, 2)
    assert e2 < e1


# ---- semantic checks ----


def test_recovers_known_phase():
    # Eigenphase placed exactly on a bin: λ·t/(2π) = j / 2^n_clock.
    n_clock, t, j = 3, 1.0, 3
    lam0 = 2 * np.pi * j / (2**n_clock)
    lam1 = 2 * np.pi * 6 / (2**n_clock)
    A = _diag([lam0, lam1])
    n = n_clock + 1
    frag = HamiltonianQPEBlock(
        pauli_decompose(A), n_clock=n_clock, time=t, steps=1
    ).build(n)
    # Start from |clock=0, state=0>: |0> is the eigenvector with eigenvalue lam0.
    sv = Circuit(_spec(frag.gates, n), provider="numpy").statevector(np.zeros(0))
    probs = np.abs(sv) ** 2
    idx = int(np.argmax(probs))
    # qubit-0-MSB: clock is the top n_clock bits, state is the LSB.
    assert probs[idx] > 0.999
    assert idx & 1 == 0  # state qubit measured |0>
    assert idx >> 1 == j  # clock register reads the expected bin


def test_inverse_uncomputes_diagonal():
    A = _diag([0.5, 1.3])
    n_clock, t, n = 3, 0.5, 4
    terms = pauli_decompose(A)
    fwd = HamiltonianQPEBlock(terms, n_clock=n_clock, time=t, steps=1).build(n).gates
    inv = (
        HamiltonianQPEBlock(terms, n_clock=n_clock, time=t, steps=1, inverse=True)
        .build(n)
        .gates
    )
    u = _unitary(_spec(tuple(fwd) + tuple(inv), n))
    assert np.allclose(u, np.eye(2**n), atol=1e-9)


# ---- honest-path invariants ----


def test_uses_only_elementary_gates_and_no_matrix_power_oracle():
    A = _random_hermitian(2, seed=4, scale=0.3)
    gates = (
        HamiltonianQPEBlock(pauli_decompose(A), n_clock=2, time=0.5, steps=2)
        .build(3)
        .gates
    )
    # A Y factor is rotated into the Z basis by RX(±π/2), matching
    # CircuitBuilder.pauli_exponential, so no discrete S/Sdg requirement lands on the
    # provider just because a Hamiltonian happens to contain Y terms.
    allowed = {"H", "RX", "RZ", "CNOT", "PhaseShift", "CPHASE", "SWAP"}
    assert {g.name for g in gates} <= allowed
    # No dense unitary black-box (which is what matrix_power would feed).
    assert all(g.name not in ("QubitUnitary", "ControlledQubitUnitary") for g in gates)


def test_identity_term_becomes_clock_phase():
    # A = c·I: no traceless part -> controlled evolution is a PhaseShift only.
    A = 0.7 * np.eye(2)
    gates = (
        HamiltonianQPEBlock(pauli_decompose(A), n_clock=2, time=0.5, steps=1)
        .build(3)
        .gates
    )
    names = {g.name for g in gates}
    assert "PhaseShift" in names
    assert "RZ" not in names  # no traceless Pauli term -> no rotation ladder


# ---- serialization / registry / validation ----


def test_serialization_roundtrip():
    terms = pauli_decompose(_random_hermitian(2, seed=9, scale=0.3))
    block = HamiltonianQPEBlock(terms, n_clock=2, time=0.8, steps=3, order=2)
    restored = HamiltonianQPEBlock.from_dict(block.to_dict())
    assert restored.n_clock == block.n_clock
    assert restored.time == block.time
    assert restored.steps == block.steps
    assert restored.order == block.order
    assert restored.inverse == block.inverse
    assert [g.name for g in restored.build(3).gates] == [
        g.name for g in block.build(3).gates
    ]


def test_registered_in_block_types():
    from eigenkit.qml._serialization import _BLOCK_TYPES

    assert _BLOCK_TYPES["HamiltonianQPEBlock"] is HamiltonianQPEBlock


def test_required_qubits():
    terms = pauli_decompose(_diag([0.5, 1.3, 0.9, 2.0]))  # 2-qubit state
    assert HamiltonianQPEBlock(terms, n_clock=3, time=1.0).required_qubits() == 5


def test_n_clock_zero_raises():
    with pytest.raises(ValueError, match="n_clock"):
        HamiltonianQPEBlock([(1.0, "Z")], n_clock=0, time=1.0)


def test_empty_terms_raises():
    with pytest.raises(ValueError, match="non-empty"):
        HamiltonianQPEBlock([], n_clock=2, time=1.0)


def test_bad_steps_and_order_raise():
    with pytest.raises(ValueError, match="steps"):
        HamiltonianQPEBlock([(1.0, "Z")], n_clock=2, time=1.0, steps=0)
    with pytest.raises(ValueError, match="order"):
        HamiltonianQPEBlock([(1.0, "Z")], n_clock=2, time=1.0, order=3)


def test_too_few_qubits_raises():
    with pytest.raises(ValueError, match="qubits"):
        HamiltonianQPEBlock([(1.0, "Z")], n_clock=2, time=1.0).build(2)
