"""Tests for TrotterBlock — gate-level e^{iAt} for the honest HHL path."""

from __future__ import annotations

from functools import reduce

import numpy as np
import pytest
import scipy.linalg

from eigenkit.algorithms._hamiltonian_sim import (
    pauli_decompose,
    pauli_string_matrix,
)
from eigenkit.core.blocks import TrotterBlock


# ---- independent big-endian unitary evaluator (qubit 0 = most significant) ----

_I2 = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _embed1(U: np.ndarray, q: int, n: int) -> np.ndarray:
    return reduce(np.kron, [U if i == q else _I2 for i in range(n)])


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def _rx(theta: float) -> np.ndarray:
    c, si = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * si], [-1j * si, c]], dtype=complex)


def _cnot(control: int, target: int, n: int) -> np.ndarray:
    dim = 2**n
    m = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        y = x
        if (x >> (n - 1 - control)) & 1:
            y = x ^ (1 << (n - 1 - target))
        m[y, x] = 1
    return m


def _gates_to_unitary(gates, n: int) -> np.ndarray:
    u = np.eye(2**n, dtype=complex)
    for g in gates:
        if g.name == "H":
            m = _embed1(_H, g.qubits[0], n)
        elif g.name in ("RZ", "RX"):
            p = g.params[0]
            angle = float(getattr(p, "value", p))
            rot = _rz(angle) if g.name == "RZ" else _rx(angle)
            m = _embed1(rot, g.qubits[0], n)
        elif g.name == "CNOT":
            m = _cnot(g.qubits[0], g.qubits[1], n)
        else:
            raise AssertionError(f"unexpected gate in TrotterBlock output: {g.name}")
        u = m @ u
    return u


def _trotter_unitary(terms, time, n, steps=1, order=1) -> np.ndarray:
    gates = TrotterBlock(terms, time=time, steps=steps, order=order).build(n).gates
    return _gates_to_unitary(gates, n)


def _exact_traceless(A: np.ndarray, time: float) -> np.ndarray:
    """expm(i·(A − c_I·I)·t): the identity term is a skipped global phase."""
    dim = A.shape[0]
    c_identity = np.trace(A) / dim
    return scipy.linalg.expm(1j * (A - c_identity * np.eye(dim)) * time)


def _random_hermitian(dim: int, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return scale * (m + m.conj().T)


# ---- single-term exactness (validates signs and basis changes) ----


@pytest.mark.parametrize("pauli", ["X", "Y", "Z", "ZZ", "XY", "YZ", "XYZ"])
def test_single_pauli_term_is_exact(pauli):
    # One Pauli term has no Trotter error: the gate sequence must equal e^{iPt}.
    A = pauli_string_matrix(pauli)
    t = 0.37
    n = len(pauli)
    u = _trotter_unitary([(1.0, pauli)], time=t, n=n, steps=1)
    expected = scipy.linalg.expm(1j * A * t)
    assert np.allclose(u, expected, atol=1e-12)


def test_negative_and_fractional_coeff_exact():
    A = -0.42 * pauli_string_matrix("Z")
    t = 1.1
    u = _trotter_unitary([(-0.42, "Z")], time=t, n=1, steps=1)
    assert np.allclose(u, scipy.linalg.expm(1j * A * t), atol=1e-12)


# ---- convergence with steps ----


@pytest.mark.parametrize("dim,seed", [(2, 0), (4, 3)])
def test_converges_to_expm_with_steps(dim, seed):
    A = _random_hermitian(dim, seed, scale=0.25)
    t = 0.6
    terms = pauli_decompose(A)
    ref = _exact_traceless(A, t)

    n = dim.bit_length() - 1
    errors = []
    for steps in (2, 8, 32, 128):
        u = _trotter_unitary(terms, time=t, n=n, steps=steps, order=1)
        errors.append(np.linalg.norm(u - ref, 2))

    # Monotone improvement and small final error.
    assert all(b < a for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 1e-2, errors


@pytest.mark.parametrize("dim,seed", [(2, 1), (4, 5)])
def test_order2_more_accurate_than_order1(dim, seed):
    A = _random_hermitian(dim, seed, scale=0.25)
    t = 0.6
    terms = pauli_decompose(A)
    ref = _exact_traceless(A, t)
    n = dim.bit_length() - 1

    err1 = np.linalg.norm(_trotter_unitary(terms, t, n, steps=16, order=1) - ref, 2)
    err2 = np.linalg.norm(_trotter_unitary(terms, t, n, steps=16, order=2) - ref, 2)
    assert err2 < err1


def test_order2_converges_faster_scaling():
    # Order-2 error should fall roughly as steps^-2 vs steps^-1 for order-1.
    A = _random_hermitian(4, seed=7, scale=0.3)
    t = 0.7
    terms = pauli_decompose(A)
    ref = _exact_traceless(A, t)
    e_lo = np.linalg.norm(_trotter_unitary(terms, t, 2, steps=8, order=2) - ref, 2)
    e_hi = np.linalg.norm(_trotter_unitary(terms, t, 2, steps=32, order=2) - ref, 2)
    # 4x the steps -> ~16x better for a second-order method (allow slack).
    assert e_hi < e_lo / 8


# ---- honest-path invariants ----


def test_uses_only_elementary_gates():
    A = _random_hermitian(4, seed=2, scale=0.3)
    gates = TrotterBlock(pauli_decompose(A), time=0.5, steps=3).build(2).gates
    # RX(±π/2) maps Y into the Z basis, matching CircuitBuilder.pauli_exponential, so
    # a Y term costs the provider no discrete S/Sdg gate.
    allowed = {"H", "RX", "CNOT", "RZ"}
    assert {g.name for g in gates} <= allowed
    # Crucially: no dense unitary black-box.
    assert all(g.name not in ("QubitUnitary", "ControlledQubitUnitary") for g in gates)


def test_identity_term_skipped():
    # Pure identity -> global phase -> no gates.
    assert TrotterBlock([(0.5, "II")], time=1.0).build(2).gates == ()
    # Identity mixed with Z: only the Z term produces gates, matching e^{i c_Z Z t}.
    u = _trotter_unitary([(0.9, "II"), (0.5, "ZI")], time=0.4, n=2)
    expected = scipy.linalg.expm(1j * 0.5 * pauli_string_matrix("ZI") * 0.4)
    assert np.allclose(u, expected, atol=1e-12)


# ---- serialization ----


def test_serialization_roundtrip():
    A = _random_hermitian(4, seed=9, scale=0.3)
    terms = pauli_decompose(A)
    block = TrotterBlock(terms, time=0.8, steps=5, order=2)
    restored = TrotterBlock.from_dict(block.to_dict())
    assert restored.time == block.time
    assert restored.steps == block.steps
    assert restored.order == block.order
    g1 = block.build(2).gates
    g2 = restored.build(2).gates
    assert [g.name for g in g1] == [g.name for g in g2]
    assert np.allclose(_gates_to_unitary(g1, 2), _gates_to_unitary(g2, 2))


def test_registered_in_block_types():
    from eigenkit.qml._serialization import _BLOCK_TYPES

    assert _BLOCK_TYPES["TrotterBlock"] is TrotterBlock


# ---- validation ----


def test_required_qubits():
    assert TrotterBlock([(1.0, "ZIZ")], time=1.0).required_qubits() == 3


def test_empty_terms_raises():
    with pytest.raises(ValueError, match="non-empty"):
        TrotterBlock([], time=1.0)


def test_mismatched_string_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        TrotterBlock([(1.0, "Z"), (1.0, "ZZ")], time=1.0)


def test_invalid_symbol_raises():
    with pytest.raises(ValueError, match="invalid Pauli string"):
        TrotterBlock([(1.0, "ZQ")], time=1.0)


def test_non_real_coeff_raises():
    with pytest.raises(ValueError, match="not\n?.*real|Hermitian"):
        TrotterBlock([(1.0 + 1.0j, "Z")], time=1.0)


def test_bad_steps_and_order_raise():
    with pytest.raises(ValueError, match="steps"):
        TrotterBlock([(1.0, "Z")], time=1.0, steps=0)
    with pytest.raises(ValueError, match="order"):
        TrotterBlock([(1.0, "Z")], time=1.0, order=3)


def test_too_few_qubits_raises():
    with pytest.raises(ValueError, match="qubits"):
        TrotterBlock([(1.0, "ZZ")], time=1.0).build(1)


def test_realness_tolerance_is_relative_to_the_operator_scale():
    """The realness check scales with the operator, not with an absolute 1e-8.

    A Hermitian A of large magnitude carries its floating-point residue in proportion
    to its own size, so an absolute threshold rejects big matrices for being big.

    The boundary is `|Im c| > 1e-8 · max |c|` over the whole term list.  A per-coefficient
    `1e-8 · max(1, |Re c|)` would anchor the tolerance at 1e-8 for every term below unit
    scale — so an operator whose coefficients are all that small could be arbitrarily
    non-real and have its imaginary parts dropped in silence.
    """
    # Imaginary part 1e-3 in absolute terms, but 1e-13 of the operator: accepted.
    TrotterBlock([(1e10 + 1e-3j, "Z")], time=1e-12, steps=1)
    # Imaginary part 1e3 — 1e-7 relative, past the boundary: rejected.
    with pytest.raises(ValueError, match="not\n?.*real|Hermitian"):
        TrotterBlock([(1e10 + 1e3j, "Z")], time=1e-12, steps=1)
    # At unit scale the relative boundary coincides with an absolute 1e-8.
    TrotterBlock([(1.0 + 1e-9j, "Z")], time=1.0, steps=1)
    with pytest.raises(ValueError, match="not\n?.*real|Hermitian"):
        TrotterBlock([(1.0 + 1e-6j, "Z")], time=1.0, steps=1)


def test_realness_tolerance_follows_the_largest_term_not_each_one():
    """A small term's residue comes from the operator it sits in, not from itself.

    Both halves matter.  A 1e-9 term inside a 1e10 Hamiltonian carries the round-off of
    the 1e10 one, so holding it to 1e-8 of *itself* — 1e-17 — would reject an honest
    decomposition; and an operator that is small throughout has a correspondingly small
    tolerance, which a `max(1, …)` anchor would remove.
    """
    # 1e-12 imaginary on a 1e-9 term: noise beside a 1e10 term, tolerance ≈ 100.
    TrotterBlock([(1e10, "Z"), (1e-9 + 1e-12j, "X")], time=1e-12, steps=1)

    # The same shape with nothing large in it: tolerance ≈ 1e-17, so this is structure.
    # `pauli_decompose(1e-9 · [[0, 1], [0, 0]])` produces exactly these two terms.
    with pytest.raises(ValueError, match="not\n?.*real|Hermitian"):
        TrotterBlock([(5e-10, "X"), (5e-10j, "Y")], time=1.0, steps=1)


def test_large_scale_hermitian_matrix_still_builds():
    """End-to-end companion: the decomposition of a big Hermitian A feeds TrotterBlock
    without tripping the realness check."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    for scale in (1.0, 1e10, 1e13):
        A = scale * (M + M.conj().T)
        TrotterBlock(tuple(pauli_decompose(A)), time=1e-3 / scale, steps=1, order=2)
