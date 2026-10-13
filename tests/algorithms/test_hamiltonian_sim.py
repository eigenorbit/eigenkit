"""Tests for the Pauli decomposition used by the honest HHL path."""

import itertools

import numpy as np
import pytest

from eigenkit.algorithms._hamiltonian_sim import (
    pauli_decompose,
    pauli_string_matrix,
)


def _reconstruct(terms: list[tuple[complex, str]]) -> np.ndarray:
    """Rebuild the dense matrix from a Pauli decomposition."""
    return sum(coeff * pauli_string_matrix(s) for coeff, s in terms)


def _random_hermitian(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return m + m.conj().T


def test_pauli_string_matrix_kron_order():
    # ZI = Z ⊗ I: acts as Z on qubit 0 (most significant), I on qubit 1.
    zi = pauli_string_matrix("ZI")
    expected = np.kron(
        np.array([[1, 0], [0, -1]], dtype=complex),
        np.eye(2, dtype=complex),
    )
    assert np.allclose(zi, expected)


def test_pauli_string_matrix_rejects_bad_symbol():
    with pytest.raises(ValueError, match="invalid Pauli symbol"):
        pauli_string_matrix("Q")
    with pytest.raises(ValueError, match="non-empty"):
        pauli_string_matrix("")


def test_known_diagonal_2x2():
    # diag(2, 1) = 1.5·I + 0.5·Z
    A = np.array([[2, 0], [0, 1]], dtype=complex)
    terms = dict((s, c) for c, s in pauli_decompose(A))
    assert set(terms) == {"I", "Z"}
    assert terms["I"].real == pytest.approx(1.5)
    assert terms["Z"].real == pytest.approx(0.5)
    assert abs(terms["I"].imag) < 1e-12
    assert abs(terms["Z"].imag) < 1e-12


def test_known_pauli_x():
    # [[0,1],[1,0]] is exactly X.
    A = np.array([[0, 1], [1, 0]], dtype=complex)
    terms = pauli_decompose(A)
    assert len(terms) == 1
    coeff, string = terms[0]
    assert string == "X"
    assert coeff.real == pytest.approx(1.0)


@pytest.mark.parametrize("dim,seed", [(2, 0), (2, 7), (4, 1), (4, 42), (8, 3)])
def test_reconstruction_random_hermitian(dim, seed):
    A = _random_hermitian(dim, seed)
    terms = pauli_decompose(A)
    assert np.allclose(_reconstruct(terms), A, atol=1e-10)


@pytest.mark.parametrize("dim,seed", [(2, 5), (4, 9)])
def test_coefficients_real_for_hermitian(dim, seed):
    A = _random_hermitian(dim, seed)
    for coeff, _ in pauli_decompose(A):
        assert abs(coeff.imag) < 1e-10


def test_term_count_is_full_basis_for_dense():
    # A generic dense Hermitian 2-qubit matrix uses (almost surely) all 4^2
    # Pauli strings — the honest cost metric for dense A.
    A = _random_hermitian(4, seed=11)
    assert len(pauli_decompose(A)) == 16


def test_term_count_small_for_structured():
    # A sparse/structured matrix decomposes into few terms.
    A = np.diag([3.0, 3.0, 1.0, 1.0]).astype(complex)
    terms = pauli_decompose(A)
    assert len(terms) < 16
    assert np.allclose(_reconstruct(terms), A, atol=1e-12)


def test_rejects_non_hermitian():
    A = np.array([[0, 1], [0, 0]], dtype=complex)
    with pytest.raises(ValueError, match="not Hermitian"):
        pauli_decompose(A)


def test_rejects_non_power_of_two():
    A = np.eye(3, dtype=complex)
    with pytest.raises(ValueError, match="power of two"):
        pauli_decompose(A)


def test_rejects_non_square():
    A = np.zeros((2, 4), dtype=complex)
    with pytest.raises(ValueError, match="square"):
        pauli_decompose(A)


def test_rtol_drops_tiny_terms():
    # I + tiny·Z: with a loose rtol the Z term is pruned.
    A = np.array([[1 + 1e-6, 0], [0, 1 - 1e-6]], dtype=complex)
    terms = pauli_decompose(A, rtol=1e-4)
    assert [s for _, s in terms] == ["I"]


def test_pruning_is_scale_invariant():
    # The threshold is relative, so rescaling A must not change which terms survive:
    # the same physics in different units has to give the same circuit.
    A = np.array([[1.0, 0.9], [0.9, 1.0]], dtype=complex)  # = 1.0*I + 0.9*X
    reference = [s for _, s in pauli_decompose(A)]
    assert reference == ["I", "X"]
    for scale in (1e12, 1e3, 1e-9, 1e-15):
        assert [s for _, s in pauli_decompose(scale * A)] == reference


def _reference_decompose(A, rtol=1e-10):
    """The O(N^4) definition: build each dense Pauli string and contract.

    The oracle for the Walsh-Hadamard implementation, which is a closed-form identity
    for the same quantity and so has to agree exactly, not approximately.
    """
    A = np.asarray(A, dtype=complex)
    dim = A.shape[0]
    threshold = rtol * float(np.max(np.abs(A)))
    out = []
    for symbols in itertools.product("IXYZ", repeat=dim.bit_length() - 1):
        s = "".join(symbols)
        coeff = complex(np.einsum("ij,ji->", pauli_string_matrix(s), A)) / dim
        if abs(coeff) > threshold:
            out.append((coeff, s))
    return out


@pytest.mark.parametrize("dim", [2, 4, 8, 16])
@pytest.mark.parametrize(
    "kind", ["dense-hermitian", "real-symmetric", "diagonal", "tridiagonal", "banded"]
)
def test_matches_the_dense_definition(dim, kind):
    """The fast path must reproduce the dense definition term for term, in order.

    `Tr(P·A)` collapses to one sum over rows because a Pauli string has exactly one
    non-zero per row, at column `r ^ x` with value `(-i)^n_Y · (-1)^popcount(r & z)`;
    taking that sum over every `z` at once is a Walsh-Hadamard transform. Nothing about
    it is approximate, so agreement is asserted at machine precision.
    """
    rng = np.random.default_rng(hash((dim, kind)) % 2**32)
    if kind == "dense-hermitian":
        m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        A = m + m.conj().T
    elif kind == "real-symmetric":
        m = rng.standard_normal((dim, dim))
        A = (m + m.T).astype(complex)
    elif kind == "diagonal":
        A = np.diag(rng.standard_normal(dim)).astype(complex)
    elif kind == "tridiagonal":
        A = (2 * np.eye(dim) - np.eye(dim, k=1) - np.eye(dim, k=-1)).astype(complex)
    else:
        A = np.diag(rng.standard_normal(dim) + dim).astype(complex)
        for i in range(dim - 1):
            A[i, i + 1] = A[i + 1, i] = rng.standard_normal()

    fast, slow = pauli_decompose(A), _reference_decompose(A)
    assert [s for _, s in fast] == [s for _, s in slow]
    peak = max((abs(c) for c, _ in slow), default=1.0)
    for (c_fast, s), (c_slow, _) in zip(fast, slow):
        assert abs(c_fast - c_slow) < 1e-13 * peak, s


def test_scales_to_a_large_system():
    """The decomposition is the honest path's binding classical cost, and at
    O(N^2 log N) a 512x512 system is routine: the dense Theta(N^4) definition needs
    minutes at N=128 alone."""
    dim = 512
    A = (2 * np.eye(dim) - np.eye(dim, k=1) - np.eye(dim, k=-1)).astype(complex)
    terms = pauli_decompose(A)
    assert len(terms) == dim  # a Toeplitz tridiagonal is Theta(N), not Theta(1)
    assert np.allclose(_reconstruct(terms), A, atol=1e-10)
