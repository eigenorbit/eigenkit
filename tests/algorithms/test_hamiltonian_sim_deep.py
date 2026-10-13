"""Component tests for the Pauli machinery the honest HHL path is built on.

``_hamiltonian_sim`` is where honest mode's cost and its numerical contract both come
from: the term list decides the circuit, its *order* decides the Trotter error, and the
anticommuting weight profile decides how many steps the selector asks for.  Every
routine here is checked against its own definition computed the slow, obvious way —
brute-force traces, dense Pauli products, the O(N²) Walsh–Hadamard sum — rather than
against another fast path, so a shared mistake cannot hide.

The claims about *term counts* are pinned too (§ term-count reality).  They refute the
tempting "sparse ⇒ O(poly n) terms" claim, which is false:
the count follows sparsity in the Pauli basis, not the number of non-zeros.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from eigenkit.algorithms import _hamiltonian_sim as hs
from eigenkit.algorithms._hamiltonian_sim import (
    anticommuting_weight_profile,
    pauli_decompose,
    pauli_string_matrix,
    trotter_drift_bound,
)


P = pauli_string_matrix


def _random_hermitian(dim: int, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return scale * (m + m.conj().T)


def _all_strings(n: int) -> list[str]:
    return ["".join(s) for s in itertools.product("IXYZ", repeat=n)]


def _brute_coefficient(A: np.ndarray, s: str) -> complex:
    """``Tr(P_s A) / 2^n`` the obvious way."""
    return complex(np.trace(P(s) @ A) / A.shape[0])


# ---------------------------------------------------------------------------
# pauli_string_matrix — the endianness the whole module is indexed by
# ---------------------------------------------------------------------------


def test_pauli_string_matrix_is_the_kronecker_product_in_qubit_order():
    # Qubit 0 is the most significant Kronecker factor. If this convention ever
    # flips, every mask in the module silently addresses the wrong qubit.
    np.testing.assert_allclose(P("XZ"), np.kron(P("X"), P("Z")))
    np.testing.assert_allclose(P("IY"), np.kron(np.eye(2), P("Y")))
    np.testing.assert_allclose(P("ZIX"), np.kron(np.kron(P("Z"), np.eye(2)), P("X")))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_every_pauli_string_is_hermitian_unitary_and_traceless(n):
    for s in _all_strings(n):
        M = P(s)
        np.testing.assert_allclose(M, M.conj().T, atol=1e-15)
        np.testing.assert_allclose(M @ M, np.eye(2**n), atol=1e-15)
        expected_trace = 2**n if set(s) == {"I"} else 0.0
        assert abs(np.trace(M) - expected_trace) < 1e-12


@pytest.mark.parametrize("n", [1, 2, 3])
def test_pauli_strings_are_orthogonal_under_hilbert_schmidt(n):
    """``Tr(P_k P_l) = 2^n δ_kl`` — the identity ``c_k = Tr(P_k A)/2^n`` rests on."""
    strings = _all_strings(n)
    mats = [P(s) for s in strings]
    for i, Mi in enumerate(mats):
        for j, Mj in enumerate(mats):
            expected = 2**n if i == j else 0.0
            assert abs(np.trace(Mi @ Mj) - expected) < 1e-12


def test_pauli_string_matrix_rejects_bad_input():
    with pytest.raises(ValueError, match="non-empty"):
        P("")
    with pytest.raises(ValueError, match="invalid Pauli symbol"):
        P("XW")
    with pytest.raises(ValueError, match="invalid Pauli symbol"):
        P("x")  # lower case is not silently accepted


# ---------------------------------------------------------------------------
# the fast transforms, against their own definitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 2, 4, 8, 16, 32])
def test_walsh_hadamard_matches_the_quadratic_definition(length):
    rng = np.random.default_rng(length)
    values = rng.standard_normal(length) + 1j * rng.standard_normal(length)
    brute = np.array(
        [
            sum((-1) ** int(bin(r & z).count("1")) * values[r] for r in range(length))
            for z in range(length)
        ]
    )
    np.testing.assert_allclose(hs._walsh_hadamard(values), brute, atol=1e-12)


def test_walsh_hadamard_is_applied_along_the_last_axis_only():
    # _pauli_coefficients feeds it a 2-D table (one row per x-shift); a transform
    # that mixed rows would scramble every coefficient.
    rng = np.random.default_rng(3)
    table = rng.standard_normal((5, 8))
    out = hs._walsh_hadamard(table)
    for i in range(5):
        np.testing.assert_allclose(out[i], hs._walsh_hadamard(table[i]), atol=1e-12)


def test_walsh_hadamard_does_not_mutate_its_input():
    values = np.arange(8.0)
    original = values.copy()
    hs._walsh_hadamard(values)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_string_masks_agree_with_the_dense_matrix(n):
    """``(x, z)`` must describe the dense matrix: one entry per row, at column ``r ⊕ x``.

    This is the identity ``_pauli_coefficients`` collapses the Hilbert–Schmidt trace
    with, so the whole ``O(N² log N)`` transform is wrong if the masks disagree with
    the dense matrix by a single bit or phase.
    """
    for s in _all_strings(n):
        x_mask, z_mask = hs._string_masks(s)
        M = P(s)
        n_y = bin(x_mask & z_mask).count("1")
        for r in range(2**n):
            row = M[r, :]
            (nz,) = np.nonzero(np.abs(row) > 1e-12)
            assert nz.tolist() == [r ^ x_mask], f"{s}: row {r}"
            expected = (-1j) ** n_y * (-1) ** bin(r & z_mask).count("1")
            assert abs(row[r ^ x_mask] - expected) < 1e-12, f"{s}: row {r}"


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_pauli_coefficients_table_matches_brute_force_traces(dim):
    A = _random_hermitian(dim, seed=dim)
    table = hs._pauli_coefficients(A)
    n = dim.bit_length() - 1
    for s in _all_strings(n):
        x_mask, z_mask = hs._string_masks(s)
        assert abs(table[x_mask, z_mask] - _brute_coefficient(A, s)) < 1e-12, s


def test_pauli_coefficients_handles_a_non_hermitian_matrix_consistently():
    # pauli_decompose rejects non-Hermitian input, but the table itself is a pure
    # linear transform and must stay exact — otherwise the Hermiticity guard is the
    # only thing standing between a user and silently wrong complex coefficients.
    rng = np.random.default_rng(11)
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    table = hs._pauli_coefficients(A)
    for s in _all_strings(2):
        x_mask, z_mask = hs._string_masks(s)
        assert abs(table[x_mask, z_mask] - _brute_coefficient(A, s)) < 1e-12, s


# ---------------------------------------------------------------------------
# pauli_decompose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim,seed", [(2, 0), (4, 1), (8, 2), (16, 3)])
def test_decomposition_reconstructs_the_matrix_exactly(dim, seed):
    A = _random_hermitian(dim, seed)
    terms = pauli_decompose(A)
    rebuilt = sum(c * P(s) for c, s in terms)
    np.testing.assert_allclose(rebuilt, A, atol=1e-11)


@pytest.mark.parametrize("dim,seed", [(2, 5), (4, 6), (8, 7)])
def test_coefficients_of_a_hermitian_matrix_are_real(dim, seed):
    for c, _ in pauli_decompose(_random_hermitian(dim, seed)):
        assert abs(complex(c).imag) < 1e-12


def test_every_coefficient_matches_its_own_trace_formula():
    A = _random_hermitian(4, seed=8)
    for c, s in pauli_decompose(A):
        assert abs(complex(c) - _brute_coefficient(A, s)) < 1e-12, s


def test_terms_come_back_in_lexicographic_order():
    """The order is part of the numerical contract: it fixes the Trotter error.

    ``trotter_drift_bound`` at order 2 is *not* permutation invariant, so a change of
    ordering silently moves every auto-selected step count and every honest-mode
    answer at a fixed ``trotter_steps``.
    """
    A = _random_hermitian(4, seed=9)
    strings = [s for _, s in pauli_decompose(A)]
    assert strings == sorted(strings, key=lambda s: [("IXYZ").index(c) for c in s])
    assert strings == _all_strings(2)  # a dense 4×4 keeps all 16


def test_known_decompositions_by_hand():
    assert [(complex(c).real, s) for c, s in pauli_decompose(np.diag([3.0, 1.0]))] == [
        (2.0, "I"),
        (1.0, "Z"),
    ]
    # [[2, 1], [1, 2]] = 2I + X — a single non-identity term, so Trotterisation of
    # e^{iAt} is *exact*.
    assert [
        (complex(c).real, s) for c, s in pauli_decompose(np.array([[2.0, 1], [1, 2]]))
    ] == [
        (2.0, "I"),
        (1.0, "X"),
    ]
    # A complex Hermitian off-diagonal is the Y channel.
    assert [
        (complex(c).real, s) for c, s in pauli_decompose(np.array([[0.0, -1j], [1j, 0]]))
    ] == [(1.0, "Y")]


def test_decomposition_rejects_malformed_input():
    with pytest.raises(ValueError, match="square"):
        pauli_decompose(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="square"):
        pauli_decompose(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="power of two"):
        pauli_decompose(np.eye(3))
    with pytest.raises(ValueError, match="not Hermitian"):
        pauli_decompose(np.array([[0.0, 1.0], [2.0, 0.0]]))


def test_hermiticity_guard_scales_up_with_the_matrix():
    """A large matrix carries large float noise; the guard must scale with it.

    An absolute tolerance would reject a perfectly good ``A`` of magnitude 1e10 whose
    symmetry is only good to ~1e-6 in absolute terms.
    """
    A = _random_hermitian(4, seed=12, scale=1e10)
    A[0, 1] += 1e-4  # ~1e-14 relative — noise at this scale
    pauli_decompose(A)  # must not raise
    A[0, 1] += 1e8  # ~1e-2 relative — genuinely not Hermitian
    with pytest.raises(ValueError, match="not Hermitian"):
        pauli_decompose(A)


def test_hermiticity_guard_is_relative_all_the_way_down():
    """``atol = 1e-8 · ‖A‖_max``, with no floor under it.

    A floor such as ``1e-8 · max(‖A‖_max, 1)`` would hold the tolerance at 1e-8 however
    small ``A`` is — so every matrix with entries at or below that would pass whatever
    its structure, and ``pauli_decompose(1e-9 · [[0, 1], [0, 0]])`` would return a
    complex coefficient for a nilpotent.  That
    :func:`~eigenkit.algorithms.hhl._make_hermitian` symmetrises before
    :meth:`~eigenkit.algorithms.hhl.HHL.solve` reaches here does not make a floor safe:
    it is true of ``solve`` and not of ``pauli_decompose``, which the example page
    imports directly — nor of :class:`~eigenkit.core.blocks.TrotterBlock`, which keeps
    ``c.real`` and would evolve only the Hermitian part of what it is handed.
    """
    B = _random_hermitian(4, seed=12, scale=1e-10)
    pauli_decompose(B)  # Hermitian is Hermitian at any scale
    B[0, 1] += 1e-10  # a 100 % relative asymmetry, whatever its absolute size
    with pytest.raises(ValueError, match="not Hermitian"):
        pauli_decompose(B)

    # A residue small *relative to the matrix* is still noise rather than structure:
    # ‖A‖_max ≈ 3.6e-10 here, so the tolerance is ≈ 3.6e-18.
    C = _random_hermitian(4, seed=12, scale=1e-10)
    C[0, 1] += 1e-19
    pauli_decompose(C)

    # A nilpotent matrix, at every scale an absolute floor would wave through.
    nilpotent = np.array([[0.0, 1.0], [0.0, 0.0]])
    for scale in (1e3, 1.0, 1e-8, 1e-9, 1e-30):
        with pytest.raises(ValueError, match="not Hermitian"):
            pauli_decompose(scale * nilpotent)


@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1.0, 1e6, 1e12])
def test_pruning_is_scale_invariant(scale):
    """An *absolute* ``atol`` would make honest mode depend on the units of ``A``.

    At an absolute 1e-10 threshold, ``1e-10 · (I + 0.9X)`` would lose its X term and
    the answer would change.  The threshold is relative to ``max|A_ij|``, so the term
    list — and every circuit built from it — is invariant under ``A → αA``.
    """
    A = np.array([[1.0, 0.9], [0.9, 1.0]])
    reference = [(complex(c).real, s) for c, s in pauli_decompose(A)]
    scaled = [(complex(c).real / scale, s) for c, s in pauli_decompose(scale * A)]
    assert [s for _, s in scaled] == [s for _, s in reference]
    np.testing.assert_allclose(
        [c for c, _ in scaled], [c for c, _ in reference], rtol=1e-12
    )


def test_rtol_prunes_relative_to_the_largest_entry():
    A = np.array([[1.0, 1e-6], [1e-6, 1.0]])
    assert [s for _, s in pauli_decompose(A, rtol=1e-10)] == ["I", "X"]
    assert [s for _, s in pauli_decompose(A, rtol=1e-4)] == ["I"]
    # rtol=0 keeps every structurally non-zero coefficient (strict inequality).
    assert len(pauli_decompose(np.eye(4), rtol=0.0)) == 1


def test_pruning_never_drops_a_term_that_carries_the_matrix():
    """Everything pruned must be negligible: the rebuild stays accurate."""
    A = _random_hermitian(8, seed=13)
    A += 1e-9 * _random_hermitian(8, seed=14)  # structural noise
    terms = pauli_decompose(A, rtol=1e-6)
    rebuilt = sum(c * P(s) for c, s in terms)
    assert np.linalg.norm(rebuilt - A) < 1e-6 * np.linalg.norm(A) * A.shape[0]


def test_one_by_one_matrix_yields_the_empty_pauli_string():
    """Documented boundary: a 1×1 system has zero qubits, hence the empty string.

    ``HHL`` never hits this — it pads a 1×1 system up to 2×2 — but the helper is
    public enough to pin, and ``TrotterBlock`` rejects the empty string rather than
    building a zero-qubit circuit.
    """
    assert pauli_decompose(np.array([[3.0 + 0j]])) == [(3.0 + 0j, "")]


# ---------------------------------------------------------------------------
# term-count reality: what actually makes the honest path cheap
# ---------------------------------------------------------------------------


def _sparse_symmetric(n: int, nnz: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = np.zeros((n, n))
    for i in range(n):
        for j in rng.choice(n, nnz, replace=False):
            M[i, j] = M[j, i] = rng.standard_normal()
    return M + np.diag(np.abs(rng.standard_normal(n)) + n)


@pytest.mark.parametrize("dim", [4, 8, 16])
def test_matrix_sparsity_does_not_bound_the_term_count(dim):
    """ "sparse ⇒ O(poly n) terms" is false, and this is the witness.

    A matrix with three non-zeros per row needs essentially as many Pauli terms as a
    dense one, and a *random diagonal* — as sparse as an invertible matrix gets —
    needs ``N``, because the Z-string coefficients are the Walsh–Hadamard transform
    of the diagonal and a generic transform is dense.
    """
    rng = np.random.default_rng(2)
    dense = rng.standard_normal((dim, dim))
    dense = dense + dense.T
    n_dense = len(pauli_decompose(dense))
    n_sparse = len(pauli_decompose(_sparse_symmetric(dim)))
    assert n_sparse >= 0.9 * n_dense, (n_sparse, n_dense)

    diagonal = np.diag(rng.standard_normal(dim))
    assert len(pauli_decompose(diagonal)) == dim


@pytest.mark.parametrize("dim", [4, 8, 16, 32])
def test_a_short_pauli_sum_by_construction_is_what_stays_cheap(dim):
    """The claim that does hold: few terms belong to matrices *built* as a short sum."""
    n = dim.bit_length() - 1
    assert len(pauli_decompose(np.eye(dim))) == 1
    # diag(1..N) is a linear ramp: n Z-strings plus the identity.
    assert len(pauli_decompose(np.diag(np.arange(1.0, dim + 1)))) == n + 1
    # Transverse-field Ising, assembled from 2n − 1 Pauli strings.
    ising = sum(P("I" * k + "ZZ" + "I" * (n - k - 2)) for k in range(n - 1)) + sum(
        P("I" * k + "X" + "I" * (n - k - 1)) for k in range(n)
    )
    assert len(pauli_decompose(ising.real.astype(complex))) == 2 * n - 1


@pytest.mark.parametrize("dim", [4, 8, 16, 32])
def test_a_toeplitz_band_still_costs_theta_n_terms(dim):
    """The canonical HHL demo matrix is ``Θ(N)`` terms — exponential in the qubits."""
    tri = 2 * np.eye(dim) - np.eye(dim, k=1) - np.eye(dim, k=-1)
    assert len(pauli_decompose(tri)) == dim


def test_a_general_tridiagonal_costs_more_than_a_toeplitz_one():
    """Variable coefficients — a realistic discretisation — cost strictly more."""
    for dim in (8, 16):
        rng = np.random.default_rng(dim)
        off = rng.standard_normal(dim - 1)
        general = np.diag(rng.standard_normal(dim)) + np.diag(off, 1) + np.diag(off, -1)
        toeplitz = 2 * np.eye(dim) - np.eye(dim, k=1) - np.eye(dim, k=-1)
        assert len(pauli_decompose(general)) > len(pauli_decompose(toeplitz))
        # Below the literature's (n+1)·2^n upper bound for a tridiagonal matrix.
        n = dim.bit_length() - 1
        assert len(pauli_decompose(general)) <= (n + 1) * 2**n


def test_the_module_does_not_claim_sparsity_buys_cheapness():
    """The docstrings state the term-count reality, not the tempting sparsity claim."""
    assert 'So "sparse ⇒ few terms" is false' in hs.__doc__
    assert "4^n" in pauli_decompose.__doc__  # there are 4^n strings, not 2^n
    assert "2^n Pauli strings" not in pauli_decompose.__doc__


def test_decomposition_scales_to_a_128_dimensional_matrix():
    """The transform is ``O(N² log N)``; the naive contraction would be ``O(N⁴)``."""
    A = _random_hermitian(128, seed=15)
    terms = pauli_decompose(A)
    assert len(terms) == 128 * 128
    # Spot-check a handful of coefficients rather than rebuilding 16384 matrices.
    rng = np.random.default_rng(16)
    for idx in rng.choice(len(terms), 12, replace=False):
        c, s = terms[int(idx)]
        assert abs(complex(c) - _brute_coefficient(A, s)) < 1e-11, s


# ---------------------------------------------------------------------------
# symplectic bookkeeping and the anticommuting weight profile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_symplectic_parity_agrees_with_dense_commutators(n):
    strings = _all_strings(n)
    x_bits, z_bits = hs._symplectic_bits(strings)
    for i, si in enumerate(strings):
        for j, sj in enumerate(strings):
            parity = int(
                np.bitwise_count(x_bits[i] & z_bits[j])
                + np.bitwise_count(x_bits[j] & z_bits[i])
            )
            Pi, Pj = P(si), P(sj)
            anticommutes_densely = np.max(np.abs(Pi @ Pj + Pj @ Pi)) < 1e-12
            assert bool(parity & 1) == anticommutes_densely, (si, sj)
            if anticommutes_densely:
                # ‖[P_k, P_l]‖ = 2 exactly — the constant the drift bound uses.
                assert abs(np.linalg.norm(Pi @ Pj - Pj @ Pi, 2) - 2.0) < 1e-12


def test_symplectic_bits_reject_input_it_cannot_pack():
    with pytest.raises(ValueError, match="same length"):
        hs._symplectic_bits(["XX", "X"])
    with pytest.raises(ValueError, match="longer than 64 qubits"):
        hs._symplectic_bits(["X" * 65])
    assert hs._symplectic_bits([])[0].shape == (0,)


def test_anticommuting_profile_matches_a_brute_force_double_loop():
    A = _random_hermitian(8, seed=17)
    terms = pauli_decompose(A)
    coeffs = [abs(complex(c)) for c, _ in terms]
    strings = [s for _, s in terms]
    brute = np.zeros(len(terms))
    mats = [P(s) for s in strings]
    for i in range(len(terms)):
        total = 0.0
        for j in range(i + 1, len(terms)):
            if np.max(np.abs(mats[i] @ mats[j] + mats[j] @ mats[i])) < 1e-12:
                total += coeffs[j]
        brute[i] = coeffs[i] * total
    np.testing.assert_allclose(anticommuting_weight_profile(terms), brute, atol=1e-12)


def test_anticommuting_profile_survives_its_own_blocking_loop():
    """The routine batches the ``L × L`` parity matrix; the batching must be exact."""
    A = _random_hermitian(32, seed=18)
    terms = pauli_decompose(A)
    assert len(terms) == 1024  # forces more than one block
    profile = anticommuting_weight_profile(terms)
    assert profile.shape == (1024,)
    # Γ₁ is a sum over unordered pairs, so it cannot depend on the blocking.
    assert profile.sum() == pytest.approx(
        anticommuting_weight_profile(terms[::-1]).sum(), rel=1e-12
    )


def test_anticommuting_profile_edge_cases():
    np.testing.assert_array_equal(anticommuting_weight_profile([]), np.zeros(0))
    np.testing.assert_array_equal(anticommuting_weight_profile([(2.0, "X")]), [0.0])
    # Only the strict upper triangle counts, so the last term is always zero.
    profile = anticommuting_weight_profile([(1.0, "X"), (1.0, "Z"), (1.0, "Y")])
    assert profile[-1] == 0.0
    np.testing.assert_allclose(profile, [2.0, 1.0, 0.0])


def test_profile_is_indexed_in_application_order():
    """``R_k`` is a *suffix* weight, so permuting the list moves the weights."""
    terms = [(1.0, "X"), (2.0, "Z"), (3.0, "I")]
    np.testing.assert_allclose(anticommuting_weight_profile(terms), [2.0, 0.0, 0.0])
    np.testing.assert_allclose(anticommuting_weight_profile(terms[::-1]), [0.0, 2.0, 0.0])


# ---------------------------------------------------------------------------
# trotter_drift_bound
# ---------------------------------------------------------------------------


def test_drift_bound_is_exactly_zero_for_a_commuting_family():
    """No Trotter error at all when every pair commutes — the selector relies on it."""
    assert trotter_drift_bound([(2.0, "I"), (1.0, "X")], 3.0, 1) == 0.0
    assert trotter_drift_bound([(1.0, "ZI"), (1.0, "IZ"), (1.0, "ZZ")], 5.0, 2) == 0.0
    assert trotter_drift_bound([], 1.0, 1) == 0.0
    assert trotter_drift_bound([(0.0, "X"), (0.0, "Z")], 1.0, 1) == 0.0


def test_drift_bound_hand_computed_constants():
    # Γ₁ = Σ_{k<l, anticommuting} |c_k c_l|.
    assert trotter_drift_bound([(0.7, "X"), (0.4, "Z")], 1.0, 1) == pytest.approx(0.28)
    # Γ₂ = (1/3)Σ S_k R_k + (1/6)Σ |c_k| R_k with S_k = Σ_{l>k} |c_l|.
    a, b = 0.7, 0.4
    assert trotter_drift_bound([(a, "X"), (b, "Z")], 1.0, 2) == pytest.approx(
        abs(a * b) * (2 * abs(b) + abs(a)) / 6.0
    )


@pytest.mark.parametrize("order,exponent", [(1, 1), (2, 2)])
def test_drift_bound_scales_as_tau_to_the_order(order, exponent):
    terms = [(1.3, "XI"), (0.7, "ZZ"), (0.4, "IY")]
    base = trotter_drift_bound(terms, 0.1, order)
    assert trotter_drift_bound(terms, 0.2, order) == pytest.approx(
        base * 2**exponent, rel=1e-12
    )
    assert trotter_drift_bound(terms, -0.1, order) == pytest.approx(base)


def test_gamma1_is_permutation_invariant_and_gamma2_is_not():
    """Documented asymmetry, and the reason the term order is pinned upstream."""
    terms = [(1.3, "XI"), (0.7, "ZZ"), (0.4, "IY"), (0.9, "YX")]
    assert trotter_drift_bound(terms[::-1], 0.3, 1) == pytest.approx(
        trotter_drift_bound(terms, 0.3, 1), rel=1e-12
    )
    assert trotter_drift_bound(terms[::-1], 0.3, 2) != pytest.approx(
        trotter_drift_bound(terms, 0.3, 2), rel=1e-9
    )


def test_drift_bound_rejects_an_unsupported_order():
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        trotter_drift_bound([(1.0, "X")], 0.1, 3)
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        trotter_drift_bound([(1.0, "X")], 0.1, 0)


@pytest.mark.parametrize("scale", [1e-100, 1e-20, 1.0, 1e20, 1e100])
def test_drift_bound_is_homogeneous_across_the_whole_float_range(scale):
    """``Γ₁ ∝ scale²`` and ``Γ₂ ∝ scale³`` without overflowing or flushing to zero.

    ``_resolve_trotter_steps`` normalises to ``max|c_k| = 1`` precisely because the
    unnormalised ``Γ₂`` leaves float64 above ``‖A‖ ≈ 1e103``; this pins the exponents
    the normalisation has to reproduce.
    """
    terms = [(scale, "X"), (scale, "Z")]
    assert trotter_drift_bound(terms, 1.0, 1) == pytest.approx(scale**2, rel=1e-12)
    assert trotter_drift_bound(terms, 1.0, 2) == pytest.approx(0.5 * scale**3, rel=1e-12)


def test_drift_bound_accepts_complex_typed_but_real_coefficients():
    """``pauli_decompose`` hands back ``complex`` coefficients; only ``|c|`` matters."""
    assert trotter_drift_bound([(0.7 + 0j, "X"), (0.4 + 0j, "Z")], 1.0, 1) == (
        pytest.approx(0.28)
    )
