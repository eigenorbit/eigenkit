"""Unit tests for every private helper ``HHL`` is assembled from.

``solve()`` is one long pipeline, and an end-to-end residual is a poor localiser: a
Gershgorin bound that is too tight, a rotation table off by a bucket, a post-selection
mask reading the wrong slice and a least-squares rescale hiding all three produce the
same symptom.  So each helper is pinned against its own definition here — the phase
window against the Nyquist geometry, the rotation table against ``2·asin(C/λ̃)``, the
uniformly controlled rotation against the multi-controlled table it implements, the
post-selection against a hand-built statevector, and the step selector against the
closed form it claims to solve.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from eigenkit import HHL
from eigenkit.algorithms import hhl as H
from eigenkit.algorithms._hamiltonian_sim import (
    pauli_decompose,
    pauli_string_matrix,
    trotter_drift_bound,
)
from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import CircuitSpec


P = pauli_string_matrix

TRI4 = np.array([[2.0, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]])
A_HEISENBERG = (P("XX") + P("YY") + P("ZZ") + P("ZI") + P("IZ")).real


def _unitary(spec: CircuitSpec) -> np.ndarray:
    circ = Circuit(spec, provider="numpy")
    dim = 2**spec.n_qubits
    return np.column_stack(
        [
            circ.statevector(np.zeros(0), initial_state=np.eye(dim, dtype=complex)[:, j])
            for j in range(dim)
        ]
    )


# ---------------------------------------------------------------------------
# numerically defensive scalars
# ---------------------------------------------------------------------------


def test_safe_norm_matches_numpy_in_the_ordinary_range():
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = rng.standard_normal(7) + 1j * rng.standard_normal(7)
        assert H._safe_norm(v) == pytest.approx(float(np.linalg.norm(v)), rel=1e-14)


@pytest.mark.parametrize("scale", [1e-320, 1e-200, 1e-8, 1.0, 1e8, 1e200, 1e300])
def test_safe_norm_survives_both_ends_of_float64(scale):
    """``np.linalg.norm`` squares first, so it overflows above ~1.3e154."""
    v = np.array([3.0, 4.0]) * scale
    assert H._safe_norm(v) == pytest.approx(5.0 * scale, rel=1e-12)


def test_safe_norm_degenerate_inputs():
    assert H._safe_norm(np.zeros(4)) == 0.0
    assert H._safe_norm(np.array([])) == 0.0
    assert H._safe_norm(np.array([np.inf, 1.0])) == float("inf")
    assert H._safe_norm(np.array([np.nan, 1.0])) == float("inf")


@pytest.mark.parametrize("scale", [1e-320, 1e-150, 1.0, 1e150, 1e300])
def test_safe_unit_returns_a_unit_vector_at_any_scale(scale):
    v = np.array([1.0, -2.0, 3.0], dtype=complex) * scale
    u = H._safe_unit(v)
    assert np.linalg.norm(u) == pytest.approx(1.0, rel=1e-12)
    # Same direction, and no NaN from a denormal complex division.
    assert np.all(np.isfinite(u))
    expected = np.array([1.0, -2.0, 3.0]) / math.sqrt(14.0)
    np.testing.assert_allclose(u.real, expected, rtol=1e-9)


def test_safe_unit_rejects_what_it_cannot_normalise():
    with pytest.raises(ValueError, match="zero or non-finite"):
        H._safe_unit(np.zeros(3))
    with pytest.raises(ValueError, match="zero or non-finite"):
        H._safe_unit(np.array([np.inf, 0.0]))


def test_is_effectively_real_is_relative_to_the_matrix_scale():
    """An absolute tolerance calls a genuinely complex 1e-9-scale matrix real.

    That decides whether the read-out casts to ``float`` and whether the imaginary
    Trotter detector runs at all, so it has to be scale free.
    """
    assert H._is_effectively_real(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert H._is_effectively_real(np.zeros((2, 2)))
    assert H._is_effectively_real(np.array([]))
    assert H._is_effectively_real(np.array([[1.0 + 1e-14j]]))
    tiny_complex = 1e-9 * np.array([[0.0, -1j], [1j, 0.0]])
    assert not H._is_effectively_real(tiny_complex)
    assert not H._is_effectively_real(1e9 * np.array([[0.0, -1j], [1j, 0.0]]))


# ---------------------------------------------------------------------------
# input validation and the Hermitian lift
# ---------------------------------------------------------------------------


def test_validate_inputs_returns_sigma_max_and_kappa():
    A = np.diag([4.0, 1.0]).astype(complex)
    sigma_max, kappa = H._validate_inputs(A, np.ones(2, dtype=complex))
    assert sigma_max == pytest.approx(4.0)
    assert kappa == pytest.approx(4.0)


def test_validate_inputs_rejects_bad_shapes_and_values():
    b2 = np.ones(2, dtype=complex)
    with pytest.raises(ValueError, match="square"):
        H._validate_inputs(np.zeros((2, 3), dtype=complex), b2)
    with pytest.raises(ValueError, match="1-D vector of length 2"):
        H._validate_inputs(np.eye(2, dtype=complex), np.ones(3, dtype=complex))
    with pytest.raises(ValueError, match="1-D vector"):
        H._validate_inputs(np.eye(2, dtype=complex), np.ones((2, 1), dtype=complex))
    with pytest.raises(ValueError, match="A contains non-finite"):
        H._validate_inputs(np.array([[np.nan, 0], [0, 1]], dtype=complex), b2)
    with pytest.raises(ValueError, match="b contains non-finite"):
        H._validate_inputs(np.eye(2, dtype=complex), np.array([np.inf, 1], dtype=complex))
    with pytest.raises(ValueError, match="empty"):
        H._validate_inputs(np.zeros((0, 0), dtype=complex), np.zeros(0, dtype=complex))
    with pytest.raises(ValueError, match="singular"):
        H._validate_inputs(np.array([[1.0, 1], [1, 1]], dtype=complex), b2)


def test_make_hermitian_passes_a_hermitian_matrix_through_untouched():
    A = np.array([[2.0, 1j], [-1j, 3.0]])
    b = np.array([1.0 + 0j, 2.0])
    A_h, b_h, is_hermitian = H._make_hermitian(A, b, kappa=3.0)
    assert is_hermitian
    assert A_h is A and b_h is b


def test_make_hermitian_forms_the_normal_equations_and_warns_when_kappa_is_large():
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    b = np.array([1.0, 1.0], dtype=complex)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        A_h, b_h, is_hermitian = H._make_hermitian(A, b, kappa=15.0)
    assert not is_hermitian
    np.testing.assert_allclose(A_h, A.conj().T @ A)
    np.testing.assert_allclose(b_h, A.conj().T @ b)
    assert any("not Hermitian" in str(w.message) for w in caught)

    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        H._make_hermitian(A, b, kappa=3.0)
    assert not quiet  # a well-conditioned lift is not worth a warning


def test_make_hermitian_tolerance_is_scale_relative():
    """The tolerance is ``1e-12·‖A‖_max``.  With ``atol=0`` a structural zero facing
    a 1e-18 partner would read as "not Hermitian" and the system would silently go
    through the normal equations, squaring ``κ``.
    """
    A = np.diag([1.0, 0.02]).astype(complex)
    A[1, 0] = 1e-18
    assert H._make_hermitian(A, np.ones(2, dtype=complex), 50.0)[2] is True
    # Scaling the whole matrix cannot change the verdict.
    for scale in (1e-6, 1.0, 1e6):
        assert H._make_hermitian(scale * A, np.ones(2, dtype=complex), 50.0)[2] is True
    # A genuine asymmetry at the same relative size is still caught.
    B = np.diag([1.0, 0.02]).astype(complex)
    B[1, 0] = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert H._make_hermitian(B, np.ones(2, dtype=complex), 50.0)[2] is False


def test_a_matrix_of_all_zeros_counts_as_hermitian():
    """Degenerate but reachable: ``solve`` rejects it later as singular, not here."""
    Z = np.zeros((2, 2), dtype=complex)
    assert H._make_hermitian(Z, np.ones(2, dtype=complex), 1.0)[2] is True


# ---------------------------------------------------------------------------
# the spectrum-free QPE parameters
# ---------------------------------------------------------------------------


def test_gershgorin_radius_is_the_max_absolute_row_sum():
    assert H._gershgorin_spectral_radius(np.diag([3.0, -5.0])) == 5.0
    assert H._gershgorin_spectral_radius(np.array([[1.0, -2], [3, 4]])) == 7.0
    assert H._gershgorin_spectral_radius(np.array([[0.0, 1j], [-1j, 0]])) == 1.0


@pytest.mark.parametrize("dim,seed", [(2, 0), (4, 1), (8, 2), (16, 3)])
def test_gershgorin_radius_bounds_the_true_spectral_radius(dim, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    A = m + m.conj().T
    assert (
        H._gershgorin_spectral_radius(A) >= np.max(np.abs(np.linalg.eigvalsh(A))) - 1e-12
    )


def test_gershgorin_radius_is_tight_on_a_diagonal_matrix():
    """Tightness matters: it is the case where an eigenvalue sits exactly on the
    edge of the guard band, which is what the Nyquist witness exploits."""
    A = np.diag([1.0, -4.0, 2.0])
    assert H._gershgorin_spectral_radius(A) == 4.0


@pytest.mark.parametrize("n_clock", range(2, 10))
def test_two_sided_phase_window_keeps_the_bound_off_the_nyquist_bucket(n_clock):
    """``t·Λ = π(N−2)/N`` puts ``±Λ`` on bucket ``±(N/2 − 1)`` exactly.

    That is a half-bucket guard band below Nyquist, which is the geometry the sign of
    ``1/λ`` depends on.
    """
    N = 2**n_clock
    window = H._honest_phase_window(N)
    bucket = window * N / (2 * math.pi)
    assert bucket == pytest.approx(N / 2 - 1)
    assert bucket < N / 2


@pytest.mark.parametrize("n_clock", range(2, 10))
def test_one_sided_phase_window_uses_the_whole_circle(n_clock):
    N = 2**n_clock
    bucket = H._honest_phase_window(N, one_sided=True) * N / (2 * math.pi)
    assert bucket == pytest.approx(N - 1)


def test_honest_qpe_params_derive_t_and_C_from_the_bound_alone():
    A = TRI4.astype(complex)
    lam_bound = H._gershgorin_spectral_radius(A)
    for n_clock in (2, 4, 6, 8):
        N = 2**n_clock
        t, C, bound = H._honest_qpe_params(A, n_clock)
        assert bound == lam_bound
        assert t == pytest.approx(math.pi * (N - 2) / N / lam_bound)
        assert C == pytest.approx(2 * math.pi / (t * N))
        # C is the bucket-1 eigenvalue, and Λ/C is the register's capacity.
        assert bound / C == pytest.approx((N - 2) / 2)


def test_one_sided_params_halve_C_for_the_same_register():
    A = TRI4.astype(complex)
    for n_clock in (3, 6):
        _, c_two, bound = H._honest_qpe_params(A, n_clock)
        _, c_one, _ = H._honest_qpe_params(A, n_clock, one_sided=True)
        assert c_one < c_two
        assert bound / c_one == pytest.approx(2**n_clock - 1)


def test_honest_qpe_params_reject_an_unusable_register_or_matrix():
    with pytest.raises(ValueError, match="n_eigenvalue_qubits >= 2"):
        H._honest_qpe_params(TRI4.astype(complex), 1)
    with pytest.raises(ValueError, match="Gershgorin radius"):
        H._honest_qpe_params(np.zeros((2, 2), dtype=complex), 4)
    with pytest.raises(ValueError, match="Gershgorin radius"):
        H._honest_qpe_params(np.array([[np.inf, 0], [0, 1]], dtype=complex), 4)


def test_explicit_t_is_accepted_up_to_the_window_and_rejected_past_it():
    A = TRI4.astype(complex)
    lam_bound = H._gershgorin_spectral_radius(A)
    window = H._honest_phase_window(16)
    t_max = window / lam_bound
    t, C, _ = H._honest_qpe_params(A, 4, t=t_max)
    assert t == t_max
    assert C == pytest.approx(2 * math.pi / (t_max * 16))
    # One ulp of slack, so the documented maximum is not rejected by rounding.
    H._honest_qpe_params(A, 4, t=t_max * (1 + 1e-15))
    with pytest.raises(ValueError, match="causes QPE aliasing"):
        H._honest_qpe_params(A, 4, t=t_max * 1.001)


def test_a_smaller_explicit_t_raises_C_above_the_register_floor():
    """Documented trade: a shorter evolution resolves a coarser grid."""
    A = TRI4.astype(complex)
    _, c_auto, _ = H._honest_qpe_params(A, 6)
    _, c_short, _ = H._honest_qpe_params(
        A, 6, t=0.1 * (math.pi * 62 / 64 / H._gershgorin_spectral_radius(A))
    )
    assert c_short == pytest.approx(10 * c_auto)


# ---------------------------------------------------------------------------
# the eigenvalue-inversion table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_clock", range(2, 11))
def test_the_honest_C_never_clamps_the_arcsine(n_clock):
    """``C/λ̃_j = 1/j_signed`` exactly, so ``|ratio| ≤ 1`` for every bucket.

    Clamping would silently replace an eigenvalue's inverse by ``±1`` — the single
    place the inversion could go wrong without touching any spectrum.
    """
    A = TRI4.astype(complex)
    for one_sided in (False, True):
        t, C, _ = H._honest_qpe_params(A, n_clock, one_sided=one_sided)
        N = 2**n_clock
        for j in range(1, N):
            j_signed = j if one_sided or j <= N // 2 else j - N
            lam_est = 2 * math.pi * j_signed / (t * N)
            ratio = C / lam_est
            assert abs(ratio) <= 1.0 + 1e-15, (n_clock, j)
            assert ratio == pytest.approx(1.0 / j_signed, rel=1e-12)


def test_rotation_angles_invert_the_estimated_eigenvalue():
    # C at the register floor 2π/(tN), which is what the honest path always passes.
    n_buckets, t, lam_max = 16, 1.0, 4.0
    C = 2 * math.pi / (t * n_buckets)
    angles = H._compute_rotation_angles(n_buckets, t, C, lam_max, epsilon=0.0)
    assert set(angles) == set(range(1, n_buckets))
    for j, theta in angles.items():
        j_signed = j if j <= n_buckets // 2 else j - n_buckets
        lam_est = 2 * math.pi * j_signed / (t * n_buckets)
        assert math.sin(theta / 2) == pytest.approx(C / lam_est)
        assert math.sin(theta / 2) == pytest.approx(1.0 / j_signed)


def test_bucket_zero_never_gets_a_rotation():
    """It is the rotation-free bucket: an eigenvalue rounding there is dropped."""
    angles = H._compute_rotation_angles(8, 1.0, 0.5, 4.0, epsilon=0.0)
    assert 0 not in angles


def test_negative_buckets_carry_a_negative_amplitude():
    """The sign of ``1/λ`` lives in the sign of the RY angle."""
    n_buckets = 16
    angles = H._compute_rotation_angles(
        n_buckets, 1.0, 2 * math.pi / 16, 4.0, epsilon=0.0
    )
    assert angles[1] > 0 and angles[n_buckets - 1] < 0
    assert angles[1] == pytest.approx(-angles[n_buckets - 1])
    # Bucket N/2 (Nyquist) is read as positive; the phase window keeps it empty.
    assert angles[n_buckets // 2] > 0


def test_one_sided_table_is_positive_everywhere():
    C = 2 * math.pi / 16
    angles = H._compute_rotation_angles(16, 1.0, C, 4.0, epsilon=0.0, one_sided=True)
    assert all(theta > 0 for theta in angles.values())
    assert angles[15] == pytest.approx(2 * math.asin(1.0 / 15))


def test_epsilon_zeroes_the_buckets_below_the_cutoff():
    """The cull is symmetric in the bucket sign and hits the largest ``1/λ`` weights."""
    n_buckets, t, lam_max = 256, 1.0, 1.0
    C = 2 * math.pi / (t * n_buckets)
    # |λ̃_j| = 2π|j_s|/(tN) = |j_s|·C ≈ 0.0245·|j_s|, so epsilon=0.03 culls |j_s| = 1.
    angles = H._compute_rotation_angles(n_buckets, t, C, lam_max, epsilon=0.03)
    assert angles[1] == 0.0
    assert angles[n_buckets - 1] == 0.0
    assert angles[2] != 0.0
    assert angles[n_buckets - 2] != 0.0
    # epsilon=0 keeps them.
    assert H._compute_rotation_angles(n_buckets, t, C, lam_max, epsilon=0.0)[1] != 0.0


def test_rotation_ratio_is_clamped_rather_than_producing_a_domain_error():
    """Defensive: a caller-supplied ``C`` above ``λ̃`` must not raise from ``asin``."""
    angles = H._compute_rotation_angles(8, 1.0, 1e6, 1.0, epsilon=0.0)
    assert all(abs(theta) == pytest.approx(math.pi) for theta in angles.values())


# ---------------------------------------------------------------------------
# the inversion circuit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_clock", [1, 2, 3, 4])
def test_uniformly_controlled_ry_equals_the_multi_controlled_table(n_clock):
    """The table is emitted as ``2^{n_l}`` RY + ``2^{n_l}`` CNOT, exactly.

    The direct construction — one multi-controlled RY per bucket — is what the honest
    gate set is not allowed to contain, so the equality has to be checked as a matrix,
    not as a gate count.
    """
    rng = np.random.default_rng(n_clock)
    angles = {j: float(rng.uniform(-math.pi, math.pi)) for j in range(1, 2**n_clock)}
    gates = H._uniformly_controlled_ry(angles, n_clock, tuple(range(1, n_clock + 1)), 0)
    assert len(gates) == 2 * 2**n_clock
    assert [g.name for g in gates] == ["RY", "CNOT"] * 2**n_clock

    got = _unitary(
        CircuitSpec(n_qubits=n_clock + 1, n_weights=0, n_inputs=0, gates=tuple(gates))
    )
    # Layout [ancilla(MSB) | clock]: bucket j is index j, ancilla-1 is j + 2^n_clock.
    N = 2**n_clock
    want = np.zeros((2 * N, 2 * N), dtype=complex)
    for j in range(N):
        c, s = math.cos(angles.get(j, 0.0) / 2), math.sin(angles.get(j, 0.0) / 2)
        want[j, j] = want[N + j, N + j] = c
        want[N + j, j] = s
        want[j, N + j] = -s
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_uniformly_controlled_ry_is_empty_when_every_angle_is_zero():
    """A fully culled table must not emit ``2^{n_l}`` no-op rotations."""
    assert (
        H._uniformly_controlled_ry({j: 0.0 for j in range(1, 8)}, 3, (1, 2, 3), 0) == []
    )
    assert H._uniformly_controlled_ry({}, 3, (1, 2, 3), 0) == []


def test_uniformly_controlled_ry_uses_only_elementary_gates():
    angles = {1: 0.3, 2: -0.7, 3: 1.1}
    names = {g.name for g in H._uniformly_controlled_ry(angles, 2, (1, 2), 0)}
    assert names == {"RY", "CNOT"}


def test_clock_qubit_zero_holds_the_most_significant_bucket_bit():
    """A flipped bucket-bit convention permutes the whole inversion table."""
    n_clock = 2
    angles = {1: 0.0, 2: 1.0, 3: 0.0}  # only bucket 2 = binary "10" rotates
    got = _unitary(
        CircuitSpec(
            n_qubits=3,
            n_weights=0,
            n_inputs=0,
            gates=tuple(H._uniformly_controlled_ry(angles, n_clock, (1, 2), 0)),
        )
    )
    # Bucket 2 means clock qubit 0 = 1, clock qubit 1 = 0.
    assert abs(got[4 + 2, 2]) == pytest.approx(math.sin(0.5))
    for j in (0, 1, 3):
        assert abs(got[4 + j, j]) < 1e-12


# ---------------------------------------------------------------------------
# post-selection
# ---------------------------------------------------------------------------


def _index(anc: int, clock: int, breg: int, n_l: int, n_b: int) -> int:
    return (anc << (n_l + n_b)) | (clock << n_b) | breg


def test_postselect_reads_the_ancilla_one_clock_zero_slice():
    n_l, n_b = 2, 1
    sv = np.zeros(2 ** (1 + n_l + n_b), dtype=complex)
    sv[_index(1, 0, 0, n_l, n_b)] = 0.6
    sv[_index(1, 0, 1, n_l, n_b)] = 0.8j
    amps, success, readout = H._postselect(sv, n_b, n_l)
    np.testing.assert_allclose(amps, [0.6, 0.8j])
    assert success == pytest.approx(1.0)
    assert readout == pytest.approx(1.0)


def test_postselect_separates_the_two_probabilities():
    """``success_probability`` is the ancilla weight over *all* clock states;
    the solution comes from the joint event, which is what a device post-selects on.
    """
    n_l, n_b = 2, 1
    sv = np.zeros(2 ** (1 + n_l + n_b), dtype=complex)
    sv[_index(1, 0, 0, n_l, n_b)] = 0.6  # in the read-out slice
    sv[_index(1, 3, 1, n_l, n_b)] = 0.8  # ancilla 1 but clock ≠ 0
    amps, success, readout = H._postselect(sv, n_b, n_l)
    assert success == pytest.approx(1.0)
    assert readout == pytest.approx(0.36)
    assert readout <= success
    np.testing.assert_allclose(amps, [1.0, 0.0])  # normalised slice


def test_postselect_ignores_the_ancilla_zero_branch():
    n_l, n_b = 2, 2
    sv = np.zeros(2 ** (1 + n_l + n_b), dtype=complex)
    sv[_index(0, 0, 2, n_l, n_b)] = 1.0
    amps, success, readout = H._postselect(sv, n_b, n_l)
    assert success == 0.0 and readout == 0.0
    np.testing.assert_allclose(amps, np.zeros(4))


def test_postselect_returns_a_unit_vector_when_there_is_anything_to_read():
    rng = np.random.default_rng(5)
    n_l, n_b = 3, 2
    sv = rng.standard_normal(2 ** (1 + n_l + n_b)) + 1j * rng.standard_normal(
        2 ** (1 + n_l + n_b)
    )
    sv /= np.linalg.norm(sv)
    amps, success, readout = H._postselect(sv, n_b, n_l)
    assert np.linalg.norm(amps) == pytest.approx(1.0)
    assert 0.0 < readout <= success <= 1.0


def test_postselect_preserves_the_relative_phases_that_carry_the_sign():
    n_l, n_b = 1, 2
    sv = np.zeros(2 ** (1 + n_l + n_b), dtype=complex)
    for breg, amp in enumerate([0.5, -0.5, 0.5j, -0.5j]):
        sv[_index(1, 0, breg, n_l, n_b)] = amp
    amps, _, _ = H._postselect(sv, n_b, n_l)
    np.testing.assert_allclose(amps, [0.5, -0.5, 0.5j, -0.5j], atol=1e-15)


# ---------------------------------------------------------------------------
# the Trotter-step selector
# ---------------------------------------------------------------------------


def _solver(**kwargs):
    return HHL(mode="honest", provider="numpy", **kwargs)


def test_selector_returns_one_step_for_a_commuting_hamiltonian():
    terms = pauli_decompose(np.array([[2.0, 1], [1, 2]], dtype=complex))
    assert _solver(n_eigenvalue_qubits=6)._resolve_trotter_steps(terms, 1.0, 0.1, 6) == 1


def test_selector_meets_its_own_criterion_and_is_minimal():
    """The closed form must be the *smallest* integer satisfying the criterion."""
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    for order in (1, 2):
        for n_clock in (4, 6):
            solver = _solver(n_eigenvalue_qubits=n_clock, trotter_order=order)
            t, C, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), n_clock)
            steps = solver._resolve_trotter_steps(terms, t, C, n_clock)
            assert trotter_drift_bound(terms, t / steps, order) <= 0.25 * C * (1 + 1e-9)
            if steps > 1:
                assert trotter_drift_bound(terms, t / (steps - 1), order) > 0.25 * C or (
                    # or the angle guard is what binds, not the drift criterion
                    (t / (steps - 1)) * sum(abs(complex(c)) for c, _ in terms)
                    > H._TROTTER_ANGLE_GUARD
                )


def test_selector_honours_the_per_step_angle_guard():
    """Past ``τ·Σ|c_k| ≈ 0.5`` the leading-order bound stops being trustworthy."""
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    l1 = sum(abs(complex(c)) for c, _ in terms)
    for n_clock in (3, 5, 7):
        solver = _solver(n_eigenvalue_qubits=n_clock)
        t, C, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), n_clock)
        steps = solver._resolve_trotter_steps(terms, t, C, n_clock)
        assert (t / steps) * l1 <= H._TROTTER_ANGLE_GUARD * (1 + 1e-9)


def test_tighter_tolerance_never_asks_for_fewer_steps():
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    t, C, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), 6)
    counts = [
        _solver(n_eigenvalue_qubits=6, trotter_tolerance=tol)._resolve_trotter_steps(
            terms, t, C, 6
        )
        for tol in (1.0, 0.5, 0.25, 0.05, 0.01)
    ]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_steps_from_criterion_solves_the_documented_closed_form():
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    unit = max(abs(complex(c)) for c, _ in terms)
    unit_terms = [(complex(c) / unit, s) for c, s in terms]
    l1_hat = sum(abs(complex(c)) for c, _ in unit_terms)
    t, C = 1.0, 0.05
    for order in (1, 2):
        gamma_hat = trotter_drift_bound(unit_terms, 1.0, order)
        solver = _solver(n_eigenvalue_qubits=6, trotter_order=order)
        steps = solver._steps_from_criterion(gamma_hat, l1_hat, unit, t, C)
        target = solver.trotter_tolerance * C
        if order == 1:
            closed = math.ceil(t * unit * gamma_hat * (unit / target))
        else:
            closed = math.ceil(t * unit * math.sqrt(gamma_hat * (unit / target)))
        assert steps == max(
            1, max(closed, math.ceil(t * unit * l1_hat / H._TROTTER_ANGLE_GUARD))
        )


@pytest.mark.parametrize("scale", [1e-40, 1e-10, 1.0, 1e10, 1e40])
def test_selector_is_scale_invariant(scale):
    """The criterion is homogeneous, so ``A → αA`` must not move the step count.

    Computed without normalisation, ``Γ₂ ∝ ‖A‖³`` leaves float64 around 1e103.
    """
    A = scale * A_HEISENBERG
    terms = pauli_decompose(A.astype(complex))
    t, C, _ = H._honest_qpe_params(A.astype(complex), 6)
    reference_terms = pauli_decompose(A_HEISENBERG.astype(complex))
    t0, c0, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), 6)
    solver = _solver(n_eigenvalue_qubits=6)
    assert solver._resolve_trotter_steps(terms, t, C, 6) == (
        solver._resolve_trotter_steps(reference_terms, t0, c0, 6)
    )


def test_selector_returns_one_when_every_coefficient_is_zero():
    """Degenerate but reachable through the public API only for a zero ``A``."""
    assert (
        _solver(n_eigenvalue_qubits=4)._resolve_trotter_steps(
            [(0.0, "X"), (0.0, "Z")], 1.0, 0.1, 4
        )
        == 1
    )
    assert (
        _solver(n_eigenvalue_qubits=4, trotter_steps=7)._resolve_trotter_steps(
            [(0.0, "X")], 1.0, 0.1, 4
        )
        == 7
    )


def test_explicit_steps_are_obeyed_and_only_warn_past_the_capture_threshold():
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    t, C, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), 6)
    coarse = _solver(n_eigenvalue_qubits=6, trotter_steps=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert coarse._resolve_trotter_steps(terms, t, C, 6) == 1
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "exceeds C/2" in message and "Nyquist" in message
    assert "trotter_steps=" in message  # names a sufficient count

    fine_steps = int(message.split("trotter_steps=")[-1].split()[0].rstrip("."))
    fine = _solver(n_eigenvalue_qubits=6, trotter_steps=fine_steps)
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        fine._resolve_trotter_steps(terms, t, C, 6)
    assert not quiet, [str(w.message) for w in quiet]


def test_gate_budget_clamps_auto_selection_with_a_warning():
    """A big register plus a fine grid can ask for tens of millions of gates."""
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    solver = _solver(n_eigenvalue_qubits=12, trotter_tolerance=1e-6)
    t, C, _ = H._honest_qpe_params(A_HEISENBERG.astype(complex), 12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        steps = solver._resolve_trotter_steps(terms, t, C, 12)
    assert steps == H._max_steps_within_budget(terms, 12, 2)
    assert len(caught) == 1
    assert "gate budget" in str(caught[0].message)
    assert "clamped" in str(caught[0].message)


def test_max_steps_within_budget_respects_the_declared_budget():
    terms = pauli_decompose(A_HEISENBERG.astype(complex))
    for n_clock in (4, 8, 12):
        steps = H._max_steps_within_budget(terms, n_clock, 2)
        assert steps >= 1
        assert (
            H._estimate_trotter_gate_count(terms, n_clock, steps, 2)
            <= H._AUTO_TROTTER_GATE_BUDGET
        )
        if steps > 1:
            assert (
                H._estimate_trotter_gate_count(terms, n_clock, steps + 1, 2)
                > H._AUTO_TROTTER_GATE_BUDGET
            )
    assert H._max_steps_within_budget([(1.0, "I")], 2, 1) >= 1


# ---------------------------------------------------------------------------
# the warnings that keep honest mode from failing silently
# ---------------------------------------------------------------------------


def test_unresolvable_spectrum_warning_names_the_register_floor():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HHL._warn_on_unresolvable_spectrum(
            sigma_min=0.001,
            C=1.0,
            lam_max_bound=10.0,
            epsilon=0.0,
            n_clock=4,
            one_sided=False,
            is_normal_eq=False,
        )
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "rotation-free bucket 0" in message
    assert "raise n_eigenvalue_qubits to" in message
    assert "epsilon" not in message  # only the floor that binds is named


def test_unresolvable_spectrum_warning_names_the_epsilon_floor():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HHL._warn_on_unresolvable_spectrum(
            sigma_min=0.05,
            C=0.01,  # register floor C/2 = 0.005 does *not* bind
            lam_max_bound=10.0,
            epsilon=0.01,  # floor 0.1 does
            n_clock=8,
            one_sided=False,
            is_normal_eq=False,
        )
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "epsilon cutoff" in message
    assert "no register size fixes this" in message
    assert "rotation-free bucket 0" not in message


def test_unresolvable_spectrum_warning_names_both_floors_when_both_bind():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HHL._warn_on_unresolvable_spectrum(
            sigma_min=1e-4,
            C=1.0,
            lam_max_bound=10.0,
            epsilon=0.01,
            n_clock=4,
            one_sided=False,
            is_normal_eq=False,
        )
    message = str(caught[0].message)
    assert "rotation-free bucket 0" in message and "epsilon cutoff" in message


def test_unresolvable_spectrum_warning_squares_epsilon_back_for_the_lift():
    """The user-facing ``epsilon`` is squared internally for ``A†A``, so the advice
    has to be given in the user's units or it is off by a square."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HHL._warn_on_unresolvable_spectrum(
            sigma_min=1e-4,
            C=1e-9,  # the register floor does not bind
            lam_max_bound=1.0,
            epsilon=1e-2,  # = (0.1)² in the user's units
            n_clock=6,
            one_sided=True,
            is_normal_eq=True,
        )
    message = str(caught[0].message)
    advised = float(message.split("lower epsilon to ≲ ")[1].split()[0])
    assert advised == pytest.approx(math.sqrt(0.5 * 1e-4 / 1.0), rel=1e-2)
    # Without the lift the same numbers give the un-square-rooted advice.
    with warnings.catch_warnings(record=True) as plain:
        warnings.simplefilter("always")
        HHL._warn_on_unresolvable_spectrum(
            sigma_min=1e-4,
            C=1e-9,
            lam_max_bound=1.0,
            epsilon=1e-2,
            n_clock=6,
            one_sided=False,
            is_normal_eq=False,
        )
    plain_advice = float(str(plain[0].message).split("lower epsilon to ≲ ")[1].split()[0])
    assert plain_advice == pytest.approx(0.5 * 1e-4, rel=1e-2)


def test_unresolvable_spectrum_stays_quiet_when_the_register_is_big_enough():
    for kwargs in (
        dict(sigma_min=1.0, C=0.1, lam_max_bound=10.0, epsilon=0.01),
        dict(sigma_min=0.0, C=1.0, lam_max_bound=10.0, epsilon=0.01),  # unknown σ_min
        dict(sigma_min=float("nan"), C=1.0, lam_max_bound=10.0, epsilon=0.01),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            HHL._warn_on_unresolvable_spectrum(
                n_clock=6, one_sided=False, is_normal_eq=False, **kwargs
            )
        assert not caught, kwargs


def test_recommended_register_size_actually_removes_the_warning():
    """A recommendation that does not work is worse than none."""
    A = np.diag([1.0, 1.0 / 60.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HHL(n_eigenvalue_qubits=4, mode="honest", provider="numpy").solve(A, np.ones(2))
    advice = [
        m for m in (str(w.message) for w in caught) if "raise n_eigenvalue_qubits to" in m
    ]
    assert advice
    needed = int(advice[0].split("raise n_eigenvalue_qubits to ≥ ")[1].split(".")[0])
    with warnings.catch_warnings(record=True) as after:
        warnings.simplefilter("always")
        result = HHL(n_eigenvalue_qubits=needed, mode="honest", provider="numpy").solve(
            A, np.ones(2)
        )
    assert not [m for m in (str(w.message) for w in after) if "bucket 0" in m]
    assert result.residual < 0.05


def test_imaginary_readout_detector_is_invariant_under_a_global_phase():
    """The exact HHL state for real input is real *up to a global phase*.

    A detector that missed that would fire on every correct answer.
    """
    real_state = np.array([0.6, -0.8, 0.0, 0.0], dtype=complex)
    for phi in (0.0, 0.7, math.pi / 2, math.pi, 3.0):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            H._warn_on_imaginary_readout(real_state * np.exp(1j * phi), 2)
        assert not caught, phi


def test_imaginary_readout_detector_fires_on_a_genuinely_complex_state():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H._warn_on_imaginary_readout(np.array([0.6, -0.8j, 0.0, 0.0]), 1)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "Trotter error in the eigenvectors" in message
    assert "trotter_order=2" in message  # order 1 gets the structural advice
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        H._warn_on_imaginary_readout(np.array([0.6, -0.8j, 0.0, 0.0]), 2)
    assert "Raise trotter_steps" in str(caught2[0].message)


def test_imaginary_readout_detector_tolerates_a_small_imaginary_part():
    """5 % is the threshold; below it the answer is dominated by other error."""
    state = np.array([1.0, 0.03j, 0.0, 0.0], dtype=complex)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H._warn_on_imaginary_readout(state, 2)
    assert not caught


def test_imaginary_readout_detector_handles_degenerate_input():
    for state in (
        np.array([], dtype=complex),
        np.zeros(4, dtype=complex),
        np.full(4, 1e-16, dtype=complex),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            H._warn_on_imaginary_readout(state, 2)
        assert not caught


def test_conditioning_hint_does_not_dress_register_capacity_up_as_kappa():
    """In honest mode the number is the register's capacity, not ``κ(A)``.

    Reporting it as "κ ≈ 127" would mislead precisely the user who needs the warning.
    """
    honest = _solver(n_eigenvalue_qubits=8)._conditioning_hint(127.0)
    assert "resolves κ up to" in honest
    assert "Gershgorin bound is loose" in honest
    reference = HHL(n_eigenvalue_qubits=8, provider="numpy")._conditioning_hint(12.5)
    assert reference == "κ ≈ 12.5"


# ---------------------------------------------------------------------------
# reference-mode time selection (shared code)
# ---------------------------------------------------------------------------


def test_choose_t_is_capped_below_nyquist_for_an_indefinite_spectrum():
    ratios = np.array([1.0, 0.5])
    t = H._choose_t(ratios, 16, has_negative=True)
    assert t <= math.pi * 14 / 16 + 1e-12


def test_choose_t_may_use_the_full_pi_for_a_positive_spectrum():
    assert H._choose_t(np.array([1.0]), 16, has_negative=False) == pytest.approx(math.pi)


def test_choose_t_prefers_a_grid_aligned_time():
    """The score is the summed squared fractional bucket error."""
    ratios = np.array([1.0, 0.25])
    t = H._choose_t(ratios, 32, has_negative=False)
    for lam in ratios:
        bucket = 32 * lam * t / (2 * math.pi)
        assert abs(bucket - round(bucket)) < 1e-9


def test_choose_t_rejects_a_two_bucket_register_for_a_negative_spectrum():
    with pytest.raises(ValueError, match="sign-ambiguous"):
        H._choose_t(np.array([1.0]), 2, has_negative=True)


def test_t_score_penalises_a_candidate_that_rounds_into_bucket_zero():
    """Bucket 0 has no rotation, so an eigenvalue landing there is dropped."""
    ratios = np.array([1.0])
    # t chosen so the eigenvalue sits at bucket 0.2 — distance is measured to 1.
    tiny = 2 * math.pi * 0.2 / 32
    assert H._t_score(tiny, ratios, 32) == pytest.approx(0.8**2)
