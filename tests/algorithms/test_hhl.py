"""Tests for the HHL quantum linear solver."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest


pytest.importorskip("scipy")

from eigenkit.algorithms.hhl import (
    HHL,
    HHLResult,
    _choose_t,
    _compute_rotation_angles,
    _make_hermitian,
    _postselect,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classical_solve(A, b):
    return np.linalg.solve(np.real(A), np.real(b))


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


class TestMakeHermitian:
    def test_non_hermitian_high_kappa_warns(self):
        """Non-Hermitian A with κ > 10 must warn about condition number squaring."""
        # Upper-triangular matrix with κ ≈ 100
        A = np.diag([100.0, 1.0]).astype(complex)
        A[0, 1] = 1.0  # break Hermitian symmetry
        b = np.array([1.0, 1.0], dtype=complex)
        with pytest.warns(RuntimeWarning, match="κ\\(A†A\\)"):
            _make_hermitian(A, b, float(np.linalg.cond(A)))

    def test_non_hermitian_low_kappa_no_warn(self):
        """Non-Hermitian A with κ ≤ 10 must not warn."""
        import warnings as _warnings

        A = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=complex)  # κ ≈ 2.6
        b = np.array([1.0, 1.0], dtype=complex)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", RuntimeWarning)
            _make_hermitian(A, b, float(np.linalg.cond(A)))  # must not raise

    def test_hermitian_input_unchanged(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([1.0, 0.0])
        A_h, b_h, is_hermitian = _make_hermitian(
            A.astype(complex), b.astype(complex), float(np.linalg.cond(A))
        )
        assert is_hermitian
        np.testing.assert_array_equal(A_h, A)
        np.testing.assert_array_equal(b_h, b)

    def test_non_hermitian_normal_equations(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 1.0])
        A_h, b_h, is_hermitian = _make_hermitian(
            A.astype(complex), b.astype(complex), float(np.linalg.cond(A))
        )
        # Non-Hermitian input -> normal equations formed, same size as original
        assert not is_hermitian
        assert A_h.shape == (2, 2)
        assert b_h.shape == (2,)
        # A†A must be Hermitian PSD
        np.testing.assert_allclose(A_h, A_h.conj().T, atol=1e-12)
        # A†A x = A†b must match the original Ax = b
        x_ref = np.linalg.solve(A, b)
        x_ne = np.linalg.solve(A_h, b_h)
        np.testing.assert_allclose(x_ne, x_ref, atol=1e-10)


class TestComputeRotationAngles:
    def test_positive_bucket_full_rotation(self):
        # C_actual = lam_max_abs → ratio = C/λ > 1 for small buckets → clamped to π
        angles = _compute_rotation_angles(
            n_buckets=4, t=2 * math.pi, C_actual=1.0, lam_max_abs=1.0, epsilon=0.01
        )
        # j=1 is the only positive bucket (1 < half=2): λ̂=0.25, ratio=4 → clamped → π
        assert 0 < angles[1] <= math.pi

    def test_negative_bucket_has_negative_angle(self):
        # j >= half encode negative eigenvalues → rotation angle should be negative
        angles = _compute_rotation_angles(
            n_buckets=4, t=2 * math.pi, C_actual=0.1, lam_max_abs=1.0, epsilon=0.01
        )
        # j=3: j_signed=3-4=-1 → λ̂=-0.25 → ratio=0.1/(-0.25)=-0.4 → angle < 0
        assert angles[3] < 0

    def test_zero_angle_below_threshold(self):
        # epsilon=0.99 → threshold=0.99 → all |λ̂_j| < 0.99 for t=2π, n=4
        angles = _compute_rotation_angles(
            n_buckets=4, t=2 * math.pi, C_actual=0.1, lam_max_abs=1.0, epsilon=0.99
        )
        # All positive and negative buckets should be below threshold
        assert all(a == 0.0 for a in angles.values())


class TestPostselect:
    def test_trivial_all_zeros(self):
        # Statevector with all weight on |000⟩ → zero success probability
        sv = np.zeros(8, dtype=complex)
        sv[0] = 1.0  # |000⟩, ancilla=0
        amps, prob, readout = _postselect(sv, n_b=1, n_l=1)
        assert prob == pytest.approx(0.0, abs=1e-12)
        assert readout == pytest.approx(0.0, abs=1e-12)

    def test_all_weight_on_ancilla_1(self):
        # |1 00⟩ for [ancilla | clock(1) | b_reg(1)] → 3 qubits, index 4 = |100⟩
        sv = np.zeros(8, dtype=complex)
        sv[4] = 1.0  # index 4 = 0b100 → ancilla=1, clock=0, b_reg=0
        amps, prob, readout = _postselect(sv, n_b=1, n_l=1)
        assert prob == pytest.approx(1.0, abs=1e-12)
        assert readout == pytest.approx(1.0, abs=1e-12)
        assert abs(amps[0]) == pytest.approx(1.0, abs=1e-12)

    def test_clock_nonzero_branches_excluded(self):
        # [ancilla(bit2) | clock(bit1) | b_reg(bit0)]. The solution is read only
        # from the clock=|0⟩ slice (indices 4, 5); amplitudes on clock=|1⟩
        # (indices 6, 7) are off-resonance leakage and must NOT leak into it.
        # A coherent sum over clock branches would corrupt the result.
        sv = np.zeros(8, dtype=complex)
        sv[4] = 0.6  # ancilla=1, clock=0, b=0  -> solution[0]
        sv[5] = 0.8  # ancilla=1, clock=0, b=1  -> solution[1]
        sv[6] = 10.0  # ancilla=1, clock=1, b=0 -> leakage, ignored
        sv[7] = -10.0  # ancilla=1, clock=1, b=1 -> leakage, ignored
        amps, prob, readout = _postselect(sv, n_b=1, n_l=1)
        # The clock=|0⟩ slice [0.6, 0.8] is already unit-norm.
        np.testing.assert_allclose(amps, np.array([0.6, 0.8]), atol=1e-12)
        # success_prob still counts the full ancilla=|1⟩ weight.
        assert prob == pytest.approx(0.36 + 0.64 + 100.0 + 100.0, abs=1e-9)
        # readout_prob counts only the slice the solution is read from — here a
        # 200-fold overstatement if one used success_prob as the shot budget.
        assert readout == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Integration tests: HHL.solve
# ---------------------------------------------------------------------------


class TestHHLSolve:
    def test_2x2_diagonal(self):
        """Diagonal 2×2 system: exact solution known."""
        A = np.diag([2.0, 1.0])
        b = np.array([1.0, 1.0])
        result = HHL(n_eigenvalue_qubits=4).solve(A, b)

        assert isinstance(result, HHLResult)
        assert result.solution.shape == (2,)
        assert result.success_probability > 0.0
        assert result.residual < 0.5  # reasonable quality for 4 clock qubits

    def test_2x2_hermitian(self):
        """2×2 Hermitian positive-definite system.

        [[2,1],[1,3]] has eigenvalues (5±√5)/2 — irrational ratio.
        n_eigenvalue_qubits=4 is insufficient; 6 qubits are required.
        """
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 0.0])
        x_ref = _classical_solve(A, b)

        result = HHL(n_eigenvalue_qubits=6).solve(A, b)

        assert result.residual < 0.5
        # Solution direction should roughly match classical
        dot = abs(np.dot(result.solution, x_ref)) / (
            np.linalg.norm(result.solution) * np.linalg.norm(x_ref) + 1e-14
        )
        assert dot > 0.5  # at least cos(60°) alignment

    def test_result_fields(self):
        """All HHLResult fields are present and typed correctly."""
        A = np.array([[2.0, 0.5], [0.5, 1.0]])
        b = np.array([1.0, 1.0])
        result = HHL(n_eigenvalue_qubits=3).solve(A, b)

        assert isinstance(result.solution, np.ndarray)
        assert isinstance(result.euclidean_norm, float)
        assert isinstance(result.residual, float)
        assert isinstance(result.success_probability, float)
        assert result.success_probability >= 0.0
        assert result.residual >= 0.0

    def test_result_is_immutable(self):
        """HHLResult must be a frozen dataclass — field assignment must raise."""
        A = np.diag([2.0, 1.0])
        b = np.array([1.0, 1.0])
        result = HHL(n_eigenvalue_qubits=3).solve(A, b)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.solution = np.zeros(2)

    def test_non_hermitian_matrix(self):
        """Non-Hermitian A is automatically lifted to Hermitian embedding."""
        A = np.array([[1.0, 2.0], [0.0, 1.0]])  # upper triangular, not Hermitian
        b = np.array([1.0, 1.0])
        # Should not raise
        result = HHL(n_eigenvalue_qubits=3).solve(A, b)
        assert result.solution.shape == (2,)

    def test_3x3_system(self):
        """3×3 Hermitian system (zero-padded to 4×4 internally)."""
        A = np.diag([3.0, 2.0, 1.0])
        b = np.array([1.0, 1.0, 1.0])
        result = HHL(n_eigenvalue_qubits=4).solve(A, b)
        assert result.solution.shape == (3,)
        assert result.residual < 1.0  # weaker bound due to padding artefacts


# ---------------------------------------------------------------------------
# Sign-indefinite spectra and the Nyquist bucket
# ---------------------------------------------------------------------------


class TestSignIndefiniteSpectra:
    """A max-magnitude negative eigenvalue must stay out of the sign-ambiguous
    Nyquist bucket N/2, where ``_compute_rotation_angles`` resolves the sign as
    positive — for a ±λ_max spectrum the 1/λ and 1/(−λ) contributions would then
    cancel into a silent zero solution at success_probability ≈ 1 and
    residual = 1.  So the auto-selected t is capped below π when the spectrum
    has a negative part.
    """

    def _alignment(self, solution, x_ref):
        return abs(np.dot(solution, x_ref)) / (
            np.linalg.norm(solution) * np.linalg.norm(x_ref) + 1e-14
        )

    def test_pauli_x_system(self):
        # A = X has eigenvalues +1 and −1; A x = [1, 0] → x = [0, 1].
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        b = np.array([1.0, 0.0])
        result = HHL(n_eigenvalue_qubits=4).solve(A, b)
        assert result.residual < 0.1
        assert self._alignment(result.solution, np.array([0.0, 1.0])) > 0.99

    def test_mixed_diagonal_spectrum(self):
        # diag(1, −0.5): ratios {1, 0.5}.  With 16 buckets and the capped
        # domain, t = 3π/4 aligns both eigenvalues exactly (buckets 6 and 3).
        A = np.diag([1.0, -0.5])
        b = np.array([1.0, 1.0])
        x_ref = _classical_solve(A, b)
        result = HHL(n_eigenvalue_qubits=4).solve(A, b)
        assert result.residual < 0.1
        assert self._alignment(result.solution, x_ref) > 0.99

    def test_choose_t_capped_for_negative_spectrum(self):
        N = 16
        t = _choose_t(np.array([1.0]), N, has_negative=True)
        assert t <= math.pi * (N - 2) / N + 1e-12

    def test_choose_t_allows_pi_for_positive_spectrum(self):
        # Positive-definite spectra keep the exact-alignment t = π endpoint.
        t = _choose_t(np.array([1.0]), 16, has_negative=False)
        assert t == pytest.approx(math.pi)

    def test_negative_spectrum_with_two_buckets_raises(self):
        with pytest.raises(ValueError, match="sign-ambiguous"):
            _choose_t(np.array([1.0]), 2, has_negative=True)

    def test_invalid_A_non_square(self):
        with pytest.raises(ValueError, match="square"):
            HHL().solve(np.ones((2, 3)), np.ones(2))

    def test_invalid_b_wrong_length(self):
        with pytest.raises(ValueError, match="length"):
            HHL().solve(np.eye(2), np.ones(3))

    def test_invalid_b_zero(self):
        with pytest.raises(ValueError, match="zero"):
            HHL().solve(np.eye(2), np.zeros(2))

    def test_invalid_t_non_positive(self):
        with pytest.raises(ValueError, match="positive, finite"):
            HHL(t=0.0)

    def test_invalid_t_non_finite(self):
        # t must be positive AND finite: a non-finite t silently bypasses the
        # `t <= 0` guard (NaN <= 0 is False) and propagates NaN/Inf into the
        # evolution unitary, so it must be rejected at construction.
        for bad_t in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValueError, match="positive, finite"):
                HHL(t=bad_t)

    def test_top_level_import(self):
        from eigenkit import HHL as HHL_top
        from eigenkit import HHLResult as HHLResult_top

        assert HHL_top is HHL
        assert HHLResult_top is HHLResult

    def test_more_eigenvalue_qubits_improves_residual(self):
        """More clock qubits must genuinely improve the residual. With the
        clock=|0⟩ projection HHL converges monotonically in n_eigenvalue_qubits;
        a coherent sum over the clock branches could make the residual *rise*
        with more qubits, which this guards against."""
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 0.0])
        r4 = HHL(n_eigenvalue_qubits=4).solve(A, b).residual
        r6 = HHL(n_eigenvalue_qubits=6).solve(A, b).residual
        assert r6 < r4

    def test_irrational_eigenvalues_need_more_qubits(self):
        """[[2,1],[1,3]] has an irrational eigenvalue ratio. With the clock=|0⟩
        projection, n>=6 drives the residual well below 1e-3 (a coherent sum over
        the clock branches would leave it near 7e-3 here)."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 0.0])
        r6 = HHL(n_eigenvalue_qubits=6).solve(A, b).residual
        assert r6 < 1e-3
