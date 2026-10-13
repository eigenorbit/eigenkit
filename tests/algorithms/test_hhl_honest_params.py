"""Tests for the spectral-free parameter estimation of the honest HHL path.

These cover the three building blocks the honest path derives its circuit from —
a Gershgorin bound on |λ|_max, an evolution time taken from that bound, and the
rotation constant C taken from the register resolution — none of which touch the
spectrum (no eigvalsh, no eigenvalue-tuned time).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eigenkit.algorithms.hhl import (
    _compute_rotation_angles,
    _gershgorin_spectral_radius,
    _honest_qpe_params,
)


def _random_hermitian(dim: int, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return scale * (m + m.conj().T)


# ---- Gershgorin bound ----


def test_gershgorin_known_values():
    assert _gershgorin_spectral_radius(np.diag([3.0, -5.0, 2.0])) == pytest.approx(5.0)
    ones = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    assert _gershgorin_spectral_radius(ones) == pytest.approx(2.0)  # each row sums to 2
    assert _gershgorin_spectral_radius(
        np.array([[0.0, 1j], [-1j, 0.0]])
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("dim,seed", [(2, 0), (8, 2)])
def test_gershgorin_upper_bounds_true_spectrum(dim, seed):
    A = _random_hermitian(dim, seed)
    bound = _gershgorin_spectral_radius(A)
    true_max = float(np.max(np.abs(np.linalg.eigvalsh(A))))
    assert bound >= true_max - 1e-9


# ---- honest (t, C) parameters ----


def test_time_avoids_aliasing_against_bound_and_truth():
    A = _random_hermitian(4, seed=7)
    for n_clock in (2, 3, 5, 8):
        t, _, lam_max_bound = _honest_qpe_params(A, n_clock)
        # Phase from the (over-)estimated radius stays strictly inside (-pi, pi).
        assert t * lam_max_bound < math.pi
        # A fortiori it is safe against the true spectral radius.
        true_max = float(np.max(np.abs(np.linalg.eigvalsh(A))))
        assert t * true_max < math.pi


# ---- integration with the rotation-angle grid ----


def test_honest_C_needs_no_clamping_in_rotation_angles():
    # With C set to the register resolution, the |j_signed|=1 buckets hit the
    # arcsin domain edge exactly (angle = ±pi) and no bucket needs clamping:
    # every eigenvalue the register can represent is >= C in magnitude.
    A = _random_hermitian(4, seed=8)
    n_clock = 5
    t, C, lam_max_bound = _honest_qpe_params(A, n_clock)
    n_buckets = 2**n_clock
    angles = _compute_rotation_angles(n_buckets, t, C, lam_max_bound, epsilon=0.0)

    # First positive and first negative buckets map to +/- pi exactly.
    assert angles[1] == pytest.approx(math.pi)
    assert angles[n_buckets - 1] == pytest.approx(-math.pi)
    # All angles are finite and within [-pi, pi]; nothing blew up.
    for theta in angles.values():
        assert math.isfinite(theta)
        assert -math.pi - 1e-9 <= theta <= math.pi + 1e-9
