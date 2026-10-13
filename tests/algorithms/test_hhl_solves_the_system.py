"""HHL must actually solve Ax = b, checked against `np.linalg.solve`.

A loose bound on HHL's accuracy -- `residual < 0.5`, `cos_sim > 0.5` -- set
against measured residuals of 1e-15 to 1e-5 cannot distinguish a working solver
from one that has drifted three orders of magnitude, and says nothing about
whether the result's dtype or its `euclidean_norm` are what they claim.

These compare against the classical solution by value, and pin what the
result's own fields claim to mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.algorithms.hhl import HHL


def _spd(n, rng, condition=3.0):
    """A symmetric positive-definite matrix with a controlled condition number."""
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = np.linspace(1.0, condition, n)
    return q @ np.diag(eigenvalues) @ q.T


class TestAgainstTheClassicalSolution:
    @pytest.mark.parametrize("seed", range(5))
    def test_two_by_two_matches_numpy(self, seed):
        rng = np.random.default_rng(seed)
        A = _spd(2, rng)
        b = rng.normal(size=2)
        result = HHL(n_eigenvalue_qubits=5).solve(A, b)
        exact = np.linalg.solve(A, b)
        relative = np.linalg.norm(np.asarray(result.solution) - exact) / np.linalg.norm(
            exact
        )
        assert relative < 5e-2, f"relative error {relative:.3e}"

    @pytest.mark.parametrize("seed", range(3))
    def test_the_direction_is_right(self, seed):
        """Direction is what HHL fundamentally produces; the scale comes from
        the classical least-squares step afterwards."""
        rng = np.random.default_rng(seed + 50)
        A = _spd(2, rng)
        b = rng.normal(size=2)
        got = np.asarray(HHL(n_eigenvalue_qubits=5).solve(A, b).solution)
        exact = np.linalg.solve(A, b)
        cosine = abs(np.dot(got, exact)) / (np.linalg.norm(got) * np.linalg.norm(exact))
        assert cosine > 0.999, f"cosine similarity {cosine:.6f}"

    def test_more_clock_qubits_do_not_make_it_worse_overall(self):
        """Not asserted per-step: the residual is not monotone in `n_l`, and a
        test claiming it would be wrong. What must hold is that the best of a
        larger budget is at least as good."""
        rng = np.random.default_rng(11)
        A, b = _spd(2, rng), rng.normal(size=2)
        residuals = [
            HHL(n_eigenvalue_qubits=n).solve(A, b).residual for n in (3, 4, 5, 6)
        ]
        assert min(residuals[2:]) <= min(residuals[:2]) + 1e-9


class TestTheResultFieldsMeanWhatTheySay:
    @staticmethod
    def _solve():
        rng = np.random.default_rng(3)
        A, b = _spd(2, rng), rng.normal(size=2)
        return HHL(n_eigenvalue_qubits=5).solve(A, b), A, b

    def test_a_real_system_gives_a_real_solution(self):
        """The result documents "Real-valued for real inputs", so the
        least-squares rescale must not promote it to complex128."""
        result, _, _ = self._solve()
        assert not np.iscomplexobj(result.solution)

    def test_a_complex_system_gives_a_complex_solution(self):
        A = np.array([[2.0, 1j], [-1j, 3.0]])
        b = np.array([1.0 + 0j, 2.0 + 1j])
        assert np.iscomplexobj(HHL(n_eigenvalue_qubits=5).solve(A, b).solution)

    def test_euclidean_norm_is_the_norm_of_the_solution(self):
        result, _, _ = self._solve()
        assert result.euclidean_norm == pytest.approx(
            float(np.linalg.norm(np.asarray(result.solution))), rel=1e-12
        )

    def test_residual_is_the_relative_residual_it_claims(self):
        result, A, b = self._solve()
        expected = np.linalg.norm(A @ np.asarray(result.solution) - b) / np.linalg.norm(b)
        assert result.residual == pytest.approx(float(expected), rel=1e-12)

    def test_success_probability_is_a_probability(self):
        result, _, _ = self._solve()
        assert 0.0 <= result.success_probability <= 1.0
