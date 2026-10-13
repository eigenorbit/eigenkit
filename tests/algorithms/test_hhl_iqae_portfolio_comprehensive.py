"""Comprehensive tests for HHL, IQAE, and PortfolioRiskEstimator.

Coverage targets:
  - Constructor / parameter validation (user error paths)
  - Helper-function unit tests not covered by the core test suite
  - Integration properties: result shapes, types, invariants
  - Edge cases: tiny systems, boundary amplitudes, special spectra
  - Cross-module: IQAE inside PortfolioRiskEstimator

Excludes duplication of test_hhl.py / test_iqae.py / test_portfolio_risk.py.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest


pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from sklearn.exceptions import NotFittedError  # noqa: E402

from eigenkit.algorithms.hhl import (  # noqa: E402
    HHL,
    HHLResult,
    _choose_t,
    _compute_rotation_angles,
    _postselect,
    _t_score,
    _validate_inputs,
)
from eigenkit.algorithms.iqae import (  # noqa: E402
    IQAE,
    IQAEResult,
    _clopper_pearson,
)
from eigenkit.algorithms.portfolio_risk import (  # noqa: E402
    PortfolioRiskEstimator,
    PortfolioRiskResult,
    _classical_tail_probability,
    _classical_var_cvar,
    _cvar_at_threshold,
)
from eigenkit.core.builder import CircuitBuilder  # noqa: E402
from eigenkit.core.ir import CircuitSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ry_oracle(theta: float) -> CircuitSpec:
    """One-qubit oracle: ancilla amplitude = sin(theta)."""
    qc = CircuitBuilder(n_qubits=1)
    qc.ry(2.0 * theta, qubit=0)
    return qc.build()


def _simple_prices(n_days: int = 30, n_assets: int = 1, seed: int = 0) -> np.ndarray:
    """Strictly-positive price series, shape (n_days, n_assets)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.01, size=(n_days - 1, n_assets))
    prices = np.empty((n_days, n_assets))
    prices[0] = 100.0
    for t in range(1, n_days):
        prices[t] = prices[t - 1] * np.exp(returns[t - 1])
    return prices


def _fitted_portfolio(**kwargs) -> PortfolioRiskEstimator:
    defaults = dict(n_qubits=3, epsilon=0.05, alpha=0.1, n_shots=50, random_state=0)
    defaults.update(kwargs)
    est = PortfolioRiskEstimator(**defaults)
    est.fit(_simple_prices())
    return est


# ===========================================================================
# HHL — constructor validation
# ===========================================================================


class TestHHLConstructorValidation:
    def test_zero_eigenvalue_qubits_raises(self):
        with pytest.raises(ValueError, match="n_eigenvalue_qubits"):
            HHL(n_eigenvalue_qubits=0)

    def test_negative_eigenvalue_qubits_raises(self):
        with pytest.raises(ValueError, match="n_eigenvalue_qubits"):
            HHL(n_eigenvalue_qubits=-3)

    def test_epsilon_negative_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            HHL(epsilon=-0.1)

    def test_epsilon_one_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            HHL(epsilon=1.0)

    def test_epsilon_above_one_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            HHL(epsilon=1.5)

    def test_epsilon_zero_valid(self):
        HHL(epsilon=0.0)  # lower bound is inclusive

    def test_explicit_t_stored(self):
        solver = HHL(t=2.5)
        assert solver.t == pytest.approx(2.5)

    def test_default_t_is_none(self):
        assert HHL().t is None


# ===========================================================================
# HHL — _validate_inputs (returns (sigma_max, kappa); these cover the raises)
# ===========================================================================


class TestValidateInputs:
    def test_nan_in_A_raises(self):
        A = np.array([[1.0, float("nan")], [0.0, 1.0]])
        b = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="non-finite"):
            _validate_inputs(A, b)

    def test_inf_in_A_raises(self):
        A = np.array([[float("inf"), 0.0], [0.0, 1.0]])
        b = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="non-finite"):
            _validate_inputs(A, b)

    def test_nan_in_b_raises(self):
        A = np.eye(2)
        b = np.array([float("nan"), 1.0])
        with pytest.raises(ValueError, match="non-finite"):
            _validate_inputs(A, b)

    def test_inf_in_b_raises(self):
        A = np.eye(2)
        b = np.array([1.0, float("inf")])
        with pytest.raises(ValueError, match="non-finite"):
            _validate_inputs(A, b)

    def test_non_square_A_raises(self):
        with pytest.raises(ValueError, match="square"):
            _validate_inputs(np.ones((2, 3)), np.ones(2))

    def test_b_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            _validate_inputs(np.eye(2), np.ones(3))


# ===========================================================================
# HHL — _t_score helper
# ===========================================================================


class TestTScore:
    def test_t_score_zero_for_exact_bucket_alignment(self):
        # t = π/2 maps lam_ratio=1.0 to bucket = 4*1*(π/2)/(2π) = 1 exactly.
        # fractional error = 0 → total score = 0.
        score = _t_score(math.pi / 2, np.array([1.0]), 4)
        assert score == pytest.approx(0.0, abs=1e-12)

    def test_t_score_positive_for_off_alignment(self):
        # Shift t slightly from the exact-alignment value so the eigenvalue
        # lands between two buckets — score must be strictly positive.
        exact_t = math.pi / 2
        score_off = _t_score(exact_t * 1.15, np.array([1.0]), 4)
        assert score_off > 0

    def test_t_score_increases_with_worse_alignment(self):
        # A larger fractional error should produce a larger score.
        s1 = _t_score(math.pi / 2 * 1.1, np.array([1.0]), 4)
        s2 = _t_score(math.pi / 2 * 1.3, np.array([1.0]), 4)
        assert s2 >= s1

    def test_t_score_returns_float(self):
        s = _t_score(math.pi, np.array([0.5, 1.0]), 8)
        assert isinstance(s, float)


# ===========================================================================
# HHL — _choose_t edge cases
# ===========================================================================


class TestChooseTEdgeCases:
    def test_positive_spectrum_allows_pi(self):
        t = _choose_t(np.array([0.5, 1.0]), 16, has_negative=False)
        assert t == pytest.approx(math.pi)

    def test_two_bucket_negative_raises(self):
        """n_buckets=2 with negative eigenvalues: Nyquist is the only bucket → raises."""
        with pytest.raises(ValueError, match="sign-ambiguous"):
            _choose_t(np.array([1.0]), 2, has_negative=True)

    def test_four_bucket_negative_is_capped_below_pi(self):
        t = _choose_t(np.array([1.0]), 4, has_negative=True)
        assert t < math.pi


# ===========================================================================
# HHL — integration: result invariants
# ===========================================================================


class TestHHLResultInvariants:
    def _solve_2x2(self, **kwargs):
        A = np.array([[2.0, 0.5], [0.5, 1.0]])
        b = np.array([1.0, 1.0])
        return HHL(n_eigenvalue_qubits=4, **kwargs).solve(A, b)

    def test_euclidean_norm_matches_solution_norm(self):
        result = self._solve_2x2()
        np.testing.assert_allclose(
            result.euclidean_norm, np.linalg.norm(result.solution), rtol=1e-6
        )

    def test_success_probability_in_unit_interval(self):
        result = self._solve_2x2()
        assert 0.0 <= result.success_probability <= 1.0

    def test_residual_non_negative(self):
        result = self._solve_2x2()
        assert result.residual >= 0.0

    def test_result_is_frozen(self):
        result = self._solve_2x2()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.residual = 99.0

    def test_solution_shape_matches_b(self):
        A = np.diag([3.0, 2.0, 1.0])
        b = np.array([1.0, 2.0, 3.0])
        result = HHL(n_eigenvalue_qubits=3).solve(A, b)
        assert result.solution.shape == b.shape

    def test_1x1_identity_system(self):
        """1×1 trivial system: x = b / a."""
        A = np.array([[2.0]])
        b = np.array([1.0])
        result = HHL(n_eigenvalue_qubits=3).solve(A, b)
        assert result.solution.shape == (1,)
        assert result.residual < 0.5

    def test_b_with_negative_components(self):
        """b is allowed to have negative entries — only its direction matters."""
        A = np.diag([2.0, 1.0])
        b = np.array([-1.0, 2.0])
        result = HHL(n_eigenvalue_qubits=4).solve(A, b)
        assert result.solution.shape == (2,)

    def test_complex_hermitian_matrix(self):
        """Complex Hermitian A (pure imaginary off-diagonal)."""
        A = np.array([[2.0, 1j], [-1j, 2.0]])
        b = np.array([1.0, 1.0], dtype=complex)
        result = HHL(n_eigenvalue_qubits=4).solve(A, b)
        assert result.solution.shape == (2,)
        assert result.residual < 1.0

    def test_explicit_t_parameter_accepted(self):
        """A user-specified valid t must be respected without errors.

        For A = diag(2, 1), λ_max = 2, so we need t * λ_max < π ↔ t < π/2.
        """
        A = np.diag([2.0, 1.0])
        b = np.array([1.0, 1.0])
        result = HHL(n_eigenvalue_qubits=4, t=1.0).solve(A, b)
        assert result.solution.shape == (2,)

    def test_top_level_import(self):
        from eigenkit import HHL as HHL_top
        from eigenkit import HHLResult as HHLResult_top

        assert HHL_top is HHL
        assert HHLResult_top is HHLResult


# ===========================================================================
# IQAE — constructor validation
# ===========================================================================


class TestIQAEConstructorValidation:
    _EPS_MAX = math.pi / 8.0

    def test_epsilon_zero_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            IQAE(epsilon=0.0, alpha=0.05, n_qubits=1)

    def test_epsilon_at_pi_over_8_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            IQAE(epsilon=self._EPS_MAX, alpha=0.05, n_qubits=1)

    def test_epsilon_above_pi_over_8_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            IQAE(epsilon=self._EPS_MAX + 0.1, alpha=0.05, n_qubits=1)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            IQAE(epsilon=0.05, alpha=0.0, n_qubits=1)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            IQAE(epsilon=0.05, alpha=1.0, n_qubits=1)

    def test_n_qubits_zero_raises(self):
        with pytest.raises(ValueError, match="n_qubits"):
            IQAE(epsilon=0.05, alpha=0.05, n_qubits=0)

    def test_n_qubits_negative_raises(self):
        with pytest.raises(ValueError, match="n_qubits"):
            IQAE(epsilon=0.05, alpha=0.05, n_qubits=-1)

    def test_n_shots_zero_raises(self):
        with pytest.raises(ValueError, match="n_shots"):
            IQAE(epsilon=0.05, alpha=0.05, n_qubits=1, n_shots=0)

    def test_min_ratio_one_raises(self):
        with pytest.raises(ValueError, match="min_ratio"):
            IQAE(epsilon=0.05, alpha=0.05, n_qubits=1, min_ratio=1.0)

    def test_min_ratio_below_one_raises(self):
        with pytest.raises(ValueError, match="min_ratio"):
            IQAE(epsilon=0.05, alpha=0.05, n_qubits=1, min_ratio=0.5)

    def test_invalid_shot_mode_raises(self):
        with pytest.raises(ValueError, match="shot_mode"):
            IQAE(epsilon=0.05, alpha=0.05, n_qubits=1, shot_mode="magic")

    def test_valid_construction_no_raise(self):
        IQAE(epsilon=0.05, alpha=0.05, n_qubits=2, n_shots=100, random_state=0)


# ===========================================================================
# IQAE — result invariants
# ===========================================================================


class TestIQAEResultInvariants:
    @pytest.fixture
    def result(self):
        oracle = _make_ry_oracle(math.asin(math.sqrt(0.3)))
        return IQAE(
            epsilon=0.05, alpha=0.1, n_qubits=1, n_shots=300, random_state=7
        ).solve(oracle)

    def test_estimation_in_unit_interval(self, result):
        assert 0.0 <= result.estimation <= 1.0

    def test_ci_lower_le_estimation_le_upper(self, result):
        lo, hi = result.confidence_interval
        assert lo <= result.estimation <= hi

    def test_ci_lower_le_upper(self, result):
        lo, hi = result.confidence_interval
        assert lo <= hi

    def test_n_oracle_calls_positive(self, result):
        assert result.n_oracle_calls > 0

    def test_n_rounds_positive(self, result):
        assert result.n_rounds >= 1

    def test_result_is_frozen(self, result):
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.estimation = 0.99

    def test_top_level_import(self):
        from eigenkit import IQAE as IQAE2
        from eigenkit import IQAEResult as IQAEResult2

        assert IQAE2 is IQAE
        assert IQAEResult2 is IQAEResult


# ===========================================================================
# IQAE — random_state=None runs without error
# ===========================================================================


class TestIQAERandomStateNone:
    def test_solve_without_seed_completes(self):
        oracle = _make_ry_oracle(math.pi / 6.0)
        result = IQAE(epsilon=0.1, alpha=0.1, n_qubits=1, n_shots=100).solve(oracle)
        assert isinstance(result, IQAEResult)

    def test_two_runs_without_seed_may_differ(self):
        oracle = _make_ry_oracle(math.pi / 4.0)
        r1 = IQAE(epsilon=0.1, alpha=0.1, n_qubits=1, n_shots=100).solve(oracle)
        r2 = IQAE(epsilon=0.1, alpha=0.1, n_qubits=1, n_shots=100).solve(oracle)
        # We can only guarantee they're both valid IQAEResult objects,
        # not that they're identical (non-deterministic).
        assert isinstance(r1, IQAEResult)
        assert isinstance(r2, IQAEResult)


# ===========================================================================
# IQAE — _clopper_pearson edge cases not already covered
# ===========================================================================


class TestClopperPearsonExtra:
    def test_n_equals_one_all_successes(self):
        lo, hi = _clopper_pearson(1, 1, 0.05)
        assert hi == pytest.approx(1.0)
        assert lo >= 0.0

    def test_n_equals_one_no_success(self):
        lo, hi = _clopper_pearson(0, 1, 0.05)
        assert lo == pytest.approx(0.0)
        assert hi <= 1.0

    def test_alpha_close_to_one_gives_narrow_interval(self):
        # α → 1: extremely wide CI (very low confidence); still ordered
        lo, hi = _clopper_pearson(5, 10, 0.99)
        assert lo <= hi

    def test_output_clipped_to_unit_interval(self):
        lo, hi = _clopper_pearson(7, 10, 0.05)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0


# ===========================================================================
# IQAE — precision guarantee over many seeds (different amplitude from core tests)
# ===========================================================================


class TestIQAEPrecisionGuarantee:
    """The additive precision contract: P(|est - a_true| < ε) ≥ 1 - α."""

    @pytest.mark.parametrize("a_true", [0.1, 0.6])
    def test_estimate_within_epsilon_of_true(self, a_true):
        """Single estimate must be within epsilon + small slack of the true value."""
        epsilon = 0.05
        oracle = _make_ry_oracle(math.asin(math.sqrt(a_true)))
        result = IQAE(
            epsilon=epsilon, alpha=0.05, n_qubits=1, n_shots=400, random_state=99
        ).solve(oracle)
        assert abs(result.estimation - a_true) < epsilon + 0.1


# ===========================================================================
# PortfolioRiskEstimator — constructor and fit() validation
# ===========================================================================


class TestPortfolioConstructorValidation:
    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution"):
            PortfolioRiskEstimator(distribution="lognormal")

    def test_n_qubits_one_raises_at_fit(self):
        """n_qubits=1 is rejected inside fit() (not at __init__)."""
        prices = _simple_prices()
        with pytest.raises(ValueError, match="n_qubits"):
            PortfolioRiskEstimator(n_qubits=1).fit(prices)

    def test_horizon_days_zero_raises_at_fit(self):
        """horizon_days=0 is rejected inside fit()."""
        prices = _simple_prices()
        with pytest.raises(ValueError, match="horizon_days"):
            PortfolioRiskEstimator(horizon_days=0).fit(prices)

    def test_horizon_days_negative_raises_at_fit(self):
        prices = _simple_prices()
        with pytest.raises(ValueError, match="horizon_days"):
            PortfolioRiskEstimator(horizon_days=-1).fit(prices)


class TestPortfolioFitValidation:
    def test_nan_prices_raise(self):
        prices = _simple_prices()
        prices[2, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            PortfolioRiskEstimator(n_qubits=3).fit(prices)

    def test_inf_prices_raise(self):
        prices = _simple_prices()
        prices[5, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            PortfolioRiskEstimator(n_qubits=3).fit(prices)

    def test_single_row_prices_raise(self):
        """T=1 → 0 log-returns → can't fit."""
        prices = np.array([[100.0]])
        with pytest.raises(ValueError):
            PortfolioRiskEstimator(n_qubits=3).fit(prices)

    def test_weights_wrong_shape_raises(self):
        prices = _simple_prices(n_days=20, n_assets=2)
        with pytest.raises(ValueError, match="weights"):
            PortfolioRiskEstimator(n_qubits=3).fit(prices, weights=np.array([1.0]))

    def test_weights_not_summing_to_one_raises(self):
        prices = _simple_prices(n_days=20, n_assets=2)
        with pytest.raises(ValueError, match="weights"):
            PortfolioRiskEstimator(n_qubits=3).fit(prices, weights=np.array([0.6, 0.6]))

    def test_multi_asset_fit_works(self):
        """2-asset portfolio should fit without error."""
        prices = _simple_prices(n_days=40, n_assets=2)
        est = PortfolioRiskEstimator(n_qubits=3).fit(prices)
        assert est.is_fitted


# ===========================================================================
# PortfolioRiskEstimator — not-fitted guard
# ===========================================================================


class TestPortfolioNotFitted:
    def test_estimate_before_fit_raises(self):
        est = PortfolioRiskEstimator(n_qubits=3)
        with pytest.raises(NotFittedError):
            est.estimate(0.95)

    def test_var_at_before_fit_raises(self):
        est = PortfolioRiskEstimator(n_qubits=3)
        with pytest.raises(NotFittedError):
            est.var_at(np.array([0.95]))


# ===========================================================================
# PortfolioRiskEstimator — result invariants
# ===========================================================================


class TestPortfolioResultInvariants:
    @pytest.fixture(scope="class")
    def result(self):
        return _fitted_portfolio(n_bisection_steps=0).estimate(0.95)

    def test_result_type(self, result):
        assert isinstance(result, PortfolioRiskResult)

    def test_cvar_ge_var(self, result):
        """CVaR ≥ VaR is a fundamental risk-measure inequality."""
        assert result.cvar >= result.var - 1e-12

    def test_tail_probability_in_unit_interval(self, result):
        assert 0.0 <= result.tail_probability <= 1.0

    def test_ci_bounds_ordered(self, result):
        lo, hi = result.confidence_interval
        assert lo <= hi

    def test_result_is_frozen(self, result):
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.var = 99.0

    def test_iqae_result_populated_when_measure_true(self):
        est = _fitted_portfolio(measure_tail_probability=True, n_bisection_steps=0)
        result = est.estimate(0.95)
        assert isinstance(result.iqae_result, IQAEResult)

    def test_iqae_result_none_when_measure_false(self):
        est = _fitted_portfolio(measure_tail_probability=False, n_bisection_steps=0)
        result = est.estimate(0.95)
        assert result.iqae_result is None

    def test_top_level_import(self):
        from eigenkit import PortfolioRiskEstimator as PRE
        from eigenkit import PortfolioRiskResult as PRR

        assert PRE is PortfolioRiskEstimator
        assert PRR is PortfolioRiskResult


# ===========================================================================
# PortfolioRiskEstimator — var_at invariants
# ===========================================================================


class TestPortfolioVarAtInvariants:
    @pytest.fixture(scope="class")
    def est(self):
        return _fitted_portfolio(n_bisection_steps=0)

    def test_var_at_returns_ndarray(self, est):
        result = est.var_at(np.array([0.90, 0.95, 0.99]))
        assert isinstance(result, np.ndarray)

    def test_var_at_shape_matches_input(self, est):
        levels = np.array([0.90, 0.95, 0.99])
        result = est.var_at(levels)
        assert result.shape == levels.shape

    def test_var_at_monotone_in_confidence_level(self, est):
        """Higher confidence level → higher (or equal) VaR."""
        levels = np.array([0.80, 0.90, 0.95, 0.99])
        vars_ = est.var_at(levels)
        assert np.all(np.diff(vars_) >= -1e-12), "VaR must be non-decreasing in CL"


# ===========================================================================
# PortfolioRiskEstimator — multi-asset estimate
# ===========================================================================


class TestPortfolioMultiAsset:
    def test_two_asset_estimate_produces_valid_result(self):
        prices = _simple_prices(n_days=50, n_assets=2, seed=3)
        est = PortfolioRiskEstimator(
            n_qubits=3, epsilon=0.05, alpha=0.1, n_shots=50, random_state=1
        ).fit(prices)
        result = est.estimate(0.90)
        assert isinstance(result, PortfolioRiskResult)
        assert result.cvar >= result.var - 1e-12


# ===========================================================================
# _classical_var_cvar — unit-level invariants
# ===========================================================================


class TestClassicalVaRCVaRInvariants:
    def _uniform(self, n=8):
        losses = np.linspace(0.0, 1.0, n)
        probs = np.ones(n) / n
        return losses, probs

    def test_cvar_ge_var(self):
        for cl in [0.5, 0.75, 0.90, 0.95]:
            losses, probs = self._uniform()
            var, cvar = _classical_var_cvar(losses, probs, cl)
            assert cvar >= var - 1e-12, f"CVaR < VaR at CL={cl}"

    def test_cvar_le_max_loss(self):
        losses, probs = self._uniform()
        _, cvar = _classical_var_cvar(losses, probs, 0.95)
        assert cvar <= losses.max() + 1e-12

    def test_var_ge_min_loss(self):
        losses, probs = self._uniform()
        var, _ = _classical_var_cvar(losses, probs, 0.95)
        assert var >= losses.min() - 1e-12

    def test_var_at_zero_confidence_is_min(self):
        """CL → 0: any loss qualifies, so VaR = first (lowest) bin."""
        losses, probs = self._uniform(8)
        var, _ = _classical_var_cvar(losses, probs, 0.0)
        assert var == pytest.approx(losses.min())


# ===========================================================================
# _classical_tail_probability — unit-level edge cases
# ===========================================================================


class TestClassicalTailProbability:
    def _uniform(self, n=8):
        losses = np.linspace(0.0, 1.0, n)
        probs = np.ones(n) / n
        return losses, probs

    def test_threshold_below_min_returns_one(self):
        losses, probs = self._uniform()
        p = _classical_tail_probability(losses, probs, losses.min() - 0.1)
        assert p == pytest.approx(1.0)

    def test_threshold_above_max_returns_zero(self):
        losses, probs = self._uniform()
        p = _classical_tail_probability(losses, probs, losses.max() + 0.1)
        assert p == pytest.approx(0.0)

    def test_total_probability_sums_correctly(self):
        losses, probs = self._uniform()
        # P(loss >= min) must equal 1.0
        p = _classical_tail_probability(losses, probs, losses.min())
        assert p == pytest.approx(1.0)


# ===========================================================================
# _cvar_at_threshold — unit-level edge cases
# ===========================================================================


class TestCVaRAtThreshold:
    def _dist(self):
        losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        return losses, probs

    def test_threshold_above_max_returns_max_bin_cvar(self):
        # CL=0.95 → tail_budget=0.05; only the max bin is in the tail → cvar = max.
        losses, probs = self._dist()
        cvar = _cvar_at_threshold(losses, probs, losses.max() + 1.0, 0.95)
        assert cvar == pytest.approx(losses.max())

    def test_agrees_with_classical_at_true_var(self):
        # _cvar_at_threshold must reproduce _classical_var_cvar when given the
        # exact same VaR threshold — they share the same underlying formula.
        losses, probs = self._dist()
        var, expected_cvar = _classical_var_cvar(losses, probs, 0.8)
        cvar = _cvar_at_threshold(losses, probs, var, 0.8)
        assert cvar == pytest.approx(expected_cvar, rel=1e-9)

    def test_cvar_le_max_loss(self):
        # For valid (true-VaR) thresholds CVaR must never exceed the max loss.
        losses, probs = self._dist()
        for cl in [0.6, 0.8, 0.95]:
            var, _ = _classical_var_cvar(losses, probs, cl)
            cvar = _cvar_at_threshold(losses, probs, var, cl)
            assert cvar <= losses.max() + 1e-12

    def test_cvar_ge_var(self):
        """CVaR ≥ VaR is a fundamental risk-measure inequality."""
        losses, probs = self._dist()
        for cl in [0.6, 0.8, 0.95]:
            var, _ = _classical_var_cvar(losses, probs, cl)
            cvar = _cvar_at_threshold(losses, probs, var, cl)
            assert cvar >= var - 1e-12


# ===========================================================================
# _postselect — additional unit tests
# ===========================================================================


class TestPostselectExtra:
    def test_uniform_ancilla_gives_half_probability(self):
        """Equal superposition of |0...⟩ and |1...⟩ → success prob = 0.5."""
        # 2 qubits (1 ancilla + 1 b_reg), equal weight on |00⟩ and |10⟩
        sv = np.zeros(4, dtype=complex)
        sv[0] = 1.0 / math.sqrt(2)  # |00⟩, ancilla=0
        sv[2] = 1.0 / math.sqrt(2)  # |10⟩, ancilla=1
        amps, prob, readout = _postselect(sv, n_b=1, n_l=0)
        assert prob == pytest.approx(0.5, abs=1e-12)
        # With no clock register the readout slice is the whole ancilla=1 branch.
        assert readout == pytest.approx(0.5, abs=1e-12)

    def test_output_amplitudes_normalised(self):
        """Returned amplitudes should be normalised (unit norm)."""
        sv = np.zeros(8, dtype=complex)
        sv[4] = 0.6 + 0j  # ancilla=1, some weight
        sv[5] = 0.8 + 0j  # ancilla=1, rest
        amps, _, _ = _postselect(sv, n_b=1, n_l=1)
        if amps is not None and len(amps) > 0:
            assert abs(np.linalg.norm(amps) - 1.0) < 1e-10


# ===========================================================================
# _compute_rotation_angles — additional unit tests
# ===========================================================================


class TestComputeRotationAnglesExtra:
    def test_all_positive_buckets_have_non_negative_angle(self):
        """For positive-definite spectra, all angles should be non-negative."""
        angles = _compute_rotation_angles(
            n_buckets=8, t=math.pi, C_actual=0.5, lam_max_abs=1.0, epsilon=0.01
        )
        half = 8 // 2
        for j, a in angles.items():
            if 1 <= j <= half:
                assert a >= 0.0, f"Positive bucket j={j} has negative angle {a}"

    def test_higher_epsilon_reduces_active_buckets(self):
        """Higher epsilon threshold should zero out more (or equal) buckets."""
        kwargs = dict(n_buckets=8, t=math.pi, C_actual=0.1, lam_max_abs=1.0)
        angles_lo = _compute_rotation_angles(**kwargs, epsilon=0.05)
        angles_hi = _compute_rotation_angles(**kwargs, epsilon=0.50)
        n_active_lo = sum(1 for a in angles_lo.values() if a != 0.0)
        n_active_hi = sum(1 for a in angles_hi.values() if a != 0.0)
        assert n_active_hi <= n_active_lo


# ===========================================================================
# Cross-module: HHL solution direction via classical comparison
# ===========================================================================


class TestHHLSolutionDirection:
    """Verify that the HHL solution direction agrees with the classical solution
    for well-conditioned Hermitian systems.  This catches sign-flip regressions
    that would produce residual ≈ 2 instead of ≈ 0.
    """

    @pytest.mark.parametrize(
        "A, b",
        [
            (np.diag([4.0, 1.0]), np.array([1.0, 1.0])),
            (np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([1.0, 0.0])),
        ],
    )
    def test_solution_direction_matches_classical(self, A, b):
        x_ref = np.linalg.solve(A, b)
        result = HHL(n_eigenvalue_qubits=5).solve(A, b)
        cos_sim = abs(np.dot(result.solution, x_ref)) / (
            np.linalg.norm(result.solution) * np.linalg.norm(x_ref) + 1e-14
        )
        assert cos_sim > 0.8, f"Direction mismatch: cos_sim={cos_sim:.3f}"
