"""Tests for PortfolioRiskEstimator."""

from __future__ import annotations

import warnings

import numpy as np
import pytest


pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from eigenkit.algorithms.portfolio_risk import (  # noqa: E402
    PortfolioRiskEstimator,
    _classical_tail_probability,
    _classical_var_cvar,
    _cvar_at_threshold,
    _discretise_gaussian,
)
from eigenkit.qml._base import BaseQuantumEstimator  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_prices(n_days: int = 30, seed: int = 0) -> np.ndarray:
    """Generate synthetic strictly-positive 1-asset price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.01, size=n_days - 1)
    prices = np.cumprod(np.concatenate([[100.0], np.exp(returns)]))
    return prices[:, np.newaxis]  # (n_days, 1)


def _fitted_estimator(**kwargs) -> PortfolioRiskEstimator:
    defaults = dict(n_qubits=3, epsilon=0.05, alpha=0.1, n_shots=50, random_state=0)
    defaults.update(kwargs)
    est = PortfolioRiskEstimator(**defaults)
    est.fit(_simple_prices())
    return est


# ---------------------------------------------------------------------------
# sklearn estimator base (not a regressor)
# ---------------------------------------------------------------------------


class TestEstimatorBase:
    def test_is_base_quantum_estimator(self):
        est = PortfolioRiskEstimator()
        assert isinstance(est, BaseQuantumEstimator)

    def test_not_regressor_mixin(self):
        from sklearn.base import RegressorMixin

        assert not isinstance(PortfolioRiskEstimator(), RegressorMixin)


# ---------------------------------------------------------------------------
# pre-fit state initialised to None
# ---------------------------------------------------------------------------


class TestPreFitState:
    def test_distribution_params_none_before_fit(self):
        est = PortfolioRiskEstimator()
        assert est.distribution_params is None

    def test_is_fitted_false_before_fit(self):
        est = PortfolioRiskEstimator()
        assert not est.is_fitted

    def test_distribution_params_populated_after_fit(self):
        est = _fitted_estimator()
        params = est.distribution_params
        assert params is not None
        assert "mu" in params and "sigma" in params


# ---------------------------------------------------------------------------
# var_at() accepts 1-D, 2-D column and scalar levels
# ---------------------------------------------------------------------------


class TestVarAtShape:
    def test_1d_input(self):
        est = _fitted_estimator()
        result = est.var_at(np.array([0.95, 0.99]))
        assert result.shape == (2,)

    def test_2d_column_input(self):
        est = _fitted_estimator()
        X = np.array([[0.95], [0.99]])
        result = est.var_at(X)
        assert result.shape == (2,)

    def test_scalar_input(self):
        est = _fitted_estimator()
        result = est.var_at(0.95)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# non-positive prices raise ValueError
# ---------------------------------------------------------------------------


class TestNonPositivePrices:
    def test_zero_price_raises(self):
        prices = _simple_prices()
        prices[5, 0] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            PortfolioRiskEstimator().fit(prices)

    def test_negative_price_raises(self):
        prices = _simple_prices()
        prices[3, 0] = -1.0
        with pytest.raises(ValueError, match="non-positive"):
            PortfolioRiskEstimator().fit(prices)

    def test_positive_prices_pass(self):
        prices = _simple_prices()
        assert (prices > 0).all()
        PortfolioRiskEstimator(n_qubits=3).fit(prices)  # should not raise


# ---------------------------------------------------------------------------
# constant prices (sigma=0) raise ValueError
# ---------------------------------------------------------------------------


class TestConstantPrices:
    def test_constant_prices_raise(self):
        prices = np.ones((20, 1)) * 100.0
        with pytest.raises(ValueError, match="constant"):
            PortfolioRiskEstimator().fit(prices)

    def test_near_constant_prices_raise(self):
        # Prices differing only by float rounding give sigma ~ 1e-14; an
        # exact-equality guard (sigma == 0.0) would let this through and fit
        # a Gaussian over bins of width ~1e-14.
        prices = 100.0 * np.array([1.0, 1.0 + 2e-14, 1.0, 1.0 + 2e-14, 1.0])
        with pytest.raises(ValueError, match="near-constant"):
            PortfolioRiskEstimator(n_qubits=3).fit(prices[:, np.newaxis])


# ---------------------------------------------------------------------------
# Independent seeds for the IQAE runs in bisection / var_at
# ---------------------------------------------------------------------------


class TestIndependentSeeds:
    """Every IQAE call in the bisection (and the final tail estimate) gets
    its own seed. Passing them all the same raw random_state would make every
    run draw the identical binomial stream, correlating the bracket updates."""

    def _record_seeds(self, monkeypatch):
        from eigenkit.algorithms import portfolio_risk as pr

        recorded = []
        orig = pr.IQAE.solve

        def spy(self, oracle):
            recorded.append(self.random_state)
            return orig(self, oracle)

        monkeypatch.setattr(pr.IQAE, "solve", spy)
        return recorded

    def test_bisection_steps_use_distinct_seeds(self, monkeypatch):
        recorded = self._record_seeds(monkeypatch)
        est = _fitted_estimator(n_bisection_steps=3)
        est.estimate(0.9)
        # 3 bisection steps + 1 final tail estimate
        assert len(recorded) == 4
        assert len(set(recorded)) == 4

    def test_var_at_levels_use_distinct_seeds(self, monkeypatch):
        recorded = self._record_seeds(monkeypatch)
        est = _fitted_estimator(n_bisection_steps=0)
        est.var_at(np.array([0.9, 0.95]))
        assert len(recorded) == 2
        assert len(set(recorded)) == 2

    def test_reproducible_with_fixed_seed(self):
        r1 = _fitted_estimator(n_bisection_steps=2).estimate(0.9)
        r2 = _fitted_estimator(n_bisection_steps=2).estimate(0.9)
        assert r1.var == r2.var
        assert r1.tail_probability == r2.tail_probability

    def test_var_at_reproducible_with_fixed_seed(self):
        p1 = _fitted_estimator(n_bisection_steps=1).var_at(np.array([0.9, 0.95]))
        p2 = _fitted_estimator(n_bisection_steps=1).var_at(np.array([0.9, 0.95]))
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Atomic fit: a failed refit must not corrupt the fitted state
# ---------------------------------------------------------------------------


class TestFailedRefitAtomicity:
    def test_failed_refit_preserves_previous_state(self):
        # fit() must validate before it writes n_features_in_/_mu/_sigma;
        # writing first would leave a still-"fitted" estimator with mixed
        # metadata and sigma=NaN after a failed second fit.
        est = PortfolioRiskEstimator(n_qubits=3).fit(
            np.array([[100.0], [101.0], [103.0], [102.0]])
        )
        params_before = dict(est.distribution_params)
        bins_before = est._bin_losses.copy()
        assert est.n_features_in_ == 1

        # Second fit: 2 price rows → 1 log-return → std(ddof=1) is NaN → raises
        with pytest.raises(ValueError, match="Cannot estimate volatility"):
            est.fit(np.array([[100.0, 200.0], [101.0, 201.0]]))

        assert est.is_fitted
        assert est.n_features_in_ == 1
        assert est.distribution_params == params_before
        np.testing.assert_array_equal(est._bin_losses, bins_before)

    def test_failed_first_fit_leaves_unfitted(self):
        est = PortfolioRiskEstimator(n_qubits=3)
        with pytest.raises(ValueError, match="Cannot estimate volatility"):
            est.fit(np.array([[100.0], [101.0]]))
        assert not est.is_fitted
        assert est.distribution_params is None
        assert not hasattr(est, "n_features_in_")


# ---------------------------------------------------------------------------
# horizon scaling uses sqrt(horizon_days)
# ---------------------------------------------------------------------------


class TestHorizonScaling:
    def test_mu_scales_linearly_with_horizon(self):
        # The square-root-of-time rule applies to volatility only; the mean
        # of i.i.d. log-returns scales linearly with the horizon.  Scaling
        # the returns themselves by sqrt(h) would bias mu by a factor sqrt(h).
        prices = _simple_prices(50)
        est1 = PortfolioRiskEstimator(n_qubits=3, horizon_days=1).fit(prices)
        est4 = PortfolioRiskEstimator(n_qubits=3, horizon_days=4).fit(prices)
        mu1 = est1.distribution_params["mu"]
        mu4 = est4.distribution_params["mu"]
        assert mu4 == pytest.approx(4.0 * mu1, rel=1e-9)

    def test_horizon_one_is_identity(self):
        # h=1 must reproduce the unscaled daily distribution exactly.
        prices = _simple_prices(50)
        est = PortfolioRiskEstimator(n_qubits=3, horizon_days=1).fit(prices)
        log_returns = np.log(prices[1:, 0] / prices[:-1, 0])
        assert est.distribution_params["mu"] == pytest.approx(
            float(np.mean(log_returns)), rel=1e-12
        )
        assert est.distribution_params["sigma"] == pytest.approx(
            float(np.std(log_returns, ddof=1)), rel=1e-12
        )

    def test_sigma_scales_with_sqrt_horizon(self):
        prices = _simple_prices(50)
        est1 = PortfolioRiskEstimator(n_qubits=3, horizon_days=1).fit(prices)
        est4 = PortfolioRiskEstimator(n_qubits=3, horizon_days=4).fit(prices)
        sigma1 = est1.distribution_params["sigma"]
        sigma4 = est4.distribution_params["sigma"]
        # sqrt(4) / sqrt(1) = 2 — sigma should double
        assert abs(sigma4 / sigma1 - 2.0) < 0.01


# ---------------------------------------------------------------------------
# _discretise_gaussian uses equal-width bins
# ---------------------------------------------------------------------------


class TestDiscretiseGaussian:
    def test_output_shape(self):
        losses, probs = _discretise_gaussian(mu=0.0, sigma=0.01, n_bins=8)
        assert losses.shape == (8,)
        assert probs.shape == (8,)

    def test_probs_sum_to_one(self):
        _, probs = _discretise_gaussian(mu=0.0, sigma=0.01, n_bins=16)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_bins_are_equal_width(self):
        losses, _ = _discretise_gaussian(mu=0.0, sigma=0.01, n_bins=8)
        diffs = np.diff(losses)
        assert np.allclose(diffs, diffs[0], atol=1e-10), "bins are not equal-width"

    def test_probs_are_non_negative(self):
        _, probs = _discretise_gaussian(mu=0.0, sigma=0.02, n_bins=16)
        assert (probs >= 0).all()


# ---------------------------------------------------------------------------
# n_features_in_ survives save/load round-trip
# ---------------------------------------------------------------------------


class TestNFeaturesInLoad:
    def test_n_features_in_preserved_after_load(self, tmp_path):
        """n_features_in_ must be set correctly after loading a saved bundle."""
        est = _fitted_estimator()
        assert hasattr(est, "n_features_in_")
        expected = est.n_features_in_

        bundle_path = tmp_path / "portfolio.eigenkit"
        est.save(bundle_path)

        loaded = PortfolioRiskEstimator.load(bundle_path)
        assert hasattr(loaded, "n_features_in_"), (
            "n_features_in_ missing after load — add it to _collect_state/_restore_state"
        )
        assert loaded.n_features_in_ == expected


# ---------------------------------------------------------------------------
# quantum VaR via bisection
# ---------------------------------------------------------------------------


class TestQuantumVaR:
    def test_default_bisection_zero_is_classical(self):
        """n_bisection_steps=0 must match the explicit classical VaR/CVaR."""
        est = _fitted_estimator(n_bisection_steps=0)
        result = est.estimate(0.95)
        var_c, cvar_c = _classical_var_cvar(est._bin_losses, est._bin_probs, 0.95)
        assert result.var == var_c
        assert result.cvar == cvar_c

    def test_quantum_var_close_to_classical(self):
        """Quantum bisection VaR must lie inside the final bracket around classical VaR."""
        est_q = _fitted_estimator(n_bisection_steps=3, epsilon=0.05)
        est_c = _fitted_estimator(n_bisection_steps=0, epsilon=0.05)
        r_q = est_q.estimate(0.95)
        r_c = est_c.estimate(0.95)
        lo = float(est_q._bin_losses.min())
        hi = float(est_q._bin_losses.max())
        bracket_half = (hi - lo) / (2**est_q.n_bisection_steps)
        assert abs(r_q.var - r_c.var) <= bracket_half + est_q.epsilon

    def test_n_bisection_steps_stored(self):
        """n_bisection_steps is exposed as an sklearn parameter."""
        est = PortfolioRiskEstimator(n_bisection_steps=5)
        assert est.n_bisection_steps == 5

    def test_measure_tail_probability_false_skips_iqae(self):
        est = _fitted_estimator(measure_tail_probability=False)
        result = est.estimate(0.95)
        assert result.iqae_result is None
        expected_tail = _classical_tail_probability(
            est._bin_losses, est._bin_probs, result.var
        )
        assert result.tail_probability == pytest.approx(expected_tail)

    def test_bisection_returns_portfolio_risk_result(self):
        """estimate() with bisection must still return a PortfolioRiskResult."""
        from eigenkit.algorithms.portfolio_risk import PortfolioRiskResult

        est = _fitted_estimator(n_bisection_steps=2, epsilon=0.05)
        result = est.estimate(0.95)
        assert isinstance(result, PortfolioRiskResult)
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)
        assert result.cvar >= result.var, "CVaR is a tail mean beyond VaR"
        assert 0.0 <= result.tail_probability <= 1.0


# ---------------------------------------------------------------------------
# distribution parameter
# ---------------------------------------------------------------------------


class TestDistributionParameter:
    """Verify all three distribution models produce valid bin distributions."""

    PRICES = _simple_prices(n_days=60, seed=7)

    @pytest.mark.parametrize("dist", ["gaussian", "empirical", "t"])
    def test_fit_does_not_raise(self, dist):
        est = PortfolioRiskEstimator(n_qubits=3, distribution=dist)
        est.fit(self.PRICES)
        assert est.is_fitted

    @pytest.mark.parametrize("dist", ["gaussian", "empirical", "t"])
    def test_bin_probs_sum_to_one(self, dist):
        est = PortfolioRiskEstimator(n_qubits=3, distribution=dist)
        est.fit(self.PRICES)
        assert abs(est._bin_probs.sum() - 1.0) < 1e-9

    @pytest.mark.parametrize("dist", ["gaussian", "empirical", "t"])
    def test_bin_shapes_match_n_qubits(self, dist):
        n_qubits = 4
        est = PortfolioRiskEstimator(n_qubits=n_qubits, distribution=dist)
        est.fit(self.PRICES)
        expected_bins = 2 ** (n_qubits - 1)
        assert est._bin_losses.shape == (expected_bins,)
        assert est._bin_probs.shape == (expected_bins,)

    @pytest.mark.parametrize("dist", ["gaussian", "empirical", "t"])
    def test_distribution_params_always_set(self, dist):
        """distribution_params (mu, sigma) must be available for all models."""
        est = PortfolioRiskEstimator(n_qubits=3, distribution=dist)
        est.fit(self.PRICES)
        params = est.distribution_params
        assert params is not None
        assert "mu" in params and "sigma" in params

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution must be one of"):
            PortfolioRiskEstimator(distribution="normal")

    def test_empirical_different_from_gaussian(self):
        """Empirical and Gaussian bin_probs should differ for real data."""
        est_g = PortfolioRiskEstimator(n_qubits=4, distribution="gaussian").fit(
            self.PRICES
        )
        est_e = PortfolioRiskEstimator(n_qubits=4, distribution="empirical").fit(
            self.PRICES
        )
        assert not np.allclose(est_g._bin_probs, est_e._bin_probs)

    def test_t_different_from_gaussian(self):
        """Student-t and Gaussian bin_probs should differ (t has heavier tails)."""
        est_g = PortfolioRiskEstimator(n_qubits=4, distribution="gaussian").fit(
            self.PRICES
        )
        est_t = PortfolioRiskEstimator(n_qubits=4, distribution="t").fit(self.PRICES)
        assert not np.allclose(est_g._bin_probs, est_t._bin_probs)


class TestDistributionSaveLoad:
    """distribution setting survives a save/load round-trip."""

    def test_distribution_preserved_after_load(self, tmp_path):
        prices = _simple_prices(n_days=40, seed=3)
        for dist in ("gaussian", "empirical", "t"):
            est = PortfolioRiskEstimator(n_qubits=3, distribution=dist).fit(prices)
            path = tmp_path / f"portfolio_{dist}.eigenkit"
            est.save(path)
            loaded = PortfolioRiskEstimator.load(path)
            assert loaded.distribution == dist
            assert np.allclose(loaded._bin_probs, est._bin_probs)
            assert np.allclose(loaded._bin_losses, est._bin_losses)

    def test_distribution_mismatch_warns(self, tmp_path):
        """_restore_state must warn when state distribution mismatches self.distribution."""
        prices = _simple_prices(n_days=40, seed=3)
        est = PortfolioRiskEstimator(n_qubits=3, distribution="gaussian").fit(prices)
        path = tmp_path / "portfolio_gaussian.eigenkit"
        est.save(path)

        # Simulate loading a bundle into an instance with a different distribution
        # by patching the distribution attribute before calling _restore_state.
        loaded = PortfolioRiskEstimator(n_qubits=3, distribution="gaussian").fit(prices)
        loaded.distribution = "empirical"  # mis-set to trigger the mismatch warning

        state = {
            "mu": est._mu,
            "sigma": est._sigma,
            "bin_losses": est._bin_losses.tolist(),
            "bin_probs": est._bin_probs.tolist(),
            "n_features_in_": est.n_features_in_,
            "distribution": "gaussian",
        }
        with pytest.warns(UserWarning, match="distribution="):
            loaded._restore_state(state)


class TestCvarThresholdSnapping:
    """_cvar_at_threshold must snap the continuous bisection
    threshold to the NEAREST discrete loss bin, not the largest bin below it.

    Flooring a threshold that lands just under a bin edge picks a bin one step
    too low, so the tail above it exceeds the (1 - confidence_level) budget and
    CVaR is overstated. The classical-quantile path is the trusted reference.
    """

    _LOSSES = np.array([-0.05, -0.02, 0.0, 0.02, 0.05, 0.08, 0.12, 0.20])
    _PROBS = np.array([0.30, 0.25, 0.20, 0.12, 0.07, 0.04, 0.015, 0.005])
    _CL = 0.95  # VaR bin is 0.08 (cumulative crosses 0.95 there)

    def test_threshold_just_below_bin_edge_snaps_to_nearest(self):
        _, cvar_ref = _classical_var_cvar(self._LOSSES, self._PROBS, self._CL)
        # A threshold a hair below the VaR bin (0.08) must snap to 0.08, not
        # floor to 0.05; flooring would return ~0.12 here.
        cvar = _cvar_at_threshold(self._LOSSES, self._PROBS, 0.08 - 1e-6, self._CL)
        assert cvar == pytest.approx(cvar_ref, abs=1e-12)

    def test_threshold_just_above_bin_edge_snaps_to_same_bin(self):
        _, cvar_ref = _classical_var_cvar(self._LOSSES, self._PROBS, self._CL)
        cvar = _cvar_at_threshold(self._LOSSES, self._PROBS, 0.08 + 1e-6, self._CL)
        assert cvar == pytest.approx(cvar_ref, abs=1e-12)

    def test_cvar_at_least_var_on_snap(self):
        # Coherence: the snapped CVaR must never fall below VaR.
        var_ref, _ = _classical_var_cvar(self._LOSSES, self._PROBS, self._CL)
        cvar = _cvar_at_threshold(self._LOSSES, self._PROBS, 0.08 - 1e-6, self._CL)
        assert cvar >= var_ref - 1e-12


class TestVarAtContract:
    """`var_at` takes confidence levels; `fit` takes prices. They differ.

    It is not called `predict`, which in sklearn means "samples from the space
    fit was given". Three guards keep the wrong array out — the
    class is not a regressor, the shape check rejects a price matrix, and
    `estimate` rejects anything outside (0, 1), which a price series is. The
    one array that slips through all three is returns or normalised features,
    whose values are legal levels and meaningless ones, so `var_at` warns.
    """

    @staticmethod
    def _fitted():
        rng = np.random.default_rng(0)
        prices = np.cumprod(1 + rng.normal(0, 0.01, size=(60, 2)), axis=0) * 100
        return PortfolioRiskEstimator(n_qubits=4, provider="numpy").fit(prices), prices

    def test_real_levels_pass_without_a_warning(self):
        est, _ = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = est.var_at([0.90, 0.95, 0.99])
        assert out.shape == (3,)

    def test_a_level_below_half_warns(self):
        est, _ = self._fitted()
        with pytest.warns(UserWarning, match="below 0.5"):
            est.var_at([0.3, 0.95])

    def test_returns_look_like_levels_and_are_caught_by_the_warning(self):
        # The only input that clears every hard guard.
        est, prices = self._fitted()
        returns = np.abs(np.diff(prices[:, 0]) / prices[:-1, 0])[:5]
        assert (returns > 0).all() and (returns < 1).all()
        with pytest.warns(UserWarning, match="returns or normalised features"):
            est.var_at(returns)

    def test_a_price_matrix_is_rejected_on_shape(self):
        est, prices = self._fitted()
        with pytest.raises(ValueError, match="not the price history"):
            est.var_at(prices)

    def test_a_price_series_is_rejected_on_range(self):
        est, prices = self._fitted()
        with pytest.raises(ValueError, match=r"confidence_level must be in \(0, 1\)"):
            est.var_at(prices[:, 0])

    def test_predict_points_to_var_at(self):
        # An sklearn predict takes samples from the space fit() was given;
        # this estimator answers for confidence levels, through var_at().
        est, _ = self._fitted()
        with pytest.raises(NotImplementedError, match=r"var_at\(confidence_levels\)"):
            est.predict([0.95])

    def test_the_class_does_not_claim_to_be_a_regressor(self):
        # What keeps sklearn tooling from reaching var_at with the wrong space.
        from sklearn.base import is_classifier, is_regressor

        est = PortfolioRiskEstimator(n_qubits=4)
        assert not is_regressor(est)
        assert not is_classifier(est)
        assert not hasattr(est, "score")
