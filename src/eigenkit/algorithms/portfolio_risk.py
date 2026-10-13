from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import NotFittedError

from ..core._results import ValueEquality
from ..core.blocks import AmplitudeEncoding, assemble
from ..core.circuit import ProviderSpec
from ..core.ir import CircuitSpec
from ..qml._base import BaseQuantumEstimator
from ..qml._registry import register_estimator
from ._provider_checks import check_tail_oracle_provider
from .iqae import IQAE, IQAEResult


@dataclass(frozen=True, eq=False)
class PortfolioRiskResult(ValueEquality):
    r"""Outcome of a [`PortfolioRiskEstimator`][eigenkit.algorithms.PortfolioRiskEstimator] run.

    The result is read-only. Assigning to a field raises `FrozenInstanceError`, so
    a mistyped `=` (instead of `==`) cannot change a result. This type holds no
    arrays, so nothing else needs protecting.

    Results compare by value. They are not hashable, so that all result types
    behave in the same way: the types that hold arrays cannot be hashed both
    correctly and efficiently.

    Attributes:
        var: The Value at Risk (VaR) at the requested `confidence_level`, as a
            fraction of the portfolio value lost. For example, `0.12` means a loss
            of 12 %.
        cvar: The Conditional Value at Risk (CVaR), also called Expected
            Shortfall, at the same confidence level.
        tail_probability: The IQAE point estimate of `P(loss ≥ VaR)`.
        confidence_interval: The `(lower, upper)` Clopper-Pearson interval for the
            tail probability, at level `1 - alpha`.
        iqae_result: The full [`IQAEResult`][eigenkit.algorithms.IQAEResult] when
            `measure_tail_probability=True`, and `None` otherwise.
    """

    var: float
    cvar: float
    tail_probability: float
    confidence_interval: tuple[float, float]
    iqae_result: IQAEResult | None


@register_estimator
class PortfolioRiskEstimator(BaseQuantumEstimator):
    """Quantum estimator of portfolio risk.

    It estimates the Value at Risk (VaR) and the Conditional Value at Risk (CVaR) of
    a portfolio. The estimator can use [`IQAE`][eigenkit.algorithms.IQAE] to estimate the tail
    probability on a quantum circuit. The oracle for IQAE is built from the fitted
    return distribution with amplitude encoding.

    !!! note
        This is not a standard scikit-learn regressor. The risk at a confidence
        level comes from `estimate()` and `var_at()`, which take confidence levels
        and not feature vectors, and the estimator has no meaningful scikit-learn
        `score()`. It inherits only from
        [`BaseQuantumEstimator`][eigenkit.qml.BaseQuantumEstimator].

    !!! warning "Not financial advice"
        The estimates come from a model fitted to the price history you pass
        in, and they carry no guarantee. They are not investment advice.

    Args:
        n_qubits: The total number of qubits, including the ancilla. The loss
            distribution is discretised into `2^(n_qubits - 1)` bins. The minimum
            is 2: one data qubit and the ancilla.
        epsilon: The additive precision of the IQAE estimate of the tail
            probability.
        alpha: The failure probability of IQAE. The confidence level is
            `1 − alpha`.
        horizon_days: The investment horizon, in trading days, used to scale daily
            log-returns to the target horizon. The default is `1`, which gives daily
            risk measures.
        provider: The backend that [`IQAE`][eigenkit.algorithms.IQAE] uses: a
            registered name (`"pennylane"`, `"qiskit"`, `"braket"` and so on), a
            config instance, or a backend object. The default is `"numpy"`.

            !!! note
                The Grover operator built internally uses the adjoint of an
                `AmplitudeEncoding` block, so the provider must support the adjoint
                of a state preparation and provide a statevector. The built-in
                `"numpy"` simulator and PennyLane both do, and their results agree
                to the last digit. A backend that does not raises an error when the
                operator is built. Calling [`IQAE`][eigenkit.algorithms.IQAE]
                directly with a `CircuitSpec` based on `QubitUnitary` avoids this
                requirement.
        n_shots: The number of shots per IQAE round. The default is `100`.
        random_state: The seed that makes the results reproducible.
        n_bisection_steps: The number of binary-search steps used to locate the
            VaR with quantum estimates of the tail probability. With `0`, the
            default, the VaR and CVaR are computed classically from the fitted
            bins. A positive value calls `_quantum_var_bisection()`, which refines
            the threshold by running IQAE at each step of the bisection.
        measure_tail_probability: When `True`, the default, `estimate()` runs IQAE
            once after the VaR and CVaR are computed, and reports the quantum
            estimate of the tail in `tail_probability` and `confidence_interval`.
            Set it to `False` to skip that final IQAE call when only the classical
            VaR and CVaR are needed and `n_bisection_steps=0`.
        distribution: The model of the loss distribution that `fit()` uses.

            * `"gaussian"` (the default) fits the mean `μ` and the standard
              deviation `σ` of the scaled log-returns, and integrates the Gaussian
              cumulative distribution function (CDF) over each bin. The
              distribution is truncated at `±6σ`. A VaR beyond that point is
              capped at the limit, and a `RuntimeWarning` is issued.
            * `"empirical"` places each historical return in the nearest bin of
              equal width, without assuming a parametric form.
            * `"t"` fits a Student-t distribution by maximum-likelihood
              estimation, and integrates its CDF over each bin. It has heavier
              tails than the Gaussian.

    Example:
        ```python
        import numpy as np
        from eigenkit.algorithms import PortfolioRiskEstimator

        rng = np.random.default_rng(0)
        returns = rng.normal(0.0005, 0.02, size=(250, 3))
        prices = 100 * np.exp(np.cumsum(returns, axis=0))

        estimator = PortfolioRiskEstimator(
            n_qubits=5, provider="numpy", random_state=0
        )
        estimator.fit(prices, weights=np.array([0.5, 0.3, 0.2]))

        risk = estimator.estimate(confidence_level=0.95)

        print(risk.var)               # 0.0237: at most a 2.37% loss on 95% of days
        print(risk.cvar)              # 0.0262: the average loss on the worst 5%
        print(risk.tail_probability)  # 0.0659: the IQAE estimate of the tail mass
        ```
    """

    _SUPPORTED_DISTRIBUTIONS: frozenset[str] = frozenset({"gaussian", "empirical", "t"})

    def __init__(
        self,
        n_qubits: int = 6,
        epsilon: float = 0.01,
        alpha: float = 0.05,
        horizon_days: int = 1,
        provider: ProviderSpec = "numpy",
        n_shots: int = 100,
        random_state: int | None = 42,
        n_bisection_steps: int = 0,
        measure_tail_probability: bool = True,
        distribution: str = "gaussian",
    ) -> None:
        super().__init__(provider=provider, random_state=random_state)
        if distribution not in self._SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of "
                f"{sorted(self._SUPPORTED_DISTRIBUTIONS)}, got {distribution!r}."
            )
        if n_bisection_steps < 0:
            raise ValueError(f"n_bisection_steps must be >= 0, got {n_bisection_steps}")
        self.n_qubits = n_qubits
        self.epsilon = epsilon
        self.alpha = alpha
        self.horizon_days = horizon_days
        self.n_shots = n_shots
        self.n_bisection_steps = n_bisection_steps
        self.measure_tail_probability = measure_tail_probability
        self.distribution = distribution

        # Pre-fit state — safe to read before fit() via distribution_params
        self._mu: float | None = None
        self._sigma: float | None = None
        self._bin_losses: np.ndarray | None = None
        self._bin_probs: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        weights: np.ndarray | None = None,
    ) -> "PortfolioRiskEstimator":
        """Fit the return distribution to historical price data.

        Args:
            X: A 2-D array of shape `(T, n_assets)` with daily closing prices, where
                `T` must be at least 2. A 1-D array of length `T` is accepted for a
                portfolio with a single asset.
            y: Ignored, because this estimator is unsupervised. It is present for
                compatibility with the scikit-learn interface.
            weights: A 1-D array of portfolio weights that sum to 1. If it is not
                given, all assets have equal weights.

        Raises:
            ValueError: If `distribution` is not one of the supported values. This can
                happen when `distribution` is changed with `set_params()` after
                construction.
        """
        if self.distribution not in self._SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of "
                f"{sorted(self._SUPPORTED_DISTRIBUTIONS)}, got {self.distribution!r}."
            )
        if self.n_qubits < 2:
            raise ValueError(f"n_qubits must be >= 2, got {self.n_qubits}")
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {self.horizon_days}")

        prices = np.asarray(X, dtype=float)
        if prices.ndim == 1:
            prices = prices[:, np.newaxis]
        if prices.ndim != 2:
            raise ValueError("prices must be 1-D or 2-D")
        T, n_assets = prices.shape
        if T < 2:
            raise ValueError("prices must have at least 2 rows")

        if weights is None:
            weights = np.ones(n_assets) / n_assets
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_assets,):
            raise ValueError(
                f"weights shape {weights.shape} does not match n_assets={n_assets}"
            )
        if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-6):
            raise ValueError("weights must sum to 1")

        if not np.all(np.isfinite(prices)):
            raise ValueError(
                "prices contains NaN or Inf values. "
                "Log-returns require finite strictly positive prices."
            )
        if (prices <= 0).any():
            raise ValueError(
                "prices contains non-positive values. Log-returns require "
                "strictly positive prices. Check for zero or negative entries in X."
            )

        log_returns = np.log(prices[1:] / prices[:-1])  # (T-1, n_assets)
        portfolio_returns = log_returns @ weights  # (T-1,)

        h = self.horizon_days
        mu_daily = float(np.mean(portfolio_returns))
        scaled_returns = (portfolio_returns - mu_daily) * math.sqrt(h) + mu_daily * h

        mu = float(np.mean(scaled_returns))
        sigma = float(np.std(scaled_returns, ddof=1))

        if not math.isfinite(sigma):
            raise ValueError(
                f"Cannot estimate volatility: only {T - 1} log-return(s) available "
                f"(T={T} price rows). Provide at least 3 price rows so that "
                "std(ddof=1) is well-defined."
            )

        if sigma < 1e-12:
            raise ValueError(
                "Portfolio returns are constant or numerically near-constant "
                f"(volatility {sigma:.3e} < 1e-12). "
                "The return distribution is degenerate; VaR/CVaR cannot be computed. "
                "Check that prices contain genuinely distinct values."
            )

        n_bins = 2 ** (self.n_qubits - 1)
        if self.distribution == "gaussian":
            bin_losses, bin_probs = _discretise_gaussian(
                mu=mu, sigma=sigma, n_bins=n_bins
            )
        elif self.distribution == "empirical":
            bin_losses, bin_probs = _discretise_empirical(
                returns=scaled_returns, n_bins=n_bins
            )
        else:  # "t"
            bin_losses, bin_probs = _discretise_student_t(
                returns=scaled_returns, n_bins=n_bins
            )

        self.n_features_in_ = n_assets
        self._mu = mu
        self._sigma = sigma
        self._bin_losses = bin_losses
        self._bin_probs = bin_probs

        return self

    def estimate(
        self,
        confidence_level: float = 0.95,
        *,
        _seed_seq: np.random.SeedSequence | None = None,
    ) -> PortfolioRiskResult:
        r"""Estimate the VaR and CVaR at `confidence_level`.

        The VaR and CVaR come from the discretised loss distribution. When
        `n_bisection_steps > 0`, the VaR is refined by a bisection that uses IQAE.
        Otherwise, the classical quantiles of the bins are used. Unless
        `measure_tail_probability=False`, a final IQAE run estimates `P(loss ≥ VaR)`,
        for checking and diagnostics.

        Args:
            confidence_level: The probability level of the risk measures, for example
                `0.95` for the 95 % VaR. It must be in `(0, 1)`.
            _seed_seq: A private override of the random seed, used by `var_at()` so
                that the shot noise of the IQAE runs for different confidence levels is
                independent. The default is `None`, which derives the seed from
                `random_state`.

        Returns:
            A [`PortfolioRiskResult`][eigenkit.algorithms.PortfolioRiskResult].

        Raises:
            RuntimeError: If `fit()` has not been called.
            ValueError: If the selected provider cannot compile the oracle for the
                tail that IQAE needs.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        needs_iqae = self.measure_tail_probability or self.n_bisection_steps > 0
        if needs_iqae:
            check_tail_oracle_provider(self.provider)
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {confidence_level}"
            )
        if self.distribution == "gaussian" and self._sigma is not None:
            from scipy.stats import norm as _norm

            if confidence_level > _norm.cdf(6.0):
                warnings.warn(
                    f"confidence_level={confidence_level} exceeds ±6σ Gaussian support; "
                    "VaR may be truncated. Consider distribution='t' for extreme quantiles.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        var_classical, cvar_classical = _classical_var_cvar(
            self._bin_losses, self._bin_probs, confidence_level
        )

        iqae_kwargs = dict(
            epsilon=self.epsilon,
            alpha=self.alpha,
            n_qubits=self.n_qubits,
            provider=self.provider,
            n_shots=self.n_shots,
        )

        seed_seq = (
            _seed_seq
            if _seed_seq is not None
            else np.random.SeedSequence(self.random_state)
        )
        n_iqae_runs = self.n_bisection_steps + int(self.measure_tail_probability)
        seeds = seed_seq.generate_state(n_iqae_runs) if n_iqae_runs > 0 else np.array([])

        if self.n_bisection_steps > 0:
            var = _quantum_var_bisection(
                bin_losses=self._bin_losses,
                bin_probs=self._bin_probs,
                confidence_level=confidence_level,
                iqae_kwargs=iqae_kwargs,
                seeds=seeds[: self.n_bisection_steps],
            )
            cvar = _cvar_at_threshold(
                self._bin_losses, self._bin_probs, var, confidence_level
            )
            seed_offset = self.n_bisection_steps
        else:
            var, cvar = var_classical, cvar_classical
            seed_offset = 0

        if self.measure_tail_probability:
            oracle = _build_tail_oracle_spec(
                bin_losses=self._bin_losses,
                bin_probs=self._bin_probs,
                threshold=var,
            )
            iqae_result: IQAEResult | None = IQAE(
                **iqae_kwargs, random_state=int(seeds[seed_offset])
            ).solve(oracle=oracle)
            tail_probability = iqae_result.estimation
            confidence_interval = iqae_result.confidence_interval
        else:
            iqae_result = None
            tail_probability = _classical_tail_probability(
                self._bin_losses, self._bin_probs, var
            )
            confidence_interval = (tail_probability, tail_probability)

        return PortfolioRiskResult(
            var=var,
            cvar=cvar,
            tail_probability=tail_probability,
            confidence_interval=confidence_interval,
            iqae_result=iqae_result,
        )

    def var_at(self, confidence_levels: np.ndarray) -> np.ndarray:
        """Return the VaR at each confidence level.

        This is the array version of `estimate()`. It runs IQAE once for each level and
        returns only the `var` of each result.

        Args:
            confidence_levels: A 1-D array of confidence levels in `(0, 1)`, for
                example `[0.95, 0.99]`. A scalar or an `(n, 1)` column is also
                accepted, and it is read as `n` levels.

        Returns:
            An array of VaR values, as fractions of the portfolio value lost, with one
                value per level.

        Raises:
            ValueError: If the argument is not 1-D after a single column is squeezed.
            NotFittedError: If `fit()` has not been called.

        Warns:
            UserWarning: If any level is below `0.5`. Such a level is valid, but values
                below `0.5` are also what a price series or a feature matrix looks
                like, and this warning catches the mistake of passing one of those.

        !!! warning
            Each level runs IQAE once, so `n` levels need `n` separate quantum
            circuits. For a single risk measure, call `estimate()` instead.

        Example:
            ```python
            estimator.var_at([0.90, 0.95, 0.99])
            ```
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        arr = np.asarray(confidence_levels)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(
                f"confidence_levels must be 1-D, got shape {arr.shape}. This "
                "method takes confidence levels, not the price history `fit()` "
                "was given."
            )
        levels = arr
        # `estimate` already rejects anything outside (0, 1), which catches a
        # price series. It cannot catch returns or normalised features, whose
        # values are legal levels and meaningless ones.
        low = levels[levels < 0.5]
        if low.size:
            warnings.warn(
                f"{low.size} confidence level(s) below 0.5, the lowest "
                f"{float(low.min()):.4g}. That is legal — VaR at 0.5 is the "
                "median loss — but it is also what returns or normalised "
                "features look like. This method takes confidence levels, not "
                "the data `fit()` was given.",
                UserWarning,
                stacklevel=2,
            )
        children = np.random.SeedSequence(self.random_state).spawn(len(levels))
        return np.array(
            [
                self.estimate(float(cl), _seed_seq=child).var
                for cl, child in zip(levels, children)
            ]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Not supported. Use `var_at()` instead.

        The estimator base class requires a `predict` method. In scikit-learn,
        `predict` takes samples from the same feature space that `fit` received, but
        the risk at a confidence level is a different question, which `var_at()`
        answers.

        Args:
            X: Ignored.

        Raises:
            NotImplementedError: Always. The message points to `var_at()`.
        """
        raise NotImplementedError(
            "PortfolioRiskEstimator has no predict(): it is not a model of "
            "the feature space fit() was given. Use var_at(confidence_levels) "
            "for the VaR at each level, or estimate(level) for VaR and CVaR."
        )

    def _collect_state(self) -> dict:
        if not self.is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before calling 'save'."
            )
        return {
            "mu": self._mu,
            "sigma": self._sigma,
            "bin_losses": self._bin_losses.tolist(),
            "bin_probs": self._bin_probs.tolist(),
            "n_features_in_": int(self.n_features_in_),
            "distribution": self.distribution,
        }

    def _restore_state(self, state: dict) -> None:
        self._mu = state["mu"]
        self._sigma = state["sigma"]
        self._bin_losses = np.array(state["bin_losses"])
        self._bin_probs = np.array(state["bin_probs"])
        if "n_features_in_" in state:
            self.n_features_in_ = state["n_features_in_"]
        saved_dist = state.get("distribution")
        if saved_dist is not None and saved_dist != self.distribution:
            warnings.warn(
                f"The loaded bundle was fitted with distribution={saved_dist!r}, "
                f"but this instance uses distribution={self.distribution!r}. "
                "The stored bin_losses/bin_probs reflect the original distribution; "
                "call fit() again to re-discretise with the current setting.",
                UserWarning,
                stacklevel=3,
            )

    @property
    def is_fitted(self) -> bool:
        """Whether `fit()` has completed successfully."""
        return getattr(self, "_bin_losses", None) is not None

    @property
    def distribution_params(self) -> dict[str, float] | None:
        """Summary statistics of the fitted return distribution.

        The value is `None` if `fit()` has not been called. Otherwise it always contains `"mu"` (the mean) and `"sigma"` (the standard deviation) of
        the scaled log-returns, whatever the `distribution` model. For `"t"`, the
        degrees of freedom are not stored here, because they are an intermediate
        result of the maximum-likelihood fit and not part of the state of the
        estimator.
        """
        if self._mu is None:
            return None
        return {"mu": self._mu, "sigma": self._sigma}  # type: ignore[dict-item]

    def _more_tags(self) -> dict:
        # sklearn 1.3–1.5: signal that fit() requires strictly positive X.
        return {"positive_only": True}

    def __sklearn_tags__(self):
        # sklearn 1.6+: same signal via the structured Tags API.
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags


def _discretise_gaussian(
    mu: float, sigma: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Discretise a Gaussian loss distribution onto *n_bins* equal-width bins.

    Bins span [mu − 6σ, mu + 6σ]; probabilities are Gaussian CDF integrals
    over each bin, renormalised to sum to 1 (±6σ truncation discards < 2e-9
    of mass).
    """
    from scipy.stats import norm  # lazy import

    lo = mu - 6.0 * sigma
    hi = mu + 6.0 * sigma
    edges = np.linspace(lo, hi, n_bins + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0

    probs = norm.cdf(edges[1:], loc=mu, scale=sigma) - norm.cdf(
        edges[:-1], loc=mu, scale=sigma
    )

    probs = probs / probs.sum()

    losses = -midpoints
    return losses.astype(float), probs.astype(float)


def _discretise_empirical(
    returns: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretise the empirical return distribution onto *n_bins* equal-width bins.

    Bins span ``[min(returns), max(returns)]``; each sample is placed into the
    nearest bin and counts are normalised to a PMF.  Raises ``ValueError`` for
    fewer than 2 observations or a degenerate (constant) return series.
    """
    if len(returns) < 2:
        raise ValueError(
            "empirical distribution requires at least 2 return observations, "
            f"got {len(returns)}."
        )

    lo, hi = float(returns.min()), float(returns.max())
    if math.isclose(lo, hi, rel_tol=1e-9):
        raise ValueError(
            "All portfolio returns are identical — empirical distribution is "
            "degenerate.  VaR/CVaR cannot be computed."
        )

    edges = np.linspace(lo, hi, n_bins + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0

    indices = np.clip(np.digitize(returns, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins).astype(float)
    probs = counts / counts.sum()

    losses = -midpoints
    return losses.astype(float), probs.astype(float)


def _discretise_student_t(
    returns: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretise a Student-t loss distribution onto *n_bins* equal-width bins.

    Fits location, scale, and degrees-of-freedom via MLE (``scipy.stats.t.fit``).
    Bins span [μ − 6σ, μ + 6σ], with σ the fitted scale, as for the Gaussian
    model.  Raises ``ValueError`` for fewer than 3 observations.
    """
    from scipy.stats import t as student_t  # lazy import

    if len(returns) < 3:
        raise ValueError(
            "Student-t distribution requires at least 3 return observations "
            f"for stable MLE, got {len(returns)}."
        )

    df, loc, scale = student_t.fit(returns)

    lo = loc - 6.0 * scale
    hi = loc + 6.0 * scale
    edges = np.linspace(lo, hi, n_bins + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0

    probs = student_t.cdf(edges[1:], df, loc=loc, scale=scale) - student_t.cdf(
        edges[:-1], df, loc=loc, scale=scale
    )
    probs = probs / probs.sum()

    losses = -midpoints
    return losses.astype(float), probs.astype(float)


def _cvar_from_var_idx(
    sorted_losses: np.ndarray,
    sorted_probs: np.ndarray,
    var_idx: int,
    confidence_level: float,
) -> float:
    """Tail expectation (CVaR) given the VaR bin index in ascending-sorted arrays.

    Integrates the loss over the tail strictly above the VaR bin, then adds the
    slice of the VaR bin needed to fill the ``1 - confidence_level`` tail budget,
    and averages over that budget. Shared by the classical-quantile and the
    quantum-bisection paths so both apply one CVaR definition.
    """
    var = float(sorted_losses[var_idx])
    tail_budget = 1.0 - confidence_level
    above_mask = sorted_losses > var
    above_weighted = float((sorted_losses[above_mask] * sorted_probs[above_mask]).sum())
    above_total = float(sorted_probs[above_mask].sum())
    var_weight = float(np.clip(tail_budget - above_total, 0.0, sorted_probs[var_idx]))
    numerator = above_weighted + var * var_weight
    return numerator / tail_budget if tail_budget > 0 else var


def _classical_var_cvar(
    losses: np.ndarray, probs: np.ndarray, confidence_level: float
) -> tuple[float, float]:
    """Compute VaR and CVaR from a discrete loss distribution."""
    idx = np.argsort(losses)
    sorted_losses = losses[idx]
    sorted_probs = probs[idx]
    cumulative = np.cumsum(sorted_probs)

    var_idx = int(np.searchsorted(cumulative, confidence_level))
    var_idx = min(var_idx, len(sorted_losses) - 1)
    var = float(sorted_losses[var_idx])
    cvar = _cvar_from_var_idx(sorted_losses, sorted_probs, var_idx, confidence_level)
    return var, cvar


def _classical_tail_probability(
    losses: np.ndarray, probs: np.ndarray, threshold: float
) -> float:
    """Return P(loss ≥ threshold) from the discretised distribution."""
    mask = losses >= threshold
    return float(probs[mask].sum())


def _cvar_at_threshold(
    losses: np.ndarray, probs: np.ndarray, threshold: float, confidence_level: float
) -> float:
    """Compute CVaR from a discrete loss distribution at a pre-computed VaR threshold."""
    idx = np.argsort(losses)
    sorted_losses = losses[idx]
    sorted_probs = probs[idx]

    # Snap the continuous bisection threshold to the NEAREST discrete loss bin,
    # not the largest bin below it: the closest bin is the VaR bucket. Flooring
    # would, when the threshold lands just under a bin edge, pick a bin one step
    # too low and let the tail exceed the budget, inflating CVaR.
    var_idx = int(np.argmin(np.abs(sorted_losses - threshold)))
    return _cvar_from_var_idx(sorted_losses, sorted_probs, var_idx, confidence_level)


def _quantum_var_bisection(
    bin_losses: np.ndarray,
    bin_probs: np.ndarray,
    confidence_level: float,
    iqae_kwargs: dict,
    seeds: np.ndarray,
) -> float:
    r"""Binary search for VaR threshold using IQAE tail-probability estimates.

    Searches for the smallest threshold ``t`` such that
    \(P(\text{loss} \ge t) \approx 1 - \text{confidence\_level}\). Each bisection step
    runs an independent IQAE instance; ``iqae_kwargs`` must not include ``random_state``
    (passed separately via ``seeds``).
    """
    lo = float(bin_losses.min())
    hi = float(bin_losses.max())
    target = 1.0 - confidence_level

    for seed in seeds:
        mid = (lo + hi) / 2.0
        oracle = _build_tail_oracle_spec(bin_losses, bin_probs, threshold=mid)
        p_tail = IQAE(**iqae_kwargs, random_state=int(seed)).solve(oracle).estimation
        if p_tail > target:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def _build_tail_oracle_spec(
    bin_losses: np.ndarray,
    bin_probs: np.ndarray,
    threshold: float,
) -> CircuitSpec:
    """Return a provider-agnostic [`CircuitSpec`][eigenkit.core.ir.CircuitSpec]
    that encodes P(loss ≥ threshold) via amplitude encoding.

    The oracle implements:

    ```text
    A|0⟩ = Σ_x √p(x) |x⟩|0⟩  (for loss(x) < threshold)
         + Σ_x √p(x) |x⟩|1⟩  (for loss(x) ≥ threshold)
    ```

    so that measuring the ancilla |1⟩ gives probability equal to the tail
    probability P(loss ≥ threshold).

    The state vector is pre-normalised; ``AmplitudeEncoding`` is called with
    ``normalize=False`` so no rounding occurs inside the IR block.
    """
    n_bins = len(bin_probs)
    amps = np.sqrt(bin_probs.astype(float))
    tail_mask = bin_losses >= threshold
    indices = 2 * np.arange(n_bins) + tail_mask.astype(int)
    state_vector = np.zeros(2 * n_bins, dtype=complex)
    state_vector[indices] = amps

    norm = float(np.linalg.norm(state_vector))
    if norm > 0:
        state_vector /= norm

    return assemble([AmplitudeEncoding(state_vector, normalize=False)])
