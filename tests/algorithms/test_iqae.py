"""Tests for the IQAE quantum amplitude estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest


pytest.importorskip("scipy")

from eigenkit.algorithms.iqae import (
    IQAE,
    IQAEResult,
    _check_amplitude_encoding_adjoint,
    _check_iqae_provider,
    _clopper_pearson,
    _find_next_k,
    _update_theta_interval,
)
from eigenkit.core.blocks import AmplitudeEncoding, QubitUnitary, assemble
from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef


# ---------------------------------------------------------------------------
# Unit tests: _clopper_pearson
# ---------------------------------------------------------------------------


class TestClopperPearson:
    def test_all_successes_upper_bound_is_one(self):
        lo, hi = _clopper_pearson(10, 10, 0.05)
        assert hi == pytest.approx(1.0)
        assert 0.0 < lo < 1.0

    def test_no_successes_lower_bound_is_zero(self):
        lo, hi = _clopper_pearson(0, 10, 0.05)
        assert lo == pytest.approx(0.0)
        assert 0.0 < hi < 1.0

    def test_interval_contains_true_proportion(self):
        # 7 successes in 10 trials, true p=0.7 should be inside a 95% CI
        lo, hi = _clopper_pearson(7, 10, 0.05)
        assert lo < 0.7 < hi

    def test_bounds_ordered(self):
        lo, hi = _clopper_pearson(3, 10, 0.05)
        assert lo <= hi

    def test_narrower_at_more_trials(self):
        lo10, hi10 = _clopper_pearson(5, 10, 0.05)
        lo100, hi100 = _clopper_pearson(50, 100, 0.05)
        assert (hi10 - lo10) > (hi100 - lo100)


# ---------------------------------------------------------------------------
# Unit tests: _update_theta_interval
# ---------------------------------------------------------------------------


class TestUpdateThetaInterval:
    def _pi2(self):
        return math.pi / 2.0

    def test_exact_measurement_narrows(self):
        # With factor=1, p = sin²(θ) exactly — zero-width p-interval should
        # pin θ to a single value (or at least shrink the window).
        theta_true = math.pi / 6.0  # a = sin²(π/6) = 0.25
        p = math.sin(theta_true) ** 2
        lo, hi = _update_theta_interval(p, p, 1, 0.0, self._pi2())
        # Result interval must contain the true theta
        assert lo <= theta_true + 1e-9
        assert hi >= theta_true - 1e-9
        # And must be strictly narrower than the original [0, π/2]
        assert (hi - lo) < self._pi2()

    def test_full_uncertainty_returns_valid_subinterval(self):
        # p_lo=0, p_hi=1 is maximally uninformative, but the function still
        # picks the narrowest intersecting branch (which may be a degenerate
        # point on a descending branch).  We only assert the result is a valid
        # sub-interval of [0, π/2].
        lo, hi = _update_theta_interval(0.0, 1.0, 1, 0.0, self._pi2())
        assert lo <= hi
        assert 0.0 <= lo
        assert hi <= self._pi2() + 1e-12

    def test_no_matching_branch_fallback(self):
        # Impossible measurement (p_lo > p_hi after clamp shouldn't normally
        # occur, but if no branch intersects we return old bounds)
        lo, hi = _update_theta_interval(0.99, 1.0, 100, 0.0, 0.01)
        # Either returns old bounds or a valid narrowed interval
        assert lo <= hi

    def test_exact_boundary_updates_keep_bounds_ordered(self):
        # An exact (zero-width) measurement can admit a candidate whose raw
        # bounds are inverted by ~1e-16 within the 1e-12 acceptance tolerance;
        # the returned interval must still satisfy lo <= hi.
        theta = 0.1
        lo, hi = 0.0, math.pi / 2
        for factor, phase in [(1, theta), (3, 3 * theta)]:
            p = math.sin(phase) ** 2
            lo, hi = _update_theta_interval(p, p, factor, lo, hi)
            assert lo <= hi
            assert lo - 1e-12 <= theta <= hi + 1e-12

    def test_narrowest_branch_selected(self):
        # High amplification factor → many branches in [0, π/2].
        # The returned interval must be the *narrowest* candidate,
        # i.e. no other intersecting branch is smaller.
        theta_true = math.pi / 8.0
        factor = 7
        p = math.sin(factor * theta_true) ** 2
        eps = 0.02
        lo, hi = _update_theta_interval(p - eps, p + eps, factor, 0.0, self._pi2())
        assert hi - lo <= self._pi2() / factor + 2 * eps / factor + 1e-9


# ---------------------------------------------------------------------------
# Unit tests: _find_next_k
# ---------------------------------------------------------------------------


class TestFindNextK:
    def test_full_interval_starts_at_zero(self):
        """Wide initial interval: FindNextK cannot increase k yet."""
        k, upper = _find_next_k(0, True, 0.0, math.pi / 2.0)
        assert k == 0
        assert upper is True

    def test_narrow_interval_increases_power(self):
        theta = math.pi / 8.0
        k, _ = _find_next_k(0, True, theta - 0.01, theta + 0.01)
        assert k >= 1

    def test_returns_same_k_when_no_feasible_increase(self):
        # Wide interval relative to current Ki: K_max < min_ratio * Ki.
        k, upper = _find_next_k(10, True, 0.0, 0.4)
        assert k == 10
        assert upper is True

    def test_min_ratio_must_exceed_one(self):
        with pytest.raises(ValueError, match="min_ratio"):
            _find_next_k(0, True, 0.0, math.pi / 2.0, min_ratio=1.0)


# ---------------------------------------------------------------------------
# Unit tests: _check_iqae_provider
# ---------------------------------------------------------------------------


class TestCheckIqaeProvider:
    def test_numpy_passes(self):
        # Should not raise for numpy, which supports all IQAE gates
        _check_iqae_provider("numpy", frozenset({"QubitUnitary", "X", "MCZ", "Z"}))

    def test_missing_gate_raises(self):
        with pytest.raises(ValueError, match="IQAE requires gate"):
            _check_iqae_provider(
                "numpy",
                frozenset({"QubitUnitary", "X", "MCZ", "Z", "NonExistentGate9999"}),
            )


# ---------------------------------------------------------------------------
# Integration tests: IQAE.solve with a known oracle
# ---------------------------------------------------------------------------


def _make_ry_oracle(theta: float) -> CircuitSpec:
    """Return a CircuitSpec that prepares sin(theta)|1⟩ on the ancilla.

    The oracle is a single RY(2θ) gate on qubit 0 (= the ancilla, n_qubits=1),
    so  a = sin²(theta)  exactly.
    """
    qc = CircuitBuilder(n_qubits=1)
    qc.ry(2.0 * theta, qubit=0)
    return qc.build()


def _make_pennylane_ry_oracle(theta: float):
    """The same oracle as a PennyLane callable, for the callable-oracle path."""
    import pennylane as qml

    def oracle(wires):
        qml.RY(2.0 * theta, wires=wires[0])

    return oracle


class TestIQAESolve:
    """Integration tests using a 1-qubit oracle with known amplitude."""

    @pytest.mark.parametrize(
        "a_true",
        [0.25, 0.5, 0.75],
    )
    def test_estimate_close_to_true(self, a_true):
        theta = math.asin(math.sqrt(a_true))
        oracle = _make_ry_oracle(theta)

        result = IQAE(
            epsilon=0.05,
            alpha=0.05,
            n_qubits=1,
            n_shots=500,
            random_state=42,
        ).solve(oracle)

        assert isinstance(result, IQAEResult)
        assert abs(result.estimation - a_true) < 0.1

    def test_result_fields(self):
        oracle = _make_ry_oracle(math.pi / 6.0)
        result = IQAE(
            epsilon=0.05, alpha=0.1, n_qubits=1, n_shots=200, random_state=0
        ).solve(oracle)

        assert isinstance(result.estimation, float)
        assert len(result.confidence_interval) == 2
        lo, hi = result.confidence_interval
        assert lo <= result.estimation <= hi
        assert result.n_rounds >= 1
        assert result.n_oracle_calls > 0

    def test_ci_contains_true_value(self):
        a_true = 0.5
        oracle = _make_ry_oracle(math.asin(math.sqrt(a_true)))
        result = IQAE(
            epsilon=0.04, alpha=0.05, n_qubits=1, n_shots=400, random_state=7
        ).solve(oracle)
        lo, hi = result.confidence_interval
        assert lo <= a_true <= hi

    @pytest.mark.parametrize("a_true", [0.15, 0.5, 0.85])
    def test_ci_coverage_over_many_seeds(self, a_true):
        """Empirical CI coverage must meet the 1-alpha contract."""
        alpha = 0.05
        n_seeds = 250
        oracle = _make_ry_oracle(math.asin(math.sqrt(a_true)))
        hits = 0
        for seed in range(n_seeds):
            result = IQAE(
                epsilon=0.05,
                alpha=alpha,
                n_qubits=1,
                n_shots=500,
                random_state=seed,
            ).solve(oracle)
            lo, hi = result.confidence_interval
            if lo <= a_true <= hi:
                hits += 1
        coverage = hits / n_seeds
        # Binomial tolerance: allow ~3σ below nominal 0.95 with n=250.
        assert coverage >= 0.95 - 0.06

    def test_circuit_spec_oracle_uses_own_n_qubits(self):
        """CircuitSpec oracle: n_qubits from spec overrides self.n_qubits."""
        from eigenkit.core.blocks import QubitUnitary, assemble

        # Build a 1-qubit RY unitary directly (avoids qml.matrix kwarg issues)
        theta = math.pi / 6.0
        c, s = math.cos(theta), math.sin(theta)
        U = np.array([[c, -s], [s, c]], dtype=complex)
        oracle_spec = assemble([QubitUnitary(unitary=U.tolist(), wires=(0,))])

        # Pass n_qubits=1 but oracle_spec.n_qubits is also 1 — should work fine
        result = IQAE(
            epsilon=0.05, alpha=0.1, n_qubits=1, n_shots=300, random_state=3
        ).solve(oracle_spec)

        assert isinstance(result, IQAEResult)
        assert abs(result.estimation - math.sin(theta) ** 2) < 0.15

    def test_non_pennylane_callable_raises(self):
        """Passing a callable with a non-PennyLane provider should raise ValueError."""

        def oracle(wires):
            pass

        estimator = IQAE(epsilon=0.05, alpha=0.1, n_qubits=1)
        with pytest.raises(
            ValueError, match="callable oracle requires provider='pennylane'"
        ):
            estimator.solve(oracle)

    def test_adaptive_schedule_default(self):
        oracle = _make_ry_oracle(math.asin(math.sqrt(0.3)))
        result = IQAE(
            epsilon=0.05,
            alpha=0.1,
            n_qubits=1,
            n_shots=300,
            random_state=1,
        ).solve(oracle)
        assert result.n_rounds >= 1


# ---------------------------------------------------------------------------
# Weighted oracle guard
# ---------------------------------------------------------------------------


@pytest.mark.pennylane
class TestNQubitsMismatch:
    def test_callable_oracle_wrong_n_qubits_raises(self, monkeypatch):
        """Oracle returning a 4×4 unitary (2 qubits) with n_qubits=1 must raise."""
        import pennylane as qml

        # Patch qml.matrix to return a 4×4 matrix regardless of the oracle,
        # simulating a 2-qubit callable passed to an IQAE with n_qubits=1.
        def fake_matrix(fn, wire_order):
            def call(**kwargs):
                return np.eye(4, dtype=complex)

            return call

        monkeypatch.setattr(qml, "matrix", fake_matrix)

        def dummy_oracle(wires):
            pass

        estimator = IQAE(
            epsilon=0.05, alpha=0.1, n_qubits=1, provider="pennylane", n_shots=50
        )
        with pytest.raises(ValueError, match="n_qubits"):
            estimator.solve(dummy_oracle)

    def test_callable_oracle_matching_n_qubits_passes(self):
        """Oracle on 1 qubit + n_qubits=1 must not raise."""
        oracle = _make_pennylane_ry_oracle(math.pi / 4.0)
        result = IQAE(
            epsilon=0.1,
            alpha=0.1,
            n_qubits=1,
            provider="pennylane",
            n_shots=100,
            random_state=0,
        ).solve(oracle)
        assert isinstance(result, IQAEResult)


class TestWeightedOracleGuard:
    def test_weighted_circuit_spec_raises(self):
        """CircuitSpec with n_weights > 0 must raise before any circuit is built."""
        weighted_spec = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=0,
            gates=(Gate("RY", (0,), (WeightRef(0),)),),
        )
        estimator = IQAE(epsilon=0.05, alpha=0.1, n_qubits=1)
        with pytest.raises(ValueError, match="trainable weight"):
            estimator.solve(weighted_spec)

    def test_zero_weights_passes(self):
        """CircuitSpec with n_weights=0 must not raise."""
        U = np.array([[0, 1], [1, 0]], dtype=complex)
        spec = assemble([QubitUnitary(unitary=U.tolist(), wires=(0,))])
        assert spec.n_weights == 0
        result = IQAE(
            epsilon=0.1, alpha=0.1, n_qubits=1, n_shots=50, random_state=0
        ).solve(spec)
        assert isinstance(result, IQAEResult)


# ---------------------------------------------------------------------------
# AmplitudeEncoding adjoint provider guard
# ---------------------------------------------------------------------------


class TestAmplitudeEncodingAdjointGuard:
    def _ae_oracle_spec(self):
        amps = np.array([1.0, 0.0], dtype=complex)
        return assemble([AmplitudeEncoding(amps, normalize=False)])

    @pytest.mark.pennylane
    def test_pennylane_with_ae_oracle_does_not_raise(self):
        """PennyLane supports adjoint(AmplitudeEncoding) — no error."""
        oracle_spec = self._ae_oracle_spec()
        # Should not raise
        _check_amplitude_encoding_adjoint(oracle_spec, "pennylane")

    def test_non_pennylane_with_ae_oracle_raises(self):
        """Any non-PennyLane provider with AE oracle must raise ValueError."""
        pytest.importorskip("qiskit")
        oracle_spec = self._ae_oracle_spec()
        with pytest.raises(ValueError, match="adjoint"):
            _check_amplitude_encoding_adjoint(oracle_spec, "qiskit")

    def test_no_ae_gate_passes_any_provider(self):
        """If oracle has no AmplitudeEncoding, provider check is skipped."""
        U = np.array([[0, 1], [1, 0]], dtype=complex)
        spec = assemble([QubitUnitary(unitary=U.tolist(), wires=(0,))])
        # Should not raise even for qiskit
        pytest.importorskip("qiskit")
        _check_amplitude_encoding_adjoint(spec, "qiskit")

    def test_solve_with_ae_oracle_and_non_pennylane_raises(self):
        """End-to-end: IQAE.solve raises upfront for AE oracle + non-PL provider."""
        pytest.importorskip("qiskit")
        oracle_spec = self._ae_oracle_spec()
        estimator = IQAE(epsilon=0.05, alpha=0.1, n_qubits=1, provider="qiskit")
        with pytest.raises(ValueError, match="adjoint"):
            estimator.solve(oracle_spec)


# ---------------------------------------------------------------------------
# Top-level import test
# ---------------------------------------------------------------------------


class TestProviderShotModeWarning:
    def test_provider_mode_with_non_pennylane_warns_at_init(self):
        """shot_mode='provider' + non-PennyLane backend must warn at __init__."""
        pytest.importorskip("qiskit")
        with pytest.warns(UserWarning, match="shot_mode='provider'"):
            IQAE(
                epsilon=0.05,
                alpha=0.1,
                n_qubits=1,
                provider="qiskit",
                shot_mode="provider",
            )

    @pytest.mark.pennylane
    def test_provider_mode_with_pennylane_no_warn(self):
        """shot_mode='provider' + PennyLane must not warn at __init__."""
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            IQAE(
                epsilon=0.05,
                alpha=0.1,
                n_qubits=1,
                provider="pennylane",
                shot_mode="provider",
            )

    def test_synthetic_mode_with_non_pennylane_no_warn(self):
        """shot_mode='synthetic' + Qiskit is valid — no warning at __init__."""
        import warnings as _warnings

        pytest.importorskip("qiskit")
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            IQAE(
                epsilon=0.05,
                alpha=0.1,
                n_qubits=1,
                provider="qiskit",
                shot_mode="synthetic",
            )


class TestProviderShotsWarning:
    """Provider-level shots add a second, unseeded noise layer on top of
    IQAE's own seeded binomial sampling — solve() must warn about it."""

    def _spec(self, a_true: float = 0.25):
        theta = math.asin(math.sqrt(a_true))
        c, s = math.cos(theta), math.sin(theta)
        return assemble(
            [QubitUnitary(np.array([[c, -s], [s, c]], dtype=complex), wires=(0,))]
        )

    @pytest.mark.pennylane
    def test_warns_on_synthetic_mode_with_provider_shots(self):
        from eigenkit import PennyLaneConfig

        iqae = IQAE(
            epsilon=0.1,
            alpha=0.05,
            n_qubits=1,
            n_shots=20,
            random_state=0,
            shot_mode="synthetic",
            provider=PennyLaneConfig(device_name="default.qubit", shots=200),
        )
        with pytest.warns(UserWarning, match="second seeded binomial layer"):
            iqae.solve(self._spec())

    @pytest.mark.pennylane
    def test_auto_mode_uses_provider_shots_without_warning(self):
        import warnings as _warnings

        from eigenkit import PennyLaneConfig

        iqae = IQAE(
            epsilon=0.1,
            alpha=0.05,
            n_qubits=1,
            n_shots=20,
            random_state=0,
            shot_mode="auto",
            provider=PennyLaneConfig(device_name="default.qubit", shots=200),
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            result = iqae.solve(self._spec())
        assert isinstance(result, IQAEResult)

    def test_no_warning_on_analytic_device(self):
        import warnings as _warnings

        iqae = IQAE(epsilon=0.1, alpha=0.05, n_qubits=1, n_shots=20, random_state=0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            iqae.solve(self._spec())


class TestBoundaryAmplitudes:
    """Amplitudes of exactly 0 or 1 must not crash solve().

    Floating-point drift in the Grover matrix products pushes the analytic
    ancilla probability marginally outside [0, 1], and ``rng.binomial``
    raises ``ValueError: p < 0, p > 1 or p is NaN`` without the clamp in
    ``_measure_ancilla_prob``.
    """

    def _uniform_ancilla_oracle(self, n_qubits: int, ancilla_value: int):
        """All probability mass on ancilla=|ancilla_value⟩ → a = 0 or 1."""
        n_bins = 2 ** (n_qubits - 1)
        state = np.zeros(2 * n_bins)
        amp = math.sqrt(1.0 / n_bins)
        for i in range(n_bins):
            state[2 * i + ancilla_value] = amp
        return assemble([AmplitudeEncoding(state, normalize=False)])

    def test_amplitude_exactly_zero(self):
        spec = self._uniform_ancilla_oracle(4, ancilla_value=0)
        result = IQAE(epsilon=0.001, alpha=0.05, n_qubits=4, random_state=0).solve(spec)
        assert result.estimation == pytest.approx(0.0, abs=0.01)

    def test_amplitude_exactly_one(self):
        spec = self._uniform_ancilla_oracle(4, ancilla_value=1)
        result = IQAE(epsilon=0.001, alpha=0.05, n_qubits=4, random_state=0).solve(spec)
        assert result.estimation == pytest.approx(1.0, abs=0.01)


def test_top_level_import():
    """IQAE and IQAEResult must be importable from the package root."""
    from eigenkit import IQAE as IQAE2  # noqa: F401
    from eigenkit import IQAEResult as IQAEResult2

    assert IQAE2 is IQAE
    assert IQAEResult2 is IQAEResult


class TestAutoShotModeNonPennyLane:
    """Provider-side ancilla sampling is implemented only for the
    PennyLane backend. `shot_mode='auto'` on a shots-capable NON-PennyLane
    backend must fall back to the synthetic sampler instead of resolving to
    'provider' and crashing later in _provider_sample_ancilla; explicit
    `shot_mode='provider'` on such a backend must fail fast with a clear error.
    A fake shots-capable backend keeps the test independent of which SDK extras
    are installed.
    """

    def _patch_fake_backend(self, monkeypatch):
        from eigenkit.algorithms import iqae as iqae_mod

        class _Caps:
            supports_shots = True
            supports_provider_shots = False

        class _Backend:
            name = "braket"
            capabilities = _Caps()

        monkeypatch.setattr(iqae_mod, "resolve_provider", lambda _provider: _Backend())
        return iqae_mod

    def test_auto_falls_back_to_synthetic(self, monkeypatch):
        iqae_mod = self._patch_fake_backend(monkeypatch)
        # Pretend the backend has shots configured: the case in which 'auto'
        # would otherwise resolve to 'provider'.
        monkeypatch.setattr(iqae_mod, "provider_configured_shots", lambda _p: 200)
        est = IQAE(epsilon=0.1, alpha=0.1, n_qubits=1, shot_mode="auto")
        assert est._resolve_shot_strategy() == "synthetic"

    def test_explicit_provider_non_pennylane_raises(self, monkeypatch):
        self._patch_fake_backend(monkeypatch)
        est = IQAE(epsilon=0.1, alpha=0.1, n_qubits=1, shot_mode="provider")
        with pytest.raises(ValueError, match="is not supported on the"):
            est._resolve_shot_strategy()
