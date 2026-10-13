from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from ..core._results import ValueEquality
from ..core.blocks import Grover, QubitUnitary, assemble
from ..core.circuit import Circuit, ProviderSpec
from ..core.ir import CircuitSpec, Observable
from ..core.provider import resolve_provider
from ..core.solver import BaseQuantumSolver
from ._provider_checks import (
    check_amplitude_encoding_adjoint,
    check_oracle_matrix_extraction,
    check_required_gates,
    provider_configured_shots,
)


ShotMode = Literal["auto", "synthetic", "provider"]


@dataclass(frozen=True, eq=False)
class IQAEResult(ValueEquality):
    """Outcome of a completed Iterative Quantum Amplitude Estimation (IQAE) run.

    The result is read-only. Assigning to a field raises `FrozenInstanceError`, so
    a mistyped `=` (instead of `==`) cannot change a result. This type holds no
    arrays, so nothing else needs protecting.

    Results compare by value. They are not hashable, so that all result types
    behave in the same way: the types that hold arrays cannot be hashed both
    correctly and efficiently.

    Attributes:
        estimation: The point estimate of the amplitude `a`.
        confidence_interval: The `(lower, upper)` Clopper-Pearson interval at the
            requested confidence level `1 - alpha`.
        n_rounds: The number of rounds that the run executed. It can also be read
            as `n_iterations`, the name that the other result types use.
        n_oracle_calls: The total number of oracle calls over all rounds.
        success: Whether the interval reached the requested precision. This is the
            same test that ends the loop over rounds, and the one reported by the
            warning that the run did not converge. `False` means that the run
            stopped at `max_rounds` with an interval still wider than
            `2 * epsilon`.
    """

    estimation: float
    confidence_interval: tuple[float, float]
    n_rounds: int
    n_oracle_calls: int
    success: bool

    @property
    def n_iterations(self) -> int:
        """The number of rounds, under the name that the other results use.

        `n_rounds` is the name that IQAE uses, and it stays available. This property
        lets one loop read the number of iterations from any result.

        Returns:
            The number of rounds that the run executed.
        """
        return self.n_rounds


class IQAE(BaseQuantumSolver):
    r"""Iterative Quantum Amplitude Estimation (IQAE).

    Estimates `a = |⟨ψ₁|A|0⟩|²` to an additive precision `epsilon`, with a failure
    probability of at most `alpha`. It uses the Grover operator
    `Q = A S₀ A† S_ψ₁`, built from the oracle `A` that you provide.

    The oracle can be given in two forms:

    * A **PennyLane callable** `oracle(wires)`, which applies the circuit `A` in
      place. The last wire (`wires[-1]`) must be the ancilla qubit:

      ```python
      def my_oracle(wires):
          qml.RY(theta, wires=wires[0])
          qml.CNOT(wires=[wires[0], wires[-1]])
      ```

      The matrix of the circuit is extracted internally with `qml.matrix`, so this
      form requires `provider="pennylane"` and the `pennylane` extra.

    * A [`CircuitSpec`][eigenkit.core.ir.CircuitSpec], which works on any backend
      that supports the `QubitUnitary`, `X`, `MCZ` and `Z` gates. With a provider
      other than PennyLane, pass `shot_mode="synthetic"` or `shot_mode="auto"`,
      because `shot_mode="provider"` works only with PennyLane.

    Args:
        epsilon: The additive precision to reach, that is, the half-width of the
            final confidence interval for `a`. It must be in `(0, π/8)`.
        alpha: The total failure probability. The algorithm applies a Bonferroni
            correction, so the confidence interval of each round is valid at level
            `alpha / T`, where `T = ceil(log2(π / (8 * epsilon)))` is the maximum
            number of rounds. It must be in `(0, 1)`.
        n_qubits: The total number of qubits, including the ancilla.
        provider: The backend to use: a registered name (`"numpy"`, `"pennylane"`,
            `"qiskit"` and so on), a config instance, or a backend object. The
            default is `"numpy"`, the built-in simulator.
        n_shots: The number of measurement shots per round. With
            `shot_mode="synthetic"`, which is used on analytic backends, it sets
            the size of a seeded binomial draw on top of the exact probability of
            the ancilla. With `shot_mode="provider"` or `"auto"`, it is the number
            of samples per round on the hardware or simulator, unless the provider
            config sets its own `shots`, which takes precedence. For example, a
            device configured for 50 shots samples 50 times, whatever the value of
            `n_shots`. To control sampling on those backends, set `shots` on the
            provider config. The default is `100`.
        shot_mode: How the ancilla measurements are obtained. With `"auto"`, the
            default, the provider's shot count is used when the backend config sets
            a finite `shots`, and synthetic binomial noise is used otherwise.
            `"provider"` always samples from the backend. `"synthetic"` always uses
            exact expectation values plus seeded binomial noise.
        min_ratio: The minimum ratio `K_{i+1} / K_i` between the Grover powers of
            consecutive rounds. It must be greater than 1.
        random_state: The seed of the random number generator used for the shot
            noise. An integer gives reproducible results, and `None` uses a
            non-deterministic seed.

    Example:
        ```python
        import numpy as np
        from eigenkit import CircuitBuilder
        from eigenkit.algorithms import IQAE

        # RY(pi/3) on the ancilla prepares an amplitude of sin^2(pi/6) = 0.25.
        qc = CircuitBuilder(n_qubits=2)
        qc.ry(np.pi / 3, qubit=1)
        oracle = qc.build()

        iqae = IQAE(epsilon=0.02, alpha=0.05, n_qubits=2, random_state=0)
        result = iqae.solve(oracle)

        print(result.estimation)           # 0.246
        print(result.confidence_interval)  # (0.2407, 0.2513), which contains 0.25
        print(result.n_oracle_calls)       # 2600
        ```
    """

    def __init__(
        self,
        epsilon: float,
        alpha: float,
        n_qubits: int,
        provider: ProviderSpec = "numpy",
        n_shots: int = 100,
        min_ratio: float = 2.0,
        shot_mode: ShotMode = "auto",
        random_state: int | None = 42,
    ) -> None:
        _epsilon_max = math.pi / 8.0
        if not 0.0 < epsilon < _epsilon_max:
            raise ValueError(
                f"epsilon must be in (0, π/8) ≈ (0, {_epsilon_max:.4f}), got {epsilon}. "
                "At ε ≥ π/8 the algorithm requires 0 rounds and cannot run."
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {n_shots}")
        if min_ratio <= 1.0:
            raise ValueError(f"min_ratio must be > 1, got {min_ratio}")
        if shot_mode not in ("auto", "synthetic", "provider"):
            raise ValueError(
                f"shot_mode must be 'auto', 'synthetic', or 'provider', got {shot_mode!r}"
            )
        if shot_mode == "provider":
            try:
                backend = resolve_provider(provider)
                if not backend.capabilities.supports_provider_shots:
                    warnings.warn(
                        f"IQAE shot_mode='provider' is not supported by the "
                        f"'{backend.name}' backend; it will raise "
                        "ValueError at solve() time. "
                        "Use shot_mode='synthetic' (analytic expectation + seeded "
                        "binomial noise) or shot_mode='auto' instead.",
                        UserWarning,
                        stacklevel=2,
                    )
            except (ImportError, ModuleNotFoundError):
                pass

        super().__init__(provider=provider)
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.min_ratio = min_ratio
        self.shot_mode = shot_mode
        self.random_state = random_state

    _REQUIRED_GATES: frozenset[str] = frozenset({"QubitUnitary", "X", "MCZ", "Z"})

    def solve(self, oracle: Callable | CircuitSpec) -> IQAEResult:
        """Estimate the amplitude `a = |⟨ψ₁|A|0⟩|²`.

        Args:
            oracle: Either a PennyLane callable `oracle(wires)` that builds the
                state-preparation circuit `A`, or a
                [`CircuitSpec`][eigenkit.core.ir.CircuitSpec] produced by
                [`CircuitBuilder`][eigenkit.core.builder.CircuitBuilder]. In both
                cases, the last qubit is treated as the ancilla.

        Returns:
            An [`IQAEResult`][eigenkit.algorithms.IQAEResult] with the point estimate,
                the confidence interval and the resource counts.

        Raises:
            ValueError: If the selected provider does not support the gates that IQAE
                needs (`QubitUnitary`, `X`, `MCZ` and `Z`).
        """
        _check_iqae_provider(self.provider, self._REQUIRED_GATES)
        shot_strategy = self._resolve_shot_strategy()
        _warn_synthetic_with_provider_shots(self.provider, shot_strategy)
        rng = np.random.default_rng(self.random_state)

        if isinstance(oracle, CircuitSpec):
            oracle_spec = oracle
            if oracle_spec.n_qubits != self.n_qubits:
                raise ValueError(
                    f"CircuitSpec oracle has {oracle_spec.n_qubits} qubit(s), "
                    f"but IQAE was created with n_qubits={self.n_qubits}. "
                    "Pass n_qubits matching the oracle's qubit count."
                )
        else:
            oracle_spec = self._oracle_to_spec(oracle)

        if oracle_spec.n_weights > 0:
            raise ValueError(
                f"IQAE oracle has {oracle_spec.n_weights} trainable weight(s). "
                "IQAE evaluates the oracle at a fixed point (weights=zeros) and "
                "cannot marginalise over weights. Bake the weight values into "
                "the oracle before calling solve(), or use a weight-free CircuitSpec."
            )

        if oracle_spec.n_inputs > 0:
            raise ValueError(
                f"IQAE oracle has {oracle_spec.n_inputs} classical input(s). "
                "IQAE evaluates the oracle at a fixed point (inputs=zeros) and "
                "cannot marginalise over inputs. Bake the input values into "
                "the oracle before calling solve(), or use an input-free CircuitSpec."
            )

        check_amplitude_encoding_adjoint(oracle_spec, self.provider)

        n_qubits = oracle_spec.n_qubits

        grover_block = Grover(a_circuit=oracle_spec)
        grover_fragment = grover_block.build(n_qubits)

        max_rounds = math.ceil(math.log2(math.pi / (8.0 * self.epsilon)))
        alpha_round = self.alpha / max_rounds

        theta_lo: float = 0.0
        theta_hi: float = math.pi / 2.0
        total_oracle_calls: int = 0
        rounds_done: int = 0
        m_k: int = 0
        upper_half_plane: bool = True
        prev_m_k: int | None = None
        cum_h: int = 0
        cum_n: int = 0

        while rounds_done < max_rounds:
            m_k, upper_half_plane = _find_next_k(
                m_k,
                upper_half_plane,
                theta_lo,
                theta_hi,
                min_ratio=self.min_ratio,
            )

            round_spec = self._build_round_spec(
                oracle_spec, grover_fragment, m_k, n_qubits
            )
            h, n_round_shots = self._measure_ancilla_counts(
                round_spec, n_qubits, shot_strategy, rng
            )
            if m_k == prev_m_k:
                cum_h += h
                cum_n += n_round_shots
            else:
                cum_h, cum_n = h, n_round_shots

            p_lo, p_hi = _clopper_pearson(cum_h, cum_n, alpha_round)

            factor = 2 * m_k + 1
            theta_lo, theta_hi = _update_theta_interval(
                p_lo, p_hi, factor, theta_lo, theta_hi
            )

            total_oracle_calls += factor * n_round_shots
            rounds_done += 1
            prev_m_k = m_k

            a_lo = math.sin(theta_lo) ** 2
            a_hi = math.sin(theta_hi) ** 2
            if a_hi - a_lo < 2.0 * self.epsilon:
                break

        a_lo = math.sin(theta_lo) ** 2
        a_hi = math.sin(theta_hi) ** 2
        a_est = (a_lo + a_hi) / 2.0

        if a_hi - a_lo >= 2.0 * self.epsilon:
            warnings.warn(
                f"IQAE did not converge: confidence interval width "
                f"{a_hi - a_lo:.6g} >= 2*epsilon={2.0 * self.epsilon:.6g} "
                f"after {rounds_done} round(s). "
                "Increase n_shots or min_ratio, or loosen epsilon.",
                RuntimeWarning,
                stacklevel=2,
            )

        return IQAEResult(
            estimation=a_est,
            confidence_interval=(a_lo, a_hi),
            n_rounds=rounds_done,
            n_oracle_calls=total_oracle_calls,
            # The same test the loop breaks on, and the one the warning above
            # reports: recorded rather than left for the caller to redo.
            success=(a_hi - a_lo) < 2.0 * self.epsilon,
        )

    def _resolve_shot_strategy(self) -> Literal["synthetic", "provider"]:
        if self.shot_mode == "synthetic":
            return "synthetic"

        backend = resolve_provider(self.provider)
        # Provider-side ancilla sampling requires a backend that advertises it
        # (see _provider_sample_ancilla); any other shots-capable backend cannot
        # serve real provider shots.
        provider_sampling_ok = (
            backend.capabilities.supports_shots
            and backend.capabilities.supports_provider_shots
        )

        if self.shot_mode == "provider":
            if not backend.capabilities.supports_shots:
                raise ValueError(
                    f"IQAE shot_mode='provider' requires a shots-capable backend, "
                    f"but '{backend.name}' does not declare supports_shots=True."
                )
            if not backend.capabilities.supports_provider_shots:
                raise ValueError(
                    f"IQAE shot_mode='provider' is not supported on the "
                    f"'{backend.name}' backend; it has no provider-side ancilla "
                    f"sampler. Use shot_mode='synthetic' (or 'auto')."
                )
            return "provider"

        # auto: use real provider shots only when the backend has shots
        # configured AND supports provider-side sampling; else fall back to the
        # synthetic sampler instead of crashing later in _provider_sample_ancilla.
        if provider_configured_shots(self.provider) is not None and provider_sampling_ok:
            return "provider"
        return "synthetic"

    def _oracle_to_spec(self, oracle: Callable) -> CircuitSpec:
        """Convert a PennyLane oracle callable to a ``CircuitSpec``.

        Requires ``capabilities.supports_oracle_matrix=True`` (PennyLane).
        For other providers a ``CircuitSpec`` can be passed directly to `solve()`.
        """
        check_oracle_matrix_extraction(self.provider)

        import pennylane as qml

        wires = list(range(self.n_qubits))
        U: np.ndarray = qml.matrix(oracle, wire_order=wires)(wires=wires)

        expected_dim = 2**self.n_qubits
        if U.shape != (expected_dim, expected_dim):
            actual_qubits = int(round(math.log2(U.shape[0])))
            raise ValueError(
                f"Oracle callable returned a {U.shape[0]}×{U.shape[1]} unitary "
                f"({actual_qubits} qubit(s)), but IQAE was created with "
                f"n_qubits={self.n_qubits} (expected {expected_dim}×{expected_dim}). "
                "Pass n_qubits matching the oracle's qubit count."
            )

        return assemble([QubitUnitary(unitary=U.tolist(), wires=tuple(wires))])

    def _build_round_spec(
        self,
        oracle_spec: CircuitSpec,
        grover_fragment,
        m_k: int,
        n_qubits: int,
    ) -> CircuitSpec:
        """Compose A · Q^{m_k} into a single ``CircuitSpec``.

        Gate ordering (left = applied first to |0⟩):
            oracle_gates  +  (grover_gates) × m_k

        All A and A† copies inside the Grover fragment share the same
        weight indices, so ``n_weights`` stays equal to the oracle's.
        """
        a_gates = oracle_spec.gates
        q_gates = grover_fragment.gates

        return CircuitSpec(
            n_qubits=n_qubits,
            n_weights=oracle_spec.n_weights,
            n_inputs=oracle_spec.n_inputs,
            gates=a_gates + q_gates * m_k,
        )

    def _measure_ancilla_counts(
        self,
        spec: CircuitSpec,
        n_qubits: int,
        shot_strategy: Literal["synthetic", "provider"],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Return ``(successes, n_shots)`` for the ancilla = |1⟩ outcome."""
        if shot_strategy == "provider":
            return _provider_sample_ancilla(
                spec, n_qubits, self.provider, self.n_shots, rng
            )
        p1 = self._measure_ancilla_prob_analytic(spec, n_qubits)
        h = int(rng.binomial(self.n_shots, p1))
        return h, self.n_shots

    def _measure_ancilla_prob_analytic(self, spec: CircuitSpec, n_qubits: int) -> float:
        """Return exact P(ancilla = |1⟩) via the Z-expectation identity."""
        ancilla = n_qubits - 1
        obs = Observable.z(ancilla)
        circuit = Circuit(spec, observables=obs, provider=self.provider)

        weights = np.zeros(spec.n_weights, dtype=np.float64)
        z_exp = float(circuit(weights))
        p1 = (1.0 - z_exp) / 2.0
        if p1 < 0.0:
            p1 = 0.0
        elif p1 > 1.0:
            p1 = 1.0
        return p1


def _find_next_k(
    k: int,
    upper_half_plane: bool,
    theta_lo: float,
    theta_hi: float,
    min_ratio: float = 2.0,
) -> tuple[int, bool]:
    r"""Greedy Grover-power selection (Grinko et al., Algorithm 2).

    Returns the largest ``k'`` such that the scaled interval
    \([(4k'+2)\theta_{lo},\ (4k'+2)\theta_{hi}] \bmod 2\pi\) lies entirely in one
    half-plane and \(K'/K \ge \text{min\_ratio}\). Falls back to the current *k* when no
    larger power is feasible; the caller accumulates shots across rounds with equal
    power.
    """
    if min_ratio <= 1.0:
        raise ValueError(f"min_ratio must be > 1, got {min_ratio}")

    delta = theta_hi - theta_lo
    if delta <= 0.0:
        return k, upper_half_plane

    ki = k
    ki_factor = 4 * ki + 2
    theta_min_i = ki_factor * theta_lo
    theta_max_i = ki_factor * theta_hi

    k_max = int(math.pi / delta)
    k_factor = k_max - (k_max - 2) % 4

    two_pi = 2.0 * math.pi
    while k_factor >= min_ratio * ki_factor:
        q = k_factor / ki_factor
        scaled_min = (q * theta_min_i) % two_pi
        scaled_max = (q * theta_max_i) % two_pi

        if scaled_max <= math.pi and scaled_min <= math.pi:
            return (k_factor - 2) // 4, True
        if scaled_max >= math.pi and scaled_min >= math.pi:
            return (k_factor - 2) // 4, False
        k_factor -= 4

    return ki, upper_half_plane


def _clopper_pearson(h: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact (Clopper-Pearson) confidence interval for a binomial proportion."""
    from scipy.stats import beta

    lo = float(beta.ppf(alpha / 2.0, h, n - h + 1)) if h > 0 else 0.0
    hi = float(beta.ppf(1.0 - alpha / 2.0, h + 1, n - h)) if h < n else 1.0
    return lo, hi


def _update_theta_interval(
    p_lo: float,
    p_hi: float,
    factor: int,
    theta_lo: float,
    theta_hi: float,
) -> tuple[float, float]:
    """Intersect the current θ-interval with the set consistent with [p_lo, p_hi].

    The preimage of [p_lo, p_hi] under P = sin²(factor·θ) is a union of
    ascending and descending branches in factor·θ space:

    ```text
    [arcsin(√p_lo) + k·π,  arcsin(√p_hi) + k·π]          (ascending)
    [π − arcsin(√p_hi) + k·π,  π − arcsin(√p_lo) + k·π]  (descending)
    ```

    Dividing by *factor* gives the corresponding θ-intervals.

    At high Grover amplification several branches may intersect the current
    interval.  The branch whose midpoint is closest to the prior midpoint is
    selected — not the narrowest one, which can collapse to a point excluding
    the true θ.  Falls back to the unchanged interval when no branch
    intersects (can happen with very few shots).
    """
    p_lo = max(0.0, min(1.0, p_lo))
    p_hi = max(0.0, min(1.0, p_hi))

    alpha_lo = math.asin(math.sqrt(p_lo))  # ∈ [0, π/2]
    alpha_hi = math.asin(math.sqrt(p_hi))  # ∈ [0, π/2]

    candidates: list[tuple[float, float]] = []

    n_branches = math.ceil(factor * theta_hi / math.pi) + 2

    for k in range(-1, n_branches + 1):
        for t_lo, t_hi in (
            (
                (alpha_lo + k * math.pi) / factor,
                (alpha_hi + k * math.pi) / factor,
            ),
            (
                (math.pi - alpha_hi + k * math.pi) / factor,
                (math.pi - alpha_lo + k * math.pi) / factor,
            ),
        ):
            lo = max(t_lo, theta_lo)
            hi = min(t_hi, theta_hi)
            if lo <= hi + 1e-12:
                # Tolerance admits zero-width candidates from exact measurements;
                # min/max normalises the order before appending.
                candidates.append((min(lo, hi), max(lo, hi)))

    if not candidates:
        return theta_lo, theta_hi

    prior_mid = (theta_lo + theta_hi) / 2.0
    new_lo, new_hi = min(candidates, key=lambda c: abs((c[0] + c[1]) / 2.0 - prior_mid))
    return float(new_lo), float(new_hi)


def _provider_sample_ancilla(
    spec: CircuitSpec,
    n_qubits: int,
    provider: ProviderSpec,
    n_shots: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample the ancilla bit ``n_shots`` times on a shots-capable backend."""
    backend = resolve_provider(provider)
    if not backend.capabilities.supports_provider_shots:
        raise ValueError(
            f"IQAE provider shot sampling is not supported by the '{backend.name}' "
            f"backend. Use shot_mode='synthetic' or pass a "
            "CircuitSpec oracle with an analytic provider."
        )

    # The sampler implementation below is PennyLane-specific; the capability
    # flag above gates entry so only backends that provide it reach this code.
    import pennylane as qml

    from ..core.backends.pennylane_backend import _apply_gate

    device_name = getattr(backend.config, "device_name", "default.qubit")
    ancilla = n_qubits - 1

    seed = int(rng.integers(0, 2**31))
    dev = qml.device(device_name, wires=n_qubits, seed=seed)

    @qml.qnode(dev)
    def _sample(weights, inputs):
        for gate in spec.gates:
            _apply_gate(gate, weights, inputs)
        return qml.sample(wires=ancilla)

    configured = provider_configured_shots(provider)
    # The provider's own shot count takes precedence: it is the device's
    # setting, and sampling more than the device was configured for is not
    # something this layer can grant.
    shots_to_use = configured if configured is not None else n_shots
    sample_qnode = qml.set_shots(_sample, shots=shots_to_use)
    weights = np.zeros(spec.n_weights, dtype=np.float64)
    inputs = np.zeros(spec.n_inputs, dtype=np.float64)
    samples = np.asarray(sample_qnode(weights, inputs)).reshape(-1)
    return int(np.count_nonzero(samples == 1)), shots_to_use


def _warn_synthetic_with_provider_shots(
    provider: ProviderSpec, shot_mode: ShotMode
) -> None:
    """Warn when synthetic shot noise stacks on provider-level shot expectations."""
    if shot_mode != "synthetic":
        return
    if provider_configured_shots(provider) is None:
        return
    backend = resolve_provider(provider)
    shots = provider_configured_shots(provider)
    warnings.warn(
        f"The selected provider '{backend.name}' is configured with shots={shots} "
        "but IQAE shot_mode='synthetic' evaluates expectations through the provider "
        "and then adds a second seeded binomial layer. Use shot_mode='auto' "
        "(default) or 'provider' for a single sampling layer, or set "
        "provider shots=None for analytic simulation.",
        UserWarning,
        stacklevel=3,
    )


def _check_iqae_provider(provider: ProviderSpec, required_gates: frozenset[str]) -> None:
    check_required_gates(
        provider,
        required_gates,
        algorithm="IQAE",
        hint=(
            "IQAE uses QubitUnitary to wrap the oracle and X / MCZ / Z "
            "inside the Grover iterate."
        ),
    )


_check_amplitude_encoding_adjoint = check_amplitude_encoding_adjoint
