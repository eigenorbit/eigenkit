"""The IQM backend, which runs circuits on IQM quantum computers through iqm-client.

The `iqm-client` package, with its `qiskit` extra, is an optional dependency
(`pip install eigenkit[iqm]`). It is imported only when it is needed, so the rest of
the library works without it. If it is missing, `compile()` raises an `ImportError`
with a clear message.

This backend connects directly to IQM Resonance, or to any IQM Server, instead of
using the same hardware through Amazon Braket. You then pay through IQM's own plans
instead of an Amazon Web Services (AWS) bill. It is also the only way to reach IQM's
fake backends, its calibration sets and its architectures with `MOVE` gates.

IQM has **no estimator primitive**: the IQM service returns histograms of measured
bitstrings and nothing else. So this backend works by sampling everywhere. The
observables are sorted into groups of qubit-wise commuting terms, one circuit runs
for each group, and the expectation values are computed locally from the counts
(see the `_sampling` module). `probabilities()` and `sample()` use the same code,
and this is what lets the quantum approximate optimisation algorithm (QAOA) read a
solution from real hardware.

There are two kinds of target:

- **Resonance or any IQM Server**: pass `url` and a `token` (or set the `IQM_TOKEN`
  environment variable). You can also pass `quantum_computer` to choose one machine
  when the server hosts several.
- **Local fake backends**: pass `IQMFakeAdonis()`, `IQMFakeApollo()`,
  `IQMFakeDeneb()` or a similar fake backend as `backend_instance`. These run on
  Qiskit Aer with a realistic IQM noise model, with no token and no network access.
  The tests of this backend run on them.

The default is `gradient_method = "none"`, because hardware is too expensive for
parameter-shift gradients: one backward pass takes at least `2 * n_weights` jobs,
each with the full number of shots. Train on a simulator, then use the trained
weights on the device. `enable_gradients=True` in the configuration turns on
parameter-shift gradients on the device. Compiling for a real device then issues a
`CostWarning` that gives the number of circuits of one backward pass.

The circuits are transpiled for the device target. The native gates are `r` and
`cz`, and also `move` on the Star architectures, which the Qiskit transpiler routes
through the plugin that iqm-client provides. `transpile_mode` chooses when this
happens:

- `"parametric"`, the default, transpiles once for each measurement group and sets
  the parameter values at each call. This is cheap, but the transpiler cannot merge
  rotations whose angles it does not know. For example, an encoding block and a
  variational block on 2 qubits, measured in one group, gave 6 `r` gates in this
  mode against 3 with the values set before transpiling.
- `"per_call"` sets the values first and transpiles at every call. The circuits are
  shallower, so they run with higher fidelity on hardware. The cost is one
  transpilation per evaluation, which is small next to the time spent in the queue,
  but not next to a local simulator.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z
    from eigenkit.configs import IQMConfig

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # Offline, against IQM's noise-modelled fake backend.
    from iqm.qiskit_iqm import IQMFakeAdonis

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=IQMConfig(backend_instance=IQMFakeAdonis(), shots=2048),
    )

    # On real hardware, by name.
    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=IQMConfig(
            url="https://resonance.iqm.tech/",
            quantum_computer="garnet",
            shots=2048,
        ),
    )
    ```

    The backend is registered as `"iqm"` when the `iqm` extra is installed, and its
    module is imported only when it is first used. A real device uses your IQM
    quota: the first compilation for a real device issues a `CostWarning`, and
    `dry_run_info()` reports the size of the circuit and the number of circuits
    that each call submits, before you submit anything.
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from ..ir import (
    ALL_GATES,
    DATA_GATES,
    CircuitSpec,
    Observable,
)
from ..provider import (
    Capabilities,
    CompiledCircuit,
    CostWarning,
    _warn_gradient_cost,
    batch_call,
)
from ..secret_fields import SecretFieldsMixin
from . import _sampling
from ._kernel_support import (
    ProbabilityJobHandle,
    job_result,
    transpiled_resources,
    zero_fraction,
)
from .qiskit_backend import _apply_gate_symbolic, _require_qiskit


if TYPE_CHECKING:  # pragma: no cover - typing only
    from qiskit import QuantumCircuit


TranspileMode = Literal["parametric", "per_call"]


def _require_iqm() -> None:
    """Probe for iqm-client's Qiskit adapter. Raise a clear error if missing."""
    try:
        import iqm.qiskit_iqm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "iqm-client (with its qiskit extra) is not installed. Install it "
            "via `pip install eigenkit[iqm]`."
        ) from e


@dataclass(frozen=True)
class IQMConfig(SecretFieldsMixin):
    """Device configuration for the IQM backend.

    There are two ways to choose a target:

    - `url`, and optionally `quantum_computer`, opens an `IQMProvider` for an IQM
      Server. This needs a token, from `token` or from the `IQM_TOKEN` environment
      variable.
    - `backend_instance` uses any IQM `BackendV2` directly: a fake backend for
      offline work, or a backend that you configured yourself. It takes precedence
      over `url`, and it needs no token and no network access.

    Args:
        url: The address of the IQM Server, for example
            `"https://resonance.iqm.tech/"`. It is ignored when `backend_instance`
            is set. With `None`, the backend reads the address from the
            `IQM_SERVER_URL` environment variable when the circuit is compiled.
        quantum_computer: The machine to use when the server hosts more than one,
            for example `"garnet"`. With `None`, the server's default is used.
        backend_instance: An IQM backend that is already configured. It takes
            precedence over `url`.
        shots: The number of shots for each circuit run. It must be positive,
            because hardware cannot compute exact values.
        optimization_level: The optimisation level passed to `qiskit.transpile`,
            from `0` to `3`.
        seed_transpiler: The seed of the random steps of the transpiler, mainly
            the `SabreLayout` and `SabreSwap` passes, which choose the layout and
            the routing. With `None`, the default, a new seed is drawn for every
            compilation, so a circuit that needs routing compiles to a different
            number of gates on every run. Within one compilation, all the
            transpilations share the seed, including those of the shifted copies
            that a hardware gradient compiles. Set it whenever two compilations
            must be comparable. Trying a few seeds and keeping the cheapest result
            is a cheap and fair way to choose a layout, because compiling submits
            nothing.
        transpile_mode: `"parametric"` transpiles once and sets the parameter
            values at each call. `"per_call"` sets the values first and transpiles
            at every call, which gives shallower circuits but costs one
            transpilation per evaluation.
        group_commuting: Whether to put Pauli terms that commute qubit by qubit
            into shared measurement circuits. With `False`, one circuit runs for
            each term: there are more jobs, but each term gets all the shots.
        token: The access token for the IQM Server. It is passed on only when it
            is set. With `None`, iqm-client handles authentication on its own,
            from `IQM_TOKEN` and from its own tokens file. Do not set both `token`
            and `IQM_TOKEN`, because iqm-client rejects a token given in two ways
            at once. This field is secret: `repr()` never shows it, and it is never
            saved in `.eigenkit` bundles, pickles or joblib checkpoints. After a
            load it is `None`, so set `IQM_TOKEN`.
        calibration_set_id: The calibration set to use, as a `UUID`. With `None`,
            the server's default is used.
        use_metrics: Whether to ask the server for the error rates of each qubit
            and each coupler, and load them into the `InstructionProperties` of
            the target. With `False` (the default of iqm-client itself), the
            target has no error rates, so the Qiskit layout pass does not see the
            noise. It then cannot prefer the better qubits, and a comparison with
            a backend that *does* have calibration data is not like for like. It
            costs one extra request at compile time, and it is ignored when
            `backend_instance` is set.
        provider_kwargs: Extra keyword arguments passed to `IQMProvider`, for
            example `tokens_file`.
        poll_timeout_seconds: How many seconds `result()` waits before giving up.
        enable_gradients: Whether to allow parameter-shift gradients on the
            device, for PyTorch `.backward()`, `jacobian()` and gradient-based
            optimisers. It is off by default, because one backward pass runs the
            circuits of a forward call at least `2 * n_weights` times. When it is
            set, compiling for a real device issues a `CostWarning` that gives
            this count.
    """

    url: str | None = None
    quantum_computer: str | None = None
    # `BackendV2`; typed as `Any` to avoid a hard import.
    backend_instance: Any = None
    shots: int = 1024
    optimization_level: int = 1
    seed_transpiler: int | None = None
    transpile_mode: TranspileMode = "parametric"
    group_commuting: bool = True
    token: str | None = field(default=None, repr=False, metadata={"secret": True})
    calibration_set_id: Any = None
    use_metrics: bool = False
    provider_kwargs: dict = field(default_factory=dict)
    poll_timeout_seconds: int = 3600
    enable_gradients: bool = False

    def __post_init__(self) -> None:
        """Reject the settings that can never work, when the configuration is created.

        If they were left to the first submission, the error would appear after
        the transpilation and the cost warning, which on a paid device is the
        wrong time to find out. Only the settings that are wrong on their own are
        checked here. Whether a *target* can be found depends on the environment
        at compile time, so `_resolve_backend` checks that.
        """
        if self.transpile_mode not in ("parametric", "per_call"):
            raise ValueError(
                f"transpile_mode must be 'parametric' or 'per_call', got "
                f"{self.transpile_mode!r}"
            )
        if self.shots <= 0:
            raise ValueError(f"IQM hardware samples; shots must be > 0, got {self.shots}")

    def make_backend(self) -> "IQMBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `IQMBackend` for this configuration.
        """
        return IQMBackend(config=self)


def _resolve_backend(config: IQMConfig):
    """Pick an IQM `BackendV2` based on config.

    `backend_instance` wins; otherwise open an `IQMProvider` against the server
    and ask it for a backend. An explicit `config.token` is forwarded; otherwise
    the token is left entirely to iqm-client, which reads `IQM_TOKEN` and its
    own tokens file.
    """
    if config.backend_instance is not None:
        return config.backend_instance

    # Checked here rather than in `IQMConfig.__post_init__`: `IQM_SERVER_URL`
    # can be set after the config is built, and the default `IQMConfig()` that
    # `provider="iqm"` constructs has to stay constructible either way.
    # Resolved here rather than left to iqm-client: it takes the url as a
    # positional argument, so passing `None` through overrides its own env
    # lookup and fails deep inside with "'NoneType' object has no attribute
    # 'isascii'" instead of saying what is missing.
    url = config.url or os.environ.get("IQM_SERVER_URL")
    if not url:
        raise ValueError(
            "no IQM target: pass url= (e.g. 'https://resonance.iqm.tech/'), "
            "set IQM_SERVER_URL, or pass backend_instance= with a backend "
            "such as IQMFakeAdonis()."
        )

    _require_iqm()
    from iqm.qiskit_iqm import IQMProvider

    provider_kwargs = dict(config.provider_kwargs)
    # Only an explicit token is forwarded. iqm-client reads IQM_TOKEN itself and
    # refuses a token supplied both ways — "Parameter sources must not be mixed"
    # — so reading the variable here and passing it on would break exactly the
    # case it was meant to support.
    if config.token is not None:
        provider_kwargs["token"] = config.token
    provider = IQMProvider(
        url,
        quantum_computer=config.quantum_computer,
        **provider_kwargs,
    )
    if config.calibration_set_id is not None:
        return provider.get_backend(
            calibration_set_id=config.calibration_set_id,
            use_metrics=config.use_metrics,
        )
    return provider.get_backend(use_metrics=config.use_metrics)


def _is_local_backend(backend) -> bool:
    """True for IQM's Aer-backed fake backends, which cost nothing to run."""
    return "fake" in type(backend).__module__.lower()


# Qiskit operation name -> the name IQM's error profiles use for it.
_ERROR_PROFILE_GATES = {"r": "prx", "cz": "cz", "move": "move"}


def _error_profile_success_probability(circuit, backend) -> float | None:
    """Estimated success probability from an IQM fake backend's error profile.

    IQM's fake backends leave the Qiskit target without error rates, but carry
    them in `error_profile` as depolarizing parameters `lambda`, keyed by qubit
    name. A depolarizing channel on `d` levels has average infidelity
    `lambda * (d - 1) / d`; readout contributes the mean of its two
    misassignment rates. `None` when the backend has no profile.
    """
    profile = getattr(backend, "error_profile", None)
    if profile is None:
        return None
    single = profile.single_qubit_gate_depolarizing_error_parameters
    double = profile.two_qubit_gate_depolarizing_error_parameters
    readout = profile.readout_errors
    probability = 1.0
    for instruction in circuit.data:
        name = instruction.operation.name
        try:
            qubits = [
                backend.index_to_qubit_name(circuit.find_bit(q).index)
                for q in instruction.qubits
            ]
        except Exception:  # noqa: BLE001 — an unnamed wire carries no error rate
            continue
        if name == "measure":
            rates = readout.get(qubits[0])
            if rates:
                probability *= 1.0 - 0.5 * (rates.get("0", 0.0) + rates.get("1", 0.0))
            continue
        key = _ERROR_PROFILE_GATES.get(name)
        if key is None:
            continue
        if len(qubits) == 1:
            lam, levels = single.get(key, {}).get(qubits[0]), 2
        else:
            table = double.get(key, {})
            lam = table.get(tuple(qubits), table.get(tuple(reversed(qubits))))
            levels = 4
        if lam is not None:
            probability *= 1.0 - float(lam) * (levels - 1) / levels
    return probability


def _counts_list(result, n_expected: int) -> list[dict[str, int]]:
    """Normalize `Result.get_counts()` into a list of `n_expected` histograms.

    Qiskit returns a bare dict when one circuit ran and a list when several
    did, so a single-group observable set and a multi-group one come back
    shaped differently from the same call.
    """
    counts = result.get_counts()
    if isinstance(counts, dict):
        counts = [counts]
    counts = list(counts)
    if len(counts) != n_expected:
        raise RuntimeError(
            f"IQM returned {len(counts)} histograms for {n_expected} submitted circuits"
        )
    return counts


class IQMJobHandle:
    """A handle to a submitted IQM job.

    `compiled.submit(weights, inputs)` returns one of these handles. Call
    `result()` to wait until the job finishes and get the array of expectation
    values, or `status()` to check the job without waiting.
    """

    def __init__(
        self,
        job,
        groups: tuple[_sampling.MeasurementGroup, ...],
        n_observables: int,
        n_qubits: int,
        poll_timeout_seconds: int,
    ):
        """Wrap a submitted IQM job in a job handle.

        `IQMCompiled.submit()` builds it. It is not meant to be created directly.

        Args:
            job (JobV1): The job returned by the IQM backend.
            groups: The measurement plan that the submitted circuits follow, in the
                order of submission.
            n_observables: The number of expectation values to compute.
            n_qubits: The number of qubits of the circuit, used to read the keys of
                the histograms.
            poll_timeout_seconds: How many seconds `result()` waits before giving up.
        """
        self._job = job
        self._groups = groups
        self._n_observables = n_observables
        self._n_qubits = n_qubits
        self._poll_timeout_seconds = poll_timeout_seconds

    def status(self) -> str:
        """Return the state of the job, as the job object reports it, without waiting.

        The state is a string such as `"QUEUED"`, `"RUNNING"`, `"DONE"`,
        `"CANCELLED"` or `"ERROR"`.
        """
        state = self._job.status()
        return state.name if hasattr(state, "name") else str(state)

    def cancel(self) -> None:
        """Ask IQM to cancel the job. It returns without waiting."""
        self._job.cancel()

    def job_id(self) -> str:
        """Return the identifier of the IQM job, to look it up in Resonance or
        through iqm-client.
        """
        return str(self._job.job_id())

    def counts(self) -> list[dict[str, int]]:
        """Wait until the job finishes, then return the raw histograms.

        There is one histogram for each measurement group, and qubit `0` is the
        leftmost bit of each key. These are the counts that the device returned.
        [`result`][eigenkit.core.backends.iqm_backend.IQMJobHandle.result]
        computes the expectation values of the observables from them.
        """
        result = self._job.result(timeout=self._poll_timeout_seconds)
        return [
            _sampling.normalize_counts(counts, self._n_qubits)
            for counts in _counts_list(result, len(self._groups))
        ]

    def result(self) -> np.ndarray:
        """Wait until the job finishes, then return one expectation value for each
        observable, in the order of the observables.
        """
        return _sampling.expectations_from_counts(
            self._groups, self.counts(), self._n_observables, self._n_qubits
        )


class IQMCompiled(CompiledCircuit):
    """A circuit prepared for an IQM quantum computer.

    This class is internal: you get it from
    [`IQMBackend.compile()`][eigenkit.core.backends.iqm_backend.IQMBackend.compile],
    and you never create it directly. The measurement plan and the parametric
    circuit are built once. `run()` and `submit()` set the parameter values, and in
    `"per_call"` mode they also transpile.
    """

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: IQMConfig,
        *,
        warn: bool = True,
    ):
        """Prepare the circuit for repeated evaluation.

        `IQMBackend.compile()` builds it, and there is no reason to create one
        directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that this handle
                measures.
            config (IQMConfig): The device settings.
            warn (bool): Whether to issue the cost warnings for a real device. It
                is `False` only for the shifted copies that `compile_shifted`
                builds. Default: `True`.
        """
        _require_qiskit()
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter as QiskitParameter

        self._config = config
        self._observables = tuple(observables)
        # One seed for every transpile of this handle and of its shifted copies,
        # `"per_call"` mode included, so the terms of a parameter-shift gradient
        # all run on the same qubits.
        self._seed = (
            config.seed_transpiler
            if config.seed_transpiler is not None
            else random.randrange(2**31)
        )
        self._n_qubits = circuit.n_qubits
        self._n_observables = len(observables)

        # Ordered Parameter symbols, named so that a value can be read back out
        # of `weights` or `inputs` from the name alone after transpilation
        # reorders them.
        self._weight_symbols = [
            QiskitParameter(f"w{i}") for i in range(circuit.n_weights)
        ]
        self._input_symbols = [QiskitParameter(f"x{i}") for i in range(circuit.n_inputs)]
        qc = QuantumCircuit(circuit.n_qubits)
        for gate in circuit.gates:
            _apply_gate_symbolic(qc, gate, self._weight_symbols, self._input_symbols)
        self._base_circuit = qc

        self._groups = _sampling.plan_measurements(
            observables, circuit.n_qubits, group_commuting=config.group_commuting
        )

        # Resolve the target (fake backend, or a live server lookup).
        self._backend = _resolve_backend(config)

        backend_name = getattr(self._backend, "name", str(self._backend))
        if warn and not _is_local_backend(self._backend):
            warnings.warn(
                f"IQM compile against '{backend_name}'. Each run/submit will "
                f"consume IQM quota at shots={config.shots} "
                f"({len(self._groups)} circuits per call).",
                CostWarning,
                stacklevel=4,
            )
            if config.enable_gradients:
                # Counted in circuits, like the warning above: a sample's
                # measurement groups share one job, and quota goes by circuit.
                _warn_gradient_cost(
                    f"IQM device '{backend_name}'",
                    circuit,
                    len(self._groups),
                    config.shots,
                    unit="circuits",
                    stacklevel=4,
                )

        # Measured circuits, keyed by basis. In "parametric" mode the group
        # bases are transpiled here, once, and reused by every call — that is
        # the work `dry_run_info()` reports the cost of. The all-Z basis that
        # `sample()` and `probabilities()` use is filled in on first use
        # instead: a caller who only ever reads expectation values should not
        # pay a transpile for a circuit they never submit. In "per_call" mode
        # this stays empty and every call transpiles its own bound circuits.
        self._measured: dict[tuple[str, ...], Any] = {}
        if config.transpile_mode == "parametric":
            # Sorted for a deterministic transpile order, which keeps
            # `dry_run_info` reproducible across compiles.
            for basis in sorted({group.basis for group in self._groups}):
                self._measured[basis] = self._transpile(self._measure_in(basis))

    def compile_shifted(self, circuit: CircuitSpec) -> "IQMCompiled":
        """Compile a copy of this circuit with one gate angle shifted.

        The chain-rule gradient compiles one copy for each shifted gate. Each copy
        reuses the backend of this handle, without asking the server for it again,
        and is transpiled with the seed of this handle, so every term of the
        gradient comes from the same qubits. The compilation of this handle already
        issued the cost warnings, so the copy issues none.

        Args:
            circuit (CircuitSpec): The shifted circuit.

        Returns:
            An `IQMCompiled` handle for `circuit`.
        """
        config = replace(
            self._config, backend_instance=self._backend, seed_transpiler=self._seed
        )
        return IQMCompiled(circuit, self._observables, config, warn=False)

    # ---- circuit construction ----

    def _measure_in(self, basis: tuple[str, ...], bindings: dict | None = None):
        """A copy of the base circuit rotated into `basis` and measured.

        `bindings` binds the parameters first, which is what `"per_call"` mode
        does so the transpiler can fuse the resulting numeric rotations.
        """
        qc = self._base_circuit.copy()
        for name, qubit in _sampling.rotation_gates(basis):
            if name == "H":
                qc.h(qubit)
            elif name == "Sdg":
                qc.sdg(qubit)
            else:  # pragma: no cover - _sampling emits only H and Sdg
                raise RuntimeError(f"unexpected basis-rotation gate {name!r}")
        qc.measure_all()
        if bindings is not None:
            qc = qc.assign_parameters(bindings)
        return qc

    def _transpile(self, qc):
        """Map `qc` onto the device's native gates and connectivity.

        Plain `qiskit.transpile` covers both IQM architectures: iqm-client
        registers the transpiler plugin that routes Star-topology circuits
        through the resonator with MOVE gates.
        """
        from qiskit import transpile

        return transpile(
            qc,
            backend=self._backend,
            optimization_level=self._config.optimization_level,
            seed_transpiler=self._seed,
        )

    def _bindings(self, transpiled, weights: np.ndarray, inputs: np.ndarray) -> dict:
        """Map each surviving Parameter of `transpiled` to its value.

        Parameters are named `w<i>` / `x<j>`, so the name says which vector to
        read — transpilation is free to reorder or drop them.
        """
        bindings = {}
        for symbol in transpiled.parameters:
            name = symbol.name
            if name.startswith("w"):
                bindings[symbol] = float(weights[int(name[1:])])
            elif name.startswith("x"):
                bindings[symbol] = float(inputs[int(name[1:])])
            else:
                raise RuntimeError(
                    f"unexpected parameter name {name!r} in transpiled circuit"
                )
        return bindings

    def _bound_circuit(
        self, basis: tuple[str, ...], weights: np.ndarray, inputs: np.ndarray
    ):
        """One submission-ready circuit: measured in `basis`, values bound,
        transpiled for the device."""
        if self._config.transpile_mode == "per_call":
            return self._transpile(
                self._measure_in(basis, self._source_bindings(weights, inputs))
            )
        transpiled = self._measured.get(basis)
        if transpiled is None:
            transpiled = self._transpile(self._measure_in(basis))
            self._measured[basis] = transpiled
        return transpiled.assign_parameters(self._bindings(transpiled, weights, inputs))

    def _source_bindings(self, weights: np.ndarray, inputs: np.ndarray) -> dict:
        """Bindings for the *untranspiled* circuit, which still carries every
        symbol the spec declared."""
        bindings = {
            symbol: float(weights[i]) for i, symbol in enumerate(self._weight_symbols)
        }
        bindings.update(
            {symbol: float(inputs[i]) for i, symbol in enumerate(self._input_symbols)}
        )
        return bindings

    # ---- inspection ----

    def dry_run_info(self) -> dict:
        """Report the size of a run, before anything is submitted.

        The report gives the target backend and whether it is one of IQM's local
        fake backends, the number of shots, the transpile mode, the optimisation
        level, the `seed_transpiler` setting, the numbers of logical and physical
        qubits, the number of observables, and the number of measurement groups
        that the observables need, which is the number of circuits that one call
        submits. In `"parametric"` mode, it also gives the gate counts and the
        depth of the transpiled circuit for each measurement basis. `"per_call"`
        mode transpiles at call time, so there is no stored circuit to report.

        Returns:
            info (dict): The values above, keyed by name.
        """
        backend = self._backend
        info = {
            "backend_name": getattr(backend, "name", str(backend)),
            "is_local": _is_local_backend(backend),
            "shots": self._config.shots,
            "transpile_mode": self._config.transpile_mode,
            "optimization_level": self._config.optimization_level,
            "seed_transpiler": self._config.seed_transpiler,
            "n_logical_qubits": self._n_qubits,
            "n_physical_qubits": getattr(backend, "num_qubits", None),
            "n_observables": self._n_observables,
            "n_measurement_groups": len(self._groups),
            "n_circuit_runs_per_call": len(self._groups),
        }
        if self._measured:
            info["transpiled_gate_counts"] = {
                "".join(basis): dict(circuit.count_ops())
                for basis, circuit in sorted(self._measured.items())
            }
            info["transpiled_depth"] = {
                "".join(basis): circuit.depth()
                for basis, circuit in sorted(self._measured.items())
            }
        return info

    # ---- run / submit ----

    def _submit_bases(
        self, bases: Sequence[tuple[str, ...]], weights: np.ndarray, inputs: np.ndarray
    ):
        """Submit one circuit per entry of `bases` as a single IQM job."""
        circuits = [self._bound_circuit(basis, weights, inputs) for basis in bases]
        return self._backend.run(circuits, shots=self._config.shots)

    def run(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return one expectation value per observable.

        It follows the `CompiledCircuit` contract: the result has shape `(n_obs,)`
        when `weights` and `inputs` are both 1-D, and `(B, n_obs)` otherwise. Each
        sample costs one job of `n_measurement_groups` circuits. The values are
        estimated from shots, so they carry the shot noise of `shots`.

        Args:
            weights (np.ndarray): The parameter vector: 1-D for one sample, or 2-D
                with a leading batch axis.
            inputs (np.ndarray): The input features, with the same 1-D or 2-D shape
                rule as `weights`.

        Returns:
            One expectation value for each observable (for each sample, in a batch).
        """
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        return batch_call(self._run_one, weights, inputs)

    def _run_one(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return self.submit(weights, inputs).result()

    def submit(self, weights: np.ndarray, inputs: np.ndarray) -> IQMJobHandle:
        """Submit the circuit without waiting, and return a job handle.

        All the measurement groups go into one job, so `handle.result()` waits once
        and returns the whole vector of expectation values.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A 1-D vector of input features.

        Returns:
            handle (JobHandle): A handle for the submitted job.
        """
        job = self._submit_bases([group.basis for group in self._groups], weights, inputs)
        return IQMJobHandle(
            job=job,
            groups=self._groups,
            n_observables=self._n_observables,
            n_qubits=self._n_qubits,
            poll_timeout_seconds=self._config.poll_timeout_seconds,
        )

    def counts(self, weights: np.ndarray, inputs: np.ndarray) -> dict[str, int]:
        """Measure one sample in the computational basis and return the histogram.

        It runs one job of `shots` shots, measured in the `Z` basis with no change
        of basis.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A 1-D vector of input features.

        Returns:
            counts (dict[str, int]): The number of shots for each measured
                bitstring. The leftmost bit of each key is qubit `0`.
        """
        z_basis = ("Z",) * self._n_qubits
        job = self._submit_bases([z_basis], weights, inputs)
        result = job_result(job, self._config.poll_timeout_seconds)
        return _sampling.normalize_counts(_counts_list(result, 1)[0], self._n_qubits)

    def sample(self, weights: np.ndarray, inputs: np.ndarray) -> dict[str, int]:
        """Return the same histogram as `counts()`.

        This is an alias of
        [`counts`][eigenkit.core.backends.iqm_backend.IQMCompiled.counts], under
        the name that `Circuit.sample()` looks for.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A 1-D vector of input features.

        Returns:
            counts (dict[str, int]): The number of shots for each measured
                bitstring.
        """
        return self.counts(weights, inputs)

    def probabilities(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Return the measurement distribution over all `2**n_qubits` basis states.

        The distribution is estimated from the shot frequencies, and it is not the
        exact one: this is hardware, and the basis states that no shot measured get
        a probability of exactly `0`. It is limited to `_sampling.MAX_DENSE_QUBITS`
        qubits. For wider circuits, read `sample()` directly.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A 1-D vector of input features.

        Returns:
            probabilities (np.ndarray): An array of shape `(2**n_qubits,)` whose
                entries add up to `1`.
        """
        return _sampling.probabilities_from_counts(
            self.counts(weights, inputs), self._n_qubits
        )

    # ---- all-zeros probabilities (quantum kernels) ----

    def _zero_fractions(self, result) -> np.ndarray:
        """All-zeros share of every histogram in `result`, in submission order."""
        counts = result.get_counts()
        if isinstance(counts, dict):
            counts = [counts]
        return np.array(
            [
                zero_fraction(
                    _sampling.normalize_counts(c, self._n_qubits), self._n_qubits
                )
                for c in counts
            ],
            dtype=np.float64,
        )

    def submit_zero_state_probabilities(
        self, weights: np.ndarray, inputs: np.ndarray
    ) -> ProbabilityJobHandle:
        """Submit the all-zeros probabilities of the rows of `inputs`, without waiting.

        The rows go into one IQM job of `B` circuits. In `"parametric"` mode, the
        circuit measured in the `Z` basis is transpiled once, and each row only
        sets the values. In `"per_call"` mode, every row is transpiled.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A batch of inputs, of shape `(B, n_inputs)`.

        Returns:
            A handle whose `result()` is the array of probabilities, of shape `(B,)`.
        """
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
        z_basis = ("Z",) * self._n_qubits
        circuits = [self._bound_circuit(z_basis, weights, row) for row in inputs]
        job = self._backend.run(circuits, shots=self._config.shots)
        return ProbabilityJobHandle(
            job, self._zero_fractions, self._config.poll_timeout_seconds
        )

    def retrieve_zero_state_probabilities(self, job_id: str) -> ProbabilityJobHandle:
        """Reconnect to a job that `submit_zero_state_probabilities()` submitted.

        Args:
            job_id (str): The identifier of the IQM job.

        Returns:
            A handle whose `result()` is the array of probabilities.

        Raises:
            NotImplementedError: For a local fake backend, which keeps no history
                of its jobs, or for a backend that cannot retrieve jobs.
        """
        retrieve = getattr(self._backend, "retrieve_job", None)
        if _is_local_backend(self._backend) or retrieve is None:
            raise NotImplementedError(
                "a local IQM fake backend keeps no job history; re-attaching by "
                "id needs a live IQM server."
            )
        return ProbabilityJobHandle(
            retrieve(job_id), self._zero_fractions, self._config.poll_timeout_seconds
        )

    def zero_state_probabilities(
        self, weights: np.ndarray, inputs: np.ndarray
    ) -> np.ndarray:
        """Return the all-zeros probability of every row of `inputs`, from one job.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A batch of inputs, of shape `(B, n_inputs)`.

        Returns:
            The probabilities estimated from the shots, of shape `(B,)`.
        """
        return self.submit_zero_state_probabilities(weights, inputs).result()

    def zero_state_probability(self, weights: np.ndarray, inputs: np.ndarray) -> float:
        """Return the all-zeros probability of one sample, estimated from the shots.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A 1-D vector of input features.

        Returns:
            The probability.
        """
        return float(
            self.zero_state_probabilities(weights, np.asarray(inputs)[None, :])[0]
        )

    # ---- resources ----

    def transpiled_circuit(self, measured: bool = True) -> "QuantumCircuit":
        """Return the circuit measured in `Z` and transpiled for this device.

        The parameters are not set. Use it to draw or inspect the circuit.

        Args:
            measured: It is accepted for consistency with the Qiskit Runtime
                backend, and it has no effect, because IQM always submits measured
                circuits.

        Returns:
            A transpiled `qiskit.QuantumCircuit`.
        """
        z_basis = ("Z",) * self._n_qubits
        transpiled = self._measured.get(z_basis)
        if transpiled is None:
            transpiled = self._transpile(self._measure_in(z_basis))
            if self._config.transpile_mode == "parametric":
                self._measured[z_basis] = transpiled
        return transpiled

    def resources(self, measured: bool = True) -> dict:
        """Report the resources that the transpiled circuit uses on this device.

        The two-qubit gate counts include `move` on Star devices, because every
        `CZ` there goes through the resonator. In `"per_call"` mode, the report
        describes the parametric circuit, and the circuits with the values set can
        merge a few more rotations.

        Args:
            measured: It is accepted for consistency with the Qiskit Runtime
                backend, and it has no effect.

        Returns:
            A dictionary with the same keys as `resources()` of the Qiskit Runtime
                backend.
        """
        z_basis = ("Z",) * self._n_qubits
        transpiled = self.transpiled_circuit()
        report = transpiled_resources(
            transpiled,
            self._backend,
            self._measure_in(z_basis),
            optimization_level=self._config.optimization_level,
            seed_transpiler=self._seed,
        )
        if report["estimated_success_probability"] is None:
            report["estimated_success_probability"] = _error_profile_success_probability(
                transpiled, self._backend
            )
        return {
            "backend_name": getattr(self._backend, "name", str(self._backend)),
            "is_fake": _is_local_backend(self._backend),
            "n_logical_qubits": self._n_qubits,
            "optimization_level": self._config.optimization_level,
            "seed_transpiler": self._config.seed_transpiler,
            **report,
        }


class IQMBackend:
    """Provider for IQM quantum computers, through iqm-client.

    It is registered as `"iqm"` when iqm-client is installed, and its module is
    imported only when it is first used. A real device uses your IQM quota: the
    first compilation for a real machine issues a
    [`CostWarning`][eigenkit.core.provider.CostWarning], and `dry_run_info()`
    reports the size of a run and the number of circuits that each call submits,
    beforehand. Compiling for one of IQM's fake backends is free and gives no
    warning.

    This class is internal: you select the backend with `provider="iqm"` or by
    passing an [`IQMConfig`][eigenkit.core.backends.iqm_backend.IQMConfig], not by
    creating this class.
    """

    def __init__(self, config: IQMConfig | None = None, name: str = "iqm"):
        """Build the backend from a device configuration.

        Args:
            config (IQMConfig): The device settings. `None` builds the default
                configuration, which reads the server address from
                `IQM_SERVER_URL`.
            name (str): The registry name, which error messages use.
        """
        self.name = name
        self.config = config or IQMConfig()

    @property
    def gradient_method(self) -> str:
        """How this backend computes gradients.

        It is `"none"`, unless the configuration sets `enable_gradients`, and then
        it is `"parameter_shift"`. A parameter-shift backward pass costs at least
        `2 * n_weights` jobs, each with the full number of shots, so gradients are
        refused unless you turn them on.
        """
        return "parameter_shift" if self.config.enable_gradients else "none"

    @property
    def capabilities(self) -> Capabilities:
        """What this backend supports.

        It supports every gate except state preparation, arbitrary unitaries and
        `CPHASE`. It samples with a finite number of shots, which is the only thing
        IQM offers. It has no native batching, because a batch takes one job for
        each sample. It has no qubit limit of its own, and the number of qubits of
        the device applies. See
        [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        return Capabilities(
            supported_gates=ALL_GATES - DATA_GATES - frozenset({"CPHASE"}),
            # One job per sample: the measurement groups of a single sample
            # share a job, but separate samples do not.
            supports_batching=False,
            supports_shots=True,
            max_qubits=None,
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> IQMCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        It is called once for each `Circuit`, and the handle is reused for every
        call. So the expensive work happens here: planning the measurement groups
        and, in `"parametric"` mode, transpiling one circuit for each group.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that the handle
                measures.

        Returns:
            An `IQMCompiled` handle.
        """
        return IQMCompiled(circuit, observables, self.config)
