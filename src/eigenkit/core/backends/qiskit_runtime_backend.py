"""The Qiskit Runtime backend, for IBM Quantum hardware.

Qiskit and qiskit-ibm-runtime are optional dependencies
(`pip install eigenkit[qiskit]`). They are imported only when they are needed, so
the rest of the library works without them. If Qiskit is missing, `compile()`
raises a clear `ImportError`. If qiskit-ibm-runtime is missing, the same kind of
error comes when it is first needed: when `compile()` looks up a backend by name,
or when the first job is submitted.

The backend runs on real IBM backends (Heron devices such as `ibm_fez`,
`ibm_marrakesh` and `ibm_torino`), on the device that `service.least_busy()`
chooses (`backend_name="least_busy"`), and, for local testing, on the fake backends
with noise models from `qiskit_ibm_runtime.fake_provider` and on
`qiskit_aer.AerSimulator`.

`QiskitRuntimeConfig(primitive=...)` chooses one of two primitives:

- `"estimator"`, the default, computes the expectation values of Pauli sums with
  `EstimatorV2`, which does the basis rotation and the averaging on the server.
  Each observable is one Primitive Unified Bloc (PUB) of a single job, and only
  the numbers come back.
- `"sampler"` computes the same expectation values from raw histograms of
  bitstrings with `SamplerV2`. The Pauli terms of the observables are sorted into
  qubit-wise commuting measurement groups, one circuit is run for each group, and
  the counts are reduced locally (see the `_sampling` module). There is one PUB
  per *group* instead of one per observable, and the shot budget is explicit.
  This is what you want when the shot cost of a result is itself what you
  measure.

`sample()` and `probabilities()` always use `SamplerV2`, whatever the configured
primitive, because they are sampling operations and `EstimatorV2` returns no
bitstrings. `primitive` only chooses how the expectation values are computed.

`submit()` runs `run`, `sample` or `probabilities` without blocking (see
`QiskitRuntimeCompiled.submit_methods`), and it can tag the IBM job so that you
can find it again. `attach()` rebuilds a handle from the id of a job submitted
earlier, for example after the process that submitted it restarted. The
`result()` of either handle returns what the blocking method of the same name
returns.

`zero_state_probability()` and its batched form `zero_state_probabilities()` are
used by quantum kernels. They run one `SamplerV2` job, measured in the `Z` basis,
for the whole batch. They are not `submit()` methods:
`submit_zero_state_probabilities()` submits the batch without blocking, and
`retrieve_zero_state_probabilities()` attaches to it again by its job id.

The credentials are looked up in this order, and the most explicit one wins:

1. `QiskitRuntimeConfig(token=..., channel=..., instance=...)`.
2. The environment variable `QISKIT_IBM_TOKEN`.
3. The saved account (`~/.qiskit/qiskit-ibm.json`).

The channel defaults to `"ibm_quantum_platform"`. The legacy `"ibm_quantum"`
channel was retired with the old IBM Quantum Platform, and `QiskitRuntimeService`
no longer accepts it at all.

For testing, pass a `FakeBackendV2` instance directly as `backend_instance`. No
token or service is needed.

By default, `gradient_method = "none"`, because runs on a quantum processing unit
(QPU) are too expensive for parameter-shift gradients: one backward step costs
at least `2 * n_weights` jobs, each with the full shot count. The PyTorch wrapper
raises a clear error on `.backward()`. Train on a simulator first, then use the
trained weights on the QPU. `enable_gradients=True` in the configuration turns on
parameter-shift gradients on the device, and compiling for a paid device then
gives a `CostWarning` that states the cost of one backward pass.

`compile()` runs `qiskit.transpile(circuit, backend=target,
optimization_level=N)` once and caches the transpiled circuit. For the estimator,
it also caches the `SparsePauliOp` operators, mapped to the layout of the
transpiled circuit. The sampler transpiles each measured circuit on its first use
and caches it too. Each `run()` or `submit()` then only binds the parameter values
in the PUBs that it passes to the primitive.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z, register
    from eigenkit.configs import QiskitRuntimeConfig

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # by a registered name (this backend is opt-in, so register it first)
    register("qiskit-runtime", QiskitRuntimeConfig(backend_name="ibm_fez"))
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="qiskit-runtime")

    # or with the configuration, when a device setting matters
    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=QiskitRuntimeConfig(backend_name="ibm_fez"),
    )
    ```

    It is opt-in, and it costs money. The first compile for a paid device gives a
    `CostWarning`, and `dry_run_info()` reports the shape of a run, including how
    many circuits a call runs, before you submit it.
"""

from __future__ import annotations

import os
import random
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence

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
from .qiskit_backend import (
    _apply_gate_symbolic,
    _require_qiskit,
)
from .qiskit_backend import (
    _observable_to_qiskit as _observable_to_sparsepauliop,
)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from qiskit import QuantumCircuit


Primitive = Literal["estimator", "sampler"]

_SUBMIT_METHODS: frozenset[str] = frozenset({"run", "sample", "probabilities"})

# IBM job-tag limits: at most 8 tags per job, each at most 86 characters
# (Qiskit Runtime REST API, "Jobs" tags).
_MAX_JOB_TAGS = 8
_MAX_JOB_TAG_LENGTH = 86

# IBM job ids are short alphanumeric strings; local testing mode uses UUIDs.
# Checked before any service call, so a malformed id never reaches the API.
_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _check_submit_method(method: str) -> None:
    if method not in _SUBMIT_METHODS:
        raise ValueError(
            f"method {method!r} cannot be submitted asynchronously on Qiskit "
            f"Runtime; submit_methods is {sorted(_SUBMIT_METHODS)}."
        )


def _check_one_sample(name: str, weights, inputs) -> None:
    if np.ndim(weights) != 1 or np.ndim(inputs) != 1:
        raise ValueError(
            f"{name}() takes one sample on Qiskit Runtime: weights and inputs "
            f"must be 1-D, got shapes {np.shape(weights)} and {np.shape(inputs)}. "
            f"Loop in Python for a batch."
        )


def _validate_job_tags(tags: Mapping[str, str] | None) -> list[str] | None:
    """Check `tags` against the IBM job-tag limits and return them as IBM
    job tags, `"key=value"` sorted by key; `None` when there are none."""
    if not tags:
        return None
    if len(tags) > _MAX_JOB_TAGS:
        raise ValueError(
            f"at most {_MAX_JOB_TAGS} job tags are allowed, got {len(tags)}."
        )
    job_tags = []
    for key in sorted(tags, key=str):
        value = tags[key]
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"job tag keys and values must be str, got {key!r}: {value!r}."
            )
        if not key or not value:
            raise ValueError(
                f"job tag keys and values must be non-empty, got {key!r}: {value!r}."
            )
        if "=" in key:
            raise ValueError(
                f"job tag key {key!r} contains '=', which separates key from value."
            )
        entry = f"{key}={value}"
        if len(entry) > _MAX_JOB_TAG_LENGTH:
            raise ValueError(
                f"job tag {key!r} is {len(entry)} characters as key=value; IBM allows "
                f"at most {_MAX_JOB_TAG_LENGTH}."
            )
        job_tags.append(entry)
    return job_tags


def _require_qiskit_ibm_runtime() -> None:
    """Probe for qiskit-ibm-runtime. Raise a clear error if missing."""
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "qiskit-ibm-runtime is not installed. Install it via "
            "`pip install eigenkit[qiskit]`."
        ) from e


@dataclass(frozen=True)
class QiskitRuntimeConfig(SecretFieldsMixin):
    """Device configuration for the Qiskit Runtime backend.

    There are two ways to choose the target backend:

    - `backend_name` is looked up with `QiskitRuntimeService`. It needs
      credentials: a token, an environment variable or a saved account.
    - `backend_instance` is a `BackendV2` that is used directly. It is useful for
      tests with a fake backend, or to pass any backend that is already
      configured.

    Args:
        backend_name: The IBM backend name, for example `"ibm_fez"`, or
            `"least_busy"` for the least busy operational device that the account
            can use, chosen once per compiled circuit. It is required unless
            `backend_instance` is set, and it is ignored when it is. Default:
            `None`.
        backend_instance: A backend that is already configured. It takes
            precedence over `backend_name`. Default: `None`.
        shots: The number of shots per circuit run. Default: `1024`.
        optimization_level: The optimisation level passed to `qiskit.transpile`,
            from `0` to `3`. Default: `1`.
        seed_transpiler: The seed of the random passes of the transpiler, mainly
            the `SabreLayout` and `SabreSwap` passes, which choose the layout and
            the routing. `None`, the default, draws a new seed for every compile,
            so a circuit that needs routing compiles to a different number of
            gates on every run. Within one compile, all the transpilations share
            the seed, including the shifted copies that a gradient on hardware
            compiles. Set it whenever two compilations must be comparable. Since
            compiling submits nothing, trying a few seeds and keeping the cheapest
            circuit is a cheap and fair way to choose a layout.
        primitive: The Runtime primitive that computes the expectation values.
            `"estimator"`, the default, uses `EstimatorV2`, with one Primitive
            Unified Bloc (PUB) per observable in each job. `"sampler"` uses
            `SamplerV2`, with one PUB per qubit-wise commuting measurement group,
            and reduces the raw counts locally. `sample()` and `probabilities()`
            use `SamplerV2` with either value.
        group_commuting: Whether the sampler puts qubit-wise commuting Pauli terms
            into shared measurement circuits. `False` runs one circuit per term,
            which runs more circuits but keeps the shot budget of each term
            independent. The estimator ignores it. Default: `True`.
        token: The token passed to `QiskitRuntimeService` when `backend_instance`
            is `None`. With `None`, the backend uses `QISKIT_IBM_TOKEN`, then the
            saved account file. It is secret: it is never shown by `repr()`, and it
            is never saved in `.eigenkit` bundles, pickles or joblib checkpoints.
            After a load it is `None`, so set `QISKIT_IBM_TOKEN`.
        channel: The service channel: `"ibm_quantum_platform"` (the default),
            `"ibm_cloud"` or `"local"`. The legacy `"ibm_quantum"` channel was
            retired, and `QiskitRuntimeService` no longer accepts it.
        instance: The service instance, as a Cloud Resource Name (CRN) or a name,
            when one applies. Default: `None`.
        service_kwargs: Extra keyword arguments passed to `QiskitRuntimeService`,
            such as `region`, `url` or `proxies`.
        poll_timeout_seconds: How long `result()` waits before it gives up, in
            seconds. Default: `3600`.
        enable_gradients: Whether to allow parameter-shift gradients on the
            device: PyTorch `.backward()`, `jacobian()` and gradient-based
            optimisers. It is off by default, because one backward pass costs
            at least `2 * n_weights` times the jobs of a forward call. When it is on,
            compiling for a paid device gives a `CostWarning` that states this
            cost.
    """

    backend_name: str | None = None
    # `BackendV2`; typed as `Any` to avoid a hard import.
    backend_instance: Any = None
    shots: int = 1024
    optimization_level: int = 1
    seed_transpiler: int | None = None
    primitive: Primitive = "estimator"
    group_commuting: bool = True
    token: str | None = field(default=None, repr=False, metadata={"secret": True})
    channel: str = "ibm_quantum_platform"
    instance: str | None = None
    service_kwargs: dict = field(default_factory=dict)
    poll_timeout_seconds: int = 3600
    enable_gradients: bool = False

    def __post_init__(self) -> None:
        """Reject an unknown primitive when the configuration is built.

        If the check waited for the first submission, the error would come as a
        confusing failure on a paid backend, after the transpilation and the cost
        warning.
        """
        if self.primitive not in ("estimator", "sampler"):
            raise ValueError(
                f"primitive must be 'estimator' or 'sampler', got {self.primitive!r}"
            )

    def make_backend(self) -> "QiskitRuntimeBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `QiskitRuntimeBackend` for this configuration.
        """
        return QiskitRuntimeBackend(config=self)


def _resolve_service(config: QiskitRuntimeConfig):
    """Open the `QiskitRuntimeService` the config describes.

    Auth resolution falls through the chain config.token > `QISKIT_IBM_TOKEN`
    > saved account file.
    """
    _require_qiskit_ibm_runtime()
    from qiskit_ibm_runtime import QiskitRuntimeService

    # QISKIT_IBM_TOKEN is the variable that qiskit-ibm-runtime itself reads.
    token = config.token or os.environ.get("QISKIT_IBM_TOKEN")
    service_kwargs = dict(config.service_kwargs)
    if token is not None:
        service_kwargs["token"] = token
    if config.channel is not None:
        service_kwargs["channel"] = config.channel
    if config.instance is not None:
        service_kwargs["instance"] = config.instance
    return QiskitRuntimeService(**service_kwargs)


def _resolve_backend(config: QiskitRuntimeConfig):
    """Pick a Qiskit `BackendV2` based on config.

    `backend_instance` wins, and no service is built; otherwise look up
    `backend_name` via the `QiskitRuntimeService` — `"least_busy"` asks the
    service for the least busy operational device. A backend from the service
    carries it as `backend.service`, which `attach()` and
    `retrieve_zero_state_probabilities()` look jobs up in.
    """
    if config.backend_instance is not None:
        return config.backend_instance
    if not config.backend_name:
        raise ValueError(
            "QiskitRuntimeConfig needs a device: pass backend_name= (e.g. "
            "'ibm_fez', or 'least_busy'), or backend_instance= with a fake "
            "backend such as FakeFez() for local runs."
        )
    service = _resolve_service(config)
    if config.backend_name == "least_busy":
        return service.least_busy(operational=True, simulator=False)
    return service.backend(config.backend_name)


def _is_local_backend(backend) -> bool:
    """True for backends that run on this machine and cost nothing: the fake
    devices from `qiskit_ibm_runtime.fake_provider` and Aer simulators."""
    module = type(backend).__module__.lower()
    return "fake" in module or module.startswith("qiskit_aer")


class QiskitRuntimeJobHandle:
    """A handle to a Qiskit Runtime job, whose `result()` blocks and decodes it.

    `compiled.submit(weights, inputs)` and `compiled.attach(job_id)` return one of
    these. What `result()` returns depends on the `method` of the handle.
    """

    def __init__(
        self,
        job,
        n_observables: int,
        poll_timeout_seconds: int,
        reduce: Callable[[Any], np.ndarray] | None = None,
        counts: Callable[[Any], list[dict[str, int]]] | None = None,
        method: str = "run",
        *,
        n_qubits: int | None = None,
    ):
        """Wrap a submitted Runtime job as a job handle.

        `QiskitRuntimeCompiled.submit()` and `QiskitRuntimeCompiled.attach()`
        build it, and it is not constructed directly.

        Args:
            job (RuntimeJob): The IBM Runtime job returned by the primitive.
            n_observables: The number of expectation values to expect back, which
                sets the shape of the result array.
            poll_timeout_seconds: How long `result()` waits before it gives up, in
                seconds.
            reduce (Callable | None): The function that turns the result of the
                primitive into the vector of expectation values. `None` reads the
                `EstimatorV2` result of each Primitive Unified Bloc (PUB)
                directly, which is what the estimator submits. The sampler passes
                its reduction of the counts instead.
            counts (Callable | None): The function that takes the histograms out
                of the result of the primitive, one per submitted circuit. Only
                sampler jobs have histograms, because the estimator returns
                expectation values and never sees a bitstring. With `None`,
                `counts()` raises `NotImplementedError`.
            method (str): The method of the compiled circuit that the job runs,
                one of `QiskitRuntimeCompiled.submit_methods`. It decides how
                `result()` decodes the job.
            n_qubits (int | None): The number of qubits of the circuit, which sets
                the size of the `"probabilities"` distribution.

        Raises:
            ValueError: If `method` is not in `QiskitRuntimeCompiled.submit_methods`,
                if `method` is `"sample"` or `"probabilities"` without `counts`,
                or if `method` is `"probabilities"` without `n_qubits`.
        """
        _check_submit_method(method)
        if method != "run" and counts is None:
            raise ValueError(f"a {method!r} handle needs the counts decoder.")
        if method == "probabilities" and n_qubits is None:
            raise ValueError("a 'probabilities' handle needs n_qubits.")
        self._job = job
        self._n_observables = n_observables
        self._poll_timeout_seconds = poll_timeout_seconds
        self._reduce = reduce
        self._counts = counts
        self._method = method
        self._n_qubits = n_qubits

    @property
    def method(self) -> str:
        """Return the method of the compiled circuit that this job runs.

        It is `"run"`, `"sample"` or `"probabilities"`.
        """
        return self._method

    def status(self) -> str:
        """Return the state of the job as IBM Quantum reports it, without blocking.

        Examples of states are `"QUEUED"`, `"RUNNING"`, `"DONE"`, `"CANCELLED"` and
        `"ERROR"`.
        """
        s = self._job.status()
        return s.name if hasattr(s, "name") else str(s)

    def cancel(self) -> None:
        """Ask IBM Quantum to cancel the job, without blocking."""
        self._job.cancel()

    def job_id(self) -> str:
        """Return the IBM Quantum job id.

        Use it to look the job up in the console or in `qiskit-ibm-runtime`.
        """
        return self._job.job_id()

    def counts(self) -> list[dict[str, int]]:
        """Block until the job finishes, then return the raw histograms.

        There is one histogram per measurement group. In each key, qubit 0 is the
        leftmost character, which is the most significant bit. A `"sample"` or
        `"probabilities"` job measured one circuit, so it has one histogram.

        These are the numbers that the device returned, and
        [`result`][eigenkit.core.backends.qiskit_runtime_backend.QiskitRuntimeJobHandle.result]
        is these numbers reduced against the observables. Because both come from a
        single job, a run can be scored again offline, against different
        observables or with a different metric, without paying for it twice.

        Raises:
            NotImplementedError: For a `"run"` job on the estimator, which returns
                expectation values and never sees a bitstring. Set
                `primitive="sampler"` to get counts.
        """
        if self._counts is None:
            raise NotImplementedError(
                "the estimator primitive returns expectation values, not "
                "bitstrings; use QiskitRuntimeConfig(primitive='sampler') to "
                "get counts back from the job"
            )
        return self._counts(job_result(self._job, self._poll_timeout_seconds))

    def result(self) -> np.ndarray | dict[str, int]:
        """Block until the job finishes, then decode it by the `method` of the handle.

        A handle from `attach()` decodes a job exactly as the handle that `submit()`
        returned would have.

        Returns:
            For `method="run"`, one float64 expectation value per observable, of
                shape `(n_obs,)`, in the order of submission, as
                `QiskitRuntimeCompiled.run()` returns. For `method="sample"`, the
                histogram that `QiskitRuntimeCompiled.sample()` returns, which maps
                each bitstring to its number of shots, with qubit 0 as the leftmost
                character. For `method="probabilities"`, the float64 distribution
                of shape `(2**n_qubits,)` that
                `QiskitRuntimeCompiled.probabilities()` returns.

        Raises:
            ValueError: If the results of the job do not have the shape that the
                `method` and the circuit of this handle produce, for example
                because the handle was attached with the wrong `method`. Also, for
                `"probabilities"`, if the circuit has more than
                `_sampling.MAX_DENSE_QUBITS` (`24`) qubits.
        """
        primitive_result = job_result(self._job, self._poll_timeout_seconds)
        if self._method == "run" and self._reduce is None:
            return self._expectation_values(primitive_result)
        try:
            if self._method == "run":
                return self._reduce(primitive_result)
            histogram = self._counts(primitive_result)[0]
        except (AttributeError, IndexError, KeyError, RuntimeError) as exc:
            raise self._mismatch(f"cannot be decoded as sampler counts ({exc})") from exc
        if self._method == "sample":
            return histogram
        return _sampling.probabilities_from_counts(histogram, self._n_qubits)

    def _expectation_values(self, primitive_result) -> np.ndarray:
        # EstimatorV2 returns one PubResult per PUB; we submit one PUB
        # per observable, so the indices line up.
        if len(primitive_result) != self._n_observables:
            raise self._mismatch(
                f"has {len(primitive_result)} results, expected one per observable "
                f"({self._n_observables})"
            )
        output = np.zeros(self._n_observables, dtype=np.float64)
        for k in range(self._n_observables):
            evs = getattr(primitive_result[k].data, "evs", None)
            if evs is None:
                raise self._mismatch("carries no expectation values")
            output[k] = float(np.asarray(evs).reshape(-1)[0])
        return output

    def _mismatch(self, detail: str) -> ValueError:
        return _job_mismatch(self._job, self._method, detail)


def _check_pub_count(result, expected: int) -> None:
    """A sampler result must hold one PUB result per submitted circuit."""
    if len(result) != expected:
        raise IndexError(f"{len(result)} PUB results, expected {expected}")


def _job_mismatch(job, method: str, detail: str) -> ValueError:
    return ValueError(
        f"IBM job {job.job_id()} {detail}; it was probably submitted with "
        f"another method, circuit or observables than this handle's "
        f"(method={method!r})."
    )


class QiskitRuntimeCompiled(CompiledCircuit):
    """A circuit transpiled for an IBM Quantum backend.

    This class is internal. It is built by
    [`QiskitRuntimeBackend.compile()`][eigenkit.core.backends.qiskit_runtime_backend.QiskitRuntimeBackend.compile]
    and obtained from it, and it is never constructed directly. The circuit is
    transpiled once and cached (each measured circuit of the sampler on its first
    use), so `run()` and `submit()` only bind the parameter values. `submit()`
    returns a
    [`QiskitRuntimeJobHandle`][eigenkit.core.backends.qiskit_runtime_backend.QiskitRuntimeJobHandle]
    without blocking, and `attach()` rebuilds that handle from a job id.
    """

    submit_methods: frozenset[str] = _SUBMIT_METHODS
    """The methods `submit()` and `attach()` accept: `"run"`, `"sample"` and
    `"probabilities"`. `zero_state_probability` is not one of them: its batched
    async form is `submit_zero_state_probabilities()`. Any method
    added to this adapter's async path must also be added here, or callers that
    check this set will refuse it."""

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: QiskitRuntimeConfig,
        *,
        warn: bool = True,
    ):
        """Prepare the circuit for repeated evaluation.

        `QiskitRuntimeBackend.compile()` builds it, and there is no reason to
        construct one directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that this handle
                measures.
            config (QiskitRuntimeConfig): The device settings.
            warn (bool): Whether to give the cost warnings for a paid device. It
                is `False` only for the shifted copies that `compile_shifted`
                builds. Default: `True`.
        """
        if config.shots <= 0:
            raise ValueError(
                f"Qiskit Runtime requires shots > 0; got shots={config.shots}"
            )

        _require_qiskit()
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import Parameter as QiskitParameter

        self._config = config
        self._observables = tuple(observables)
        # One seed for every transpile of this handle and of its shifted copies,
        # so the terms of a parameter-shift gradient all run on the same qubits.
        self._seed = (
            config.seed_transpiler
            if config.seed_transpiler is not None
            else random.randrange(2**31)
        )
        self._n_qubits_logical = circuit.n_qubits

        # Build the parametric circuit using ordered Parameter symbols.
        # Order matters: parameter values are passed to EstimatorV2 in
        # this order.
        self._weight_symbols = [
            QiskitParameter(f"w{i}") for i in range(circuit.n_weights)
        ]
        self._input_symbols = [QiskitParameter(f"x{i}") for i in range(circuit.n_inputs)]
        qc = QuantumCircuit(circuit.n_qubits)
        for gate in circuit.gates:
            _apply_gate_symbolic(qc, gate, self._weight_symbols, self._input_symbols)

        # Kept unmeasured: the sampler path appends its own basis rotation and
        # measurements to a copy of this, and transpiles that instead.
        self._base_circuit = qc
        self._n_observables = len(observables)

        # Resolve the target backend (FakeBackend, real QPU, or service lookup),
        # once: a service lookup costs several round trips to IBM.
        self._backend = _resolve_backend(config)

        # The sampler runs one circuit per measurement group; the estimator runs
        # one per observable. Plan first, so the cost warning can state the real
        # per-call job count rather than the one the other primitive would pay.
        self._groups: tuple[_sampling.MeasurementGroup, ...] = ()
        if config.primitive == "sampler":
            self._groups = _sampling.plan_measurements(
                observables,
                circuit.n_qubits,
                group_commuting=config.group_commuting,
            )
        self._n_runs_per_call = (
            len(self._groups) if config.primitive == "sampler" else self._n_observables
        )

        # Emit the cost warning once at compile time, before any billable
        # submission can happen.
        backend_name = getattr(self._backend, "name", str(self._backend))
        if warn and not _is_local_backend(self._backend):
            warnings.warn(
                f"Qiskit Runtime compile against '{backend_name}' via "
                f"{config.primitive}. Each run/submit will consume IBM Quantum "
                f"compute time at shots={config.shots} "
                f"(one job of {self._n_runs_per_call} PUBs per call).",
                CostWarning,
                stacklevel=4,
            )
            if config.enable_gradients:
                _warn_gradient_cost(
                    f"Qiskit Runtime backend '{backend_name}'",
                    circuit,
                    self._n_runs_per_call,
                    config.shots,
                    unit="PUBs",
                    stacklevel=4,
                )

        # Compile-time transpilation. Heavy operation; cached after this.
        self._transpiled_circuit = transpile(
            qc,
            backend=self._backend,
            optimization_level=config.optimization_level,
            seed_transpiler=self._seed,
        )

        # Operators are in the logical layout; relayout them to match
        # the transpiled circuit, since transpile may permute qubits.
        # Sampling never evaluates them, and `apply_layout` is not free on a
        # 100+ qubit device, so only the estimator path pays for them.
        self._operators = (
            [
                _observable_to_sparsepauliop(obs, circuit.n_qubits).apply_layout(
                    self._transpiled_circuit.layout
                )
                for obs in observables
            ]
            if config.primitive == "estimator"
            else []
        )

        # `transpile()` can reorder parameters, so read the canonical
        # order from the transpiled circuit rather than the source.
        self._param_order = list(self._transpiled_circuit.parameters)

        # Measured circuits are built on first use: a caller running the
        # estimator never needs them, and each one costs its own transpile.
        # Keyed by measurement basis, so the Z-basis circuit that `sample()`
        # and `probabilities()` use is shared with any all-Z group.
        self._measured: dict[tuple[str, ...], tuple[Any, list]] = {}

    def compile_shifted(self, circuit: CircuitSpec) -> "QiskitRuntimeCompiled":
        """Compile a copy of this circuit with one gate angle shifted.

        The chain-rule gradient compiles one copy per shifted gate. Each copy runs
        on the device of this handle, never on a new `"least_busy"` choice, and is
        transpiled with the seed of this handle, so every term of the gradient
        comes from the same qubits. The compile of this handle already gave the
        cost warnings, so the copy gives none.

        Args:
            circuit (CircuitSpec): The shifted circuit.

        Returns:
            A `QiskitRuntimeCompiled` handle for `circuit`.
        """
        config = replace(
            self._config, backend_instance=self._backend, seed_transpiler=self._seed
        )
        return QiskitRuntimeCompiled(circuit, self._observables, config, warn=False)

    # ---- inspection ----

    def dry_run_info(self) -> dict:
        """Report the size of a run, before anything is submitted.

        It reports the target backend and whether it is a fake one, the primitive
        in use, the number of shots, the optimisation level, the transpiler seed
        set in the configuration, the numbers of logical and physical qubits, the
        gate counts and the depth of the transpiled circuit, the numbers of
        observables and of measurement groups, and how many circuits one call
        runs: one per observable with the estimator, and one per measurement group
        with the sampler.

        `transpiled_gate_counts` and `transpiled_depth` describe the circuit that
        is actually submitted. With the estimator, this is the circuit without
        measurements, which carries its observables separately, so both are
        single values. With the sampler, there is one circuit per measurement
        group, with the basis rotation and the measurements added *before*
        transpiling, so that the layout can place them on the qubits that the
        circuit ran on. Both are then dictionaries keyed by basis, as in the IQM
        backend. Reporting the circuit without measurements there would
        understate a routed circuit, because the transpiler sees a different input
        and chooses a different layout.

        Returns:
            info (dict): The fields above, keyed by name.
        """
        backend = self._backend
        gate_counts = dict(self._transpiled_circuit.count_ops())
        depth = self._transpiled_circuit.depth()
        if self._config.primitive == "sampler":
            submitted = {
                group.basis: self._measured_circuit(group.basis)[0]
                for group in self._groups
            }
            gate_counts = {
                "".join(basis): dict(qc.count_ops()) for basis, qc in submitted.items()
            }
            depth = {"".join(basis): qc.depth() for basis, qc in submitted.items()}
        return {
            "backend_name": getattr(backend, "name", str(backend)),
            "is_fake": _is_local_backend(backend),
            "primitive": self._config.primitive,
            "shots": self._config.shots,
            "optimization_level": self._config.optimization_level,
            "seed_transpiler": self._config.seed_transpiler,
            "n_logical_qubits": self._n_qubits_logical,
            "n_physical_qubits": getattr(backend, "num_qubits", None),
            "transpiled_gate_counts": gate_counts,
            "transpiled_depth": depth,
            "n_observables": self._n_observables,
            "n_measurement_groups": len(self._groups),
            "n_circuit_runs_per_call": self._n_runs_per_call,
        }

    # ---- run / submit ----

    def _param_values(
        self, weights: np.ndarray, inputs: np.ndarray, param_order=None
    ) -> list[float]:
        """Build the parameter-value list in the order the transpiled
        circuit expects. Each Parameter is named `w<i>` or `x<i>`, which
        determines whether to read from `weights` or `inputs`.

        `param_order` defaults to the bare circuit's. Each measured circuit is
        transpiled separately and can come back with its parameters in a
        different order, so the sampler path passes its own.
        """
        values = []
        for symbol in self._param_order if param_order is None else param_order:
            name = symbol.name
            if name.startswith("w"):
                values.append(float(weights[int(name[1:])]))
            elif name.startswith("x"):
                values.append(float(inputs[int(name[1:])]))
            else:
                raise RuntimeError(
                    f"unexpected parameter name {name!r} in transpiled circuit"
                )
        return values

    def _build_pubs(self, weights: np.ndarray, inputs: np.ndarray):
        param_values = self._param_values(weights, inputs)
        return [(self._transpiled_circuit, op, param_values) for op in self._operators]

    def _estimator(self):
        _require_qiskit_ibm_runtime()
        from qiskit_ibm_runtime import EstimatorV2

        estimator = EstimatorV2(mode=self._backend)
        estimator.options.default_shots = self._config.shots
        return estimator

    # ---- sampler path ----

    def _sampler(self):
        _require_qiskit_ibm_runtime()
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=self._backend)
        sampler.options.default_shots = self._config.shots
        return sampler

    def _measured_circuit(self, basis: tuple[str, ...]):
        """Transpiled circuit measuring in `basis`, plus its parameter order.

        Built once per basis and cached: appending the rotation before
        transpiling is what lets the layout map the measurements onto the same
        physical qubits the circuit ran on.
        """
        cached = self._measured.get(basis)
        if cached is not None:
            return cached

        from qiskit import transpile

        transpiled = transpile(
            self._logical_measured(basis),
            backend=self._backend,
            optimization_level=self._config.optimization_level,
            seed_transpiler=self._seed,
        )
        entry = (transpiled, list(transpiled.parameters))
        self._measured[basis] = entry
        return entry

    def _logical_measured(self, basis: tuple[str, ...]):
        """The untranspiled circuit rotated into `basis` and measured."""
        qc = self._base_circuit.copy()
        for name, qubit in _sampling.rotation_gates(basis):
            if name == "H":
                qc.h(qubit)
            elif name == "Sdg":
                qc.sdg(qubit)
            else:  # pragma: no cover - _sampling emits only H and Sdg
                raise RuntimeError(f"unexpected basis-rotation gate {name!r}")
        qc.measure_all()
        return qc

    def _counts_from_pub(self, pub_result, index: int | None = None) -> dict[str, int]:
        """Pull one histogram out of a `SamplerV2` PUB result, qubit-0-MSB.

        `measure_all()` names its classical register `meas`, but the register
        is read by position rather than by name so a differently-named one
        still works. `index` selects a row when the PUB broadcast a parameter
        batch.
        """
        data = pub_result.data
        fields = list(data.keys())
        if len(fields) != 1:
            raise RuntimeError(
                f"expected one classical register in the sampler result, got {fields}"
            )
        bit_array = data[fields[0]]
        counts = bit_array.get_counts() if index is None else bit_array.get_counts(index)
        return _sampling.normalize_counts(counts, self._n_qubits_logical)

    def _sampler_pubs(self, bases, weights: np.ndarray, inputs: np.ndarray):
        pubs = []
        for basis in bases:
            circuit, param_order = self._measured_circuit(basis)
            pubs.append((circuit, self._param_values(weights, inputs, param_order)))
        return pubs

    def _z_basis_pubs(self, weights: np.ndarray, inputs: np.ndarray):
        """The single Z-basis PUB `sample()` and `probabilities()` run."""
        z_basis = ("Z",) * self._n_qubits_logical
        return self._sampler_pubs([z_basis], weights, inputs)

    def _z_counts(self, result) -> list[dict[str, int]]:
        """The one histogram of a Z-basis job, as a one-element list."""
        _check_pub_count(result, 1)
        return [self._counts_from_pub(result[0])]

    def _counts_per_group(self, result) -> list[dict[str, int]]:
        """One histogram per measurement group, in submission order."""
        _check_pub_count(result, len(self._groups))
        return [self._counts_from_pub(result[k]) for k in range(len(self._groups))]

    def _reduce_sampler_result(self, result) -> np.ndarray:
        counts_per_group = self._counts_per_group(result)
        return _sampling.expectations_from_counts(
            self._groups,
            counts_per_group,
            self._n_observables,
            self._n_qubits_logical,
        )

    def _run_sampler_broadcast(
        self, weights: np.ndarray, inputs: np.ndarray
    ) -> np.ndarray:
        """Batched sampler evaluation: one job per measurement group, whatever
        the batch size, via the same `(B, n_params)` broadcasting the estimator
        path uses."""
        batch_size = inputs.shape[0]
        pubs = []
        for group in self._groups:
            circuit, param_order = self._measured_circuit(group.basis)
            pubs.append(
                (circuit, self._param_values_batched(weights, inputs, param_order))
            )
        result = job_result(self._sampler().run(pubs), self._config.poll_timeout_seconds)
        output = np.empty((batch_size, self._n_observables), dtype=np.float64)
        for row in range(batch_size):
            counts_per_group = [
                self._counts_from_pub(result[k], row) for k in range(len(self._groups))
            ]
            output[row] = _sampling.expectations_from_counts(
                self._groups,
                counts_per_group,
                self._n_observables,
                self._n_qubits_logical,
            )
        return output

    def counts(self, weights: np.ndarray, inputs: np.ndarray) -> dict[str, int]:
        """Return the histogram in the computational basis for one sample.

        It is always a `SamplerV2` job, whatever `primitive` the configuration
        names, because `EstimatorV2` has no bitstrings to return. It is one job of
        `shots` shots.

        Args:
            weights (np.ndarray): The parameter vector, which must be 1-D.
            inputs (np.ndarray): The input features, which must be 1-D.

        Returns:
            counts (dict[str, int]): The number of shots for each measured
                bitstring. The first character of each key is qubit 0, which is
                the most significant bit.
        """
        job = self._sampler().run(self._z_basis_pubs(weights, inputs))
        return self._z_counts(job_result(job, self._config.poll_timeout_seconds))[0]

    def sample(self, weights: np.ndarray, inputs: np.ndarray) -> dict[str, int]:
        """Return the same histogram as `counts()`.

        This is an alias of
        [`counts`][eigenkit.core.backends.qiskit_runtime_backend.QiskitRuntimeCompiled.counts],
        under the name that the `Circuit.sample()` dispatch looks for.

        Args:
            weights (np.ndarray): The parameter vector, which must be 1-D.
            inputs (np.ndarray): The input features, which must be 1-D.

        Returns:
            counts (dict[str, int]): The number of shots for each measured
                bitstring.
        """
        return self.counts(weights, inputs)

    def probabilities(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Return the measurement distribution over all `2**n_qubits` basis states.

        It is the estimate from the shot frequencies, not the exact distribution:
        on hardware, the basis states that were never sampled come back as exact
        zeros. It is limited to circuits of at most `_sampling.MAX_DENSE_QUBITS`
        (`24`) qubits. For a wider circuit, read `sample()` directly.

        Args:
            weights (np.ndarray): The parameter vector, which must be 1-D.
            inputs (np.ndarray): The input features, which must be 1-D.

        Returns:
            probabilities (np.ndarray): The distribution, of shape
                `(2**n_qubits,)`, which sums to `1`.
        """
        return _sampling.probabilities_from_counts(
            self.counts(weights, inputs), self._n_qubits_logical
        )

    # ---- all-zeros probabilities (quantum kernels) ----

    def _zero_state_pub(self, weights: np.ndarray, inputs: np.ndarray):
        z_basis = ("Z",) * self._n_qubits_logical
        circuit, param_order = self._measured_circuit(z_basis)
        return (circuit, self._param_values_batched(weights, inputs, param_order))

    def _zero_fractions(self, result) -> np.ndarray:
        """All-zeros share of every row of the single PUB in `result`."""
        pub_result = result[0]
        n_rows = int(np.prod(pub_result.data[list(pub_result.data.keys())[0]].shape))
        return np.array(
            [
                zero_fraction(
                    self._counts_from_pub(pub_result, row), self._n_qubits_logical
                )
                for row in range(n_rows)
            ],
            dtype=np.float64,
        )

    def submit_zero_state_probabilities(
        self, weights: np.ndarray, inputs: np.ndarray
    ) -> ProbabilityJobHandle:
        """Submit the all-zeros probability of every row of `inputs` as one job.

        It submits one `SamplerV2` job and returns without waiting. The circuit is
        transpiled once and measured in the `Z` basis, and every row binds one set
        of parameter values of the same Primitive Unified Bloc (PUB), so `B`
        kernel entries cost one job of `B * shots` shots. It works whatever
        `primitive` the configuration names.

        Args:
            weights (np.ndarray): The parameter vector, which must be 1-D.
            inputs (np.ndarray): A batch, of shape `(B, n_inputs)`.

        Returns:
            A handle whose `result()` is the probability array, of shape `(B,)`.
        """
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
        job = self._sampler().run([self._zero_state_pub(weights, inputs)])
        return ProbabilityJobHandle(
            job, self._zero_fractions, self._config.poll_timeout_seconds
        )

    def retrieve_zero_state_probabilities(self, job_id: str) -> ProbabilityJobHandle:
        """Attach again to a job that `submit_zero_state_probabilities()` submitted.

        The job may have been submitted from another session.

        Args:
            job_id (str): The IBM Quantum job id.

        Returns:
            A handle whose `result()` is the probability array.

        Raises:
            ValueError: If `job_id` is not made of 1 to 64 letters, digits, `_` or
                `-`.
            NotImplementedError: For a backend without a `service` to look the job
                up in, such as a fake backend or Aer, which keep no job history.
        """
        if not (isinstance(job_id, str) and _JOB_ID_PATTERN.fullmatch(job_id)):
            raise ValueError(
                f"job_id must be an IBM job id of 1-64 letters, digits, _ or -, "
                f"got {job_id!r}."
            )
        service = getattr(self._backend, "service", None)
        if service is None:
            raise NotImplementedError(
                "a local backend_instance keeps no job history; re-attaching by "
                "id needs a backend_name looked up through QiskitRuntimeService."
            )
        return ProbabilityJobHandle(
            service.job(job_id),
            self._zero_fractions,
            self._config.poll_timeout_seconds,
        )

    def zero_state_probabilities(
        self, weights: np.ndarray, inputs: np.ndarray
    ) -> np.ndarray:
        """Return the all-zeros probability of every row of `inputs`, from one job.

        Args:
            weights (np.ndarray): The parameter vector, which must be 1-D.
            inputs (np.ndarray): A batch, of shape `(B, n_inputs)`.

        Returns:
            The estimates from the shots, of shape `(B,)`.
        """
        return self.submit_zero_state_probabilities(weights, inputs).result()

    def zero_state_probability(self, weights: np.ndarray, inputs: np.ndarray) -> float:
        """Return the all-zeros probability for one sample, estimated from shots.

        Args:
            weights (np.ndarray): The parameter vector, which must be 1-D.
            inputs (np.ndarray): The input features, which must be 1-D.

        Returns:
            The probability.
        """
        return float(
            self.zero_state_probabilities(weights, np.asarray(inputs)[None, :])[0]
        )

    # ---- resources ----

    def transpiled_circuit(self, measured: bool = True) -> "QuantumCircuit":
        """Return the Qiskit circuit that this backend submits, to draw or inspect it.

        Args:
            measured: If `True`, the default, return the circuit measured in the
                `Z` basis that `sample()`, `probabilities()` and the kernel path
                run. If `False`, return the transpiled circuit without
                measurements that the estimator runs.

        Returns:
            A transpiled `qiskit.QuantumCircuit`.
        """
        if not measured:
            return self._transpiled_circuit
        return self._measured_circuit(("Z",) * self._n_qubits_logical)[0]

    def resources(self, measured: bool = True) -> dict:
        """Report the resources that the transpiled circuit uses on this device.

        Args:
            measured: If `True`, the default, report the circuit measured in the
                `Z` basis, which is what kernels and sampling submit. If `False`,
                report the circuit without measurements that the estimator runs.

        Returns:
            A dictionary with `backend_name`, `is_fake`, `n_logical_qubits`,
                `optimization_level`, `seed_transpiler`, `gate_counts`,
                `n_2q_gates`, `two_qubit_gate_counts`, `depth`, `depth_2q`,
                `n_2q_gates_unrouted` (the same circuit on the native gates with
                all-to-all connectivity), `routing_overhead_2q`, `physical_qubits`,
                `n_physical_qubits_device`, and `estimated_success_probability`,
                computed from the error rates of the target.
        """
        z_basis = ("Z",) * self._n_qubits_logical
        logical = self._logical_measured(z_basis) if measured else self._base_circuit
        report = transpiled_resources(
            self.transpiled_circuit(measured),
            self._backend,
            logical,
            optimization_level=self._config.optimization_level,
            seed_transpiler=self._seed,
        )
        return {
            "backend_name": getattr(self._backend, "name", str(self._backend)),
            "is_fake": _is_local_backend(self._backend),
            "n_logical_qubits": self._n_qubits_logical,
            "optimization_level": self._config.optimization_level,
            "seed_transpiler": self._config.seed_transpiler,
            **report,
        }

    def run(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return one expectation value per observable.

        It follows the `CompiledCircuit` contract: the result has shape `(n_obs,)`
        when `weights` and `inputs` are both 1-D, and `(B, n_obs)` otherwise.

        Args:
            weights (np.ndarray): The parameter vector: a 1-D array for one sample,
                or a 2-D array with a leading batch axis.
            inputs (np.ndarray): The input features, with the same 1-D or 2-D
                shapes as `weights`.

        Returns:
            One expectation value per observable.
        """
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        sampling = self._config.primitive == "sampler"
        if weights.ndim == 1 and inputs.ndim == 2:
            return (
                self._run_sampler_broadcast(weights, inputs)
                if sampling
                else self._run_broadcast(weights, inputs)
            )
        # `_run_one` goes through `submit`, which already routes to whichever
        # primitive the config names, so the per-sample loop is shared.
        return batch_call(self._run_one, weights, inputs)

    def _run_one(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return self.submit(weights, inputs).result()

    def _run_broadcast(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Native batched evaluation via EstimatorV2 parameter broadcasting.

        EstimatorV2 PUBs accept a `(B, n_params)` array — the primitive
        evaluates all B parameter assignments inside one job per
        observable, so we go from `B * n_obs` jobs down to `n_obs`.
        """
        param_array = self._param_values_batched(weights, inputs)
        pubs = [(self._transpiled_circuit, op, param_array) for op in self._operators]
        primitive_result = job_result(
            self._estimator().run(pubs), self._config.poll_timeout_seconds
        )
        output = np.stack(
            [
                np.asarray(primitive_result[k].data.evs, dtype=np.float64).reshape(-1)
                for k in range(self._n_observables)
            ]
        )
        # Stack is (n_obs, B); contract is (B, n_obs).
        return output.T

    def _param_values_batched(
        self, weights: np.ndarray, inputs: np.ndarray, param_order=None
    ) -> np.ndarray:
        """Build a `(B, n_params)` array in transpiled-circuit parameter order.

        Weight columns are broadcast across all B rows (shared across
        the batch); input columns vary per row. `param_order` defaults to the
        bare circuit's — the sampler path passes the measured circuit's.
        """
        batch_size = inputs.shape[0]
        order = self._param_order if param_order is None else param_order
        n_params = len(order)
        arr = np.empty((batch_size, n_params), dtype=np.float64)
        for col, symbol in enumerate(order):
            name = symbol.name
            if name.startswith("w"):
                arr[:, col] = float(weights[int(name[1:])])
            elif name.startswith("x"):
                arr[:, col] = inputs[:, int(name[1:])]
            else:
                raise RuntimeError(
                    f"unexpected parameter name {name!r} in transpiled circuit"
                )
        return arr

    def submit(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
        *,
        method: str = "run",
        tags: Mapping[str, str] | None = None,
    ) -> QiskitRuntimeJobHandle:
        """Submit the circuit without blocking, and return a job handle.

        Nothing blocks: `handle.result()` waits for the job and decodes it in the
        same way as the blocking method of the same name. For `"run"`, the handle
        returns expectation values with either primitive: the counts of the
        sampler are reduced when the result is collected, so the two paths look
        the same to the caller. `"sample"` and `"probabilities"` are always one
        `SamplerV2` job measured in the `Z` basis, whatever the configured
        primitive.

        Args:
            weights (np.ndarray): The parameter vector for one sample (1-D).
            inputs (np.ndarray): The input features for one sample (1-D).
            method (str): What the job computes, one of `submit_methods`.
                `"run"`, the default, measures the observables, like `run()`.
                `"sample"` returns the measurement histogram, like `sample()`.
                `"probabilities"` returns the distribution over every basis state,
                like `probabilities()`.
            tags (Mapping[str, str] | None): Tags to put on the IBM job, so that
                you can find it again later with
                `QiskitRuntimeService.jobs(job_tags=[...])`. Each one becomes the
                job tag `"key=value"`, sorted by key. IBM allows at most 8 tags of
                at most 86 characters each. Keys and values must not be empty, and
                a key must not contain `=`.

        Returns:
            handle (JobHandle): A handle for the submitted job.

        Raises:
            ValueError: If `method` is not in `submit_methods`, if `weights` or
                `inputs` is not 1-D, or if `tags` breaks the IBM job tag limits.
                In all these cases, no primitive is built and nothing is
                submitted.
        """
        _check_submit_method(method)
        _check_one_sample("submit", weights, inputs)
        job_tags = _validate_job_tags(tags)
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        if method != "run":
            pubs = self._z_basis_pubs(weights, inputs)
            primitive = self._sampler()
        elif self._config.primitive == "sampler":
            bases = [group.basis for group in self._groups]
            pubs = self._sampler_pubs(bases, weights, inputs)
            primitive = self._sampler()
        else:
            pubs = self._build_pubs(weights, inputs)
            primitive = self._estimator()
        if job_tags:
            primitive.options.environment.job_tags = job_tags
        return self._handle(primitive.run(pubs), method)

    def _handle(self, job, method: str) -> QiskitRuntimeJobHandle:
        """Wrap `job` with the decoders `method` and the primitive need."""
        reduce = counts = None
        if method != "run":
            counts = self._z_counts
        elif self._config.primitive == "sampler":
            reduce, counts = self._reduce_sampler_result, self._counts_per_group
        return QiskitRuntimeJobHandle(
            job=job,
            n_observables=self._n_observables,
            poll_timeout_seconds=self._config.poll_timeout_seconds,
            reduce=reduce,
            counts=counts,
            method=method,
            n_qubits=self._n_qubits_logical,
        )

    def attach(self, job_id: str, *, method: str = "run") -> QiskitRuntimeJobHandle:
        """Rebuild the handle of a job submitted earlier from this circuit.

        Use it to pick up a job from a process other than the one that submitted
        it, for example after a crash. The handle decodes results exactly as the
        handle that `submit()` returned would have, provided that the job was
        submitted from the same circuit, observables, primitive and `method`.

        Args:
            job_id (str): The IBM job id, as returned by
                `QiskitRuntimeJobHandle.job_id()`.
            method (str): The `method` that the job was submitted with, one of
                `submit_methods`.

        Returns:
            handle (JobHandle): A handle for the existing job.

        Raises:
            ValueError: If `job_id` is not made of 1 to 64 letters, digits, `_` or
                `-`, if `method` is not in `submit_methods`, or if the backend is
                not an IBM Runtime backend with a `service` to look the job up in
                (a fake backend has none). In all these cases, the service is not
                contacted.
        """
        if not (isinstance(job_id, str) and _JOB_ID_PATTERN.fullmatch(job_id)):
            raise ValueError(
                f"job_id must be an IBM job id of 1-64 letters, digits, _ or -, "
                f"got {job_id!r}."
            )
        _check_submit_method(method)
        service = getattr(self._backend, "service", None)
        if service is None:
            raise ValueError(
                "attach() needs an IBM Runtime backend, which carries the "
                "QiskitRuntimeService to look the job up in; "
                f"{type(self._backend).__name__} has none."
            )
        return self._handle(service.job(job_id), method)


class QiskitRuntimeBackend:
    """The provider for IBM Quantum hardware through Qiskit Runtime.

    It is opt-in: the name `"qiskit-runtime"` exists only after the caller
    registers it, for example with
    `register("qiskit-runtime", QiskitRuntimeConfig(...))`.

    It costs money: the first compile for a paid device gives a
    [`CostWarning`][eigenkit.core.provider.CostWarning], and `dry_run_info()`
    reports the shape of a run beforehand. It does not estimate a price.

    This class is internal. Select the backend with `provider="qiskit-runtime"` or
    by passing a
    [`QiskitRuntimeConfig`][eigenkit.core.backends.qiskit_runtime_backend.QiskitRuntimeConfig],
    instead of constructing this class.
    """

    def __init__(
        self,
        config: QiskitRuntimeConfig | None = None,
        name: str = "qiskit-runtime",
    ):
        """Build the backend from a device configuration.

        Args:
            config (QiskitRuntimeConfig | None): The device settings. `None` uses
                `QiskitRuntimeConfig()`.
            name (str): The registry name, used in error messages.
        """
        self.name = name
        self.config = config or QiskitRuntimeConfig()

    @property
    def gradient_method(self) -> str:
        """Return the gradient method, which depends on `enable_gradients`.

        It is `"parameter_shift"` when the configuration sets `enable_gradients`,
        and `"none"` otherwise. A parameter-shift backward pass on a quantum
        processing unit (QPU) costs at least `2 * n_weights` jobs, each with the
        full shot count, so gradients are refused unless the caller opts in.
        """
        return "parameter_shift" if self.config.enable_gradients else "none"

    @property
    def capabilities(self) -> Capabilities:
        """Return what this backend supports.

        It supports every gate except `AmplitudeEncoding` (state preparation),
        `QubitUnitary` and `ControlledQubitUnitary` (arbitrary unitaries) and
        `CPHASE`. It supports finite-shot sampling, which a quantum processing unit
        (QPU) always uses. It supports batching, because the primitives evaluate a
        batch of parameter values in one job when `weights` is 1-D and `inputs`
        is 2-D. It has no limit of its own on the number of qubits: the limit of
        the device applies. See
        [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        return Capabilities(
            supported_gates=ALL_GATES - DATA_GATES - frozenset({"CPHASE"}),
            # EstimatorV2 broadcasts parameters across PUBs when weights
            # are 1-D and inputs are 2-D; other shapes still loop.
            supports_batching=True,
            supports_shots=True,
            max_qubits=None,
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> QiskitRuntimeCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        It is called once per `Circuit`, and the handle is reused for every call, so
        the expensive work is done here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that the handle
                measures.

        Returns:
            A `QiskitRuntimeCompiled` handle.
        """
        return QiskitRuntimeCompiled(circuit, observables, self.config)
