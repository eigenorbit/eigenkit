"""The provider protocol, its registry, and the parameter-shift fallback.

A provider compiles a circuit separately from running it, which matters for
speed. Building a `qml.device` and a QNode (PennyLane), or a parametrised
`QuantumCircuit` and a `SparsePauliOp` (Qiskit), takes far longer than one
execution. Doing that work once per `Circuit` and reusing the compiled handle for
every forward pass is what makes repeated evaluations fast.

The `run` and `run_torch` methods of each backend accept batched arrays
directly. The first axis is the batch, and `weights` and `inputs` can each be 1-D
(one sample) or 2-D (a batch). Backends that can run a batch at once override
`run` to submit one job. The others call `batch_call` and pay for a loop in
Python, but the loop is inside the backend and not in the `Circuit` that calls
it. The `Capabilities.supports_batching` flag only gives information, and
nothing chooses a path based on it: `Circuit` always passes the batch on
unchanged.

Example:
    ```python
    from eigenkit.core.provider import get, registered_names

    print(registered_names())   # ['numpy'] when no extra is installed

    backend = get("numpy")
    print(backend.gradient_method)                 # adjoint
    print(backend.capabilities.supports_batching)  # True
    ```
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from .ir import CircuitSpec, Observable, weights_needing_four_term_shift


# "adjoint": backend ships its own analytic reverse-mode jacobian (the numpy
# reference sim); routed like "parameter_shift" via the provider's `jacobian`.
GradientMethod = Literal["native", "parameter_shift", "adjoint", "none"]


class CostWarning(UserWarning):
    """Warning issued before a submission to a paid quantum backend.

    A paid backend is a quantum processing unit (QPU) or a cloud simulator that
    charges for use. The first compilation for such a device issues this warning,
    so that costs cannot build up without the user noticing."""


def _warn_gradient_cost(
    target: str,
    circuit: CircuitSpec,
    jobs_per_call: int,
    shots: int,
    *,
    unit: str = "jobs",
    stacklevel: int,
) -> None:
    """Warn that gradients are enabled on a paid device, quoting what one
    backward pass costs there.

    Hardware backends refuse gradients unless their config sets
    `enable_gradients=True`; this is the warning they emit at compile when it
    is set. The count is the weight-only parameter-shift pass: two forward
    calls per weight, four for a weight feeding a multi-frequency gate.
    Input gradients and the chain-rule path cost more, so the figure is a
    floor.

    Args:
        target (str): The device, as the warning names it.
        circuit (CircuitSpec): The circuit being compiled.
        jobs_per_call (int): Billable units one forward call submits.
        shots (int): Shots per unit.
        unit (str): The billable unit, as the device's own cost warning
            counts it — `"jobs"`, `"PUBs"`, `"tasks"` or `"circuits"`. Default:
            `"jobs"`.
        stacklevel (int): `warnings.warn` stacklevel as seen by the caller.
    """
    if circuit.n_weights == 0:
        # A kernel or reservoir circuit: no weight pass to put a price on.
        return
    four_term = len(weights_needing_four_term_shift(circuit))
    evaluations = 2 * (circuit.n_weights - four_term) + 4 * four_term
    warnings.warn(
        f"gradients are enabled on {target}: one backward pass costs at least "
        f"{evaluations * jobs_per_call} {unit} per sample ({evaluations} "
        f"parameter-shift evaluations x {jobs_per_call} {unit} each) at "
        f"shots={shots}. Input gradients, and weights inside Scale/Multiply/Sum "
        f"angles, cost more.",
        CostWarning,
        stacklevel=stacklevel + 1,
    )


@runtime_checkable
class JobHandle(Protocol):
    """A handle to a job running on a quantum processing unit (QPU) or in the cloud.

    `compiled.submit(...)` returns it without waiting for the job. Call `.result()`
    to wait for the job and combine its results."""

    def result(self) -> np.ndarray:
        """Wait until the job finishes, then return its expectation values.

        The shape follows the same rule as
        [`CompiledCircuit.run()`][eigenkit.core.provider.CompiledCircuit]:
        `(n_obs,)` for a single sample, and `(B, n_obs)` for a batch.
        """
        ...

    def status(self) -> str:
        """Return the state of the job as the provider reports it, without waiting.

        The strings are the provider's own. For example, Braket returns
        `"CREATED"`, `"QUEUED"`, `"RUNNING"`, `"COMPLETED"`, `"FAILED"` or
        `"CANCELLED"`. Compare them within one provider, not across providers.
        """
        ...


# ---- capabilities ----


@dataclass(frozen=True)
class Capabilities:
    """The description that a provider gives of what it supports.

    It is used to check a circuit before compilation, and to choose between code
    paths at run time.

    Attributes:
        supported_gates: The names of the gates that the backend can run. A circuit
            that uses any other gate is rejected before compilation, with a message
            that names that gate.
        supports_batching: Whether the backend evaluates a 2-D `inputs` batch in
            one call, instead of looping over it internally. It only gives
            information: `Circuit` passes the batch on in both cases and never
            reads this flag. It describes the strategy of the backend, and nothing
            chooses a path based on it.
        supports_shots: Whether the backend can sample, instead of returning exact
            expectation values.
        max_qubits: The maximum number of qubits of the hardware. `None` means
            that there is no limit, which is the case for every simulator.
        supported_adjoint_gates: The gates whose adjoint the backend can also run.
            `None` means all of `supported_gates`.
        supports_oracle_matrix: Whether the dense unitary of a callable oracle can
            be extracted. Iterative quantum amplitude estimation (IQAE) uses this
            to accept oracles that are PennyLane callables.
        supports_provider_shots: Whether the backend can sample an ancilla qubit
            with its own shots, which is what IQAE needs for
            `shot_mode="provider"`.
    """

    supported_gates: frozenset[str]
    supports_batching: bool
    supports_shots: bool
    max_qubits: int | None = None
    supported_adjoint_gates: frozenset[str] | None = None
    supports_oracle_matrix: bool = False
    supports_provider_shots: bool = False


# ---- compiled handle ----


@runtime_checkable
class CompiledCircuit(Protocol):
    """A handle to a circuit that a backend has prepared, cheap to call many times.

    `run` accepts `weights` and `inputs` that are each either 1-D (one sample) or
    2-D (a batch along the first axis). The output has shape `(n_obs,)` when both
    are 1-D, and `(B, n_obs)` otherwise. Each backend decides whether to evaluate
    the batch at once or in a loop (see `batch_call`).

    Implementations can also provide `run_torch(...)`, for backends that support
    PyTorch automatic differentiation themselves. When the `gradient_method` of
    the backend is `"native"`, the PyTorch interface detects this method with
    `hasattr` and uses it instead of the parameter-shift fallback. The same shape
    rule applies.
    """

    def run(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return one expectation value for each observable.

        See the class description for the shapes.

        Args:
            weights (np.ndarray | Sequence[float]): The weight vector, in any
                container that can be indexed: a NumPy array, a PyTorch tensor or
                a list.
            inputs (np.ndarray | Sequence[float]): The input vector, in the same
                kinds of container as `weights`.
        """
        ...


# ---- provider ----


@runtime_checkable
class Provider(Protocol):
    """The protocol that every backend follows.

    A provider turns a `CircuitSpec` into something that can run. Anything with
    these members can be passed as `provider=`: the built-in backends, the backend
    built by a configuration object, or your own class. To select it by a string
    instead, register it under a short name with `register`.

    Attributes:
        name: A short name, used in error messages and in `repr`.
        gradient_method: How the gradient code should differentiate this backend:
            `"adjoint"`, `"parameter_shift"`, `"native"`, or `"none"` for hardware
            that only returns values.
    """

    name: str
    gradient_method: GradientMethod

    @property
    def capabilities(self) -> Capabilities:
        """Return what this backend supports, as a `Capabilities` object.

        It covers the supported gates, shots, batching and the other flags."""
        ...

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> CompiledCircuit:
        """Prepare `circuit` so that `observables` can be evaluated many times.

        A `Circuit` calls it once, and reuses the returned handle for every call,
        so the expensive work belongs here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.
        """
        ...


# ---- registry and provider resolution ----
#
# `provider=` on the user-facing Circuit accepts three shapes:
#   str     - looked up in the registry ("pennylane"/"qiskit" pre-registered)
#   Config  - any object with a `make_backend()` factory (e.g. PennyLaneConfig)
#   Backend - a Provider instance, used directly
#
# Registration is optional aliasing: register a config or backend under a
# short name when you want to reuse it. The zero-config and inline-config
# paths never touch the registry.


_REGISTRY: dict[str, Provider] = {}
# Lazy entries: `name -> loader()` that imports the backend's SDK and returns
# a Provider only when the name is first resolved. This is what keeps
# `import eigenkit` (and `eigenkit.core`) from pulling
# torch / pennylane / qiskit — those SDKs load on first use, not on import.
_LAZY: dict[str, Callable[[], Provider]] = {}


def register(name: str, provider_or_config) -> None:
    """Register a backend, or a configuration that can build one, under `name`.

    Later calls to `Circuit(..., provider=name)` then use that backend. If a
    configuration is passed, its `make_backend()` method is called to build the
    backend.

    Args:
        name (str): The name to register.
        provider_or_config (Provider | object): A backend instance, or a
            configuration with a `make_backend()` method.
    """
    _REGISTRY[name] = _materialize(provider_or_config)
    _LAZY.pop(name, None)


def register_lazy(name: str, loader: Callable[[], Provider]) -> None:
    """Register a backend under `name` without importing its package yet.

    `loader` is called the first time the name is looked up, and its result is
    kept for later lookups. The optional backends are registered in this way, so
    that naming them imports nothing.

    It replaces whatever `name` currently refers to, as `register` does. If the
    registration were skipped when the name was already taken, a caller could not
    replace a built-in backend with a lazy one, and would get no sign that the
    loader had been ignored.

    Args:
        name (str): The name to register.
        loader (Callable[[], Provider]): A function with no arguments that returns
            the backend. It is called the first time the name is looked up.
    """
    _LAZY[name] = loader
    _REGISTRY.pop(name, None)


# What to tell someone whose `provider=` name did not resolve. A name is
# lazily registered only when its SDK is importable, so the usual cause is a
# missing extra. One name is different: `cudaq` is never registered by import
# at all, opt-in by design, so pointing at the extra alone would send the user
# to install something and meet the identical error afterwards.
_EXTRA_HINT = {
    "pennylane": "install with `pip install eigenkit[pennylane]`",
    "qiskit": "install with `pip install eigenkit[qiskit]`",
    "braket": "install with `pip install eigenkit[braket]`",
    "qulacs": "install with `pip install eigenkit[qulacs]`",
    "qsim": "install with `pip install eigenkit[qsim]`",
    "tensorcircuit": "install with `pip install eigenkit[tensorcircuit]`",
    "iqm": "install with `pip install eigenkit[iqm]`",
    "qiskit-aer": "install with `pip install eigenkit[qiskit]`",
    "qiskit-aer-mps": "install with `pip install eigenkit[qiskit]`",
    "cudaq": (
        "opt-in and Linux/arm64-macOS only: after installing it, register it "
        "yourself with `register('cudaq', CudaQuantumBackend())`"
    ),
}


def load_failure_hint(exc: ImportError) -> str:
    """Name the fix for a package that is installed but fails to load, if known.

    Args:
        exc: The `ImportError` that loading the package raised.

    Returns:
        A parenthesised hint to append to an error message, or `""`.
    """
    if "static TLS" in str(exc):
        # Linux ARM: PyTorch, scikit-learn, CUDA-Q and PennyLane Lightning each
        # bundle an OpenMP runtime, and with Qiskit in the same process they
        # outgrow glibc's static TLS reserve. The reserve is fixed when the
        # process starts, so only the environment of a new process can raise it.
        # 256 KiB is what the whole test suite needed with every extra installed.
        return (
            " (on Linux ARM, the native libraries of PyTorch, scikit-learn, Qiskit "
            "and CUDA-Q together exceed glibc's static TLS reserve: start Python "
            "with GLIBC_TUNABLES=glibc.rtld.optional_static_tls=262144)"
        )
    return ""


def get(name: str) -> Provider:
    """Return the provider registered under `name`.

    A backend registered with `register_lazy` imports its package at the first
    lookup, and the backend is kept for later lookups.

    Args:
        name: A registered short name, such as `"numpy"` or `"pennylane"`.

    Returns:
        The provider instance.

    Raises:
        ImportError: If `name` is a registered backend whose package is not
            installed, or is installed but fails to import. For a missing
            package the message names the extra that installs it. The
            registration is kept, so a later call works once the package is
            installed.
        KeyError: If nothing is registered under `name`. The message lists the
            available names and, for a known backend, the extra that installs it.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name in _LAZY:
        try:
            provider = _LAZY[name]()  # import the SDK now, then cache
        except ImportError as exc:
            # Do not drop the registration: `name` is a backend this build
            # knows about, it is only missing its package. Popping first would
            # make the *second* attempt report the name as unknown, which is a
            # different and untrue problem.
            if not isinstance(exc, ModuleNotFoundError):
                # The package is there but did not load: a missing system
                # library, say. Naming the extra would send the user to
                # install what they already have.
                raise ImportError(
                    f"provider '{name}' is installed but its package failed to "
                    f"import{load_failure_hint(exc)}. Original error: {exc}"
                ) from exc
            hint = _EXTRA_HINT.get(name)
            suffix = f" ({hint})" if hint else ""
            raise ImportError(
                f"provider '{name}' is registered but its package is not "
                f"installed{suffix}. Original error: {exc}"
            ) from exc
        _LAZY.pop(name, None)
        _REGISTRY[name] = provider
        return provider
    hint = _EXTRA_HINT.get(name)
    suffix = f" ({hint})" if hint else ""
    raise KeyError(
        f"no provider registered under name '{name}'{suffix}. "
        f"registered: {sorted(registered_names())}. "
        f"Pass a config directly (e.g. PennyLaneConfig(...)) to skip registration."
    )


def registered_names() -> list[str]:
    """Return every provider name that `get` accepts, including the lazy ones."""
    return sorted(set(_REGISTRY) | set(_LAZY))


def resolve_provider(spec) -> Provider:
    """Turn a provider (a name, a configuration or a backend) into a backend instance.

    A string is looked up in the registry. Any object with a `make_backend()`
    method is treated as a configuration. A backend is used as it is.

    Args:
        spec (str | object | Provider): A registered name, a configuration with a
            `make_backend()` method, or a backend instance.

    Returns:
        The backend instance.
    """
    if isinstance(spec, str):
        return get(spec)
    return _materialize(spec)


def _materialize(spec) -> Provider:
    """Accept either a Backend instance or a Config with `make_backend()`."""
    if hasattr(spec, "make_backend"):
        return spec.make_backend()
    if isinstance(spec, Provider):
        return spec
    raise TypeError(
        f"provider spec {spec!r} is not a string, a Backend, or a config "
        f"with .make_backend(). Got {type(spec).__name__}."
    )


# ---- validation ----


def validate_circuit(circuit: CircuitSpec, capabilities: Capabilities) -> None:
    """Raise a `ValueError` if the circuit cannot run on the provider.

    The message says clearly why: the circuit uses a gate, or the adjoint of a
    gate, that the provider does not support, or it needs more qubits than the
    provider has.

    Args:
        circuit (CircuitSpec): The circuit to check.
        capabilities (Capabilities): What the provider says it supports.
    """
    unsupported = {gate.name for gate in circuit.gates} - capabilities.supported_gates
    if unsupported:
        raise ValueError(
            f"gates {sorted(unsupported)} not supported by provider "
            f"(supported: {sorted(capabilities.supported_gates)})"
        )
    adjoint_gates = (
        capabilities.supported_adjoint_gates
        if capabilities.supported_adjoint_gates is not None
        else capabilities.supported_gates
    )
    unsupported_adjoint = {
        gate.name
        for gate in circuit.gates
        if gate.adjoint and gate.name not in adjoint_gates
    }
    if unsupported_adjoint:
        raise ValueError(
            f"adjoint of gates {sorted(unsupported_adjoint)} not supported by provider"
        )
    if capabilities.max_qubits is not None and circuit.n_qubits > capabilities.max_qubits:
        raise ValueError(
            f"circuit needs {circuit.n_qubits} qubits; provider supports "
            f"{capabilities.max_qubits}"
        )


# ---- initial-state validation (shared by state-IO backends) ----


def _check_normalized(amplitudes: np.ndarray) -> None:
    norm_sq = float(np.vdot(amplitudes, amplitudes).real)
    if abs(norm_sq - 1.0) > 1e-6:
        raise ValueError(
            f"initial_state must be L2-normalized (got squared norm {norm_sq:.6g})"
        )


def validate_initial_state(
    initial_state, n_qubits: int, *, batch: int | None = None
) -> np.ndarray:
    """Check and flatten a statevector that the user gives as a starting state.

    It returns a contiguous `complex128` array in the flat order of the library,
    where qubit `0` is the most significant bit. The shape is `(2**n,)` when
    `batch` is `None`, and `(batch, 2**n)` otherwise. A 1-D state passed with a
    `batch` is repeated for every row of the batch. It raises a `ValueError` if
    the length is wrong or the state is not normalised. A unitary evolution keeps
    the norm, so a state that is not normalised is almost always a mistake of the
    caller. Each backend reshapes the returned flat array into its own state
    layout. Doing the checks here, in one place, keeps the error messages the same
    for the NumPy, PennyLane and Qiskit backends.

    Args:
        initial_state (np.ndarray): The statevector to start from.
        n_qubits (int): The number of qubits of the circuit.
        batch (int | None): The batch size that the state must match, or `None`
            for a single state. Default: `None`.

    Returns:
        The flattened statevector, after the checks.
    """
    arr = np.asarray(initial_state, dtype=np.complex128)
    dim = 1 << n_qubits
    if batch is None:
        arr = arr.reshape(-1)
        if arr.shape[0] != dim:
            raise ValueError(
                f"initial_state must have length {dim} (2**{n_qubits}), "
                f"got {arr.shape[0]}"
            )
        _check_normalized(arr)
        return np.ascontiguousarray(arr)
    if arr.ndim <= 1:
        single = arr.reshape(-1)
        if single.shape[0] != dim:
            raise ValueError(
                f"initial_state must have length {dim} (2**{n_qubits}), "
                f"got {single.shape[0]}"
            )
        _check_normalized(single)
        return np.broadcast_to(single, (batch, dim)).copy()
    if arr.shape != (batch, dim):
        raise ValueError(
            f"batched initial_state must have shape ({batch}, {dim}), got {arr.shape}"
        )
    norm_sq = np.einsum("bi,bi->b", arr.conjugate(), arr).real
    if not np.allclose(norm_sq, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("every initial_state row must be L2-normalized")
    return np.ascontiguousarray(arr)


# ---- batched-call helpers (used by backends that don't natively batch) ----


def _batch_pair(weights, inputs):
    """Resolve broadcast shape of (weights, inputs) along the leading axis.

    Returns `(B, weights_is_2d, inputs_is_2d)`. `B` is None when both are
    1-D (no batch). Raises if both are 2-D with mismatched batch sizes.
    """
    Bw = weights.shape[0] if weights.ndim == 2 else None
    Bi = inputs.shape[0] if inputs.ndim == 2 else None
    if Bw is None and Bi is None:
        return None, False, False
    if Bw is not None and Bi is not None and Bw != Bi:
        raise ValueError(f"cannot broadcast weights batch {Bw} against inputs batch {Bi}")
    return (Bw if Bw is not None else Bi), (Bw is not None), (Bi is not None)


def batch_call(
    run_one: Callable[[np.ndarray, np.ndarray], np.ndarray],
    weights: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    """Call `run_one(w_i, x_i)` for each sample along the first (batch) axis.

    It is meant for backends that cannot run a batch at once. It holds the loop
    over the samples, and if one sample fails, it raises a `RuntimeError` that
    gives the index of that sample, so the caller can find which sample failed.

    Args:
        run_one (Callable): A function that evaluates one sample.
        weights (np.ndarray | Sequence[float]): The weights, 1-D for one sample or
            2-D for a batch. They must be a NumPy array or a PyTorch tensor,
            because the batch is detected from `ndim` and `shape`. A list is not
            accepted.
        inputs (np.ndarray | Sequence[float]): The inputs, 1-D or 2-D, in the same
            kinds of container as `weights`.

    Returns:
        The results of the samples, stacked.
    """
    B, w_is_batch, x_is_batch = _batch_pair(weights, inputs)
    if B is None:
        return run_one(weights, inputs)
    outputs = []
    for i in range(B):
        w_i = weights[i] if w_is_batch else weights
        x_i = inputs[i] if x_is_batch else inputs
        try:
            outputs.append(run_one(w_i, x_i))
        except Exception as exc:
            raise RuntimeError(f"failed on batch sample index {i}") from exc
    return np.stack(outputs)


# ---- parameter-shift gradient ----

# The two-term rule df/dtheta = [f(theta + pi/2) - f(theta - pi/2)] / 2 is exact
# only when the gate's generator has one eigenvalue gap, so its expectation
# carries one frequency. A controlled rotation has two ({1/2, 1}) and needs four
# evaluations. These are the standard shifts and coefficients for that spectrum.
# The four-term rule is also exact for a single-frequency gate -- {1} sits
# inside {1/2, 1} -- so it is only ever a cost question, never a correctness
# one; it is used exactly where `ir.MULTI_FREQUENCY_GATES` says it is needed.
_SQRT2 = np.sqrt(2.0)
FOUR_TERM_RULE: tuple[tuple[float, float], ...] = (
    (np.pi / 2, (_SQRT2 + 1.0) / (4.0 * _SQRT2)),
    (-np.pi / 2, -(_SQRT2 + 1.0) / (4.0 * _SQRT2)),
    (3 * np.pi / 2, -(_SQRT2 - 1.0) / (4.0 * _SQRT2)),
    (-3 * np.pi / 2, (_SQRT2 - 1.0) / (4.0 * _SQRT2)),
)
TWO_TERM_RULE: tuple[tuple[float, float], ...] = (
    (np.pi / 2, 0.5),
    (-np.pi / 2, -0.5),
)


def _shift_column(evaluate, vector: np.ndarray, index: int, four_term: bool):
    """One jacobian column: sum of `coefficient * f(vector shifted at index)`.

    Args:
        evaluate: Callable taking the shifted vector and returning `(n_obs,)`.
        vector (np.ndarray): The weights or inputs being differentiated.
        index (int): Which entry to shift.
        four_term (bool): Use the four-term rule instead of the two-term one.

    Returns:
        The column, shape `(n_obs,)`.
    """
    rule = FOUR_TERM_RULE if four_term else TWO_TERM_RULE
    column = None
    for shift, coefficient in rule:
        shifted = vector.copy()
        shifted[index] += shift
        term = coefficient * evaluate(shifted)
        column = term if column is None else column + term
    return column


def parameter_shift_jacobian(
    compiled: CompiledCircuit,
    weights: np.ndarray,
    inputs: np.ndarray,
    four_term_indices: frozenset[int] | None = None,
) -> np.ndarray:
    """Return the jacobian with respect to the weights, by the parameter-shift rule.

    Each entry is `d<O_k>/dθ_i`, computed with the parameter-shift rule. For a
    weight whose gates have a single frequency, the rule takes two evaluations,
    `(f(θ + π/2) - f(θ - π/2)) / 2`. For a weight listed in `four_term_indices`,
    it uses the four-term rule.

    Args:
        compiled (CompiledCircuit): The prepared circuit to differentiate.
        weights (np.ndarray | Sequence[float]): The weight vector, as a NumPy
            array or a list. A PyTorch tensor is not accepted, because the vector
            is shifted on a copy made with `.copy()`.
        inputs (np.ndarray | Sequence[float]): The input vector, in any container
            that `compiled.run` accepts: a NumPy array, a PyTorch tensor or a list.
        four_term_indices (frozenset[int] | None): The weights that feed a gate
            with several frequencies, as returned by
            `ir.weights_needing_four_term_shift`. The two-term rule is not exact
            for those weights (on a controlled rotation it is wrong by tens of
            percent), so leaving them out here returns wrong gradients, not an
            error.

    Returns:
        The jacobian, of shape `(n_observables, n_weights)`.
    """
    n_weights = len(weights)
    if n_weights == 0:
        # No columns to fill, but the caller still expects the observable axis.
        return np.zeros((len(compiled.run(weights, inputs)), 0), dtype=np.float64)
    four_term = four_term_indices or frozenset()
    # Sized from the first shifted evaluation rather than from a baseline run.
    # A baseline is not part of the parameter-shift rule — it would serve only to
    # read the observable count — and on a metered or shots backend it would be
    # a paid submission per gradient.
    jacobian: np.ndarray | None = None
    for i in range(n_weights):
        column = _shift_column(
            lambda shifted: compiled.run(shifted, inputs), weights, i, i in four_term
        )
        if jacobian is None:
            jacobian = np.zeros((len(column), n_weights), dtype=np.float64)
        jacobian[:, i] = column
    return jacobian


def parameter_shift_input_jacobian(
    compiled: CompiledCircuit,
    weights: np.ndarray,
    inputs: np.ndarray,
    four_term_indices: frozenset[int] | None = None,
) -> np.ndarray:
    """Return the jacobian with respect to the inputs, by shifting each input.

    Each entry is `d<O_k>/dx_j`, computed by shifting `inputs[j]` directly by
    `±π/2` (and also by `±3π/2` for an input listed in `four_term_indices`). The
    result is correct only when the angle of every gate that depends on an input
    is exactly `InputRef(j)`, not wrapped in `Scale`, `Multiply`, `Sum` or
    `Negate`, and when each `InputRef(j)` feeds a single gate. When that is not
    the case, use the chain-rule path instead.

    This simple path does not use `shift_gate_angle` (which wraps the angle in a
    `Sum`), so it works on backends that do not support symbolic `Sum`
    expressions, such as CUDA Quantum.

    Args:
        compiled (CompiledCircuit): The prepared circuit to differentiate.
        weights (np.ndarray | Sequence[float]): The weight vector, in any
            container that `compiled.run` accepts: a NumPy array, a PyTorch tensor
            or a list.
        inputs (np.ndarray | Sequence[float]): The input vector, as a NumPy array
            or a list. A PyTorch tensor is not accepted, because the vector is
            shifted on a copy made with `.copy()`.
        four_term_indices (frozenset[int] | None): The inputs that feed a gate
            with several frequencies, as returned by
            `ir.inputs_needing_four_term_shift`.

    Returns:
        The jacobian, of shape `(n_observables, n_inputs)`.
    """
    n_inputs = len(inputs)
    if n_inputs == 0:
        return np.zeros((len(compiled.run(weights, inputs)), 0), dtype=np.float64)
    four_term = four_term_indices or frozenset()
    # Sized from the first shifted evaluation — see `parameter_shift_jacobian`.
    jacobian: np.ndarray | None = None
    for j in range(n_inputs):
        column = _shift_column(
            lambda shifted: compiled.run(weights, shifted), inputs, j, j in four_term
        )
        if jacobian is None:
            jacobian = np.zeros((len(column), n_inputs), dtype=np.float64)
        jacobian[:, j] = column
    return jacobian
