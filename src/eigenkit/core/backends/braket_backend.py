"""The Amazon Braket backend.

The `amazon-braket-sdk` package is an optional dependency
(`pip install eigenkit[braket]`). It is imported only when it is needed, so the
rest of the library works without it. If it is missing, `compile()` raises an
`ImportError` with a clear message.

`BraketConfig` selects one of three modes:

- The local simulator, the default, with `device_arn=None`. The circuit runs on
  `LocalSimulator`, in the same process. Exact values (`shots=0`) are supported.
  Gradients use the general parameter-shift method.
- A cloud simulator, when `device_arn` names SV1, DM1 or TN1. Amazon Web Services
  bills each task, and a single `CostWarning` is issued when the circuit is
  compiled.
- A quantum processing unit (QPU), when `device_arn` contains `/qpu/` (Rigetti,
  IonQ, IQM, AQT and others). `shots > 0` is required. Gradients are turned off
  (`gradient_method="none"`), and calling `.backward()` raises an error whose
  message recommends training on a simulator and then using the weights on the
  device. `enable_gradients=True` turns on parameter-shift gradients on the QPU,
  and compiling then issues a `CostWarning` that gives the number of tasks of one
  backward pass.

Both cloud modes can submit without waiting, through `submit()`, which returns a
`BraketJobHandle`. Its `result()` waits for the task, and its `status()` checks the
task without waiting. `submit()` can run `run`, `sample`, `probabilities` or
`zero_state_probability` (see `BraketCompiled.submit_methods`), and it can add
Amazon Web Services (AWS) tags to the task. `attach()` rebuilds a handle from the
Amazon Resource Name (ARN) of a task submitted earlier, for example after the
process that submitted it restarted.

`sample` reads the measurement counts of the task, so it needs `shots > 0`.
`probabilities` and `zero_state_probability` read the first result type that the
device publishes for the configured shots, in the order `Probability`,
`StateVector`, `Amplitude`, `DensityMatrix` (`zero_state_probability` skips
`StateVector`). With `shots > 0`, that is always `Probability`. At `shots=0`, the
managed simulators publish no `Probability`, so SV1 gives the exact answer through
`Amplitude`, and DM1 through `DensityMatrix`. The local simulator uses
`Probability` at any number of shots. Three of these paths are limited to the
number of qubits that Braket accepts, and wider circuits are refused before any
task is created. On a cloud device, `Probability` covers at most 15 qubits
([List of result types](https://docs.aws.amazon.com/braket/latest/developerguide/braket-result-types.html)),
so both methods stop there with `shots > 0`. On the local simulator, only the
memory limits `Probability`. At `shots=0`, `probabilities` on SV1 lists every
basis state in the program, so it stops at 17 qubits (because of the program limit
of 5 megabytes), and DM1 returns a density matrix of at most 8 qubits. `sample`
has no limit.

`compile()` builds a parametric `braket.circuits.Circuit`, whose gates use
`FreeParameter` symbols, and adds one expectation result type for each distinct
Pauli term of all the observables. With `shots=0`, one circuit measures every
term. With `shots > 0`, one task can measure only terms that commute qubit-wise,
so the terms are split into such groups, with one circuit, and one task per call,
for each group. Each `run` or `submit` sets the parameter values through the
`inputs={...}` dictionary of the Braket package.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # By registered name.
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="braket")

    # Or with the configuration, when a device setting matters.
    from eigenkit.configs import BraketConfig

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=BraketConfig(
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1"
        ),
    )
    ```

    Without a `device_arn`, this is the local simulator. With one, the circuit is
    submitted to AWS. The ARN of a QPU turns gradients off (unless
    `enable_gradients=True`) and requires shots.
"""

from __future__ import annotations

import unicodedata
import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..ir import (
    ALL_GATES,
    CircuitSpec,
    Constant,
    Gate,
    InputRef,
    Multiply,
    Negate,
    Observable,
    Parameter,
    Scale,
    Sum,
    WeightRef,
    _data_to_array,
)
from ..provider import (
    Capabilities,
    CompiledCircuit,
    CostWarning,
    _warn_gradient_cost,
    batch_call,
)
from . import _sampling


def _require_braket() -> None:
    """Probe for amazon-braket-sdk. Raise a clear error if missing.

    Call this once before issuing local `from braket... import ...`
    statements in code paths that need it — the guard centralises the
    install hint so each call site stays a plain Python import.
    """
    try:
        import braket.circuits  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "amazon-braket-sdk is not installed. Install it via "
            "`pip install eigenkit[braket]`."
        ) from e


def _require_braket_aws() -> None:
    """Probe for braket.aws (cloud / QPU paths only)."""
    try:
        import braket.aws  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "amazon-braket-sdk is not installed. Install it via "
            "`pip install eigenkit[braket]`."
        ) from e


@dataclass(frozen=True)
class BraketConfig:
    """Device configuration for the Braket backend.

    Args:
        device_arn: `None` (the default) for the local simulator, or the Amazon
            Resource Name (ARN) of an Amazon Web Services (AWS) device. An example
            for a cloud simulator is
            `arn:aws:braket:::device/quantum-simulator/amazon/sv1`. Quantum
            processing units (QPUs) have ARNs of the form
            `arn:aws:braket:<region>::device/qpu/<vendor>/<name>`.
        local_backend: The `LocalSimulator` backend to use when `device_arn` is
            `None`. Common values are `"braket_sv"` (state vector) and
            `"braket_dm"` (density matrix).
        shots: `0` means exact values (simulators only), and a positive integer
            turns on sampling with that number of shots. AWS QPUs require
            `shots > 0`.
        s3_bucket: The Amazon Simple Storage Service (S3) bucket for the results
            of cloud tasks. With `None`, the Braket package picks the default of
            the account.
        s3_prefix: The key prefix inside `s3_bucket`.
        poll_timeout_seconds: How many seconds `result()` waits for a submitted
            task.
        enable_gradients: Whether to allow parameter-shift gradients on a QPU, for
            PyTorch `.backward()`, `jacobian()` and gradient-based optimisers. It is
            off by default, because one backward pass costs at least
            `2 * n_weights` paid tasks. When it is set, compiling for a QPU issues
            a `CostWarning` that gives this count. Simulators compute gradients in
            either case.
    """

    device_arn: str | None = None
    local_backend: str = "braket_sv"
    shots: int = 0
    s3_bucket: str | None = None
    s3_prefix: str = "braket-results"
    poll_timeout_seconds: int = 3600
    enable_gradients: bool = False

    def make_backend(self) -> "BraketBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `BraketBackend` for this configuration.
        """
        return BraketBackend(config=self)

    @property
    def is_cloud(self) -> bool:
        """Whether `device_arn` is set.

        If it is, the circuit runs on Amazon Web Services (AWS), on a cloud
        simulator or a quantum processing unit (QPU), instead of on the local
        simulator.
        """
        return self.device_arn is not None

    @property
    def is_qpu(self) -> bool:
        """Whether `device_arn` names real hardware (it contains `/qpu/`)."""
        return self.device_arn is not None and "/qpu/" in self.device_arn


def _pauli_term_to_braket(term, pauli_factory):
    """Translate one `PauliTerm` to `(braket_observable, target_qubits)`.

    `pauli_factory` maps "X" / "Y" / "Z" to the corresponding braket
    observable class; passed in by the caller because the classes can't
    be imported at module load.
    """
    targets: list[int] = []
    factors = []
    for op, qubit in term.paulis:
        targets.append(qubit)
        factors.append(pauli_factory[op]())
    observable = factors[0]
    for factor in factors[1:]:
        observable = observable @ factor
    return observable, targets


def _resolve_symbolic(
    parameter: Parameter,
    weight_symbols: list,
    input_symbols: list,
):
    """Translate a Parameter into a Braket `FreeParameter` or
    `FreeParameterExpression`. Braket supports `+`, `-`, scalar `*`, and
    symbol `*` symbol, so all expression nodes map cleanly."""
    if isinstance(parameter, Constant):
        return parameter.value
    if isinstance(parameter, WeightRef):
        return weight_symbols[parameter.index]
    if isinstance(parameter, InputRef):
        return input_symbols[parameter.index]
    if isinstance(parameter, Scale):
        return parameter.factor * _resolve_symbolic(
            parameter.expression, weight_symbols, input_symbols
        )
    if isinstance(parameter, Negate):
        return -_resolve_symbolic(parameter.expression, weight_symbols, input_symbols)
    if isinstance(parameter, Sum):
        parts = [
            _resolve_symbolic(e, weight_symbols, input_symbols)
            for e in parameter.expressions
        ]
        total = parts[0]
        for p in parts[1:]:
            total = total + p
        return total
    if isinstance(parameter, Multiply):
        # FreeParameterExpression supports * between two FreeParameters.
        return _resolve_symbolic(
            parameter.a, weight_symbols, input_symbols
        ) * _resolve_symbolic(parameter.b, weight_symbols, input_symbols)
    raise TypeError(parameter)


def _apply_mcz(bk_circuit, qubits: list[int]) -> None:
    """Multi-controlled Z from native gates.

    Braket's `ctrl @` modifier (`Instruction(..., control=...)`) is avoided on
    purpose: the local simulator silently skips such a gate for some qubit
    orderings on four or more qubits, and QPUs do not accept the modifier.
    """
    if len(qubits) == 2:
        bk_circuit.cz(*qubits)
    elif len(qubits) == 3:
        bk_circuit.h(qubits[-1])
        bk_circuit.ccnot(*qubits)
        bk_circuit.h(qubits[-1])
    else:
        diagonal = np.ones(2 ** len(qubits))
        diagonal[-1] = -1.0
        bk_circuit.unitary(matrix=np.diag(diagonal), targets=qubits)


def _apply_gate_symbolic(
    bk_circuit,
    gate: Gate,
    weight_symbols: list,
    input_symbols: list,
) -> None:
    if gate.params:
        angle = _resolve_symbolic(gate.params[0], weight_symbols, input_symbols)
    match gate.name:
        case "RX":
            bk_circuit.rx(gate.qubits[0], angle)
        case "RY":
            bk_circuit.ry(gate.qubits[0], angle)
        case "RZ":
            bk_circuit.rz(gate.qubits[0], angle)
        case "H":
            bk_circuit.h(gate.qubits[0])
        case "X":
            bk_circuit.x(gate.qubits[0])
        case "S":
            bk_circuit.s(gate.qubits[0])
        case "Sdg":
            bk_circuit.si(gate.qubits[0])  # Braket spells S-dagger "Si"
        case "CNOT":
            bk_circuit.cnot(gate.qubits[0], gate.qubits[1])
        case "Z":
            bk_circuit.z(gate.qubits[0])
        case "MCZ":
            _apply_mcz(bk_circuit, list(gate.qubits))
        case "MCX":
            target = gate.qubits[-1]
            bk_circuit.h(target)
            _apply_mcz(bk_circuit, list(gate.qubits))
            bk_circuit.h(target)
        case "PhaseShift":
            bk_circuit.phaseshift(gate.qubits[0], angle)
        case "SWAP":
            bk_circuit.swap(gate.qubits[0], gate.qubits[1])
        case "CRY":
            # CRY(t) = CNOT . RY(-t/2) . CNOT . RY(t/2) on the target.
            control, target = gate.qubits
            bk_circuit.ry(target, angle / 2)
            bk_circuit.cnot(control, target)
            bk_circuit.ry(target, -angle / 2)
            bk_circuit.cnot(control, target)
        case "QubitUnitary":
            bk_circuit.unitary(
                matrix=_data_to_array(gate.data[0]), targets=list(gate.qubits)
            )
        case "AmplitudeEncoding":
            raise NotImplementedError(
                "AmplitudeEncoding is not supported by the Braket backend: "
                "Braket circuits have no native state-initialization primitive. "
                "Use the PennyLane or Qiskit backend instead."
            )
        case _:
            raise NotImplementedError(f"gate '{gate.name}' not implemented for Braket")


_SUBMIT_METHODS: frozenset[str] = frozenset(
    {"run", "sample", "probabilities", "zero_state_probability"}
)

# The result types `probabilities` and `zero_state_probability` can read their
# answer from, in order of preference. `Probability` comes first, so every
# shots > 0 run reads the shot frequencies; the others are exact and only
# published at shots=0 (SV1 publishes `Amplitude`, DM1 `DensityMatrix`, and
# neither a shots=0 `Probability`). `sample` reads the measurement counts and
# needs no result type at all.
_METHOD_RESULT_TYPES: dict[str, tuple[str, ...]] = {
    "probabilities": ("Probability", "StateVector", "Amplitude", "DensityMatrix"),
    "zero_state_probability": ("Probability", "Amplitude", "DensityMatrix"),
}
# The device action whose capabilities apply: the SDK submits circuits as
# OpenQASM programs.
_OPENQASM_ACTION = "braket.ir.openqasm.program"

# Width caps on the result types whose size grows with the circuit. All three
# are Braket-side limits, so a wider circuit could only fail at
# CreateQuantumTask; `_check_result_width` refuses it before any task instead.
#
# `Probability` over every qubit, which both methods read whenever the shots
# are > 0 on a cloud device: "Managed simulators and QPUs are limited to 15
# qubits maximum, and local simulators are limited to the system's memory
# size." (Amazon Braket Developer Guide, "List of result types",
# https://docs.aws.amazon.com/braket/latest/developerguide/braket-result-types.html)
_MAX_PROBABILITY_QUBITS = 15
# Exact `probabilities()` on SV1 names every basis state in its `Amplitude`
# result type, inside the program. Braket limits a quantum-task action to 5 MB
# (Amazon Braket Developer Guide, "Amazon Braket Quotas"). The SDK serialises
# the state list alone to 3.0 MB of action at 17 qubits, leaving 2 MB for the
# gates, and to 6.3 MB at 18. `zero_state_probability()` names one state and
# is not capped.
_MAX_AMPLITUDE_LISTING_QUBITS = 17
# DM1 returns the density matrix of at most 8 target qubits (Developer Guide,
# "Submitting quantum tasks to simulators": "Reduced density matrix: Shots = 0,
# up to max 8 qubits"), and both exact paths need the one over every qubit:
# 4**8 = 65,536 complex entries.
_MAX_DENSITY_MATRIX_QUBITS = 8

# AWS resource-tag limits for CreateQuantumTask.
_MAX_TAGS = 50
_MAX_TAG_KEY_LENGTH = 128
_MAX_TAG_VALUE_LENGTH = 256
# AWS allows Unicode letters, numbers and separators (the `\p{L}\p{N}\p{Z}`
# classes of its tag pattern) plus these symbols, and reserves the `aws:` key
# prefix in any letter case.
_TAG_SYMBOLS = frozenset("_.:/=+-@")
_RESERVED_TAG_PREFIX = "aws:"

# `AwsQuantumTask.result()` returns None for these, instead of a result.
_NO_RESULT_STATES = frozenset({"FAILED", "CANCELLED"})


def _check_submit_method(method: str) -> None:
    if method not in _SUBMIT_METHODS:
        raise ValueError(
            f"method {method!r} cannot be submitted asynchronously on Braket; "
            f"submit_methods is {sorted(_SUBMIT_METHODS)}."
        )


def _check_one_sample(name: str, weights, inputs) -> None:
    if np.ndim(weights) != 1 or np.ndim(inputs) != 1:
        raise ValueError(
            f"{name}() takes one sample: weights and inputs must be 1-D, got "
            f"shapes {np.shape(weights)} and {np.shape(inputs)}. Loop in "
            f"Python for a batch."
        )


def _select_result_type(
    method: str,
    shots: int,
    limits: Mapping[str, tuple[int, int]] | None,
    device: str,
) -> str | None:
    """Pick the result type `method` reads its answer from on a device.

    Deterministic, so a caller that knows a device's capabilities can predict
    which result type a task will carry.

    Args:
        method (str): `"sample"`, `"probabilities"` or
            `"zero_state_probability"`.
        shots (int): The configured shot count; 0 means exact.
        limits (Mapping[str, tuple[int, int]] | None): The result types the
            device publishes, name to `(minShots, maxShots)`, as
            `_result_type_limits` reads them. `None` when the device publishes
            nothing readable, as with the local simulator; `Probability` is
            used then, which the local simulator supports at any shot count.
        device (str): Device name, for the error message.

    Returns:
        The result-type name, or `None` for `sample`, which reads the task's
        measurement counts instead of a result type.

    Raises:
        ValueError: If `method` is `"sample"` and `shots <= 0`, or the device
            publishes none of the method's result types for `shots`.
    """
    if method == "sample":
        if shots <= 0:
            raise ValueError(
                f"sample() needs shots > 0, got shots={shots}: a histogram is "
                f"made of shots. Use probabilities() for the exact distribution."
            )
        return None
    candidates = _METHOD_RESULT_TYPES[method]
    if limits is None:
        return "Probability"
    for name in candidates:
        low, high = limits.get(name, (1, 0))
        if low <= shots <= high:
            return name
    raise ValueError(
        f"device {device!r} publishes none of {list(candidates)} for "
        f"{method}() at shots={shots}; use a shot count inside the device's range."
    )


def _check_result_width(
    method: str, result_type: str | None, n_qubits: int, device: str, *, cloud: bool
) -> None:
    """Refuse a read wider than Braket returns it.

    Deterministic, like `_select_result_type`, so a caller that predicts the
    result type can predict the refusal too.

    Args:
        method (str): `"sample"`, `"probabilities"` or
            `"zero_state_probability"`.
        result_type (str | None): What `_select_result_type` picked for it.
        n_qubits (int): Circuit width.
        device (str): Device name, for the error message.
        cloud (bool): Whether the device is a managed simulator or QPU. Only
            those cap `Probability`; the local simulator has no limit.

    Raises:
        ValueError: If either method is read through `Probability` on a cloud
            device on more than `_MAX_PROBABILITY_QUBITS` qubits, `method` is
            `"probabilities"` read through `Amplitude` on more than
            `_MAX_AMPLITUDE_LISTING_QUBITS`, or either method is read through
            `DensityMatrix` on more than `_MAX_DENSITY_MATRIX_QUBITS`.
    """
    if result_type == "Probability":
        if cloud and n_qubits > _MAX_PROBABILITY_QUBITS:
            if method == "probabilities":
                cap = (
                    f", a tighter cap than the {_sampling.MAX_DENSE_QUBITS}-qubit "
                    f"dense cap of probabilities(). Use sample() for the "
                    f"histogram, which has no width limit."
                )
            else:
                cap = (
                    ". Use sample() for the histogram, which has no width limit, "
                    "and divide its all-zeros count by the shots."
                )
            raise ValueError(
                f"{method}() on {device!r} reads Braket's Probability result type "
                f"over all {n_qubits} qubits, and managed simulators and QPUs "
                f"return it for at most {_MAX_PROBABILITY_QUBITS} qubits{cap}"
            )
    elif result_type == "Amplitude" and method == "probabilities":
        if n_qubits > _MAX_AMPLITUDE_LISTING_QUBITS:
            raise ValueError(
                f"probabilities() at shots=0 on {device!r} lists all 2**{n_qubits} "
                f"basis states in the program, which is over the 5 MB Braket "
                f"accepts above {_MAX_AMPLITUDE_LISTING_QUBITS} qubits. Use "
                f"shots > 0 for the shot-frequency estimate, or sample() for the "
                f"histogram."
            )
    elif result_type == "DensityMatrix" and n_qubits > _MAX_DENSITY_MATRIX_QUBITS:
        raise ValueError(
            f"{method}() at shots=0 on {device!r} reads the density matrix of all "
            f"{n_qubits} qubits, and Braket returns one of at most "
            f"{_MAX_DENSITY_MATRIX_QUBITS}. Use shots > 0 for the shot-frequency "
            f"estimate, sample() for the histogram, or SV1 for an exact answer."
        )


def _result_type_limits(device) -> dict[str, tuple[int, int]] | None:
    """Read the result types a device publishes, with their shot ranges.

    Reads the OpenQASM action of `device.properties`, the capability document
    `AwsDevice` fetches from GetDevice. Returns `None` when there is nothing
    readable in it.
    """
    try:
        entries = device.properties.action[_OPENQASM_ACTION].supportedResultTypes
    except (AttributeError, KeyError, TypeError):
        return None
    limits: dict[str, tuple[int, int]] = {}
    for entry in entries or ():
        name = getattr(entry, "name", None)
        low = getattr(entry, "minShots", None)
        high = getattr(entry, "maxShots", None)
        if isinstance(name, str) and isinstance(low, int) and isinstance(high, int):
            limits[name] = (low, high)
    return limits or None


def _carried_result_type(result) -> str:
    """Name of the result type a task result carries, e.g. `"Probability"`.

    A result that does not list its result types reads as `Probability`, the
    one result type every probability task can carry.
    """
    result_types = getattr(result, "result_types", None)
    if not result_types:
        return "Probability"
    return type(result_types[0].type).__name__


def _is_tag_text(text: str) -> bool:
    return all(unicodedata.category(ch)[0] in "LNZ" or ch in _TAG_SYMBOLS for ch in text)


def _validate_tags(tags: Mapping[str, str] | None) -> dict[str, str] | None:
    """Check `tags` against the AWS tag rules and return a plain-dict copy."""
    if tags is None:
        return None
    if len(tags) > _MAX_TAGS:
        raise ValueError(f"at most {_MAX_TAGS} tags are allowed, got {len(tags)}.")
    for key, value in tags.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"tag keys and values must be str, got {key!r}: {value!r}.")
        if not 1 <= len(key) <= _MAX_TAG_KEY_LENGTH:
            raise ValueError(
                f"tag key length must be 1..{_MAX_TAG_KEY_LENGTH}, got {len(key)}."
            )
        if len(value) > _MAX_TAG_VALUE_LENGTH:
            raise ValueError(
                f"tag value length must be at most {_MAX_TAG_VALUE_LENGTH}, "
                f"got {len(value)} for key {key!r}."
            )
        if key.lower().startswith(_RESERVED_TAG_PREFIX):
            raise ValueError(f"tag key {key!r} uses the prefix aws:, which AWS reserves.")
        if not _is_tag_text(key) or not _is_tag_text(value):
            raise ValueError(
                f"tag {key!r}: {value!r} has a character AWS does not allow in "
                f"tags; use letters, digits, spaces and _ . : / = + - @ only."
            )
    return dict(tags)


class BraketJobHandle:
    """A handle to a submitted Braket task.

    It holds the Braket `QuantumTask` and the plan that turns the raw expectation
    values of the task into one value for each `Observable`.
    `compiled.submit(weights, inputs)` and `compiled.attach(task_id)` return one of
    these handles. Call `result()` to wait until the task finishes, or `status()`
    to check the task without waiting.
    """

    def __init__(
        self,
        task,
        aggregation_plan: list[list[tuple[float, int]]],
        poll_timeout_seconds: int,
        method: str = "run",
        *,
        n_qubits: int | None = None,
    ):
        """Wrap a submitted Braket task in a job handle.

        `BraketCompiled.submit()` or `BraketCompiled.attach()` builds it. It is not
        meant to be created directly.

        Args:
            task (QuantumTask): The Braket quantum task returned by the Braket
                package.
            aggregation_plan: How the raw result values map to the requested
                observables. For each observable, it lists the
                `(coefficient, result index)` pairs to add up.
            poll_timeout_seconds: How many seconds `result()` waits before giving up.
            method (str): The method of the compiled circuit that the task runs,
                one of `BraketCompiled.submit_methods`. It decides how `result()`
                decodes the values of the task.
            n_qubits (int | None): The number of qubits of the circuit. When it is
                given, `result()` checks that a `"probabilities"` or
                `"zero_state_probability"` task returned a distribution over
                `2**n_qubits` states, and it writes the keys of a `"sample"`
                histogram as bitstrings of `n_qubits` characters.

        Raises:
            ValueError: If `method` is not in `BraketCompiled.submit_methods`.
        """
        _check_submit_method(method)
        self._task = task
        self._aggregation_plan = aggregation_plan
        self._poll_timeout_seconds = poll_timeout_seconds
        self._method = method
        self._n_qubits = n_qubits
        # One result value per unique Pauli term, which is how the plan's
        # indices were assigned.
        self._n_expectation_values = 1 + max(
            (index for plan in aggregation_plan for _, index in plan), default=-1
        )

    @property
    def method(self) -> str:
        """The method of the compiled circuit that this task runs.

        It is `"run"`, `"sample"`, `"probabilities"` or `"zero_state_probability"`.
        """
        return self._method

    def status(self) -> str:
        """Return the state of the task, without waiting.

        The state is a string such as `"CREATED"`, `"QUEUED"`, `"RUNNING"`,
        `"COMPLETED"`, `"FAILED"` or `"CANCELLED"`.
        """
        return self._task.state()

    def cancel(self) -> None:
        """Ask Braket to cancel the quantum task. It returns without waiting."""
        self._task.cancel()

    def task_id(self) -> str:
        """Return the identifier of the Braket quantum task, to look it up in the
        Amazon Web Services (AWS) console.
        """
        return self._task.id

    def result(self) -> np.ndarray | dict[str, int]:
        """Wait until the task finishes, then decode its values.

        It waits for at most `poll_timeout_seconds`. The decoding follows the
        `method` of this handle and the result type that the result of the task
        actually holds, so a handle from `attach()` decodes a task exactly as the
        handle returned by `submit()` would have.

        Returns:
            For `method="run"`, one `float64` expectation value for each
                observable, of shape `(n_obs,)`, exactly as `BraketCompiled.run()`
                returns. For `method="sample"`, the measurement histogram as
                `BraketCompiled.sample()` returns it: the number of shots for each
                bitstring, with qubit `0` as the leftmost character. For
                `method="probabilities"`, the `float64` distribution of shape
                `(2**n_qubits,)` that `BraketCompiled.probabilities()` returns. For
                `method="zero_state_probability"`, a zero-dimensional `float64`
                array that holds the value that
                `BraketCompiled.zero_state_probability()` returns.

        Raises:
            RuntimeError: If the task has no result, because it ended `FAILED` or
                `CANCELLED`, or because the wait timed out. The message names the
                task and its state.
            ValueError: If the values of the task do not have the shape that the
                `method` and the circuit of this handle produce, for example
                because the handle was attached with the wrong `method`.
        """
        result = self._task.result()
        if result is None:
            raise RuntimeError(self._no_result_message())
        if self._method == "sample":
            return self._decode_counts(result)
        if self._method == "run":
            values = result.values
            self._check_values(values)
            return _aggregate_values(values, self._aggregation_plan)
        return self._decode_distribution(result)

    def _no_result_message(self) -> str:
        state = self._task.state()
        if state in _NO_RESULT_STATES:
            return f"Braket task {self._task.id} ended {state} and has no result."
        return (
            f"Braket task {self._task.id} returned no result: it is still {state} "
            f"after poll_timeout_seconds={self._poll_timeout_seconds}. Call "
            f"result() again to keep waiting."
        )

    def _mismatch(self, detail: str) -> ValueError:
        return ValueError(
            f"Braket task {self._task.id} {detail}; the task was probably "
            f"submitted with another method, circuit or observables than this "
            f"handle's (method={self._method!r})."
        )

    def _check_values(self, values) -> None:
        if len(values) != self._n_expectation_values or any(
            np.ndim(value) != 0 for value in values
        ):
            raise self._mismatch(
                f"returned {len(values)} result values, expected "
                f"{self._n_expectation_values} scalar expectation values"
            )

    def _decode_counts(self, result) -> dict[str, int]:
        """The task's measurement histogram, keyed qubit-0-MSB over the whole
        circuit.

        Braket keys counts by `measured_qubits`, in that order, and measures
        only the qubits the circuit touches; a qubit no gate acts on stays in
        `|0>`, so it is filled in as `0`.
        """
        counts = getattr(result, "measurement_counts", None)
        if getattr(result, "result_types", None) or not counts:
            raise self._mismatch("returned no plain measurement counts")
        first_key = next(iter(counts))
        measured = [int(q) for q in (getattr(result, "measured_qubits", None) or ())]
        if not measured:
            measured = list(range(len(first_key)))
        width = self._n_qubits if self._n_qubits is not None else max(measured) + 1
        if max(measured) >= width:
            raise self._mismatch(
                f"measured qubit {max(measured)}, outside the {width}-qubit circuit"
            )
        padded: dict[str, int] = {}
        for key, count in counts.items():
            if len(key) != len(measured):
                raise self._mismatch(
                    f"returned measurement key {key!r} for {len(measured)} "
                    f"measured qubits"
                )
            bits = ["0"] * width
            for position, qubit in enumerate(measured):
                bits[qubit] = key[position]
            bitstring = "".join(bits)
            padded[bitstring] = padded.get(bitstring, 0) + int(count)
        return _sampling.normalize_counts(padded, width, little_endian=False)

    def _decode_distribution(self, result) -> np.ndarray:
        """Decode a `probabilities` or `zero_state_probability` task from the
        one result type it carries, into qubit-0-MSB basis-state order.

        `Probability` is the distribution itself, `StateVector` gives it as
        `|amplitude|**2`, a `DensityMatrix` as the real part of its diagonal,
        and `Amplitude` as `|amplitude|**2` of each requested state. The
        complex values are accepted both as complex numbers (the local
        simulator) and as `[real, imag]` pairs (the managed simulators; see
        `_complex_from_wire`).
        """
        values = result.values
        kind = _carried_result_type(result)
        if len(values) != 1:
            raise self._mismatch(f"returned {len(values)} result values, not one {kind}")
        value = values[0]
        n_qubits = self._n_qubits
        zero_state_only = self._method == "zero_state_probability"
        if kind == "Amplitude":
            if not isinstance(value, Mapping) or not value:
                raise self._mismatch("returned an Amplitude result with no states")
            n_qubits = n_qubits if n_qubits is not None else len(next(iter(value)))
            if zero_state_only:
                amplitude = value.get("0" * n_qubits)
                if amplitude is None:
                    raise self._mismatch("returned no amplitude for the all-zeros state")
                return np.asarray(self._probability_of(amplitude), dtype=np.float64)
            if len(value) != 2**n_qubits:
                raise self._mismatch(
                    f"returned {len(value)} amplitudes, expected 2**{n_qubits} = "
                    f"{2**n_qubits}"
                )
            distribution = np.zeros(2**n_qubits, dtype=np.float64)
            for state, amplitude in value.items():
                distribution[int(state, 2)] = self._probability_of(amplitude)
            return distribution
        if kind == "DensityMatrix":
            matrix = _complex_from_wire(value, ndim=2)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise self._mismatch("returned a density matrix that is not square")
            distribution = np.real(np.diagonal(matrix)).astype(np.float64)
        elif kind == "StateVector":
            vector = _complex_from_wire(value, ndim=1)
            if vector.ndim != 1:
                raise self._mismatch("returned a state vector that is not 1-D")
            distribution = (np.abs(vector) ** 2).astype(np.float64)
        else:
            if np.ndim(value) != 1:
                raise self._mismatch(
                    f"returned {len(values)} result values, not one probability vector"
                )
            distribution = np.asarray(value, dtype=np.float64)
        if n_qubits is not None and len(distribution) != 2**n_qubits:
            raise self._mismatch(
                f"returned a distribution of length {len(distribution)}, expected "
                f"2**{n_qubits} = {2**n_qubits}"
            )
        if zero_state_only:
            # Index 0 is the all-zeros state.
            return np.asarray(distribution[0], dtype=np.float64)
        return distribution

    def _probability_of(self, amplitude) -> float:
        amplitude = _complex_from_wire(amplitude, ndim=0)
        if amplitude.ndim != 0:
            raise self._mismatch("returned an amplitude that is not a complex number")
        return float(np.abs(amplitude) ** 2)


def _complex_from_wire(value, ndim: int) -> np.ndarray:
    """`value` as an array, with the managed simulators' encoding of complex
    numbers undone.

    SV1 and DM1 serialize each complex number of a result as a `[real, imag]`
    pair, and the SDK decodes the pairs of only some result types:
    amazon-braket-sdk 1.117 does so for `Amplitude` and `StateVector`, not for
    `DensityMatrix`, whose value arrives as nested lists of shape `(d, d, 2)`.
    The local simulator hands over complex arrays. So a real array of `ndim + 1`
    dimensions whose last axis has length 2 is read as pairs; anything else is
    returned as is, for the caller to check its shape.
    """
    array = np.asarray(value)
    if array.ndim == ndim + 1 and array.shape[-1] == 2 and np.isrealobj(array):
        return array[..., 0] + 1j * array[..., 1]
    return array


def _qubitwise_commuting_groups(
    terms: list[tuple[tuple[str, int], ...]],
) -> list[list[int]]:
    """Indices of `terms` split greedily into qubit-wise-commuting groups."""
    groups: list[tuple[dict[int, str], list[int]]] = []
    for index, paulis in enumerate(terms):
        for basis, members in groups:
            if all(basis.get(qubit, letter) == letter for letter, qubit in paulis):
                basis.update({qubit: letter for letter, qubit in paulis})
                members.append(index)
                break
        else:
            groups.append(({qubit: letter for letter, qubit in paulis}, [index]))
    return [members for _, members in groups]


def _aggregate_values(
    values, aggregation_plan: list[list[tuple[float, int]]]
) -> np.ndarray:
    output = np.zeros(len(aggregation_plan), dtype=np.float64)
    for k, plan in enumerate(aggregation_plan):
        output[k] = sum(coeff * float(values[idx]) for coeff, idx in plan)
    return output


class BraketCompiled(CompiledCircuit):
    """A circuit prepared for Amazon Braket.

    This class is internal: you get it from
    [`BraketBackend.compile()`][eigenkit.core.backends.braket_backend.BraketBackend.compile],
    and you never create it directly. `run()` evaluates the observables and waits
    for the result. `submit()` returns a
    [`BraketJobHandle`][eigenkit.core.backends.braket_backend.BraketJobHandle] for
    a cloud or quantum processing unit (QPU) task, without waiting. `attach()`
    rebuilds that handle from the Amazon Resource Name (ARN) of a task.
    `dry_run_info()` reports how many circuits a call runs, without submitting
    anything.
    """

    submit_methods: frozenset[str] = _SUBMIT_METHODS
    """The methods `submit()` and `attach()` accept: `"run"`, `"sample"`,
    `"probabilities"` and `"zero_state_probability"`. Any method added to this
    adapter's async path must also be added here, or callers that check this
    set will refuse it."""

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: BraketConfig,
        *,
        warn: bool = True,
    ):
        """Prepare the circuit for repeated evaluation.

        `BraketBackend.compile()` builds it, and there is no reason to create one
        directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that this handle
                measures.
            config (BraketConfig): The device settings.
            warn (bool): Whether to issue the cost warnings for a cloud device. It
                is `False` only for the shifted copies that `compile_shifted`
                builds. Default: `True`.
        """
        if config.is_qpu and config.shots <= 0:
            raise ValueError(
                f"shots must be > 0 for QPU runs (device_arn={config.device_arn!r}). "
                f"Set BraketConfig(shots=N, ...)."
            )
        if warn and config.is_cloud:
            warnings.warn(
                f"Braket cloud run will incur AWS charges (device_arn="
                f"{config.device_arn!r}, shots={config.shots}). "
                f"Use compiled.dry_run_info() to inspect the circuit before "
                f"submitting.",
                CostWarning,
                stacklevel=4,
            )

        _require_braket()
        from braket.circuits import Circuit as BraketCircuit
        from braket.circuits import FreeParameter
        from braket.circuits.observables import X as BraketX
        from braket.circuits.observables import Y as BraketY
        from braket.circuits.observables import Z as BraketZ
        from braket.devices import LocalSimulator

        pauli_factory = {"X": BraketX, "Y": BraketY, "Z": BraketZ}
        self._LocalSimulator = LocalSimulator

        self._config = config
        self._observables = tuple(observables)
        self._n_qubits = circuit.n_qubits
        self._weight_symbols = [FreeParameter(f"w{i}") for i in range(circuit.n_weights)]
        self._input_symbols = [FreeParameter(f"x{i}") for i in range(circuit.n_inputs)]

        def gates_only():
            bk_circuit = BraketCircuit()
            for gate in circuit.gates:
                _apply_gate_symbolic(
                    bk_circuit, gate, self._weight_symbols, self._input_symbols
                )
            return bk_circuit

        # One expectation result per unique Pauli term, then aggregated per
        # observable with coefficients: a term that appears in several
        # observables shares one index in the values vector.
        self._aggregation_plan: list[list[tuple[float, int]]] = []
        seen_term_index: dict[tuple[tuple[str, int], ...], int] = {}
        unique_terms: list = []
        for observable in observables:
            plan: list[tuple[float, int]] = []
            for term in observable.terms:
                # No identity-only term to skip here: `PauliTerm` rejects one at
                # construction, so `term.paulis` is never empty.
                index = seen_term_index.get(term.paulis)
                if index is None:
                    index = len(seen_term_index)
                    seen_term_index[term.paulis] = index
                    unique_terms.append(term)
                plan.append((float(term.coeff), index))
            self._aggregation_plan.append(plan)

        # With shots, one task can only measure terms that commute qubit-wise
        # (Braket refuses the rest), so the terms are split into such groups,
        # one circuit each. Exact (shots=0) results take every term at once.
        self._term_groups = (
            _qubitwise_commuting_groups([t.paulis for t in unique_terms])
            if config.shots > 0
            else [list(range(len(unique_terms)))]
        )
        self._group_circuits = []
        for members in self._term_groups:
            bk_circuit = gates_only()
            for index in members:
                bk_obs, targets = _pauli_term_to_braket(
                    unique_terms[index], pauli_factory
                )
                bk_circuit.expectation(observable=bk_obs, target=targets)
            self._group_circuits.append(bk_circuit)
        self._parametric_circuit = self._group_circuits[0]
        # Kept for lazy probability-circuit construction.
        self._ir_for_zsp = circuit
        # The probability circuit is separate because mixing expectation
        # and probability result types would change the order Braket
        # returns values in. Built lazily on first ZSP call.
        self._zsp_circuit = None
        # The sample and exact shots=0 circuits, keyed (method, result type)
        # and built lazily the same way. The `Probability` circuit is
        # `_zsp_circuit`, shared by `probabilities` and ZSP.
        self._measurement_circuits: dict[tuple[str, str | None], object] = {}
        self._n_gates = len(circuit.gates)
        self._n_unique_pauli_terms = len(unique_terms)
        self._n_observables = len(observables)
        if warn and config.is_qpu and config.enable_gradients:
            # One task per group of qubit-wise-commuting terms per call.
            _warn_gradient_cost(
                f"Braket QPU {config.device_arn!r}",
                circuit,
                len(self._term_groups),
                config.shots,
                unit="tasks",
                stacklevel=4,
            )
        # Device construction is deferred to first use. `AwsDevice()`
        # makes a live boto3 call to populate properties, which we don't
        # want during compile so that `dry_run_info()` works offline and
        # tests don't hit AWS.
        self._device = None

    @property
    def device(self):
        """The local simulator or the `AwsDevice`, created when first read.

        Authentication with Amazon Web Services (AWS) and network calls happen
        only when this property is first read.
        """
        if self._device is None:
            if self._config.device_arn is None:
                self._device = self._LocalSimulator(self._config.local_backend)
            else:
                _require_braket_aws()
                from braket.aws import AwsDevice

                self._device = AwsDevice(self._config.device_arn)
        return self._device

    def compile_shifted(self, circuit: CircuitSpec) -> "BraketCompiled":
        """Compile a copy of this circuit with one gate angle shifted.

        The chain-rule gradient compiles one copy for each shifted gate. The
        compilation of this handle already issued the cost warnings, so the copy
        issues none.

        Args:
            circuit (CircuitSpec): The shifted circuit.

        Returns:
            A `BraketCompiled` handle for `circuit`.
        """
        return BraketCompiled(circuit, self._observables, self._config, warn=False)

    # ---- inspection ----

    def dry_run_info(self) -> dict:
        """Report the size of the circuit without submitting anything.

        Use it to check a cloud submission before you submit it. It reports how many
        circuits a call runs. It does not estimate a price.
        """
        return {
            "device_arn": self._config.device_arn,
            "local_backend": self._config.local_backend,
            "shots": self._config.shots,
            "n_qubits": self._n_qubits,
            "n_gates": self._n_gates,
            "n_observables": self._n_observables,
            "n_unique_pauli_terms": self._n_unique_pauli_terms,
            "n_circuit_runs_per_call": len(self._term_groups),
            "is_cloud": self._config.is_cloud,
            "is_qpu": self._config.is_qpu,
        }

    # ---- run / submit ----

    def _binding(self, weights: np.ndarray, inputs: np.ndarray) -> dict:
        binding: dict[str, float] = {}
        for i, symbol in enumerate(self._weight_symbols):
            binding[symbol.name] = float(weights[i])
        for i, symbol in enumerate(self._input_symbols):
            binding[symbol.name] = float(inputs[i])
        return binding

    def _run_kwargs(self) -> dict:
        kwargs: dict = {}
        if self._config.shots > 0:
            kwargs["shots"] = self._config.shots
        if self._config.is_cloud and self._config.s3_bucket is not None:
            kwargs["s3_destination_folder"] = (
                self._config.s3_bucket,
                self._config.s3_prefix,
            )
        if self._config.is_cloud:
            kwargs["poll_timeout_seconds"] = self._config.poll_timeout_seconds
        return kwargs

    def run(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return one expectation value per observable.

        It follows the `CompiledCircuit` contract: the result has shape `(n_obs,)`
        when `weights` and `inputs` are both 1-D, and `(B, n_obs)` otherwise. With
        `shots > 0`, each sample runs one task for each group of terms that commute
        qubit-wise (see `dry_run_info()["n_circuit_runs_per_call"]`).

        Args:
            weights (np.ndarray): The parameter vector: 1-D for one sample, or 2-D
                with a leading batch axis.
            inputs (np.ndarray): The input features, with the same 1-D or 2-D shape
                rule as `weights`.

        Returns:
            One expectation value for each observable (for each sample, in a batch).
        """
        return batch_call(self._run_one, weights, inputs)

    def _run_one(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        values = np.empty(self._n_unique_pauli_terms)
        for circuit, members in zip(self._group_circuits, self._term_groups):
            result = self.device.run(
                circuit, inputs=self._binding(weights, inputs), **self._run_kwargs()
            ).result()
            values[members] = result.values
        return _aggregate_values(values, self._aggregation_plan)

    def submit(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
        *,
        method: str = "run",
        tags: Mapping[str, str] | None = None,
    ) -> BraketJobHandle:
        """Submit the circuit without waiting, and return a `BraketJobHandle`.

        The handle holds the Braket `QuantumTask`.

        Args:
            weights (np.ndarray): The parameter vector of one sample, 1-D.
            inputs (np.ndarray): The input features of one sample, 1-D.
            method (str): What the task computes, one of `submit_methods`.
                `"run"` (the default) measures the observables, like `run()`.
                `"sample"` returns the measurement histogram, like `sample()`.
                `"probabilities"` returns the distribution over every basis
                state, like `probabilities()`. `"zero_state_probability"`
                measures the all-zeros probability, like
                `zero_state_probability()`.
            tags (Mapping[str, str] | None): Amazon Web Services (AWS) tags to put
                on the quantum task, to find it again later. There can be at most
                50 tags. Keys have 1 to 128 characters and values at most 256
                characters, made of letters, digits, spaces and `_ . : / = + - @`.
                A key cannot start with `aws:`. Only cloud tasks are tagged, and
                the local simulator ignores the tags.

        Returns:
            handle (JobHandle): A handle for the submitted job.

        Raises:
            ValueError: If `method` is not in `submit_methods`, if `weights` or
                `inputs` is not 1-D, if `tags` breaks the AWS tag rules, if
                `method` is `"sample"` with `shots=0`, if the device publishes
                no result type that `method` can read at the configured shots,
                or if the result type it reads is limited to fewer qubits than
                the circuit has. The limits are 15 qubits for `Probability` on a
                cloud device
                ([List of result types](https://docs.aws.amazon.com/braket/latest/developerguide/braket-result-types.html)),
                which `"probabilities"` and `"zero_state_probability"` read with
                `shots > 0`, and the exact limits listed under `probabilities()`.
                It is also raised if `method` is `"run"` and the observables need
                more than one task, because a handle tracks one task. This happens
                with `shots > 0` when the terms do not all commute qubit-wise. Use
                `run()`, or split the observables into sets that commute. Nothing
                is submitted in any of these cases.
        """
        _check_submit_method(method)
        _check_one_sample("submit", weights, inputs)
        checked_tags = _validate_tags(tags)
        kwargs = self._run_kwargs()
        if checked_tags and self._config.is_cloud:
            kwargs["tags"] = checked_tags
        # Chosen before the device is touched: a sample with shots=0 fails
        # without constructing an AwsDevice.
        circuit = self._circuit_for(method)
        task = self.device.run(circuit, inputs=self._binding(weights, inputs), **kwargs)
        return self._handle(task, method)

    def _check_one_task(self) -> None:
        if len(self._term_groups) > 1:
            raise ValueError(
                f"these observables need {len(self._term_groups)} tasks at "
                f"shots={self._config.shots}, one per group of terms that "
                "commute qubit-wise, and a handle tracks one task. Use run(), "
                "or split the observables into commuting sets."
            )

    def _handle(self, task, method: str) -> BraketJobHandle:
        return BraketJobHandle(
            task=task,
            aggregation_plan=self._aggregation_plan,
            poll_timeout_seconds=self._config.poll_timeout_seconds,
            method=method,
            n_qubits=self._n_qubits,
        )

    def attach(self, task_id: str, *, method: str = "run") -> BraketJobHandle:
        """Rebuild a job handle for a task submitted earlier from this circuit.

        Use it to pick up a task after the process that submitted it has stopped,
        for example after a crash. The handle decodes the results exactly as the
        handle returned by `submit()` would have, provided that the task was
        submitted from the same circuit, observables and `method`.

        Args:
            task_id (str): The Amazon Resource Name (ARN) of the Braket quantum
                task, as returned by `BraketJobHandle.task_id()`.
            method (str): The `method` that the task was submitted with, one of
                `submit_methods`.

        Returns:
            handle (JobHandle): A handle for the existing task.

        Raises:
            ValueError: If the configuration is not a cloud configuration, if
                `task_id` is not the ARN of a Braket quantum task, if `method`
                is not in `submit_methods`, or if `method` is `"run"` and the
                observables need more than one task (see `submit()`). The Braket
                package is not called in any of these cases.
        """
        if not self._config.is_cloud:
            raise ValueError(
                "attach() needs a cloud config (BraketConfig(device_arn=...)); "
                "local-simulator tasks cannot be re-attached."
            )
        if not (
            isinstance(task_id, str)
            and task_id.startswith("arn:aws:braket:")
            and ":quantum-task/" in task_id
        ):
            raise ValueError(
                f"task_id must be a Braket quantum-task ARN "
                f"(arn:aws:braket:<region>:<account>:quantum-task/<id>), "
                f"got {task_id!r}."
            )
        _check_submit_method(method)
        if method == "run":
            self._check_one_task()
        _require_braket_aws()
        from braket.aws import AwsQuantumTask

        task = AwsQuantumTask(
            task_id, poll_timeout_seconds=self._config.poll_timeout_seconds
        )
        return self._handle(task, method)

    # ---- per-method circuits ----

    def _result_type_for(self, method: str) -> str | None:
        """The result type `method` reads on this device; see
        `_select_result_type`. Only a cloud device publishes capabilities; the
        local simulator keeps the `Probability` path at any shot count."""
        limits = None
        if method != "sample" and self._config.is_cloud:
            limits = _result_type_limits(self.device)
        return _select_result_type(
            method, self._config.shots, limits, self._device_label()
        )

    def _device_label(self) -> str:
        return (
            self._config.device_arn or f"LocalSimulator({self._config.local_backend!r})"
        )

    def _circuit_for(self, method: str):
        if method == "run":
            self._check_one_task()
            return self._parametric_circuit
        if method == "probabilities" and self._n_qubits > _sampling.MAX_DENSE_QUBITS:
            raise ValueError(
                f"probabilities() would return 2**{self._n_qubits} floats; the cap "
                f"is {_sampling.MAX_DENSE_QUBITS} qubits. Use sample() and read "
                f"the histogram directly."
            )
        result_type = self._result_type_for(method)
        # Before the circuit is built: the SV1 state list is itself 2**n long.
        _check_result_width(
            method,
            result_type,
            self._n_qubits,
            self._device_label(),
            cloud=self._config.is_cloud,
        )
        if result_type == "Probability":
            return self._zero_state_circuit()
        key = (method, result_type)
        if key not in self._measurement_circuits:
            self._measurement_circuits[key] = self._build_measurement_circuit(
                method, result_type
            )
        return self._measurement_circuits[key]

    def _gate_circuit(self, *, cover_every_qubit: bool):
        """The gates alone, bound to the same FreeParameter symbols as the
        expectation-value circuit so the binding logic is shared.

        With `cover_every_qubit`, a qubit no gate acts on gets an identity, so
        a result type over every qubit has all of them to read: Braket sizes
        the qubit register from the qubits the instructions use.
        """
        from braket.circuits import Circuit as BraketCircuit

        bk_circuit = BraketCircuit()
        for gate in self._ir_for_zsp.gates:
            _apply_gate_symbolic(
                bk_circuit,
                gate,
                self._weight_symbols,
                self._input_symbols,
            )
        if cover_every_qubit:
            used = {int(qubit) for qubit in bk_circuit.qubits}
            for qubit in range(self._n_qubits):
                if qubit not in used:
                    bk_circuit.i(qubit)
        return bk_circuit

    def _zero_state_circuit(self):
        if self._zsp_circuit is None:
            self._zsp_circuit = self._build_zsp_circuit()
        return self._zsp_circuit

    def _build_zsp_circuit(self):
        """Lazily build a parametric circuit with a `Probability` result
        on every qubit. `probabilities` and `zero_state_probability` share it."""
        bk_circuit = self._gate_circuit(cover_every_qubit=True)
        bk_circuit.probability(target=list(range(self._n_qubits)))
        return bk_circuit

    def _build_measurement_circuit(self, method: str, result_type: str | None):
        """Lazily build the circuit for `sample` (no result type: the task
        returns measurement counts) or an exact shots=0 result type."""
        if result_type is None:
            # Braket measures the qubits the circuit uses; the decode fills
            # the others in as 0, so no identity is needed on hardware.
            return self._gate_circuit(cover_every_qubit=False)
        bk_circuit = self._gate_circuit(cover_every_qubit=True)
        n_qubits = self._n_qubits
        if result_type == "StateVector":
            bk_circuit.state_vector()
        elif result_type == "DensityMatrix":
            bk_circuit.density_matrix(target=list(range(n_qubits)))
        elif result_type == "Amplitude":
            if method == "zero_state_probability":
                states = ["0" * n_qubits]
            else:
                states = [format(index, f"0{n_qubits}b") for index in range(2**n_qubits)]
            bk_circuit.amplitude(state=states)
        else:
            raise ValueError(f"no circuit for result type {result_type!r}")
        return bk_circuit

    def _evaluate(self, method: str, weights: np.ndarray, inputs: np.ndarray):
        """Run `method` as one blocking task and decode it like its handle."""
        circuit = self._circuit_for(method)
        task = self.device.run(
            circuit, inputs=self._binding(weights, inputs), **self._run_kwargs()
        )
        return self._handle(task, method).result()

    def sample(self, weights: np.ndarray, inputs: np.ndarray) -> dict[str, int]:
        """Measure one sample in the computational basis and return the histogram.

        It runs one task with the configured shots and no result type, and reads
        the measurement counts of the task. A qubit that no gate acts on is
        reported as `0`.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A 1-D vector of input features.

        Returns:
            counts (dict[str, int]): The number of shots for each measured
                bitstring, with qubit `0` as the first character of each key. The
                counts add up to the configured shots.

        Raises:
            ValueError: If the configuration has `shots=0` (a histogram needs
                shots), or if `weights` or `inputs` is not 1-D. Nothing runs then.
        """
        _check_one_sample("sample", weights, inputs)
        return self._evaluate("sample", weights, inputs)

    def probabilities(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Return the measurement distribution over all `2**n_qubits` basis states.

        With `shots > 0`, it is estimated from the shot frequencies, through the
        `Probability` result type. With `shots=0`, it is the exact distribution:
        `Probability` on the local simulator, the squared amplitudes of every basis
        state on SV1 (`Amplitude`), and the diagonal of the density matrix on DM1
        (`DensityMatrix`). It is limited to `_sampling.MAX_DENSE_QUBITS` qubits,
        and every cloud path has a lower limit. The limit is 15 qubits with
        `shots > 0`, the widest `Probability` that the managed simulators and the
        quantum processing units (QPUs) of Braket return
        ([List of result types](https://docs.aws.amazon.com/braket/latest/developerguide/braket-result-types.html)).
        It is 17 qubits on SV1 at `shots=0`, because the program lists all
        `2**n_qubits` states and must stay under the Braket limit of 5 megabytes.
        It is 8 qubits on DM1 at `shots=0`, the widest density matrix that DM1
        returns. `sample()` has no limit on the number of qubits.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A 1-D vector of input features.

        Returns:
            probabilities (np.ndarray): A `float64` array of shape
                `(2**n_qubits,)`, ordered with qubit `0` as the most significant
                bit, whose entries add up to `1`.

        Raises:
            ValueError: If the device publishes none of the result types above
                for the configured shots, if the circuit has more qubits than the
                limit of the path taken, or if `weights` or `inputs` is not 1-D.
                Nothing runs then.
        """
        _check_one_sample("probabilities", weights, inputs)
        return self._evaluate("probabilities", weights, inputs)

    def zero_state_probability(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
    ) -> float:
        """Return the probability of measuring all zeros at the end of the circuit.

        It uses the `Probability` result type of Braket, which every Braket
        simulator and quantum processing unit (QPU) supports with `shots > 0`, and
        which the local simulator supports at any number of shots. The managed
        simulators and the QPUs return it for at most 15 qubits
        ([List of result types](https://docs.aws.amazon.com/braket/latest/developerguide/braket-result-types.html)),
        so a wider circuit with `shots > 0` on a cloud device is refused. In that
        case, use `sample()`, which has no limit on the number of qubits, and
        divide the all-zeros count by the shots. The local simulator has no limit.
        With `shots=0` on a cloud simulator, which publishes no `Probability` at
        `shots=0`, the value is exact instead: the squared all-zeros amplitude on
        SV1 (`Amplitude`), for any number of qubits, and the `[0, 0]` entry of the
        density matrix on DM1 (`DensityMatrix`), which DM1 returns for at most 8
        qubits.

        Args:
            weights (np.ndarray): A 1-D parameter vector.
            inputs (np.ndarray): A 1-D vector of input features.

        Returns:
            The probability, a float in `[0, 1]`.

        Raises:
            ValueError: If the device publishes none of those result types for
                the configured shots, or if the circuit has more than 15 qubits on
                the cloud `Probability` path or more than 8 qubits on the DM1 path.
                Nothing runs then.
        """
        return float(self._evaluate("zero_state_probability", weights, inputs))


class BraketBackend:
    """Provider for Amazon Braket, registered as `"braket"`.

    It runs on the local simulator by default, or on a cloud simulator or a
    quantum processing unit (QPU) once `device_arn` is set in the configuration.

    This class is internal: you select the backend with `provider="braket"` or by
    passing a [`BraketConfig`][eigenkit.core.backends.braket_backend.BraketConfig],
    not by creating this class.
    """

    def __init__(
        self,
        config: BraketConfig | None = None,
        name: str = "braket",
    ):
        """Build the backend from a device configuration.

        Args:
            config (BraketConfig): The device settings.
            name (str): The registry name, which error messages use.
        """
        self.name = name
        self.config = config or BraketConfig()

    @property
    def gradient_method(self) -> str:
        """How this backend computes gradients.

        It is `"parameter_shift"` on a simulator. On a quantum processing unit
        (QPU), it is `"none"`, unless the configuration sets `enable_gradients`. A
        parameter-shift backward pass costs at least two paid tasks per weight on
        real hardware, so gradients are refused there unless you turn them on.
        """
        # Local and cloud simulators always allow parameter-shift; a QPU only
        # when the config opts in to 2*n_weights paid tasks per backward pass.
        if self.config.is_qpu and not self.config.enable_gradients:
            return "none"
        return "parameter_shift"

    @property
    def capabilities(self) -> Capabilities:
        """What this backend supports.

        It supports every gate except `AmplitudeEncoding`, `CPHASE` and
        `ControlledQubitUnitary`, which this backend does not translate into Braket
        circuits. It can sample with a finite number of shots. It has no native
        batching, so `run()` loops over the samples itself. It has no limit on the
        number of qubits. See
        [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        # AmplitudeEncoding has no native state-initialization primitive in
        # Braket circuits. Exclude DATA_GATES that are unsupported so the
        # dispatch layer surfaces a clear error before attempting to compile.
        return Capabilities(
            supported_gates=ALL_GATES
            - frozenset({"AmplitudeEncoding", "CPHASE", "ControlledQubitUnitary"}),
            supports_batching=False,
            supports_shots=True,
            max_qubits=None,
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> BraketCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        It is called once for each `Circuit`, and the handle is reused for every
        call, so the expensive work happens here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that the handle
                measures.

        Returns:
            A `BraketCompiled` handle.
        """
        return BraketCompiled(circuit, observables, self.config)
