"""The CUDA Quantum backend (the `cuda-quantum` package).

`cudaq` is an optional dependency (`pip install eigenkit[cudaquantum]`). The
package is imported only when it is needed, so the rest of the library works
without it, and `compile()` raises a clear `ImportError` if it is missing.

It declares `gradient_method = "parameter_shift"`. `cudaq` has its own
parameter-shift tools, but this backend uses the fallback of the gradient code
instead, for consistency across backends. This keeps Qiskit, Braket and `cudaq`
on a single gradient code path.

`compile()` builds a parametric kernel with `cudaq.make_kernel` once. Each
`run()` calls `cudaq.observe(kernel, hamiltonian, *params)` for each observable.

Limitation: only the parameter expressions `Constant`, `WeightRef`, `InputRef`,
`Scale` and `Negate` are supported. `Sum` and `Multiply` expressions raise
`NotImplementedError`, because the arithmetic of the `cudaq` `QuakeValue` is more
restricted than that of the Qiskit `ParameterExpression`.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z
    from eigenkit import CudaQuantumBackend, register

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # by registered name, after registering it once
    register("cudaq", CudaQuantumBackend())
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="cudaq")

    # or with the config, when a device setting matters
    from eigenkit.configs import CudaQuantumConfig

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=CudaQuantumConfig(target="nvidia"),
    )
    ```

    This backend is optional: register it once with
    `register("cudaq", CudaQuantumBackend())` before the name can be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..ir import (
    ALL_GATES,
    DATA_GATES,
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
)
from ..provider import Capabilities, CompiledCircuit, batch_call, load_failure_hint


def _require_cudaq() -> None:
    """Probe for cudaq. Raise a clear error if it is not installed.

    Call this once before issuing local `import cudaq` statements in code
    paths that need it — the guard centralises the install hint so each
    call site stays a plain Python import.

    Also loads triton (torch's compiler on Linux) first: cudaq >= 0.15 puts
    its LLVM/MLIR into the global symbol scope, and triton, which bundles
    another LLVM, crashes in `dlopen` if it loads after them.
    """
    try:
        import triton  # noqa: F401
    except ImportError:
        pass
    try:
        import cudaq  # noqa: F401
    except ModuleNotFoundError as e:
        raise ImportError(
            "cuda-quantum is not installed. Install it via "
            "`pip install eigenkit[cudaquantum]`."
        ) from e
    except ImportError as e:
        # Installed but not loadable, e.g. its bundled OpenMP runtime on Linux ARM.
        raise ImportError(
            f"cuda-quantum is installed but failed to import{load_failure_hint(e)}. "
            f"Original error: {e}"
        ) from e


@dataclass(frozen=True)
class CudaQuantumConfig:
    """Device configuration for the CUDA Quantum backend.

    Args:
        target: The `cudaq` target. Common values are `"qpp-cpu"`, a statevector
            simulator on the central processing unit (CPU), which is the
            default; `"nvidia"`, on one graphics processing unit (GPU);
            `"nvidia-mgpu"`, on several GPUs; `"nvidia-mqpu"`, with several
            quantum processing units (QPUs); and `"tensornet"`. Default:
            `"qpp-cpu"`.
        shots: `None` for exact expectation values, or a positive integer for a
            finite number of shots. Default: `None`.
    """

    target: str = "qpp-cpu"
    shots: int | None = None

    def make_backend(self) -> "CudaQuantumBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `CudaQuantumBackend` for this configuration.
        """
        return CudaQuantumBackend(config=self)


def _resolve_for_kernel(
    parameter: Parameter,
    weight_arg,
    input_arg,
):
    """Resolve a Parameter against cudaq `make_kernel` arguments.

    cudaq's `QuakeValue` supports scalar multiplication and unary
    negation but not addition between two QuakeValues, so we raise
    `NotImplementedError` for `Sum` (and `Multiply`) over symbols.
    """
    if isinstance(parameter, Constant):
        return parameter.value
    if isinstance(parameter, WeightRef):
        if weight_arg is None:
            raise ValueError("circuit has no weights, but a WeightRef was used")
        return weight_arg[parameter.index]
    if isinstance(parameter, InputRef):
        if input_arg is None:
            raise ValueError("circuit has no inputs, but an InputRef was used")
        return input_arg[parameter.index]
    if isinstance(parameter, Scale):
        return parameter.factor * _resolve_for_kernel(
            parameter.expression, weight_arg, input_arg
        )
    if isinstance(parameter, Negate):
        return -_resolve_for_kernel(parameter.expression, weight_arg, input_arg)
    if isinstance(parameter, Sum):
        raise NotImplementedError(
            "Sum parameter expressions are not supported by the cudaq "
            "backend (QuakeValue arithmetic is restricted). Use "
            "Constant/WeightRef/InputRef/Scale/Negate, or expand the Sum "
            "at construction time."
        )
    if isinstance(parameter, Multiply):
        raise NotImplementedError(
            "Multiply parameter expressions are not supported by the "
            "cudaq backend. This blocks `Reuploading(trainable_scale=True)` "
            "and any other symbol-by-symbol product. Use a different "
            "backend (PennyLane, Qiskit and Braket all support "
            "Multiply), or set `trainable_scale=False`."
        )
    raise TypeError(parameter)


def _observable_to_cudaq(observable: Observable):
    """Translate an `Observable` to a `cudaq.SpinOperator`."""
    import cudaq

    pauli_factory = {
        "X": cudaq.spin.x,
        "Y": cudaq.spin.y,
        "Z": cudaq.spin.z,
    }
    hamiltonian = None
    for term in observable.terms:
        product = None
        for op, qubit in term.paulis:
            factor = pauli_factory[op](qubit)
            product = factor if product is None else product * factor
        weighted = term.coeff * product
        hamiltonian = weighted if hamiltonian is None else hamiltonian + weighted
    if hamiltonian is None:
        raise ValueError("observable reduced to zero or identity-only")
    return hamiltonian


class CudaQuantumCompiled(CompiledCircuit):
    """A circuit prepared for CUDA Quantum.

    It is internal:
    [`CudaQuantumBackend.compile()`][eigenkit.core.backends.cudaquantum_backend.CudaQuantumBackend.compile]
    builds it and returns it, and it is never built directly.
    """

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: CudaQuantumConfig,
    ):
        """Prepare the circuit for repeated evaluation.

        `CudaQuantumBackend.compile()` builds this object, so there is no reason
        to build one directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.
            config (CudaQuantumConfig): The device settings.
        """
        _require_cudaq()
        import cudaq

        cudaq.set_target(config.target)
        self._config = config

        n_w, n_x = circuit.n_weights, circuit.n_inputs
        if n_w > 0 and n_x > 0:
            kernel, weight_arg, input_arg = cudaq.make_kernel(list, list)
        elif n_w > 0:
            kernel, weight_arg = cudaq.make_kernel(list)
            input_arg = None
        elif n_x > 0:
            kernel, input_arg = cudaq.make_kernel(list)
            weight_arg = None
        else:
            kernel = cudaq.make_kernel()
            weight_arg = None
            input_arg = None

        qubits = kernel.qalloc(circuit.n_qubits)

        for gate in circuit.gates:
            self._apply_gate(kernel, gate, qubits, weight_arg, input_arg)

        self._kernel = kernel
        self._has_weights = n_w > 0
        self._has_inputs = n_x > 0
        self._hamiltonians = [_observable_to_cudaq(o) for o in observables]

    @staticmethod
    def _apply_gate(kernel, gate: Gate, qubits, weight_arg, input_arg) -> None:
        match gate.name:
            case "RX":
                kernel.rx(
                    _resolve_for_kernel(gate.params[0], weight_arg, input_arg),
                    qubits[gate.qubits[0]],
                )
            case "RY":
                kernel.ry(
                    _resolve_for_kernel(gate.params[0], weight_arg, input_arg),
                    qubits[gate.qubits[0]],
                )
            case "RZ":
                kernel.rz(
                    _resolve_for_kernel(gate.params[0], weight_arg, input_arg),
                    qubits[gate.qubits[0]],
                )
            case "H":
                kernel.h(qubits[gate.qubits[0]])
            case "X":
                kernel.x(qubits[gate.qubits[0]])
            case "S":
                kernel.s(qubits[gate.qubits[0]])
            case "Sdg":
                kernel.sdg(qubits[gate.qubits[0]])
            case "CNOT":
                kernel.cx(qubits[gate.qubits[0]], qubits[gate.qubits[1]])
            case "Z":
                kernel.z(qubits[gate.qubits[0]])
            case "MCX" | "MCZ":
                raise NotImplementedError(
                    f"gate '{gate.name}' is not supported by the cudaq backend: "
                    "the cudaq kernel builder API does not provide a generic "
                    "multi-controlled gate primitive."
                )
            case "PhaseShift":
                kernel.r1(
                    _resolve_for_kernel(gate.params[0], weight_arg, input_arg),
                    qubits[gate.qubits[0]],
                )
            case "SWAP":
                kernel.swap(qubits[gate.qubits[0]], qubits[gate.qubits[1]])
            case "CRY":
                kernel.cry(
                    _resolve_for_kernel(gate.params[0], weight_arg, input_arg),
                    qubits[gate.qubits[0]],
                    qubits[gate.qubits[1]],
                )
            case "AmplitudeEncoding" | "QubitUnitary":
                raise NotImplementedError(
                    f"'{gate.name}' is not supported by the CUDA-Q backend: "
                    "cudaq kernels compile to hardware IR and do not support "
                    "arbitrary state-preparation or unitary-matrix gates. "
                    "Use the PennyLane or Qiskit backend instead."
                )
            case _:
                raise NotImplementedError(f"gate '{gate.name}' not implemented for cudaq")

    def run(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return one expectation value per observable.

        It follows the `CompiledCircuit` contract: the result has shape `(n_obs,)`
        when `weights` and `inputs` are both 1-D, and `(B, n_obs)` otherwise.

        Args:
            weights (np.ndarray): The parameter vector: 1-D for one sample, or 2-D
                with a leading batch axis.
            inputs (np.ndarray): The input features, with the same 1-D or 2-D
                shape convention as `weights`.

        Returns:
            The expectation values, one for each observable.
        """
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        self._select_target()
        if weights.ndim == 1 and inputs.ndim == 2 and self._has_inputs:
            return self._run_broadcast(weights, inputs)
        return batch_call(self._run_one, weights, inputs)

    def _select_target(self) -> None:
        """Point CUDA-Q at this circuit's target before running it.

        `cudaq.set_target` is process-global, so setting it only at compile
        time would let the last circuit built win for every circuit: one
        handle built against `qpp-cpu` and then another against
        `density-matrix-cpu` would leave the first silently executing on the
        second's target -- which for an exact-versus-noisy pair means different
        numbers from the handle the user still holds.

        Selecting per run removes that build-order dependence but does not make
        the backend thread-safe: this is a read-then-write on process-global
        state, so two threads running handles with different targets still
        race, and one can execute under the other's target. Use one target per
        process when running concurrently.
        """
        import cudaq

        if cudaq.get_target().name != self._config.target:
            cudaq.set_target(self._config.target)

    def _run_one(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        import cudaq

        args = self._build_args(weights, inputs)
        results = np.zeros(len(self._hamiltonians), dtype=np.float64)
        for k, hamiltonian in enumerate(self._hamiltonians):
            if self._config.shots is None:
                result = cudaq.observe(self._kernel, hamiltonian, *args)
            else:
                result = cudaq.observe(
                    self._kernel, hamiltonian, *args, shots_count=self._config.shots
                )
            results[k] = float(result.expectation())
        return results

    def _run_broadcast(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Native batched evaluation via `cudaq.observe` argument broadcasting.

        cudaq detects a broadcast when *any* positional kernel argument
        is a 2-D `numpy.ndarray` whose leading axis is the batch (see
        `cudaq.runtime.utils.__isBroadcast`). Each ndarray row is then
        marshalled into one kernel call. We must pass weights as a 2-D
        array too (broadcast-replicated), since for a kernel whose first
        argument is `list[float]` cudaq only triggers the broadcast path
        on a 2-D ndarray. cudaq returns a flat list of `ObserveResult`
        instances of length B; we collect per-observable expectations.

        Note: with kernels built via `make_kernel(...)` (as here), cudaq's
        broadcast helper is a Python-level loop, so the gain over our
        own loop is marginal — the real native-vectorization win is only
        available for kernels declared with `@cudaq.kernel` plus 2-D
        ndarray args (which lets cudaq cache the JIT'd kernel across the
        batch). Keeping this path because it exercises cudaq's batch API
        consistently and benefits from any future internal improvements.
        """
        import cudaq

        batch_size = inputs.shape[0]
        batched_args = self._build_batched_args(weights, inputs, batch_size)
        output = np.zeros((batch_size, len(self._hamiltonians)), dtype=np.float64)
        for k, hamiltonian in enumerate(self._hamiltonians):
            if self._config.shots is None:
                results = cudaq.observe(self._kernel, hamiltonian, *batched_args)
            else:
                results = cudaq.observe(
                    self._kernel,
                    hamiltonian,
                    *batched_args,
                    shots_count=self._config.shots,
                )
            for b in range(batch_size):
                output[b, k] = float(results[b].expectation())
        return output

    def _build_args(self, weights: np.ndarray, inputs: np.ndarray) -> tuple:
        if self._has_weights and self._has_inputs:
            return ([float(w) for w in weights], [float(x) for x in inputs])
        if self._has_weights:
            return ([float(w) for w in weights],)
        if self._has_inputs:
            return ([float(x) for x in inputs],)
        return ()

    def _build_batched_args(
        self, weights: np.ndarray, inputs: np.ndarray, batch_size: int
    ) -> tuple:
        """Pack args as 2-D `numpy.ndarray`s so cudaq triggers its
        broadcast path. Weights are shared across the batch, so we
        replicate them along the leading axis; inputs vary per sample.

        Why ndarray and not list-of-lists: when the kernel's first arg
        type is a flat `list[float]` (as ours is, from
        `make_kernel(list, ...)`), cudaq's `__isBroadcast` returns False
        for list-of-lists and treats them as a single value. Only a 2-D
        ndarray reliably triggers broadcasting.
        """
        inputs_2d = np.ascontiguousarray(inputs, dtype=np.float64)
        if self._has_weights and self._has_inputs:
            weights_2d = np.ascontiguousarray(
                np.broadcast_to(
                    np.asarray(weights, dtype=np.float64).reshape(1, -1),
                    (batch_size, len(weights)),
                )
            )
            return (weights_2d, inputs_2d)
        if self._has_inputs:
            return (inputs_2d,)
        raise RuntimeError(
            "broadcast path requires inputs; weight-only kernels can't vary per sample"
        )

    def zero_state_probability(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
    ) -> float:
        """Return the probability of `|0…0⟩`, read from the statevector of `cudaq`.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features of one sample (1-D).

        Returns:
            The probability, a float in `[0, 1]`.
        """
        import cudaq

        # `run()` is not the only entry point that executes a kernel: this one
        # goes through `cudaq.get_state`, and it needs the same target
        # selection, or the fidelity kernels and
        # `Circuit.zero_state_probability` / `zero_state_probabilities` would
        # run a handle built against `qpp-cpu` on `density-matrix-cpu` because
        # a second circuit was built after it.
        self._select_target()
        if self._has_weights and self._has_inputs:
            args = ([float(w) for w in weights], [float(x) for x in inputs])
        elif self._has_weights:
            args = ([float(w) for w in weights],)
        elif self._has_inputs:
            args = ([float(x) for x in inputs],)
        else:
            args = ()
        state = cudaq.get_state(self._kernel, *args)
        amplitude = complex(state[0])
        return float(abs(amplitude) ** 2)


class CudaQuantumBackend:
    """The provider for NVIDIA CUDA Quantum, under the name `"cudaq"`.

    It is not registered automatically. Register it with
    `register("cudaq", CudaQuantumBackend())` to select it with
    `provider="cudaq"`, or pass a
    [`CudaQuantumConfig`][eigenkit.core.backends.cudaquantum_backend.CudaQuantumConfig]
    as `provider=`.
    """

    gradient_method = "parameter_shift"

    def __init__(
        self,
        config: CudaQuantumConfig | None = None,
        name: str = "cudaq",
    ):
        """Build the backend from a device configuration.

        Args:
            config (CudaQuantumConfig): The device settings. Default: `None`, which
                uses `CudaQuantumConfig()`.
            name (str): The registry name, used in error messages. Default:
                `"cudaq"`.
        """
        self.name = name
        self.config = config or CudaQuantumConfig()

    @property
    def capabilities(self) -> Capabilities:
        """What this backend supports.

        It runs every gate except the state preparation gates, the arbitrary
        unitaries, the generic multi-controlled gates and `CPHASE`. A CUDA Quantum
        kernel compiles to a hardware intermediate representation, so it cannot
        express the first three, and this backend does not translate `CPHASE`.
        It can sample a finite number of shots, evaluates a batch in one call
        through `cudaq.observe`, and has no limit on the number of qubits. See
        [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        # DATA_GATES (AmplitudeEncoding, QubitUnitary) and MCX/MCZ are not
        # supported by the CUDA-Q backend: cudaq kernels compile to hardware
        # IR and cannot represent arbitrary state-preparation, unitary-matrix,
        # or generic multi-controlled gates.
        return Capabilities(
            supported_gates=ALL_GATES - DATA_GATES - frozenset({"MCX", "MCZ", "CPHASE"}),
            # Native broadcasting through `cudaq.observe` when weights
            # are 1-D and inputs are 2-D; other shapes fall back to loop.
            supports_batching=True,
            supports_shots=True,
            max_qubits=None,
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> CudaQuantumCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        A `Circuit` calls it once and reuses the result for every call, so the
        expensive work belongs here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.

        Returns:
            The prepared circuit, as a `CudaQuantumCompiled` object.
        """
        return CudaQuantumCompiled(circuit, observables, self.config)
