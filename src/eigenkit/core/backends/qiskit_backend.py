"""The Qiskit backend.

Qiskit is an optional dependency (`pip install eigenkit[qiskit]`). It is imported
only when it is needed, so the rest of the library works without it, and
`compile()` raises a clear `ImportError` if it is missing.

The backend has `gradient_method = "parameter_shift"`. Qiskit has no automatic
differentiation for statevector expectation values, so the dispatch layer computes
the gradients with the parameter-shift rule.

`compile()` builds a single parametrised `QuantumCircuit` whose gates use
`qiskit.circuit.Parameter` symbols. Each `run()` calls `assign_parameters` on this
circuit, so the circuit is not rebuilt at each call. The `SparsePauliOp` operators
are also built only once.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # by its registered name
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="qiskit")

    # or with the configuration, when a device setting matters
    from eigenkit.configs import QiskitConfig

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=QiskitConfig(),
    )
    ```

    The name `"qiskit"` is registered when the `qiskit` extra is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    batch_call,
    validate_initial_state,
)
from ._state_prep import state_prep_unitary


def _require_qiskit() -> None:
    """Probe for qiskit. Raise a clear error if it is not installed.

    Call this once before issuing local `from qiskit import ...` statements
    in code paths that need it — the guard centralises the install hint
    so each call site stays a plain Python import.
    """
    try:
        import qiskit  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "qiskit is not installed. Install it via `pip install eigenkit[qiskit]`."
        ) from e


@dataclass(frozen=True)
class QiskitConfig:
    """Device configuration for the Qiskit backend.

    Pass it to `Circuit(provider=QiskitConfig(...))`, or register it under a name
    with `register()` to use it again.

    Args:
        shots: `None` computes exact expectation values with `Statevector`.
            Finite-shot sampling is not supported on this backend yet, and any
            other value makes the compilation raise `NotImplementedError`. For
            sampling, use the Qiskit Runtime backend, which can also run locally on
            a fake device. Default: `None`.
    """

    shots: int | None = None

    def make_backend(self) -> "QiskitBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `QiskitBackend` for this configuration.
        """
        return QiskitBackend(config=self)


def _observable_to_qiskit(observable: Observable, n_qubits: int):
    """Qiskit Pauli strings are little-endian: rightmost char is qubit 0."""
    from qiskit.quantum_info import SparsePauliOp

    rows = []
    for term in observable.terms:
        chars = ["I"] * n_qubits
        for op, qubit in term.paulis:
            chars[qubit] = op
        rows.append(("".join(reversed(chars)), term.coeff))
    return SparsePauliOp.from_list(rows)


def _resolve_symbolic(
    parameter: Parameter,
    weight_symbols: list,
    input_symbols: list,
):
    """Translate a Parameter into a qiskit `ParameterExpression`.

    Qiskit's Parameter supports arithmetic (`0.5 * w0`, `w0 + x0`,
    `-w0`), so Scale, Negate, Sum, and Multiply map cleanly onto it.
    """
    if isinstance(parameter, Constant):
        return parameter.value
    if isinstance(parameter, WeightRef):
        return weight_symbols[parameter.index]
    if isinstance(parameter, InputRef):
        return input_symbols[parameter.index]
    if isinstance(parameter, Scale):
        inner = _resolve_symbolic(parameter.expression, weight_symbols, input_symbols)
        return parameter.factor * inner
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
        # Qiskit's ParameterExpression supports symbol * symbol natively.
        return _resolve_symbolic(
            parameter.a, weight_symbols, input_symbols
        ) * _resolve_symbolic(parameter.b, weight_symbols, input_symbols)
    raise TypeError(parameter)


def _bit_reversal_perm(n: int) -> list[int]:
    """Index permutation mapping this library's qubit-0-MSB ordering to
    Qiskit's qubit-0-LSB ordering for ``n`` qubits. Applied to data-gate
    payloads (AmplitudeEncoding amplitudes, QubitUnitary matrices) so the same
    logical state/operator lands on the same physical basis states in both."""
    return [int(format(k, f"0{n}b")[::-1], 2) for k in range(2**n)]


def _apply_gate_symbolic(
    qiskit_circuit,
    gate: Gate,
    weight_symbols: list,
    input_symbols: list,
) -> None:
    values = [_resolve_symbolic(p, weight_symbols, input_symbols) for p in gate.params]
    match gate.name:
        case "RX":
            qiskit_circuit.rx(values[0], gate.qubits[0])
        case "RY":
            qiskit_circuit.ry(values[0], gate.qubits[0])
        case "RZ":
            qiskit_circuit.rz(values[0], gate.qubits[0])
        case "H":
            qiskit_circuit.h(gate.qubits[0])
        case "X":
            qiskit_circuit.x(gate.qubits[0])
        case "S":
            qiskit_circuit.s(gate.qubits[0])
        case "Sdg":
            qiskit_circuit.sdg(gate.qubits[0])
        case "CNOT":
            qiskit_circuit.cx(gate.qubits[0], gate.qubits[1])
        case "Z":
            qiskit_circuit.z(gate.qubits[0])
        case "MCZ":
            controls = list(gate.qubits[:-1])
            target = gate.qubits[-1]
            qiskit_circuit.h(target)
            qiskit_circuit.mcx(controls, target)
            qiskit_circuit.h(target)
        case "MCX":
            controls = list(gate.qubits[:-1])
            target = gate.qubits[-1]
            qiskit_circuit.mcx(controls, target)
        case "PhaseShift":
            qiskit_circuit.p(values[0], gate.qubits[0])
        case "SWAP":
            qiskit_circuit.swap(gate.qubits[0], gate.qubits[1])
        case "CRY":
            qiskit_circuit.cry(values[0], gate.qubits[0], gate.qubits[1])
        case "CPHASE":
            qiskit_circuit.cp(values[0], gate.qubits[0], gate.qubits[1])
        case "AmplitudeEncoding":
            if gate.adjoint:
                raise NotImplementedError(
                    "adjoint(AmplitudeEncoding) is not supported by the Qiskit "
                    "backend. Use the PennyLane backend for circuits that contain "
                    "Grover iterates built from AmplitudeEncoding oracles, or "
                    "replace AmplitudeEncoding with QubitUnitary inside the oracle "
                    "(QubitUnitary adjoint is computed in the IR and works on all "
                    "backends)."
                )
            # The shared completion, applied as an ordinary unitary rather than
            # through `prepare_state`: the SDK's own synthesis picks different
            # columns beyond the first, which only agrees with the other
            # backends when the target register is |0...0>.
            mat = state_prep_unitary(_data_to_array(gate.data[0]))
            n = len(gate.qubits)
            if n > 1:
                # unitary() is LSB-first; permute rows and columns to reconcile
                # with this library's MSB-first matrix indexing.
                perm = _bit_reversal_perm(n)
                mat = mat[np.ix_(perm, perm)]
            qiskit_circuit.unitary(mat, list(gate.qubits))
        case "QubitUnitary":
            mat = _data_to_array(gate.data[0])
            n = len(gate.qubits)
            if n > 1:
                # unitary() is LSB-first; permute rows and columns to reconcile
                # with this library's MSB-first matrix indexing.
                perm = _bit_reversal_perm(n)
                mat = mat[np.ix_(perm, perm)]
            qiskit_circuit.unitary(mat, list(gate.qubits))
        case "ControlledQubitUnitary":
            import math

            from qiskit.circuit.library import UnitaryGate

            mat = _data_to_array(gate.data[0])
            n_target = int(math.log2(mat.shape[0]))
            n_ctrl = len(gate.qubits) - n_target
            ctrl_wires = list(gate.qubits[:n_ctrl])
            tgt_wires = list(gate.qubits[n_ctrl:])
            if n_target > 1:
                perm = _bit_reversal_perm(n_target)
                mat = mat[np.ix_(perm, perm)]
            # Qiskit ctrl_state is LSB-first (rightmost char = ctrl_wires[0]);
            # IR control_values activate ctrl_wires[i] in order, so reverse them.
            cv = gate.data[1]
            ctrl_state = "".join(str(v) for v in reversed(cv)) if cv is not None else None
            ctrl_gate = UnitaryGate(mat).control(n_ctrl, ctrl_state=ctrl_state)
            qiskit_circuit.append(ctrl_gate, ctrl_wires + tgt_wires)
        case _:
            raise NotImplementedError(f"gate '{gate.name}' not implemented")


class QiskitCompiled(CompiledCircuit):
    """A circuit prepared for Qiskit.

    This class is internal. It is built by
    [`QiskitBackend.compile()`][eigenkit.core.backends.qiskit_backend.QiskitBackend.compile]
    and obtained from it, and it is never constructed directly.
    """

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: QiskitConfig,
    ):
        """Prepare the circuit for repeated evaluation.

        `QiskitBackend.compile()` builds it, and there is no reason to construct one
        directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that this handle
                measures.
            config (QiskitConfig): The device settings.
        """
        if config.shots is not None:
            raise NotImplementedError(
                "finite-shot Qiskit execution is not wired yet; set shots=None"
            )
        _require_qiskit()
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter as QiskitParameter
        from qiskit.quantum_info import Statevector

        self._Statevector = Statevector

        self._config = config
        self._n_qubits = circuit.n_qubits
        self._weight_symbols = [
            QiskitParameter(f"w{i}") for i in range(circuit.n_weights)
        ]
        self._input_symbols = [QiskitParameter(f"x{i}") for i in range(circuit.n_inputs)]
        qc = QuantumCircuit(circuit.n_qubits)
        for gate in circuit.gates:
            _apply_gate_symbolic(qc, gate, self._weight_symbols, self._input_symbols)
        self._parametric_circuit = qc
        # The circuit `_binding` measures and the circuit it binds are the same
        # one here, so this is only about not rebuilding the set on every call.
        self._present_parameters = set(qc.parameters)
        self._operators = [
            _observable_to_qiskit(obs, circuit.n_qubits) for obs in observables
        ]

    def _binding(self, weights: np.ndarray, inputs: np.ndarray) -> dict:
        """Map this circuit's symbols to values.

        A symbol is created for every declared weight and input, but only the
        ones a gate actually references reach the `QuantumCircuit`, and
        `assign_parameters` is strict about extras. Binding the full list
        would raise `CircuitError: Cannot bind parameters (w1) not present in
        the circuit` for a spec declaring a weight it does not use — which
        every other backend runs without complaint.
        """
        present = self._present_parameters
        binding: dict = {}
        for i, symbol in enumerate(self._weight_symbols):
            if symbol in present:
                binding[symbol] = float(weights[i])
        for i, symbol in enumerate(self._input_symbols):
            if symbol in present:
                binding[symbol] = float(inputs[i])
        return binding

    def _bound_statevector(self, weights: np.ndarray, inputs: np.ndarray):
        bound = self._parametric_circuit.assign_parameters(self._binding(weights, inputs))
        return self._Statevector.from_instruction(bound)

    def _swap_endianness(self, data: np.ndarray) -> np.ndarray:
        """Convert between Qiskit's qubit-0-LSB amplitude order and the
        library's qubit-0-MSB flat order (the convention numpy / PennyLane use).
        Reversing the qubit axes is its own inverse, so this maps both ways."""
        n = self._n_qubits
        if n < 2:
            return data
        return data.reshape((2,) * n).transpose(tuple(range(n - 1, -1, -1))).reshape(-1)

    def statevector(
        self, weights: np.ndarray, inputs: np.ndarray, initial_state=None
    ) -> np.ndarray:
        r"""Return the full complex statevector after the circuit.

        In the result, qubit 0 is the most significant bit. The amplitudes are
        reordered from the Qiskit convention, where qubit 0 is the least significant
        bit, so that they match the NumPy and PennyLane backends. `initial_state`,
        in the same order as the result, is evolved through the bound circuit, so
        the evolution starts from it instead of `|0…0⟩`. When `inputs` is 2-D, the
        method loops over its rows and returns an array of shape
        `(B, 2**n_qubits)`.

        Args:
            weights (np.ndarray): The parameter vector, which must be 1-D.
            inputs (np.ndarray): The input features: a 1-D array for one sample, or
                a 2-D array with a leading batch axis.
            initial_state (np.ndarray | None): The state to evolve from instead of
                `|0…0⟩`. Default: `None`.

        Returns:
            The statevector, of shape `(2**n_qubits,)`, or `(B, 2**n_qubits)` for
                a batch.
        """
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        if weights.ndim == 1 and inputs.ndim == 2:
            batch = inputs.shape[0]
            init = (
                None
                if initial_state is None
                else validate_initial_state(initial_state, self._n_qubits, batch=batch)
            )
            return np.stack(
                [
                    self._statevector_one(weights, row, None if init is None else init[i])
                    for i, row in enumerate(inputs)
                ]
            )
        init = (
            None
            if initial_state is None
            else validate_initial_state(initial_state, self._n_qubits)
        )
        return self._statevector_one(weights, inputs, init)

    def _statevector_one(self, weights, inputs, init_flat) -> np.ndarray:
        bound = self._parametric_circuit.assign_parameters(self._binding(weights, inputs))
        if init_flat is None:
            sv = self._Statevector.from_instruction(bound)
        else:
            qiskit_init = self._swap_endianness(
                np.asarray(init_flat, dtype=np.complex128)
            )
            sv = self._Statevector(qiskit_init).evolve(bound)
        return self._swap_endianness(np.asarray(sv.data, dtype=np.complex128))

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
        return batch_call(self._run_one, weights, inputs)

    def _run_one(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        statevector = self._bound_statevector(weights, inputs)
        output = np.zeros(len(self._operators), dtype=np.float64)
        for k, operator in enumerate(self._operators):
            output[k] = float(statevector.expectation_value(operator).real)
        return output

    def zero_state_probability(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
    ) -> float:
        """Return the probability of the all-zeros state for one sample.

        It is the squared magnitude of the all-zeros amplitude of the simulated
        statevector.

        Args:
            weights (np.ndarray): The parameter vector, which must be 1-D.
            inputs (np.ndarray): The input features of one sample, which must be
                1-D.

        Returns:
            The probability, a float in `[0, 1]`.
        """
        statevector = self._bound_statevector(weights, inputs)
        return float(np.abs(statevector.data[0]) ** 2)


class QiskitBackend:
    """The provider for Qiskit, registered as `"qiskit"`.

    This class is internal. Select the backend with `provider="qiskit"` or by
    passing a [`QiskitConfig`][eigenkit.core.backends.qiskit_backend.QiskitConfig],
    instead of constructing this class.
    """

    gradient_method = "parameter_shift"

    def __init__(
        self,
        config: QiskitConfig | None = None,
        name: str = "qiskit",
    ):
        """Build the backend from a device configuration.

        Args:
            config (QiskitConfig | None): The device settings. `None` uses
                `QiskitConfig()`.
            name (str): The registry name, used in error messages.
        """
        self.name = name
        self.config = config or QiskitConfig()

    @property
    def capabilities(self) -> Capabilities:
        """Return what this backend supports.

        It supports every gate, but only exact expectation values, because
        finite-shot sampling is not supported yet. It does not batch, and it has no
        limit on the number of qubits. The adjoint of `AmplitudeEncoding` is
        excluded, because the Qiskit gate translation does not implement it. See
        [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        return Capabilities(
            supported_gates=ALL_GATES,
            supports_batching=False,
            # Finite-shot sampling is not wired yet.
            supports_shots=False,
            max_qubits=None,
            # AmplitudeEncoding adjoint uses prepare_state() which is not invertible here.
            supported_adjoint_gates=ALL_GATES - frozenset({"AmplitudeEncoding"}),
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> QiskitCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        It is called once per `Circuit`, and the handle is reused for every call, so
        the expensive work is done here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that the handle
                measures.

        Returns:
            A `QiskitCompiled` handle.
        """
        return QiskitCompiled(circuit, observables, self.config)
