"""The Qiskit Aer backend, with the statevector and matrix product state (MPS) methods.

Aer comes with the `qiskit` extra (`pip install eigenkit[qiskit]`), so this backend
needs no new dependency. It reuses, unchanged, the translation of the plain Qiskit
backend: the gate translation, the binding of the symbolic `Parameter` values and
the `SparsePauliOp` observables. Only the execution is different. Instead of
`Statevector.from_instruction`, it runs the bound circuit on an `AerSimulator` and
reads the expectation values with the Aer `save_expectation_value` instruction. As
a result, the **same** code measures with both the exact statevector simulator and
the MPS simulator, which `method` selects.

`method="statevector"` is the dense Aer simulator, which is exact.
`method="matrix_product_state"` is the Aer MPS simulator. It is exact when the bond
dimension is not bounded, which is the Aer default. Otherwise, it is a fast
approximation for circuits with little entanglement. The provider names are
`qiskit-aer` for the statevector method and `qiskit-aer-mps` for the MPS method.

Both methods get the circuit translated first to gates that they can run (see
`_to_native_basis`). The MPS method rejects `cry` with "contains invalid
instructions" instead of decomposing it, and the statevector method cannot run the
controlled unitary that `ControlledQubitUnitary` becomes.

The backend has `gradient_method = "parameter_shift"`. Aer has no cheap analytic
reverse-mode Jacobian for these expectation values, so the dispatch layer computes
the gradients with the parameter-shift rule, as for plain Qiskit. Expectation
values do not depend on the qubit ordering convention, as long as the circuit and
the operator use the same one, and here both use the Qiskit convention. So `run`
needs no reordering. Only the statevectors that are read out are reordered to the
convention of the library, where qubit 0 is the most significant bit.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # by its registered name
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="qiskit-aer")

    # or with the configuration, when a device setting matters
    from eigenkit.configs import QiskitAerConfig

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=QiskitAerConfig(method="matrix_product_state"),
    )
    ```

    The name `"qiskit-aer-mps"` selects the same backend with the MPS method already
    set. The backend comes with the `qiskit` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..ir import ALL_GATES, CircuitSpec, Observable
from ..provider import (
    Capabilities,
    CompiledCircuit,
    batch_call,
    validate_initial_state,
)
from .qiskit_backend import (
    _apply_gate_symbolic,
    _observable_to_qiskit,
    _require_qiskit,
)


_AER_METHODS = ("statevector", "matrix_product_state")


def _require_qiskit_aer() -> None:
    """Probe for qiskit-aer. Raise a clear install hint if it is missing."""
    _require_qiskit()
    try:
        import qiskit_aer  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "qiskit-aer is not installed. Install it via `pip install eigenkit[qiskit]`."
        ) from e


@dataclass(frozen=True)
class QiskitAerConfig:
    """Device configuration for the Qiskit Aer backend.

    Args:
        method: The Aer simulation method. `"statevector"` selects the exact dense
            simulator. `"matrix_product_state"` selects the matrix product state
            (MPS) simulator, which is exact when the bond dimension is not bounded,
            and an approximation for circuits with little entanglement otherwise.
            The provider name `"qiskit-aer-mps"` is this configuration with the MPS
            method already set. Any other value raises `ValueError`. Default:
            `"statevector"`.
        shots: `None` returns exact expectation values. Finite-shot sampling is
            not supported yet, and any other value makes the compilation raise
            `NotImplementedError`. Default: `None`.
    """

    method: str = "statevector"
    shots: int | None = None

    def __post_init__(self) -> None:
        if self.method not in _AER_METHODS:
            raise ValueError(
                f"unknown Aer method {self.method!r}; choose from {_AER_METHODS}"
            )

    def make_backend(self) -> "QiskitAerBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `QiskitAerBackend` for this configuration.
        """
        return QiskitAerBackend(config=self)


class QiskitAerCompiled(CompiledCircuit):
    """A circuit prepared for Qiskit Aer.

    This class is internal. It is built by
    [`QiskitAerBackend.compile()`][eigenkit.core.backends.qiskit_aer_backend.QiskitAerBackend.compile]
    and obtained from it, and it is never constructed directly.
    """

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: QiskitAerConfig,
    ):
        """Prepare the circuit for repeated evaluation.

        `QiskitAerBackend.compile()` builds it, and there is no reason to construct
        one directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that this handle
                measures.
            config (QiskitAerConfig): The device settings.
        """
        if config.shots is not None:
            raise NotImplementedError(
                "finite-shot Aer execution is not wired yet; set shots=None"
            )
        _require_qiskit_aer()
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter as QiskitParameter
        from qiskit_aer import AerSimulator

        self._config = config
        self._n_qubits = circuit.n_qubits
        self._sim = AerSimulator(method=config.method)
        self._weight_symbols = [
            QiskitParameter(f"w{i}") for i in range(circuit.n_weights)
        ]
        self._input_symbols = [QiskitParameter(f"x{i}") for i in range(circuit.n_inputs)]

        base = QuantumCircuit(circuit.n_qubits)
        for gate in circuit.gates:
            _apply_gate_symbolic(base, gate, self._weight_symbols, self._input_symbols)
        base = self._to_native_basis(base)
        # Every circuit built below is `base` plus a parameter-free save
        # instruction, so this is the parameter set of all of them. Recorded
        # once, here, rather than read off one of the copies at bind time:
        # reading `_sv_circuit.parameters` to bind `_expval_circuit` would be
        # correct only while the two happen to agree, and a parametric save
        # instruction on either would produce an unbound-parameter run or the
        # very `Cannot bind parameters` error `_binding` exists to avoid.
        self._present_parameters = set(base.parameters)
        self._operators = [
            _observable_to_qiskit(obs, circuit.n_qubits) for obs in observables
        ]

        # One parametric circuit carrying a save_expectation_value per observable (labels e0..e{K-1}),
        # bound per call. save_expectation_value works under both the statevector and MPS methods.
        qubits = list(range(circuit.n_qubits))
        expval = base.copy()
        for k, op in enumerate(self._operators):
            expval.save_expectation_value(op, qubits, label=f"e{k}")
        self._expval_circuit = expval

        # A second parametric circuit that saves the full statevector (small-n spot-checks / QFIM
        # finite-difference / fidelity kernels). For the MPS method Aer reconstructs the dense state.
        sv = base.copy()
        sv.save_statevector()
        self._sv_circuit = sv

    def _to_native_basis(self, base):
        """Translate gates the chosen Aer method cannot execute directly.

        Both methods need this. MPS rejects ``cry`` outright ("contains invalid
        instructions") rather than decomposing it. The statevector core cannot
        execute ``c-unitary`` — the lowering of `ControlledQubitUnitary`, which
        this backend declares support for — and would raise `AerError: unknown
        instruction` at run time, after `validate_circuit` has already accepted
        the circuit.

        There is no coupling map on a simulator target, so
        ``optimization_level=0`` performs a pure basis rewrite with no routing
        and no qubit permutation, leaving observable qubit indices and the
        endianness swap valid.
        """
        from qiskit import transpile

        return transpile(base, self._sim, optimization_level=0)

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

    def _swap_endianness(self, data: np.ndarray) -> np.ndarray:
        """Aer amplitudes are qubit-0-LSB; the library convention is qubit-0-MSB (numpy/PennyLane).
        Reversing the qubit axes is its own inverse, so this maps both ways."""
        n = self._n_qubits
        if n < 2:
            return data
        return data.reshape((2,) * n).transpose(tuple(range(n - 1, -1, -1))).reshape(-1)

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
        bound = self._expval_circuit.assign_parameters(self._binding(weights, inputs))
        data = self._sim.run(bound).result().data(0)
        out = np.zeros(len(self._operators), dtype=np.float64)
        for k in range(len(self._operators)):
            out[k] = float(np.real(data[f"e{k}"]))
        return out

    def statevector(
        self, weights: np.ndarray, inputs: np.ndarray, initial_state=None
    ) -> np.ndarray:
        """Return the full complex statevector after the circuit.

        In the result, qubit 0 is the most significant bit. When `inputs` is 2-D,
        the method loops over its rows and returns an array of shape
        `(B, 2**n_qubits)`. `initial_state`, in the same order as the result, is
        loaded with the Aer `set_statevector` instruction.

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
        if init_flat is None:
            circ = self._sv_circuit
        else:
            from qiskit import QuantumCircuit

            circ = QuantumCircuit(self._n_qubits)
            circ.set_statevector(
                self._swap_endianness(np.asarray(init_flat, dtype=np.complex128))
            )
            circ = circ.compose(self._sv_circuit)
        bound = circ.assign_parameters(self._binding(weights, inputs))
        sv = np.asarray(
            self._sim.run(bound).result().data(0)["statevector"], dtype=np.complex128
        )
        return self._swap_endianness(sv)

    def zero_state_probability(self, weights: np.ndarray, inputs: np.ndarray) -> float:
        """Return the probability of the all-zeros state for one sample.

        It is the squared magnitude of the all-zeros amplitude of the simulated
        statevector, and it is the basic operation of fidelity kernels.

        Args:
            weights (np.ndarray): The parameter vector, which must be 1-D.
            inputs (np.ndarray): The input features of one sample, which must be
                1-D.

        Returns:
            The probability, a float in `[0, 1]`.
        """
        sv = self._statevector_one(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
            None,
        )
        return float(np.abs(sv[0]) ** 2)


class QiskitAerBackend:
    """The provider for Qiskit Aer.

    It is registered as `"qiskit-aer"` for the statevector method and as
    `"qiskit-aer-mps"` for the matrix product state method.

    This class is internal. Select the backend with `provider="qiskit-aer"` or by
    passing a
    [`QiskitAerConfig`][eigenkit.core.backends.qiskit_aer_backend.QiskitAerConfig],
    instead of constructing this class.
    """

    gradient_method = "parameter_shift"

    def __init__(self, config: QiskitAerConfig | None = None, name: str | None = None):
        """Build the backend from a device configuration.

        Args:
            config (QiskitAerConfig | None): The device settings. `None` uses
                `QiskitAerConfig()`.
            name (str | None): The registry name, used in error messages. `None`
                gives `"qiskit-aer-mps"` for the matrix product state method and
                `"qiskit-aer"` otherwise.
        """
        self.config = config or QiskitAerConfig()
        # default id tracks the method so qiskit-aer vs qiskit-aer-mps are self-describing
        self.name = name or (
            "qiskit-aer-mps"
            if self.config.method == "matrix_product_state"
            else "qiskit-aer"
        )

    @property
    def capabilities(self) -> Capabilities:
        """Return what this backend supports.

        It supports every gate, but only exact expectation values. It does not
        batch, and it has no limit on the number of qubits. The adjoint of
        `AmplitudeEncoding` is excluded, because the Qiskit gate translation does
        not implement it. See [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        return Capabilities(
            supported_gates=ALL_GATES,
            supports_batching=False,
            supports_shots=False,
            max_qubits=None,
            supported_adjoint_gates=ALL_GATES - frozenset({"AmplitudeEncoding"}),
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> QiskitAerCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        It is called once per `Circuit`, and the handle is reused for every call, so
        the expensive work is done here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables that the handle
                measures.

        Returns:
            A `QiskitAerCompiled` handle.
        """
        return QiskitAerCompiled(circuit, observables, self.config)
