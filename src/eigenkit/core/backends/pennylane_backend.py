"""The PennyLane backend.

It declares `gradient_method = "native"`: the PyTorch interface of PennyLane
carries the gradients through the tape, so the parameter-shift rule is not needed
here.

`compile()` builds the device, both qnodes (one for NumPy and one for PyTorch)
and the Pauli operators once. Each `run()` only binds the parameters into the
cached qnodes. A finite number of shots is set on each qnode with
`qml.set_shots`, because passing `shots=` to `qml.device` was deprecated in
PennyLane 0.43.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # by registered name
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="pennylane")

    # or with the config, when a device setting matters
    from eigenkit.configs import PennyLaneConfig

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=PennyLaneConfig(device_name="lightning.qubit"),
    )
    ```

    This backend needs the `pennylane` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np


try:
    import pennylane as qml
except ImportError:  # the config stays importable; the backend raises on use
    qml = None


if TYPE_CHECKING:  # torch is an optional extra — type-only import, never at runtime
    import torch

from ..ir import (
    ALL_GATES,
    CircuitSpec,
    Gate,
    Observable,
    _data_to_array,
    chain_factor,
    chain_factor_input,
    collect_input_dependencies,
    collect_weight_dependencies,
    resolve,
)
from ..provider import (
    Capabilities,
    CompiledCircuit,
    batch_call,
    validate_initial_state,
)
from ._state_prep import state_prep_unitary


@dataclass(frozen=True)
class PennyLaneConfig:
    """Device settings, used for every `qml.device()` call.

    Pass it directly to `Circuit(provider=PennyLaneConfig(...))`, or pass it to
    `register()` under a name to reuse it.

    Args:
        device_name: The identifier of the PennyLane simulator plugin:
            `"default.qubit"`, `"lightning.qubit"`, `"lightning.gpu"`, or any other
            installed device. Default: `"default.qubit"`.
        shots: `None` computes exact expectation values, and a positive integer
            samples that many shots for each expectation value. Default: `None`.
        diff_method: The PennyLane differentiation method. With `"best"`,
            PennyLane chooses for each device, usually backpropagation on a
            simulator and the parameter-shift rule on hardware. Default: `"best"`.
    """

    device_name: str = "default.qubit"
    shots: int | None = None
    # PennyLane's own differentiation method for the numpy gradient path
    # (`jacobian`): "best" lets PennyLane pick per device (adjoint/backprop on
    # default.qubit & lightning, parameter-shift with shots). Override with
    # "adjoint", "backprop", or "parameter-shift". Torch training is unaffected
    # — it flows autograd through the tape via `run_torch`.
    diff_method: str = "best"

    def make_backend(self) -> "PennyLaneBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `PennyLaneBackend` for this configuration.
        """
        return PennyLaneBackend(config=self)


def _observable_to_pennylane(observable: Observable):
    pauli_map = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    accumulator = None
    for term in observable.terms:
        product = None
        for op, qubit in term.paulis:
            factor = pauli_map[op](qubit)
            product = factor if product is None else product @ factor
        weighted = term.coeff * product
        accumulator = weighted if accumulator is None else accumulator + weighted
    return accumulator


def _apply_controlled_qubit_unitary(gate: Gate) -> None:
    """Apply a ControlledQubitUnitary via ``qml.ctrl(qml.QubitUnitary)``.

    ``gate.data = (matrix, control_values)`` and ``gate.qubits`` lists the
    control wires followed by the target wires; the target count is inferred
    from the matrix dimension. Data gates carry no differentiable angle, so the
    forward and gate-angle-jacobian paths apply them identically.
    """
    import math

    mat = _data_to_array(gate.data[0])
    n_target = int(math.log2(mat.shape[0]))
    n_ctrl = len(gate.qubits) - n_target
    ctrl_wires = list(gate.qubits[:n_ctrl])
    tgt_wires = list(gate.qubits[n_ctrl:])
    control_values = list(gate.data[1]) if gate.data[1] is not None else None
    qml.ctrl(qml.QubitUnitary, control=ctrl_wires, control_values=control_values)(
        mat, wires=tgt_wires
    )


def _apply_gate(gate: Gate, weights, inputs) -> None:
    values = [resolve(p, weights, inputs) for p in gate.params]
    match gate.name:
        case "RX":
            qml.RX(values[0], wires=gate.qubits[0])
        case "RY":
            qml.RY(values[0], wires=gate.qubits[0])
        case "RZ":
            qml.RZ(values[0], wires=gate.qubits[0])
        case "H":
            qml.Hadamard(wires=gate.qubits[0])
        case "X":
            qml.PauliX(wires=gate.qubits[0])
        case "S":
            qml.S(wires=gate.qubits[0])
        case "Sdg":
            qml.adjoint(qml.S)(wires=gate.qubits[0])
        case "CNOT":
            qml.CNOT(wires=list(gate.qubits))
        case "Z":
            qml.PauliZ(wires=gate.qubits[0])
        case "MCZ":
            qml.ctrl(qml.PauliZ, control=list(gate.qubits[:-1]))(wires=gate.qubits[-1])
        case "MCX":
            qml.MultiControlledX(wires=list(gate.qubits))
        case "PhaseShift":
            qml.PhaseShift(values[0], wires=gate.qubits[0])
        case "SWAP":
            qml.SWAP(wires=list(gate.qubits))
        case "CRY":
            qml.CRY(values[0], wires=list(gate.qubits))
        case "CPHASE":
            qml.ControlledPhaseShift(values[0], wires=list(gate.qubits))
        case "AmplitudeEncoding":
            # The shared completion as a plain unitary, not AmplitudeEmbedding:
            # PennyLane's decomposition fixes only the first column, so it
            # would agree with the other backends only from |0...0>.
            matrix = state_prep_unitary(_data_to_array(gate.data[0]))
            if gate.adjoint:
                matrix = matrix.conj().T
            qml.QubitUnitary(matrix, wires=list(gate.qubits))
        case "QubitUnitary":
            qml.QubitUnitary(_data_to_array(gate.data[0]), wires=list(gate.qubits))
        case "ControlledQubitUnitary":
            _apply_controlled_qubit_unitary(gate)
        case _:
            raise NotImplementedError(f"gate '{gate.name}' not implemented")


def _apply_gate_with_angle(gate: Gate, angle) -> None:
    """Apply a gate using an explicit `angle` (for parametric gates) instead
    of resolving from weights/inputs. Used by the gate-angle jacobian qnodes,
    which differentiate directly with respect to the gate angles."""
    match gate.name:
        case "RX":
            qml.RX(angle, wires=gate.qubits[0])
        case "RY":
            qml.RY(angle, wires=gate.qubits[0])
        case "RZ":
            qml.RZ(angle, wires=gate.qubits[0])
        case "H":
            qml.Hadamard(wires=gate.qubits[0])
        case "X":
            qml.PauliX(wires=gate.qubits[0])
        case "S":
            qml.S(wires=gate.qubits[0])
        case "Sdg":
            qml.adjoint(qml.S)(wires=gate.qubits[0])
        case "CNOT":
            qml.CNOT(wires=list(gate.qubits))
        case "Z":
            qml.PauliZ(wires=gate.qubits[0])
        case "MCZ":
            qml.ctrl(qml.PauliZ, control=list(gate.qubits[:-1]))(wires=gate.qubits[-1])
        case "MCX":
            qml.MultiControlledX(wires=list(gate.qubits))
        case "PhaseShift":
            qml.PhaseShift(angle, wires=gate.qubits[0])
        case "SWAP":
            qml.SWAP(wires=list(gate.qubits))
        case "CRY":
            qml.CRY(angle, wires=list(gate.qubits))
        case "CPHASE":
            qml.ControlledPhaseShift(angle, wires=list(gate.qubits))
        case "AmplitudeEncoding":
            # Data gates carry no differentiable angle; apply them unchanged so
            # the gate-angle jacobian qnode can rebuild circuits that contain
            # them alongside trainable rotations.
            # The shared completion as a plain unitary, not AmplitudeEmbedding:
            # PennyLane's decomposition fixes only the first column, so it
            # would agree with the other backends only from |0...0>.
            matrix = state_prep_unitary(_data_to_array(gate.data[0]))
            if gate.adjoint:
                matrix = matrix.conj().T
            qml.QubitUnitary(matrix, wires=list(gate.qubits))
        case "QubitUnitary":
            qml.QubitUnitary(_data_to_array(gate.data[0]), wires=list(gate.qubits))
        case "ControlledQubitUnitary":
            _apply_controlled_qubit_unitary(gate)
        case _:
            raise NotImplementedError(f"gate '{gate.name}' not implemented")


class PennyLaneCompiled(CompiledCircuit):
    """A circuit prepared for PennyLane.

    It is internal:
    [`PennyLaneBackend.compile()`][eigenkit.core.backends.pennylane_backend.PennyLaneBackend.compile]
    builds it and returns it, and it is never built directly. It has a
    `run_torch()` method, so the PyTorch interface differentiates through
    PennyLane itself instead of falling back to the parameter-shift rule.
    """

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: PennyLaneConfig,
    ):
        """Prepare the circuit for repeated evaluation.

        `PennyLaneBackend.compile()` builds this object, so there is no reason to
        build one directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.
            config (PennyLaneConfig): The device settings.
        """
        self._circuit = circuit
        self._shots = config.shots
        self._diff_method = config.diff_method
        self._device = qml.device(
            config.device_name,
            wires=circuit.n_qubits,
        )
        self._pauli_observables = [_observable_to_pennylane(o) for o in observables]

        # Cached qnodes — one per interface. The closure over `circuit` and
        # The qnode tape still rebuilds per call (PennyLane needs that for
        # parameter binding), but device + decorator setup runs once.
        ir = circuit
        pauli_obs = self._pauli_observables

        @qml.qnode(self._device, interface="auto")
        def qnode_numpy(weights, inputs):
            for gate in ir.gates:
                _apply_gate(gate, weights, inputs)
            return [qml.expval(o) for o in pauli_obs]

        @qml.qnode(self._device, interface="torch")
        def qnode_torch(weights, inputs):
            for gate in ir.gates:
                _apply_gate(gate, weights, inputs)
            return [qml.expval(o) for o in pauli_obs]

        if self._shots is not None:
            qnode_numpy = qml.set_shots(qnode_numpy, shots=self._shots)
            qnode_torch = qml.set_shots(qnode_torch, shots=self._shots)

        self._qnode_numpy = qnode_numpy
        self._qnode_torch = qnode_torch
        # Built lazily on first use to avoid decorator overhead for callers
        # that only need expvals.
        self._zsp_qnode = None
        self._probs_qnode = None
        self._state_qnode = None
        self._jac_qnodes = None

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
        weights_np = np.asarray(weights, dtype=np.float64)
        inputs_np = np.asarray(inputs, dtype=np.float64)
        if self._can_broadcast(weights_np, inputs_np):
            # PennyLane parameter broadcasting: feed each input column as a
            # (B,) array so every gate angle broadcasts across the batch in
            # a single qnode call. Transpose to (n_inputs, B) because
            # `resolve(InputRef(j), w, x)` reads `x[j]`, and we want that
            # to return the j-th feature across all samples.
            output = self._qnode_numpy(weights_np, inputs_np.T)
            return np.stack([np.asarray(x, dtype=np.float64) for x in output]).T
        return batch_call(self._run_one_numpy, weights_np, inputs_np)

    def _run_one_numpy(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        output = self._qnode_numpy(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
        )
        return np.asarray([float(x) for x in output], dtype=np.float64)

    def run_torch(
        self,
        weights: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # torch is imported lazily: the numpy `run` path (and module import)
        # must work on a torch-free install (`pip install ...[pennylane]`),
        # while `run_torch` is only reached through the torch interface.
        """Evaluate the circuit on PyTorch tensors, keeping the result on the tape.

        This is the path of native automatic differentiation: the gradients flow
        through the simulator itself instead of through the parameter-shift
        fallback.

        Args:
            weights (torch.Tensor): The parameter vector, 1-D or batched.
            inputs (torch.Tensor): The input features, 1-D or batched.

        Returns:
            A PyTorch tensor of expectation values, which can still be
                differentiated.
        """
        import torch

        from eigenkit.torch.interface import batch_call_torch

        if self._can_broadcast(weights, inputs):
            # Same broadcast strategy as the numpy path. The transpose is
            # a view, so autograd flows cleanly back to the input tensor.
            output = self._qnode_torch(weights, inputs.transpose(0, 1))
            stacked = torch.stack([x if x.ndim > 0 else x.reshape(()) for x in output])
            return stacked.transpose(0, 1)
        return batch_call_torch(self._run_one_torch, weights, inputs)

    def _run_one_torch(
        self,
        weights: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        import torch

        output = self._qnode_torch(weights, inputs)
        return torch.stack([x if x.ndim > 0 else x.reshape(()) for x in output])

    @staticmethod
    def _can_broadcast(weights, inputs) -> bool:
        """Single-call broadcasting needs scalar weights and a batched
        inputs row. Anything else (batched weights, single-sample inputs)
        falls back to the loop."""
        return weights.ndim == 1 and inputs.ndim == 2

    def zero_state_probability(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
    ) -> float:
        """Return the probability of measuring `|0…0⟩` after the circuit has run.

        On first use, it builds a separate qnode that returns `qml.probs(...)`,
        and the method returns entry `0` of that distribution. The qnode uses the
        same device as the expectation-value qnodes.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features of one sample (1-D).

        Returns:
            The probability, a float in `[0, 1]`.
        """
        if self._zsp_qnode is None:
            ir = self._circuit
            wires = list(range(ir.n_qubits))

            @qml.qnode(self._device, interface="auto")
            def zsp_qnode(weights, inputs):
                for gate in ir.gates:
                    _apply_gate(gate, weights, inputs)
                return qml.probs(wires=wires)

            if self._shots is not None:
                zsp_qnode = qml.set_shots(zsp_qnode, shots=self._shots)

            self._zsp_qnode = zsp_qnode

        probs = self._zsp_qnode(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
        )
        return float(probs[0])

    def probabilities(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Return the full measurement distribution over the `2**n_qubits` basis states.

        Qubit `0` is the most significant bit. The distribution comes from
        `qml.probs`, through a qnode that is built on first use and then cached.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features of one sample (1-D).

        Returns:
            The probabilities, of shape `(2**n_qubits,)`.
        """
        if self._probs_qnode is None:
            ir = self._circuit
            wires = list(range(ir.n_qubits))

            @qml.qnode(self._device, interface="auto")
            def probs_qnode(weights, inputs):
                for gate in ir.gates:
                    _apply_gate(gate, weights, inputs)
                return qml.probs(wires=wires)

            if self._shots is not None:
                probs_qnode = qml.set_shots(probs_qnode, shots=self._shots)
            self._probs_qnode = probs_qnode

        probs = self._probs_qnode(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
        )
        return np.asarray(probs, dtype=np.float64)

    def statevector(
        self, weights: np.ndarray, inputs: np.ndarray, initial_state=None
    ) -> np.ndarray:
        r"""Return the full complex statevector, computed with `qml.state()`.

        Qubit `0` is the most significant bit. `initial_state` is added at the
        start of the circuit as a `qml.StatePrep`, so the evolution starts from
        that state instead of `|0…0⟩`. A 2-D `inputs` batch returns an array of
        shape `(B, 2**n_qubits)`. The batch is evaluated in a loop, because
        reading the state does not use the fast path of parameter broadcasting.
        This method needs exact mode, because `qml.state` is not defined with a
        finite number of shots.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features: 1-D for one sample, or 2-D
                with a leading batch axis.
            initial_state (np.ndarray | None): The state to start from, instead
                of `|0…0⟩`. Default: `None`.

        Returns:
            The statevector, of shape `(2**n_qubits,)`, or `(B, 2**n_qubits)` for
                a batch.

        Raises:
            NotImplementedError: If the configuration sets a finite number of
                `shots`.
        """
        if self._shots is not None:
            raise NotImplementedError(
                "statevector() requires analytic mode (shots=None); "
                "qml.state() is undefined under finite shots."
            )
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        n = self._circuit.n_qubits
        if self._state_qnode is None:
            self._state_qnode = self._build_state_qnode()
        if weights.ndim == 1 and inputs.ndim == 2:
            batch = inputs.shape[0]
            init = (
                None
                if initial_state is None
                else validate_initial_state(initial_state, n, batch=batch)
            )
            return np.stack(
                [
                    self._state_one(weights, row, None if init is None else init[i])
                    for i, row in enumerate(inputs)
                ]
            )
        init = None if initial_state is None else validate_initial_state(initial_state, n)
        return self._state_one(weights, inputs, init)

    def _build_state_qnode(self):
        ir = self._circuit
        wires = list(range(ir.n_qubits))

        @qml.qnode(self._device, interface="auto")
        def state_qnode(weights, inputs, init):
            if init is not None:
                qml.StatePrep(init, wires=wires)
            for gate in ir.gates:
                _apply_gate(gate, weights, inputs)
            return qml.state()

        return state_qnode

    def _state_one(self, weights, inputs, init) -> np.ndarray:
        state = self._state_qnode(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
            init,
        )
        return np.asarray(state, dtype=np.complex128)

    def jacobian(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
        *,
        need_input_grad: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute the gradient with PennyLane, using the configured `diff_method`.

        The gradient code of the core package looks for a `jacobian` method, and
        prefers it to its own parameter-shift rule, which works on any backend.
        Here PennyLane differentiates the gate angles with its own engine (adjoint,
        backpropagation or parameter shift, as `diff_method` selects). These
        gate-angle gradients are then mapped onto the weights and inputs with the
        shared symbolic chain-rule factors, so angles built with `Scale`,
        `Multiply`, `Sum` or `Negate` are handled. It takes a single sample (1-D)
        only, and the caller loops over a batch.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features of one sample (1-D).
            need_input_grad (bool): If True, also compute the jacobian with
                respect to the inputs. Default: `False`.

        Returns:
            weight_jacobian: The jacobian with respect to the weights.
            input_jacobian: The jacobian with respect to the inputs, or `None`
                when `need_input_grad` is `False`.
        """
        from pennylane import numpy as pnp

        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        ir = self._circuit
        n_obs = len(self._pauli_observables)
        weight_jac = np.zeros((n_obs, ir.n_weights), dtype=np.float64)
        input_jac = (
            np.zeros((n_obs, ir.n_inputs), dtype=np.float64) if need_input_grad else None
        )

        param_gate_indices = [i for i, gate in enumerate(ir.gates) if gate.params]
        if not param_gate_indices:
            return weight_jac, input_jac

        if self._jac_qnodes is None:
            self._jac_qnodes = self._build_jac_qnodes(param_gate_indices)

        angles = pnp.array(
            [
                float(resolve(ir.gates[i].params[0], weights, inputs))
                for i in param_gate_indices
            ],
            requires_grad=True,
        )
        # d<O_k>/d(angle vector) for each observable k, shape (n_param_gates,).
        gate_angle_grads = [
            np.atleast_1d(np.asarray(qml.grad(qnode)(angles), dtype=np.float64))
            for qnode in self._jac_qnodes
        ]

        for column, gate_index in enumerate(param_gate_indices):
            angle = ir.gates[gate_index].params[0]
            # d<O_k>/dtheta_g across observables, for this gate g.
            gate_grad = np.array(
                [grad[column] for grad in gate_angle_grads], dtype=np.float64
            )
            for weight_index in collect_weight_dependencies(angle):
                factor = chain_factor(angle, weight_index, weights, inputs)
                if factor != 0.0:
                    weight_jac[:, weight_index] += factor * gate_grad
            if input_jac is not None:
                for input_index in collect_input_dependencies(angle):
                    factor = chain_factor_input(angle, input_index, weights, inputs)
                    if factor != 0.0:
                        input_jac[:, input_index] += factor * gate_grad

        return weight_jac, input_jac

    def _build_jac_qnodes(self, param_gate_indices: list[int]) -> list:
        """One scalar autograd qnode per observable, parametrized by the flat
        vector of parametric-gate angles, so PennyLane differentiates them
        directly. Built once and cached."""
        ir = self._circuit
        position = {gate_index: col for col, gate_index in enumerate(param_gate_indices)}
        qnodes = []
        for observable in self._pauli_observables:

            @qml.qnode(self._device, interface="autograd", diff_method=self._diff_method)
            def qnode(angles, _observable=observable):
                for index, gate in enumerate(ir.gates):
                    angle = angles[position[index]] if gate.params else None
                    _apply_gate_with_angle(gate, angle)
                return qml.expval(_observable)

            if self._shots is not None:
                qnode = qml.set_shots(qnode, shots=self._shots)
            qnodes.append(qnode)
        return qnodes


class PennyLaneBackend:
    """The provider for PennyLane devices. It is registered as `"pennylane"`.

    It is internal: you select this backend with `provider="pennylane"` or by
    passing a
    [`PennyLaneConfig`][eigenkit.core.backends.pennylane_backend.PennyLaneConfig],
    not by building this class.
    """

    gradient_method = "native"

    def __init__(
        self,
        config: PennyLaneConfig | None = None,
        name: str = "pennylane",
    ):
        """Build the backend from a device configuration.

        Args:
            config (PennyLaneConfig): The device settings. Default: `None`, which
                uses `PennyLaneConfig()`.
            name (str): The registry name, used in error messages. Default:
                `"pennylane"`.
        """
        if qml is None:
            raise ImportError(
                "pennylane is not installed. Install it via "
                "`pip install eigenkit[pennylane]`."
            )
        self.name = name
        self.config = config or PennyLaneConfig()

    @property
    def capabilities(self) -> Capabilities:
        """What this backend supports.

        It runs every gate, evaluates a batch of inputs in one call, can sample a
        finite number of shots, and has no limit on the number of qubits. It also
        provides the two features that Iterative Quantum Amplitude Estimation
        (IQAE) needs: the dense unitary of a callable oracle, through
        `qml.matrix`, and sampling of the ancilla qubit by the provider, for
        `shot_mode="provider"`. See
        [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        return Capabilities(
            supported_gates=ALL_GATES,
            # Native parameter broadcasting in `run` / `run_torch` when
            # weights are 1-D and inputs are 2-D. Other shapes still loop.
            supports_batching=True,
            supports_shots=True,
            max_qubits=None,
            # PennyLane can extract a callable oracle's dense unitary via
            # qml.matrix (used by IQAE's callable-oracle path).
            supports_oracle_matrix=True,
            # PennyLane provides provider-side ancilla shot sampling (used by
            # IQAE's shot_mode='provider' path).
            supports_provider_shots=True,
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> PennyLaneCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        A `Circuit` calls it once and reuses the result for every call, so the
        expensive work belongs here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.

        Returns:
            The prepared circuit, as a `PennyLaneCompiled` object.
        """
        return PennyLaneCompiled(circuit, observables, self.config)
