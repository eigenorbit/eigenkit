r"""The Qulacs backend.

Qulacs is an optional dependency (`pip install eigenkit[qulacs]`). The package is
imported only when it is needed, so the rest of the library works without it,
and `compile()` raises a clear `ImportError` if it is missing.

It declares `gradient_method = "adjoint"`. Qulacs provides an analytic
reverse-mode gradient (`ParametricQuantumCircuit.backprop`), so, as with the
NumPy reference simulator, the compiled circuit has a `jacobian` method, and the
gradient code of the core package prefers it to the parameter-shift rule, whose
cost is `O(n_params)`.

## Compile and run

`compile()` builds one `ParametricQuantumCircuit` and the Qulacs `Observable`
objects once. Every rotation whose angle depends on the weights or inputs becomes
a *parametric slot*. `run()` sets the slots with `set_parameter`, without
rebuilding the circuit at each call, and `jacobian()` reads the gradients of all
slots from one `backprop` pass for each observable.

## Conventions matched with the NumPy and PennyLane reference

- **Rotation sign.** Qulacs rotations are `e^{+iθP/2}`, while the library and
  PennyLane use `e^{-iθP/2}`. Every rotation angle is negated when it is passed
  to Qulacs: the Qulacs parameter of a slot is `φ = -(coeff · angle)`. As a
  result, `backprop` returns `d⟨O⟩/dφ`, and the chain rule through `φ` gives
  `d⟨O⟩/d(weights)` with the sign handled automatically (see `jacobian`).
- **Bit order.** Qulacs stores the amplitudes with qubit `0` as the least
  significant bit (as Qiskit does), while the library uses qubit `0` as the most
  significant bit. The statevector input and output, and the matrices of
  multi-qubit gates, are bit-reversed to match the two. The Pauli *observables*
  need no reindexing, because the qubit labels match: only the storage order of
  the amplitudes differs.

## Gate coverage

`add_parametric_R{X,Y,Z}` covers the trainable rotations directly. The gates that
have no parametric primitive in Qulacs are decomposed into these rotations, so
that they stay differentiable:

- `PhaseShift(θ)` becomes a parametric `RZ`. The two are equal up to a global
  phase that cannot be observed.
- `CRY(θ)` becomes `RY(θ/2) CX RY(-θ/2) CX`, where `CX` is a controlled-NOT
  gate. This decomposition is exact.
- `CPHASE(θ)` becomes `P(θ/2)_c CX P(-θ/2)_t CX P(θ/2)_t`, with each `P` as an
  `RZ` slot. This decomposition is exact up to a global phase that cannot be
  observed.

`PhaseShift`, `CPHASE` and `CRY` with a constant angle, and the non-parametric
gates (`H`, `X`, `Z`, `S`, `Sdg`, `CNOT`, `SWAP`, `MCX`, `MCZ`, `QubitUnitary`,
`ControlledQubitUnitary`, `AmplitudeEncoding`), are added once as exact Qulacs
gates. `backprop` accepts these fixed gates, so native adjoint gradients are
available for every circuit.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # by registered name
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="qulacs")

    # or with the config, when a device setting matters
    from eigenkit.configs import QulacsConfig

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=QulacsConfig(),
    )
    ```

    Gradients are computed natively in reverse mode, with `backprop`. This backend
    needs the `qulacs` extra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..ir import (
    ALL_GATES,
    CircuitSpec,
    Gate,
    Observable,
    Parameter,
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


def _require_qulacs() -> None:
    """Probe for qulacs. Raise a clear error if it is not installed.

    Call this once before issuing local `from qulacs import ...` statements
    in code paths that need it — the guard centralises the install hint so
    each call site stays a plain Python import.
    """
    try:
        import qulacs  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "qulacs is not installed. Install it via `pip install eigenkit[qulacs]`."
        ) from e


@dataclass(frozen=True)
class QulacsConfig:
    """Device configuration for the Qulacs backend.

    The Qulacs statevector simulator for the central processing unit (CPU) needs
    no settings, so this configuration is currently only a marker. It makes
    `Circuit(provider=QulacsConfig())` select this backend, and it leaves room for
    future fields, for example a state on a graphics processing unit (GPU).

    Pass it to `Circuit(provider=QulacsConfig())`, or pass it to `register()` under
    a name to reuse it.
    """

    def make_backend(self) -> "QulacsBackend":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `QulacsBackend` for this configuration.
        """
        return QulacsBackend(config=self)


# ---- pure helpers (no qulacs import) ----


def _bit_reversal_perm(n: int) -> list[int]:
    """Index permutation mapping the library's qubit-0-MSB ordering to qulacs'
    qubit-0-LSB ordering for ``n`` qubits. Applied to multi-qubit gate matrices
    (QubitUnitary, state-prep isometries) so the same logical operator lands on
    the same physical basis states in both conventions."""
    return [int(format(k, f"0{n}b")[::-1], 2) for k in range(2**n)]


def _swap_endianness(data: np.ndarray, n: int) -> np.ndarray:
    """Convert a flat statevector between qulacs' qubit-0-LSB order and the
    library's qubit-0-MSB order. Reversing the qubit axes is its own inverse, so
    this maps both ways."""
    if n < 2:
        return data
    return data.reshape((2,) * n).transpose(tuple(range(n - 1, -1, -1))).reshape(-1)


def _ry_matrix(theta: float) -> np.ndarray:
    r"""Standard \(RY(\theta) = e^{-i\theta Y/2}\) (library convention)."""
    cos, sin = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[cos, -sin], [sin, cos]], dtype=np.complex128)


def _is_constant(parameter: Parameter) -> bool:
    """True when the angle depends on neither weights nor inputs."""
    return not (
        collect_weight_dependencies(parameter) or collect_input_dependencies(parameter)
    )


def _const_value(parameter: Parameter) -> float:
    """Resolve a constant angle to a float (never indexes weights/inputs)."""
    return float(resolve(parameter, np.empty(0), np.empty(0)))


class QulacsCompiled(CompiledCircuit):
    """A compiled circuit: one `ParametricQuantumCircuit`, reused across calls.

    `_slots[k] = (expr, coeff)` records that the parametric Qulacs gate `k`
    implements the library angle `coeff · expr`. Its Qulacs parameter is set to
    `-(coeff · resolve(expr))`, because Qulacs uses the opposite rotation sign.
    """

    # Enable the batch-capable jacobian entry point: `jacobian` accepts a 2-D
    # inputs batch and returns a leading-batch-axis result (it loops internally,
    # since qulacs `backprop` is single-sample).
    supports_batched_jacobian = True

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: QulacsConfig,
    ):
        """Prepare the circuit for repeated evaluation.

        `QulacsBackend.compile()` builds this object, so there is no reason to
        build one directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.
            config (QulacsConfig): The device settings.
        """
        _require_qulacs()
        from qulacs import Observable as QulacsObservable
        from qulacs import ParametricQuantumCircuit, QuantumState
        from qulacs.gate import DenseMatrix, to_matrix_gate

        self._QuantumState = QuantumState
        self._DenseMatrix = DenseMatrix
        self._to_matrix_gate = to_matrix_gate

        self._config = config
        self._n_qubits = circuit.n_qubits
        self._n_weights = circuit.n_weights
        self._n_inputs = circuit.n_inputs

        self._pqc = ParametricQuantumCircuit(circuit.n_qubits)
        self._slots: list[tuple[Parameter, float]] = []
        # (expression, coefficient) for every global phase the RZ-for-phase
        # substitutions above introduce; `statevector` divides them back out.
        self._phase_terms: list[tuple[Parameter, float]] = []
        for gate in circuit.gates:
            self._add_gate(gate)

        # Split each observable into a qulacs Pauli operator plus a constant
        # (identity) offset. qulacs `Observable` holds only genuine Pauli terms;
        # identity terms (e.g. a chemistry Hamiltonian's nuclear-repulsion
        # constant) are added back after `get_expectation_value` and contribute
        # nothing to the gradient.
        self._qulacs_obs = []
        self._obs_offsets = np.zeros(len(observables), dtype=np.float64)
        for k, observable in enumerate(observables):
            qulacs_obs = QulacsObservable(circuit.n_qubits)
            offset = 0.0
            for term in observable.terms:
                if term.paulis:
                    pauli_str = " ".join(f"{op} {qubit}" for op, qubit in term.paulis)
                    qulacs_obs.add_operator(float(term.coeff), pauli_str)
                else:
                    offset += float(term.coeff)
            self._qulacs_obs.append(qulacs_obs)
            self._obs_offsets[k] = offset

    # ---- gate translation (compile-time) ----

    def _add_param_rotation(
        self, kind: str, qubit: int, expr: Parameter, coeff: float
    ) -> None:
        """Add a parametric R{X,Y,Z} slot; its parameter is bound at run time."""
        getattr(self._pqc, f"add_parametric_{kind}_gate")(qubit, 0.0)
        self._slots.append((expr, coeff))

    def _add_controlled_dense(
        self,
        controls: Sequence[int],
        control_values: Sequence[int] | None,
        targets: Sequence[int],
        matrix: np.ndarray,
    ) -> None:
        """Add a (multi-)controlled dense matrix gate. ``matrix`` is in the
        library's MSB convention over ``targets``; it is bit-reversed for the
        qulacs LSB convention when it spans more than one target qubit."""
        if len(targets) > 1:
            perm = _bit_reversal_perm(len(targets))
            matrix = matrix[np.ix_(perm, perm)]
        gate = self._to_matrix_gate(self._DenseMatrix(list(targets), matrix))
        values = control_values if control_values is not None else [1] * len(controls)
        for control, value in zip(controls, values):
            gate.add_control_qubit(int(control), int(value))
        self._pqc.add_gate(gate)

    def _add_gate(self, gate: Gate) -> None:
        name = gate.name
        qubits = gate.qubits
        if name in ("RX", "RY", "RZ"):
            expr = gate.params[0]
            if _is_constant(expr):
                # Exact fixed rotation; negate the angle for qulacs' +i sign.
                getattr(self._pqc, f"add_{name}_gate")(qubits[0], -_const_value(expr))
            else:
                self._add_param_rotation(name, qubits[0], expr, 1.0)
        elif name == "PhaseShift":
            expr = gate.params[0]
            if _is_constant(expr):
                self._pqc.add_U1_gate(qubits[0], _const_value(expr))  # exact diag(1,e^iθ)
            else:
                # qulacs has no parametric U1, so a trainable phase becomes an
                # RZ slot: RZ(θ) = e^{-iθ/2}·PhaseShift(θ). The factor is
                # unobservable in an expectation value but not in the
                # statevector, and `LeakyCarry` mixes an evolved state with a
                # reset one, which turns it into a *relative* phase and moves
                # even <Z>. Record it so `statevector` can divide it back out.
                self._add_param_rotation("RZ", qubits[0], expr, 1.0)
                self._phase_terms.append((expr, 0.5))
        elif name == "CRY":
            expr = gate.params[0]
            control, target = qubits[0], qubits[1]
            if _is_constant(expr):
                self._add_controlled_dense(
                    [control], [1], [target], _ry_matrix(_const_value(expr))
                )
            else:
                # RY(θ/2)·CX·RY(-θ/2)·CX (exact controlled-RY decomposition).
                self._add_param_rotation("RY", target, expr, 0.5)
                self._pqc.add_CNOT_gate(control, target)
                self._add_param_rotation("RY", target, expr, -0.5)
                self._pqc.add_CNOT_gate(control, target)
        elif name == "CPHASE":
            expr = gate.params[0]
            control, target = qubits[0], qubits[1]
            if _is_constant(expr):
                phase = np.exp(1j * _const_value(expr))
                self._add_controlled_dense(
                    [control], [1], [target], np.diag([1.0, phase]).astype(np.complex128)
                )
            else:
                # P(θ/2)_c · CX · P(-θ/2)_t · CX · P(θ/2)_t, each P as an RZ
                # slot. The three substitutions leave e^{-iθ/4} overall; see
                # the PhaseShift branch above.
                self._add_param_rotation("RZ", control, expr, 0.5)
                self._pqc.add_CNOT_gate(control, target)
                self._add_param_rotation("RZ", target, expr, -0.5)
                self._pqc.add_CNOT_gate(control, target)
                self._add_param_rotation("RZ", target, expr, 0.5)
                self._phase_terms.append((expr, 0.25))
        elif name == "H":
            self._pqc.add_H_gate(qubits[0])
        elif name == "X":
            self._pqc.add_X_gate(qubits[0])
        elif name == "Z":
            self._pqc.add_Z_gate(qubits[0])
        elif name == "S":
            self._pqc.add_S_gate(qubits[0])
        elif name == "Sdg":
            self._pqc.add_Sdag_gate(qubits[0])
        elif name == "CNOT":
            self._pqc.add_CNOT_gate(qubits[0], qubits[1])
        elif name == "SWAP":
            self._pqc.add_SWAP_gate(qubits[0], qubits[1])
        elif name == "MCX":
            x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
            self._add_controlled_dense(qubits[:-1], None, [qubits[-1]], x)
        elif name == "MCZ":
            z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
            self._add_controlled_dense(qubits[:-1], None, [qubits[-1]], z)
        elif name == "QubitUnitary":
            matrix = _data_to_array(gate.data[0])
            if gate.adjoint:
                matrix = matrix.conj().T
            n = len(qubits)
            if n > 1:
                perm = _bit_reversal_perm(n)
                matrix = matrix[np.ix_(perm, perm)]
            self._pqc.add_gate(self._DenseMatrix(list(qubits), matrix))
        elif name == "ControlledQubitUnitary":
            matrix = _data_to_array(gate.data[0])
            if gate.adjoint:
                matrix = matrix.conj().T
            n_target = int(round(math.log2(matrix.shape[0])))
            n_ctrl = len(qubits) - n_target
            controls = qubits[:n_ctrl]
            targets = qubits[n_ctrl:]
            control_values = gate.data[1] if len(gate.data) > 1 else None
            self._add_controlled_dense(controls, control_values, targets, matrix)
        elif name == "AmplitudeEncoding":
            if gate.adjoint:
                raise NotImplementedError(
                    "adjoint(AmplitudeEncoding) is not supported by the qulacs "
                    "backend (state preparation is not invertible as a gate). Use "
                    "the PennyLane backend for Grover iterates built from "
                    "AmplitudeEncoding oracles, or replace AmplitudeEncoding with "
                    "QubitUnitary inside the oracle."
                )
            amplitudes = _data_to_array(gate.data[0])
            unitary = state_prep_unitary(amplitudes)
            n = len(qubits)
            if n > 1:
                perm = _bit_reversal_perm(n)
                unitary = unitary[np.ix_(perm, perm)]
            self._pqc.add_gate(self._DenseMatrix(list(qubits), unitary))
        else:
            raise NotImplementedError(f"gate '{name}' not implemented")

    # ---- run-time parameter binding ----

    def _bind(self, weights: np.ndarray, inputs: np.ndarray) -> None:
        """Bind every parametric slot from ``(weights, inputs)``."""
        for k, (expr, coeff) in enumerate(self._slots):
            angle = float(resolve(expr, weights, inputs))
            self._pqc.set_parameter(k, -(coeff * angle))

    def _run_state(self, weights: np.ndarray, inputs: np.ndarray, initial_flat=None):
        r"""Evolve a fresh `QuantumState` through the bound circuit.

        ``initial_flat`` (a validated flat vector in the library's MSB order)
        seeds the state instead of \(|0\cdots 0\rangle\).
        """
        state = self._QuantumState(self._n_qubits)
        if initial_flat is None:
            state.set_zero_state()
        else:
            state.load(
                _swap_endianness(np.ascontiguousarray(initial_flat), self._n_qubits)
            )
        self._bind(weights, inputs)
        self._pqc.update_quantum_state(state)
        return state

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
        return batch_call(self._run_one, weights, inputs)

    def _run_one(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        state = self._run_state(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
        )
        output = np.empty(len(self._qulacs_obs), dtype=np.float64)
        for k, observable in enumerate(self._qulacs_obs):
            value = observable.get_expectation_value(state)
            output[k] = float(np.real(value)) + self._obs_offsets[k]
        return output

    # ---- native adjoint gradient (backprop) ----

    def jacobian(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
        *,
        need_input_grad: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        r"""Compute the gradient with the native Qulacs method, the analytic adjoint.

        The gradient code of the core package looks for a `jacobian` method. When
        a backend has one, it is used in preference to the parameter-shift rule,
        which works on any backend and is the fallback. Here the gradient comes
        from `backprop`: one pass for each observable gives `d⟨O⟩/dφ` for every
        parametric slot. These gradients are mapped onto the weights (and the
        inputs) with the same symbolic chain-rule factors as the parameter-shift
        code, so angles built with `Scale`, `Multiply`, `Sum` or `Negate` (for
        example in `Reuploading(trainable_scale=True)`) are also handled.

        For a single sample, the two jacobians have shapes `(n_obs, n_weights)`
        and `(n_obs, n_inputs)`. For a 2-D `inputs` batch, they have a leading
        batch axis, and the batch is evaluated in a loop.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features: 1-D for one sample, or 2-D
                with a leading batch axis.
            need_input_grad (bool): If True, also compute the jacobian with
                respect to the inputs. Default: `False`.

        Returns:
            weight_jacobian: The jacobian with respect to the weights.
            input_jacobian: The jacobian with respect to the inputs, or `None`
                when `need_input_grad` is `False`.
        """
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim == 2:
            per_sample = [
                self._jacobian_one(weights, row, need_input_grad) for row in inputs
            ]
            weight_jac = np.stack([wj for wj, _ in per_sample])
            input_jac = (
                np.stack([xj for _, xj in per_sample]) if need_input_grad else None
            )
            return weight_jac, input_jac
        return self._jacobian_one(weights, inputs, need_input_grad)

    def _jacobian_one(
        self, weights: np.ndarray, inputs: np.ndarray, need_input_grad: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        n_obs = len(self._qulacs_obs)
        weight_jac = np.zeros((n_obs, self._n_weights), dtype=np.float64)
        input_jac = (
            np.zeros((n_obs, self._n_inputs), dtype=np.float64)
            if need_input_grad
            else None
        )
        if not self._slots:
            return weight_jac, input_jac
        self._bind(weights, inputs)
        for obs_index, observable in enumerate(self._qulacs_obs):
            if observable.get_term_count() == 0:
                continue  # identity-only observable: gradient is zero
            slot_grads = self._pqc.backprop(observable)  # d<O>/dφ per slot
            for k, (expr, coeff) in enumerate(self._slots):
                # φ_k = -(coeff · expr) ⇒ dφ_k/dθ = -coeff, so the standard
                # chain rule d<O>/dw = Σ_k (d<O>/dφ_k)(dφ_k/dw) becomes:
                gate_grad = slot_grads[k]
                for w_index in collect_weight_dependencies(expr):
                    factor = chain_factor(expr, w_index, weights, inputs)
                    weight_jac[obs_index, w_index] += -coeff * factor * gate_grad
                if input_jac is not None:
                    for x_index in collect_input_dependencies(expr):
                        factor = chain_factor_input(expr, x_index, weights, inputs)
                        input_jac[obs_index, x_index] += -coeff * factor * gate_grad
        return weight_jac, input_jac

    # ---- state-vector I/O ----

    def zero_state_probability(self, weights: np.ndarray, inputs: np.ndarray) -> float:
        r"""Return `|⟨0…0|ψ⟩|²`, the squared amplitude of `|0…0⟩`.

        The basis state `0` is `|0…0⟩` whether qubit `0` is the least or the most
        significant bit, so no reindexing is needed. The fidelity quantum kernel
        is built on this method.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features of one sample (1-D).

        Returns:
            The probability, a float in `[0, 1]`.
        """
        state = self._run_state(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
        )
        amplitude = state.get_vector()[0]
        return float((amplitude.conjugate() * amplitude).real)

    def statevector(
        self, weights: np.ndarray, inputs: np.ndarray, initial_state=None
    ) -> np.ndarray:
        r"""Return the full complex statevector after the circuit has run.

        Qubit `0` is the most significant bit. `initial_state`, in the same
        order, is the starting state instead of `|0…0⟩`. A 2-D `inputs` batch is
        evaluated in a loop and returns an array of shape `(B, 2**n)`.
        `initial_state` can then be one state of shape `(2**n,)`, shared by all
        samples, or a stack of shape `(B, 2**n)` with one state for each sample.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features: 1-D for one sample, or 2-D
                with a leading batch axis.
            initial_state (np.ndarray | None): The state to start from, instead
                of `|0…0⟩`. Default: `None`.

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
        state = self._run_state(weights, inputs, initial_flat=init_flat)
        vector = _swap_endianness(
            np.asarray(state.get_vector(), dtype=np.complex128), self._n_qubits
        )
        return vector * self._phase_correction(weights, inputs)

    def _phase_correction(self, weights, inputs) -> complex:
        """The factor that turns qulacs' state back into the circuit's own.

        Each trainable phase gate is lowered to RZ slots, which agree with the
        gate up to `e^{-i·c·theta}`. Multiplying by its inverse here makes
        `statevector()` return the same amplitudes as every other backend --
        phases included, which matters wherever the state itself is the output
        rather than an expectation value.
        """
        if not self._phase_terms:
            return 1.0 + 0.0j
        total = sum(
            coefficient * float(resolve(expression, weights, inputs))
            for expression, coefficient in self._phase_terms
        )
        return complex(np.exp(1j * total))


class QulacsBackend:
    """The provider for the Qulacs statevector simulator.

    It is optional: it is registered as `"qulacs"` when the package is installed,
    and the package is imported only on first use. It declares
    `gradient_method = "adjoint"`, so the gradient code of the core package uses
    [`QulacsCompiled.jacobian()`][eigenkit.core.backends.qulacs_backend.QulacsCompiled.jacobian]
    (analytic, in reverse mode, through Qulacs `backprop`) instead of the
    parameter-shift rule, whose cost is `O(n_params)`.
    """

    gradient_method = "adjoint"

    def __init__(
        self,
        config: QulacsConfig | None = None,
        name: str = "qulacs",
    ):
        """Build the backend from a device configuration.

        Args:
            config (QulacsConfig): The device settings. Default: `None`, which
                uses `QulacsConfig()`.
            name (str): The registry name, used in error messages. Default:
                `"qulacs"`.
        """
        self.name = name
        self.config = config or QulacsConfig()

    @property
    def capabilities(self) -> Capabilities:
        """What this backend supports.

        It runs every gate, returns exact expectation values only, does not
        evaluate a batch in one call, and has no limit on the number of qubits.
        The adjoint of `AmplitudeEncoding` is excluded, because a state
        preparation cannot be inverted as a Qulacs gate. See
        [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        return Capabilities(
            supported_gates=ALL_GATES,
            supports_batching=False,
            supports_shots=False,
            max_qubits=None,
            # State preparation is not invertible as a qulacs gate, so the
            # adjoint of AmplitudeEncoding is rejected (mirrors Qiskit).
            supported_adjoint_gates=ALL_GATES - frozenset({"AmplitudeEncoding"}),
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> QulacsCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        A `Circuit` calls it once and reuses the result for every call, so the
        expensive work belongs here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.

        Returns:
            The prepared circuit, as a `QulacsCompiled` object.
        """
        return QulacsCompiled(circuit, observables, self.config)
