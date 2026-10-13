"""The reference NumPy statevector simulator, which is the default backend.

This backend uses only NumPy. It needs neither PyTorch nor an external quantum
software development kit. It exists for two reasons:

1. **It works out of the box.** The library runs (`Circuit(provider="numpy")`)
   without installing a large simulator or one that needs a licence. This
   backend is always available.
2. **It is the reference for correctness.** Every other backend is tested
   against it (and against PennyLane) for equal results, so it also serves as
   the reference that the others are checked against.

It is written to be easy to read, not tuned for speed. For speed, install one of
the optional backends. It does use two simple NumPy optimisations. A batch of
inputs evolves in one vectorised pass, with a leading batch axis on the state
tensor, instead of a Python loop over the samples. A controlled-NOT (CNOT) gate is
applied by swapping amplitudes (`np.flip` on the subspace where the control is
`1`) instead of by contracting a `4x4` matrix.

## State representation

The state is a `complex128` tensor of shape `(2,) * n_qubits`, and qubit `q` is
on axis `q`. In C order, this makes qubit `0` the most significant bit, as in
PennyLane, where wire `0` is the most significant bit. As a result,
`state.reshape(-1)[0]` is the amplitude of `|0…0⟩`, and the computed values agree
with PennyLane to machine precision.

## Gradients

The backend declares `gradient_method = "adjoint"` and implements
[`NumpyCompiled.jacobian()`][eigenkit.core.backends.numpy_backend.NumpyCompiled.jacobian].
This method computes analytic gradients in reverse mode, with one forward pass
and one backward pass, so the number of passes is `O(1)` in the number of
parameters. The gradient code of the core package uses it in preference to the
parameter-shift rule, which is the fallback, and it differentiates a 2-D `inputs`
batch in one batched pass. The gradients agree with the parameter-shift rule and
with PennyLane to machine precision.

Example:
    ```python
    from eigenkit import Circuit, Encoding, Variational, all_z

    blocks = [Encoding(in_features=4), Variational(depth=2)]

    # by registered name
    circuit = Circuit(blocks=blocks, observables=all_z(4), provider="numpy")

    # or with the config, when a device setting matters
    from eigenkit.configs import NumpyConfig

    circuit = Circuit(
        blocks=blocks,
        observables=all_z(4),
        provider=NumpyConfig(),
    )
    ```

    This backend is always available. It is the default, and it needs no extra.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

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


# ---- gate / Pauli matrices ----

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_H = np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=np.complex128)

_PAULI: dict[str, np.ndarray] = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}

# Generator Pauli for each rotation: U(theta) = exp(-i theta/2 * P), so
# dU/dtheta = -i/2 * P @ U. Used by the adjoint sweep.
_ROTATION_PAULI = {"RX": "X", "RY": "Y", "RZ": "Z"}

# Non-parametric single-qubit gate matrices.
_CONSTANT_MATRICES: dict[str, np.ndarray] = {
    "H": _H,
    "X": _PAULI["X"],
    "Z": _PAULI["Z"],
    "S": np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128),
    "Sdg": np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128),
}

# Projector onto |1> for a single qubit. Used as the PhaseShift generator:
# PhaseShift(phi) = diag(1, e^{i phi}) = exp(i phi |1><1|), so
# (dU/dphi) U^-1 = i |1><1|.
_PROJ1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)


def _phaseshift_matrix(phi: float) -> np.ndarray:
    """PhaseShift(phi) = diag(1, e^{i phi}): identity on |0>, phase on |1>."""
    return np.array([[1.0, 0.0], [0.0, cmath.exp(1j * phi)]], dtype=np.complex128)


def _phaseshift_matrix_batch(phis: np.ndarray) -> np.ndarray:
    """Per-sample PhaseShift matrices of shape ``(B, 2, 2)`` for angles ``(B,)``."""
    matrix = np.zeros((phis.shape[0], 2, 2), dtype=np.complex128)
    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = np.exp(1j * phis)
    return matrix


def _rotation_matrix(name: str, theta: float) -> np.ndarray:
    half = theta / 2.0
    cos = math.cos(half)
    sin = math.sin(half)
    if name == "RX":
        return np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=np.complex128)
    if name == "RY":
        return np.array([[cos, -sin], [sin, cos]], dtype=np.complex128)
    if name == "RZ":
        return np.array(
            [[cmath.exp(-1j * half), 0.0], [0.0, cmath.exp(1j * half)]],
            dtype=np.complex128,
        )
    raise NotImplementedError(f"not a rotation gate: {name!r}")


# ---- statevector primitives ----


def _zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros((2,) * n_qubits, dtype=np.complex128)
    state[(0,) * n_qubits] = 1.0
    return state


def _apply_single(state: np.ndarray, matrix: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a 2x2 unitary to `qubit` (axis `qubit`) of the state tensor."""
    out = np.tensordot(matrix, state, axes=([1], [qubit]))
    return np.moveaxis(out, 0, qubit)


def _cnot_swap(state: np.ndarray, control_axis: int, target_axis: int) -> np.ndarray:
    """CNOT as an amplitude swap, not a matrix multiply.

    CNOT leaves the control=0 subspace alone and flips the target bit on the
    control=1 subspace. On a length-2 axis, flipping the bit is just reversing
    the axis — so we copy the state and reverse the target axis within the
    control=1 slice. No complex arithmetic. The reversed read is taken from the
    original `state` and written into the fresh copy, so there is no aliasing.
    """
    out = state.copy()
    selector = [slice(None)] * state.ndim
    selector[control_axis] = 1
    selector = tuple(selector)
    # Indexing the control axis to 1 drops it, shifting later axes down by one.
    target_in_slice = target_axis if target_axis < control_axis else target_axis - 1
    out[selector] = np.flip(state[selector], axis=target_in_slice)
    return out


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    return _cnot_swap(state, control, target)


# ---- batched statevector primitives ----
#
# The batched state carries a leading batch axis: shape ``(B, 2, ..., 2)`` with
# qubit ``q`` on axis ``q + 1``. Evolving the whole batch in one sweep avoids the
# Python per-sample loop. Gate angles that depend only on weights are uniform
# across the batch (a single 2x2 matrix, broadcast); angles that depend on the
# per-sample inputs become a stack of ``(B, 2, 2)`` matrices.


def _zero_state_batch(batch: int, n_qubits: int) -> np.ndarray:
    state = np.zeros((batch,) + (2,) * n_qubits, dtype=np.complex128)
    state[(slice(None),) + (0,) * n_qubits] = 1.0
    return state


def _apply_single_batch(state: np.ndarray, matrix: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a single-qubit unitary to `qubit` of a batched state.

    `matrix` is either a shared ``(2, 2)`` (same gate for every sample) or a
    per-sample ``(B, 2, 2)`` stack. `qubit` ``q`` lives on axis ``q + 1``.
    """
    axis = qubit + 1
    if matrix.ndim == 2:
        out = np.tensordot(matrix, state, axes=([1], [axis]))  # (2_out, B, rest)
        return np.moveaxis(out, 0, axis)
    moved = np.moveaxis(state, axis, 1)  # (B, 2_target, rest)
    # optimize=False: this is a 2-operand contraction, so einsum's path search
    # cannot improve the (single) contraction order and is pure per-call overhead.
    out = np.einsum("bxy,by...->bx...", matrix, moved, optimize=False)
    return np.moveaxis(out, 1, axis)


def _apply_cnot_batch(state: np.ndarray, control: int, target: int) -> np.ndarray:
    # Batch axis is 0, so qubit q lives on axis q + 1; same swap, offset axes.
    return _cnot_swap(state, control + 1, target + 1)


def _rotation_matrix_batch(name: str, angles: np.ndarray) -> np.ndarray:
    """Per-sample rotation matrices of shape ``(B, 2, 2)`` for angles ``(B,)``."""
    half = angles / 2.0
    cos = np.cos(half)
    sin = np.sin(half)
    matrix = np.zeros((angles.shape[0], 2, 2), dtype=np.complex128)
    if name == "RX":
        matrix[:, 0, 0] = cos
        matrix[:, 1, 1] = cos
        matrix[:, 0, 1] = -1j * sin
        matrix[:, 1, 0] = -1j * sin
    elif name == "RY":
        matrix[:, 0, 0] = cos
        matrix[:, 1, 1] = cos
        matrix[:, 0, 1] = -sin
        matrix[:, 1, 0] = sin
    elif name == "RZ":
        matrix[:, 0, 0] = np.exp(-1j * half)
        matrix[:, 1, 1] = np.exp(1j * half)
    else:  # pragma: no cover - only rotation gates reach here
        raise NotImplementedError(f"not a rotation gate: {name!r}")
    return matrix


def _as_batch_factor(factor) -> np.ndarray:
    """Shape a chain factor for broadcasting against a ``(B, n_obs)`` gradient.

    A weight/input chain factor is a scalar unless it runs through a `Multiply`
    over an input (then it's per-sample, shape ``(B,)``). Scalars broadcast as
    is; a ``(B,)`` becomes ``(B, 1)`` so it scales every observable column.
    """
    arr = np.asarray(factor, dtype=np.float64)
    return arr if arr.ndim == 0 else arr[:, None]


# ---- multi-qubit gate primitives (lead-aware) ----
#
# Qubit ``q`` lives on axis ``q + lead``: ``lead == 0`` for the single-sample
# state ``(2,)*n`` and ``lead == 1`` for the batched state ``(B, 2, ..., 2)``.
# The leading batch axis is always a `slice(None)`, so the same integer-index
# bookkeeping serves both modes.


def _apply_swap(state: np.ndarray, q0: int, q1: int, lead: int) -> np.ndarray:
    """SWAP: exchange the two qubit axes (a pure axis swap, no arithmetic)."""
    return state.swapaxes(q0 + lead, q1 + lead)


def _target_axis_after_controls(qubits, target: int, lead: int) -> int:
    """Axis of ``target`` once the control axes have been integer-indexed away
    (each control axis positioned before the target shifts it down by one)."""
    controls = qubits[:-1]
    return target + lead - sum(1 for c in controls if c < target)


def _apply_mcx(state: np.ndarray, qubits: tuple, lead: int) -> np.ndarray:
    """Multi-controlled X: on the all-controls-=1 subspace, flip the target
    (reverse its length-2 axis); leave every other amplitude untouched."""
    *controls, target = qubits
    out = state.copy()
    selector = [slice(None)] * state.ndim
    for control in controls:
        selector[control + lead] = 1
    selector = tuple(selector)
    target_axis = _target_axis_after_controls(qubits, target, lead)
    out[selector] = np.flip(state[selector], axis=target_axis)
    return out


def _apply_mcz(state: np.ndarray, qubits: tuple, lead: int) -> np.ndarray:
    """Multi-controlled Z: negate the amplitude where every listed qubit is 1
    (symmetric in its qubits — no distinguished target)."""
    out = state.copy()
    selector = [slice(None)] * state.ndim
    for qubit in qubits:
        selector[qubit + lead] = 1
    out[tuple(selector)] *= -1.0
    return out


def _apply_controlled_single(
    state: np.ndarray, control: int, target: int, matrix: np.ndarray, lead: int
) -> np.ndarray:
    """Apply a single-qubit ``matrix`` to ``target`` only on the ``control``=1
    subspace (the control=0 subspace is left unchanged). ``matrix`` is a shared
    ``(2, 2)`` or a per-sample ``(B, 2, 2)`` stack. Used for CRY."""
    out = state.copy()
    selector = [slice(None)] * state.ndim
    selector[control + lead] = 1
    selector = tuple(selector)
    sub = state[selector]
    axis = _target_axis_after_controls((control, target), target, lead)
    if matrix.ndim == 2:
        applied = np.tensordot(matrix, sub, axes=([1], [axis]))
        applied = np.moveaxis(applied, 0, axis)
    else:  # per-sample (B, 2, 2); batch axis stayed at 0 (slice on the lead axis)
        moved = np.moveaxis(sub, axis, 1)
        applied = np.einsum("bxy,by...->bx...", matrix, moved, optimize=False)
        applied = np.moveaxis(applied, 1, axis)
    out[selector] = applied
    return out


def _apply_controlled_generator(
    state: np.ndarray, control: int, target: int, matrix: np.ndarray, lead: int
) -> np.ndarray:
    r"""\((P_1^{\text{control}} \otimes U_{\text{target}})\,|\text{state}\rangle\): apply ``matrix`` to ``target``
    on the control=1 subspace and *zero* the control=0 subspace. This is the
    CRY gradient generator \((dU/d\theta)\,U^{-1}\) (with \(\text{matrix} = -\tfrac{i}{2}\,Y\))."""
    out = np.zeros_like(state)
    selector = [slice(None)] * state.ndim
    selector[control + lead] = 1
    selector = tuple(selector)
    sub = state[selector]
    axis = _target_axis_after_controls((control, target), target, lead)
    if matrix.ndim == 2:
        applied = np.tensordot(matrix, sub, axes=([1], [axis]))
        applied = np.moveaxis(applied, 0, axis)
    else:
        moved = np.moveaxis(sub, axis, 1)
        applied = np.einsum("bxy,by...->bx...", matrix, moved, optimize=False)
        applied = np.moveaxis(applied, 1, axis)
    out[selector] = applied
    return out


# ---- data-carrying gate primitives (lead-aware) ----
#
# QubitUnitary applies an arbitrary k-qubit unitary; AmplitudeEncoding prepares
# a state via the unitary that maps |0...0> to the target amplitudes. Both reuse
# the general k-qubit applier below, in the backend's qubit-0-MSB ordering (no
# bit-reversal: numpy is MSB-first natively, unlike the Qiskit backend).


def _apply_unitary_axes(
    state: np.ndarray, matrix: np.ndarray, axes: list[int]
) -> np.ndarray:
    """Apply a ``2**k x 2**k`` unitary to the ``k`` tensor ``axes`` of ``state``.

    ``matrix`` is MSB-ordered: ``axes[0]`` is the most-significant index. The
    matrix is reshaped to a ``(2,)*2k`` tensor whose input legs are contracted
    against ``axes``, then the output legs are moved back into place."""
    k = len(axes)
    tensor = np.asarray(matrix, dtype=np.complex128).reshape((2,) * (2 * k))
    out = np.tensordot(tensor, state, axes=(list(range(k, 2 * k)), axes))
    # tensordot places the k output legs first; restore them to their axes.
    return np.moveaxis(out, list(range(k)), axes)


def _apply_unitary(
    state: np.ndarray, matrix: np.ndarray, qubits: tuple, lead: int
) -> np.ndarray:
    """Apply a ``2**k x 2**k`` unitary to the ``k`` target ``qubits`` (qubit-0
    MSB ordered). Lead-aware: qubit ``q`` lives on axis ``q + lead``."""
    return _apply_unitary_axes(state, matrix, [q + lead for q in qubits])


def _apply_controlled_unitary(
    state: np.ndarray,
    controls: tuple,
    control_values: tuple,
    targets: tuple,
    matrix: np.ndarray,
    lead: int,
) -> np.ndarray:
    """Apply ``matrix`` (a ``2**n_target`` unitary, qubit-0-MSB over ``targets``)
    to ``targets`` only on the subspace where each control equals its
    ``control_values`` entry; leave the rest of the state unchanged."""
    out = state.copy()
    selector = [slice(None)] * state.ndim
    for c, v in zip(controls, control_values):
        selector[c + lead] = v
    selector = tuple(selector)
    # Once the control axes are integer-indexed away, each target axis shifts
    # down by the number of control axes positioned before it.
    control_axes = sorted(c + lead for c in controls)
    target_axes = [
        (t + lead) - sum(1 for ca in control_axes if ca < t + lead) for t in targets
    ]
    out[selector] = _apply_unitary_axes(state[selector], matrix, target_axes)
    return out


def _controlled_unitary_args(gate: Gate) -> tuple[np.ndarray, tuple, tuple, tuple]:
    """Unpack a ControlledQubitUnitary gate into (matrix, controls,
    control_values, targets). ``gate.data = (matrix, control_values)``;
    ``gate.qubits`` lists the controls then the targets, with the target count
    inferred from the matrix dimension."""
    matrix = _data_to_array(gate.data[0])
    n_target = int(round(math.log2(matrix.shape[0])))
    n_ctrl = len(gate.qubits) - n_target
    controls = gate.qubits[:n_ctrl]
    targets = gate.qubits[n_ctrl:]
    control_values = gate.data[1] if gate.data[1] is not None else (1,) * n_ctrl
    return matrix, controls, control_values, targets


@dataclass(frozen=True)
class NumpyConfig:
    """Configuration for the reference NumPy simulator.

    It has no settings yet. It exists so that `Circuit(provider=NumpyConfig())`
    works in the same way as with the configurations of the other backends.
    """

    def make_backend(self) -> "NumpySimulator":
        """Build the backend that this configuration describes.

        The provider layer calls it when a configuration is passed as `provider=`.
        There is no reason to call it directly.

        Returns:
            The `NumpySimulator` for this configuration.
        """
        return NumpySimulator(config=self)


class NumpyCompiled(CompiledCircuit):
    """A circuit prepared for the reference NumPy simulator.

    [`NumpySimulator.compile()`][eigenkit.core.backends.numpy_backend.NumpySimulator.compile]
    builds it, and it is cheap to call many times. `run()` evaluates the
    observables, and `jacobian()` computes their analytic gradient.
    `statevector()`, `probabilities()` and `zero_state_probability()` read the
    state directly, which the Harrow-Hassidim-Lloyd (HHL) linear solver needs.
    """

    # The adjoint `jacobian` accepts a 2-D inputs batch and returns batched
    # jacobians in one sweep, so the gradient dispatch routes batches to it
    # instead of looping the single-sample path.
    supports_batched_jacobian = True

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: NumpyConfig,
    ):
        """Prepare the circuit for repeated evaluation.

        `NumpySimulator.compile()` builds this object, so there is no reason to
        build one directly.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.
            config (NumpyConfig): The device settings.
        """
        self._spec = circuit
        self._observables = list(observables)
        self._n = circuit.n_qubits

    # ---- forward ----

    # The single-sample and batched paths share their control flow (gate walk,
    # observable accumulation, adjoint sweep, chain-rule mapping) and differ
    # only in the injected leaf primitives: single uses a bare ``(2,)*n`` state
    # with ``_apply_single`` (qubit on axis q); batched prepends a batch axis
    # and uses ``_apply_single_batch`` (axis q+1, per-sample ``(B,2,2)`` matrices
    # for input-dependent angles). Keeping the loops in one place is what stops
    # the two modes from drifting apart.

    def _rotation_for(self, gate: Gate, weights, resolve_inputs) -> np.ndarray:
        """Rotation matrix for ``gate``: a shared ``(2, 2)`` when the angle is
        batch-uniform (single sample, or a weight-only angle on a batch), or a
        per-sample ``(B, 2, 2)`` stack when it depends on batched inputs.
        ``resolve_inputs`` is the 1-D inputs (single) or the ``(n_inputs, B)``
        transpose (batched, so an ``InputRef`` resolves to ``(B,)``)."""
        angle = resolve(gate.params[0], weights, resolve_inputs)
        if np.ndim(angle) == 0:
            return _rotation_matrix(gate.name, float(angle))
        return _rotation_matrix_batch(gate.name, np.asarray(angle))

    def _phaseshift_for(self, gate: Gate, weights, resolve_inputs) -> np.ndarray:
        """PhaseShift matrix for ``gate``: shared ``(2, 2)`` or per-sample
        ``(B, 2, 2)``, mirroring `_rotation_for()`."""
        angle = resolve(gate.params[0], weights, resolve_inputs)
        if np.ndim(angle) == 0:
            return _phaseshift_matrix(float(angle))
        return _phaseshift_matrix_batch(np.asarray(angle))

    def _cry_matrix(self, gate: Gate, weights, resolve_inputs) -> np.ndarray:
        """The RY block applied to the CRY target on the control=1 subspace."""
        angle = resolve(gate.params[0], weights, resolve_inputs)
        if np.ndim(angle) == 0:
            return _rotation_matrix("RY", float(angle))
        return _rotation_matrix_batch("RY", np.asarray(angle))

    def _walk(self, state, weights, resolve_inputs, apply_single, apply_cnot, lead):
        """Apply the circuit's gates to ``state`` using the injected single or
        batched single-qubit/CNOT primitives plus the lead-aware multi-qubit
        helpers. ``lead`` is 0 (single) or 1 (batched). The gate dispatch lives
        here once."""
        for gate in self._spec.gates:
            name = gate.name
            match name:
                case "CNOT":
                    state = apply_cnot(state, gate.qubits[0], gate.qubits[1])
                case n if n in _ROTATION_PAULI:
                    matrix = self._rotation_for(gate, weights, resolve_inputs)
                    state = apply_single(state, matrix, gate.qubits[0])
                case "PhaseShift":
                    matrix = self._phaseshift_for(gate, weights, resolve_inputs)
                    state = apply_single(state, matrix, gate.qubits[0])
                case "CRY":
                    matrix = self._cry_matrix(gate, weights, resolve_inputs)
                    state = _apply_controlled_single(
                        state, gate.qubits[0], gate.qubits[1], matrix, lead
                    )
                case "CPHASE":
                    # Controlled phase = PhaseShift on the target over the
                    # control=1 subspace (symmetric in control/target).
                    matrix = self._phaseshift_for(gate, weights, resolve_inputs)
                    state = _apply_controlled_single(
                        state, gate.qubits[0], gate.qubits[1], matrix, lead
                    )
                case "SWAP":
                    state = _apply_swap(state, gate.qubits[0], gate.qubits[1], lead)
                case "MCX":
                    state = _apply_mcx(state, gate.qubits, lead)
                case "MCZ":
                    state = _apply_mcz(state, gate.qubits, lead)
                case n if n in _CONSTANT_MATRICES:
                    state = apply_single(state, _CONSTANT_MATRICES[n], gate.qubits[0])
                case "QubitUnitary":
                    matrix = _data_to_array(gate.data[0])
                    state = _apply_unitary(state, matrix, gate.qubits, lead)
                case "ControlledQubitUnitary":
                    mat, controls, cv, targets = _controlled_unitary_args(gate)
                    state = _apply_controlled_unitary(
                        state, controls, cv, targets, mat, lead
                    )
                case "AmplitudeEncoding":
                    matrix = state_prep_unitary(_data_to_array(gate.data[0]))
                    if gate.adjoint:
                        matrix = matrix.conj().T
                    state = _apply_unitary(state, matrix, gate.qubits, lead)
                case _:  # pragma: no cover - validate_circuit blocks unknown gates
                    raise NotImplementedError(f"gate {name!r} not supported")
        return state

    # ---- initial-state injection ----
    #
    # By default a circuit starts from |0...0>. `initial_state` lets a caller
    # inject an arbitrary normalized statevector instead — the primitive behind
    # quantum reservoir carry-over (read state out, re-inject next timestep) and
    # amplitude encoding. The flat vector is in the same qubit-0-MSB order that
    # `statevector()` / `probabilities()` return, so a read-out state round-trips
    # back in unchanged.

    def _coerce_state(self, initial_state: np.ndarray) -> np.ndarray:
        """Validate a single flat initial state and reshape to ``(2,)*n``."""
        flat = validate_initial_state(initial_state, self._n)
        return flat.reshape((2,) * self._n)

    def _coerce_state_batch(self, initial_state: np.ndarray, batch: int) -> np.ndarray:
        """Validate a batched initial state and reshape to ``(B, 2, ..., 2)``.

        Accepts a shared ``(2**n,)`` state (broadcast to every sample) or a
        per-sample ``(B, 2**n)`` stack."""
        flat = validate_initial_state(initial_state, self._n, batch=batch)
        return flat.reshape((batch,) + (2,) * self._n)

    def _evolve(
        self, weights: np.ndarray, inputs: np.ndarray, initial_state=None
    ) -> np.ndarray:
        state = (
            _zero_state(self._n)
            if initial_state is None
            else self._coerce_state(initial_state)
        )
        return self._walk(state, weights, inputs, _apply_single, _apply_cnot, 0)

    def _accumulate_observable(self, state, observable: Observable, apply_single):
        """``O|state>`` for a Pauli-sum observable, with the injected single or
        batched single-qubit applier."""
        total = None
        for term in observable.terms:
            acted = state
            for op, qubit in term.paulis:
                acted = apply_single(acted, _PAULI[op], qubit)
            contribution = term.coeff * acted
            total = contribution if total is None else total + contribution
        if total is None:  # pragma: no cover - observables always have terms
            raise ValueError("observable has no Pauli terms")
        return total

    def _run_one(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        state = self._evolve(weights, inputs)
        return np.array(
            [
                float(
                    np.vdot(
                        state, self._accumulate_observable(state, o, _apply_single)
                    ).real
                )
                for o in self._observables
            ],
            dtype=np.float64,
        )

    # ---- vectorized batch forward ----

    def _evolve_batch(
        self, weights: np.ndarray, inputs: np.ndarray, initial_state=None
    ) -> np.ndarray:
        """Evolve a whole batch in one sweep. ``inputs`` is ``(B, n_inputs)``;
        transposed to ``(n_inputs, B)`` so an input-dependent angle resolves to
        a per-sample ``(B,)`` (weight-only angles stay a single broadcast 2x2)."""
        batch = inputs.shape[0]
        state = (
            _zero_state_batch(batch, self._n)
            if initial_state is None
            else self._coerce_state_batch(initial_state, batch)
        )
        return self._walk(
            state, weights, inputs.T, _apply_single_batch, _apply_cnot_batch, 1
        )

    def _run_batch(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        state = self._evolve_batch(weights, inputs)
        conj_state = np.conjugate(state)  # shared across observables
        qubit_axes = tuple(range(1, self._n + 1))
        columns = [
            np.real(
                np.sum(
                    conj_state
                    * self._accumulate_observable(state, o, _apply_single_batch),
                    axis=qubit_axes,
                )
            )
            for o in self._observables
        ]
        return np.stack(columns, axis=1)  # (B, n_obs)

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
        # Fast path: shared weights + a batch of inputs -> one vectorized sweep.
        if weights.ndim == 1 and inputs.ndim == 2:
            return self._run_batch(weights, inputs)
        return batch_call(self._run_one, weights, inputs)

    def zero_state_probability(self, weights: np.ndarray, inputs: np.ndarray) -> float:
        """Return the probability of measuring the state `|0…0⟩`.

        Fidelity kernels and swap tests are built on it. It does not go through
        `Observable`, because it is not the expectation value of a Pauli operator.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features of one sample (1-D).

        Returns:
            The probability, a float in `[0, 1]`.
        """
        state = self._evolve(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
        )
        amplitude = state.reshape(-1)[0]
        return float((amplitude.conjugate() * amplitude).real)

    def zero_state_probabilities(
        self, weights: np.ndarray, inputs: np.ndarray
    ) -> np.ndarray:
        """Return the probability of `|0…0⟩` for every input of a batch, vectorised.

        `inputs` has shape `(B, n_inputs)`, and the result has shape `(B,)`. The
        whole batch evolves in one pass. The quantum kernel uses this method to
        build a fidelity matrix without a Python loop over pairs.

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The batch of inputs, of shape `(B, n_inputs)`.

        Returns:
            The probabilities, one for each row, of shape `(B,)`.
        """
        state = self._evolve_batch(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
        )
        amplitudes = state.reshape(state.shape[0], -1)[:, 0]  # (B,)
        return (amplitudes.conjugate() * amplitudes).real

    def statevector(
        self, weights: np.ndarray, inputs: np.ndarray, initial_state=None
    ) -> np.ndarray:
        r"""Return the full complex statevector after the circuit has run.

        The result is a flat array of `2**n_qubits` amplitudes, with qubit `0` as
        the most significant bit (the same order as `probabilities()`).
        `initial_state`, a normalised flat vector in the same order, makes the
        evolution start from that state instead of `|0…0⟩`. Reading a state out
        and passing it back in this way is what quantum reservoir carry-over and
        amplitude encoding are built on. A 2-D `inputs` batch evolves in one pass
        and returns an array of shape `(B, 2**n_qubits)`. `initial_state` can then
        be one state of shape `(2**n,)`, shared by all samples, or a stack of
        shape `(B, 2**n)` with one state for each sample.

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
            state = self._evolve_batch(weights, inputs, initial_state=initial_state)
            return state.reshape(state.shape[0], -1)
        state = self._evolve(weights, inputs, initial_state=initial_state)
        return state.reshape(-1)

    def probabilities(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Return the measurement probabilities of all `2**n_qubits` basis states.

        Qubit `0` is the most significant bit, as in PennyLane's `qml.probs`. The
        probabilities are used to decode samples from the prepared state, for
        example the bitstrings of the Quantum Approximate Optimization Algorithm
        (QAOA).

        Args:
            weights (np.ndarray): The parameter vector (1-D).
            inputs (np.ndarray): The input features of one sample (1-D).

        Returns:
            The probabilities, of shape `(2**n_qubits,)`.
        """
        state = self._evolve(
            np.asarray(weights, dtype=np.float64),
            np.asarray(inputs, dtype=np.float64),
        )
        amplitudes = state.reshape(-1)
        return np.abs(amplitudes) ** 2

    # ---- analytic gradients (adjoint / reverse-mode) ----

    @staticmethod
    def _dagger_matrix(matrix: np.ndarray) -> np.ndarray:
        """Conjugate transpose of a shared ``(2, 2)`` or per-sample ``(B, 2, 2)``."""
        if matrix.ndim == 2:
            return matrix.conj().T
        return np.conjugate(np.transpose(matrix, (0, 2, 1)))

    def _dagger(
        self, gate: Gate, weights, resolve_inputs, apply_single, apply_cnot, lead
    ):
        """A function applying ``gate``'s inverse to a state, via the injected
        single or batched primitives. Rotations / PhaseShift / CRY use the
        conjugate-transpose of their (shared 2x2 or per-sample (B,2,2)) matrix;
        CNOT/X/H/Z/SWAP/MCX/MCZ are self-inverse; S† = Sdg, Sdg† = S."""
        name = gate.name
        if name == "CNOT":
            control, target = gate.qubits[0], gate.qubits[1]
            return lambda s: apply_cnot(s, control, target)  # self-inverse
        if name == "SWAP":
            q0, q1 = gate.qubits[0], gate.qubits[1]
            return lambda s: _apply_swap(s, q0, q1, lead)  # self-inverse
        if name == "MCX":
            qubits = gate.qubits
            return lambda s: _apply_mcx(s, qubits, lead)  # self-inverse
        if name == "MCZ":
            qubits = gate.qubits
            return lambda s: _apply_mcz(s, qubits, lead)  # self-inverse
        if name == "CRY":
            dagger = self._dagger_matrix(self._cry_matrix(gate, weights, resolve_inputs))
            control, target = gate.qubits[0], gate.qubits[1]
            return lambda s: _apply_controlled_single(s, control, target, dagger, lead)
        if name == "CPHASE":
            dagger = self._dagger_matrix(
                self._phaseshift_for(gate, weights, resolve_inputs)
            )
            control, target = gate.qubits[0], gate.qubits[1]
            return lambda s: _apply_controlled_single(s, control, target, dagger, lead)
        if name == "QubitUnitary":
            # U is stored explicitly; its inverse is U† (always non-adjoint here:
            # adjoint(QubitUnitary) is folded into the data by the IR transform).
            inverse = _data_to_array(gate.data[0]).conj().T
            qubits = gate.qubits
            return lambda s: _apply_unitary(s, inverse, qubits, lead)
        if name == "ControlledQubitUnitary":
            # Inverse is the controlled-U† (always non-adjoint here: the IR
            # adjoint transform folds the dagger into the stored matrix).
            mat, controls, cv, targets = _controlled_unitary_args(gate)
            inverse = mat.conj().T
            return lambda s: _apply_controlled_unitary(
                s, controls, cv, targets, inverse, lead
            )
        if name == "AmplitudeEncoding":
            # Forward applies U (or U† when gate.adjoint); the inverse is the
            # opposite, so the back-sweep undoes the state preparation.
            u = state_prep_unitary(_data_to_array(gate.data[0]))
            inverse = u if gate.adjoint else u.conj().T
            qubits = gate.qubits
            return lambda s: _apply_unitary(s, inverse, qubits, lead)
        qubit = gate.qubits[0]
        if name == "PhaseShift":
            dagger = self._dagger_matrix(
                self._phaseshift_for(gate, weights, resolve_inputs)
            )
            return lambda s: apply_single(s, dagger, qubit)
        if name in _CONSTANT_MATRICES:
            dagger = _CONSTANT_MATRICES[name].conj().T
            return lambda s: apply_single(s, dagger, qubit)
        dagger = self._dagger_matrix(self._rotation_for(gate, weights, resolve_inputs))
        return lambda s: apply_single(s, dagger, qubit)

    def _adjoint_sweep(self, weights, inputs, *, batched):
        """Per-gate angle gradients ``d<O_k>/dtheta_g`` via one forward + one
        backward sweep (cost independent of the parameter count). Single and
        batched share this loop; they differ only in the injected primitives and
        the observable reduction (``vdot`` vs sum over the qubit axes). ``inputs``
        is the natural form (1-D, or ``(B, n_inputs)``). Returns
        ``{gate_index: (n_obs,) | (B, n_obs)}``.
        """
        lead = 1 if batched else 0
        if batched:
            apply_single, apply_cnot = _apply_single_batch, _apply_cnot_batch
            resolve_inputs = inputs.T
            ket = self._evolve_batch(weights, inputs)
            qubit_axes = tuple(range(1, self._n + 1))

            def reduce(lambdas, d_ket):  # -> (B, n_obs)
                return np.stack(
                    [
                        2.0 * np.real(np.sum(np.conjugate(lam) * d_ket, axis=qubit_axes))
                        for lam in lambdas
                    ],
                    axis=1,
                )
        else:
            apply_single, apply_cnot = _apply_single, _apply_cnot
            resolve_inputs = inputs
            ket = self._evolve(weights, inputs)

            def reduce(lambdas, d_ket):  # -> (n_obs,)
                return np.array(
                    [2.0 * np.vdot(lam, d_ket).real for lam in lambdas],
                    dtype=np.float64,
                )

        lambdas = [
            self._accumulate_observable(ket, o, apply_single) for o in self._observables
        ]
        gradients: dict[int, np.ndarray] = {}
        for index in range(len(self._spec.gates) - 1, -1, -1):
            gate = self._spec.gates[index]
            name = gate.name
            if name in _ROTATION_PAULI:
                pauli = _PAULI[_ROTATION_PAULI[name]]
                # dU/dtheta |before> = -i/2 P U |before> = -i/2 P |ket>.
                d_ket = -0.5j * apply_single(ket, pauli, gate.qubits[0])
                gradients[index] = reduce(lambdas, d_ket)
            elif name == "PhaseShift":
                # PhaseShift(phi) = exp(i phi |1><1|): (dU/dphi) U^-1 = i |1><1|.
                d_ket = 1j * apply_single(ket, _PROJ1, gate.qubits[0])
                gradients[index] = reduce(lambdas, d_ket)
            elif name == "CRY":
                # (dU/dtheta) U^-1 = |1><1|_control (x) (-i/2 Y_target).
                generator = -0.5j * _PAULI["Y"]
                d_ket = _apply_controlled_generator(
                    ket, gate.qubits[0], gate.qubits[1], generator, lead
                )
                gradients[index] = reduce(lambdas, d_ket)
            elif name == "CPHASE":
                # CPHASE(phi) = exp(i phi |11><11|):
                # (dU/dphi) U^-1 = |1><1|_control (x) (i |1><1|_target).
                d_ket = _apply_controlled_generator(
                    ket, gate.qubits[0], gate.qubits[1], 1j * _PROJ1, lead
                )
                gradients[index] = reduce(lambdas, d_ket)
            dagger = self._dagger(
                gate, weights, resolve_inputs, apply_single, apply_cnot, lead
            )
            ket = dagger(ket)
            lambdas = [dagger(lam) for lam in lambdas]
        return gradients

    def jacobian(
        self,
        weights: np.ndarray,
        inputs: np.ndarray,
        *,
        need_input_grad: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute the gradient with this backend's own method, the analytic adjoint.

        The gradient code of the core package looks for a `jacobian` method. When
        a backend has one, it is used in preference to the parameter-shift rule,
        which works on any backend and is the fallback. Here it is the adjoint
        method in reverse mode: one forward pass and one backward pass, so the
        number of passes is `O(1)` in the number of parameters.

        For a single sample (1-D `inputs`), the two jacobians have shapes
        `(n_obs, n_weights)` and `(n_obs, n_inputs)`. For a 2-D `inputs` batch,
        they have a leading batch axis (`(B, n_obs, n_weights)` and
        `(B, n_obs, n_inputs)`), and the whole batch is differentiated in one
        pass. This is what makes the training of a hybrid quantum neural network
        with PyTorch fast on this backend. The gradients with respect to the gate
        angles are mapped onto the weights and inputs with the same symbolic
        chain-rule factors as the parameter-shift code, so angles built with
        `Scale`, `Multiply`, `Sum` or `Negate` are also differentiated correctly.

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
        batched = inputs.ndim == 2
        lead = (inputs.shape[0],) if batched else ()  # leading batch axis, or none
        resolve_inputs = inputs.T if batched else inputs
        n_obs = len(self._observables)
        weight_jac = np.zeros(lead + (n_obs, self._spec.n_weights), dtype=np.float64)
        input_jac = (
            np.zeros(lead + (n_obs, self._spec.n_inputs), dtype=np.float64)
            if need_input_grad
            else None
        )
        gate_grads = self._adjoint_sweep(weights, inputs, batched=batched)
        self._map_gate_grads(gate_grads, weight_jac, input_jac, weights, resolve_inputs)
        return weight_jac, input_jac

    def _map_gate_grads(self, gate_grads, weight_jac, input_jac, weights, resolve_inputs):
        """Scatter per-gate angle gradients onto the weight (and input) jacobian
        columns via the symbolic chain factors. The ``...`` index plus
        ``_as_batch_factor`` let one body serve both the single-sample
        ``(n_obs, n_*)`` and batched ``(B, n_obs, n_*)`` shapes."""
        for index, gate_grad in gate_grads.items():
            angle = self._spec.gates[index].params[0]
            for weight_index in collect_weight_dependencies(angle):
                factor = chain_factor(angle, weight_index, weights, resolve_inputs)
                weight_jac[..., weight_index] += _as_batch_factor(factor) * gate_grad
            if input_jac is not None:
                for input_index in collect_input_dependencies(angle):
                    factor = chain_factor_input(
                        angle, input_index, weights, resolve_inputs
                    )
                    input_jac[..., input_index] += _as_batch_factor(factor) * gate_grad


class NumpySimulator:
    """The provider for the reference NumPy statevector simulator.

    It is registered automatically as `"numpy"`. It uses only NumPy and no
    optional dependency, so it is always available.

    It declares `gradient_method = "adjoint"`, so the gradient code of the core
    package uses
    [`NumpyCompiled.jacobian()`][eigenkit.core.backends.numpy_backend.NumpyCompiled.jacobian]
    (analytic, in reverse mode, with one forward pass and one backward pass
    whatever the number of parameters) instead of the parameter-shift rule, whose
    cost is `O(n_params)`.
    """

    gradient_method = "adjoint"

    def __init__(
        self,
        config: NumpyConfig | None = None,
        name: str = "numpy",
    ):
        """Build the backend from a device configuration.

        Args:
            config (NumpyConfig): The device settings. Default: `None`, which uses
                `NumpyConfig()`.
            name (str): The registry name, used in error messages. Default:
                `"numpy"`.
        """
        self.name = name
        self.config = config or NumpyConfig()

    @property
    def capabilities(self) -> Capabilities:
        """What this backend supports.

        It runs every gate of the library, evaluates a batch of inputs in one
        call, returns exact expectation values only (it does not sample shots),
        and has no limit on the number of qubits. See
        [`Capabilities`][eigenkit.core.provider.Capabilities].
        """
        return Capabilities(
            supported_gates=ALL_GATES,
            # The forward path evolves a whole batch of inputs in one vectorized
            # sweep (leading batch axis on the state tensor).
            supports_batching=True,
            supports_shots=False,
            max_qubits=None,
        )

    def compile(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
    ) -> NumpyCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`.

        A `Circuit` calls it once and reuses the result for every call, so the
        expensive work belongs here.

        Args:
            circuit (CircuitSpec): The circuit to prepare.
            observables (Sequence[Observable]): The observables to measure.

        Returns:
            The prepared circuit, as a `NumpyCompiled` object.
        """
        return NumpyCompiled(circuit, observables, self.config)
