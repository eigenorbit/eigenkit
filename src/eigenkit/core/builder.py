"""A builder that creates a `CircuitSpec` one gate at a time.

The style follows Qiskit, Amazon Braket and CUDA Quantum: `qc.rx(angle, qubit)`
adds a gate to `qc` itself and returns `qc`, so that calls can be chained.

`CircuitBuilder` sits between the ready-made blocks and writing the circuit data
by hand. You can write a circuit gate by gate when that is the natural way, and
then, if you want, call `.to_block()` to combine the result with other blocks.
"""

from __future__ import annotations

import math
from typing import Sequence

from .ir import (
    CircuitSpec,
    Gate,
    GateName,
    InputRef,
    Parameter,
    ParameterLike,
    Scale,
    WeightRef,
    _coerce_parameter,
)


__all__ = ["CircuitBuilder", "ParameterLike"]


class CircuitBuilder:
    """Build a `CircuitSpec` one gate at a time.

    Example:
        ```python
        qc = CircuitBuilder(n_qubits=3)
        weights = qc.add_weights(6)
        inputs = qc.add_inputs(3)
        for qubit in range(3):
            qc.ry(inputs[qubit], qubit=qubit)
        for depth in range(2):
            for qubit in range(3):
                qc.ry(weights[depth * 3 + qubit], qubit=qubit)
            for qubit in range(2):
                qc.cnot(qubit, qubit + 1)
        spec = qc.build()
        ```

    Args:
        n_qubits: The number of qubits of the circuit being built. Every qubit
            index that a gate uses must be between `0` and `n_qubits - 1`.

    Raises:
        ValueError: If `n_qubits` is not positive.
    """

    def __init__(self, n_qubits: int):
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")
        self._n_qubits = n_qubits
        self._n_weights = 0
        self._n_inputs = 0
        self._gates: list[Gate] = []

    # ---- parameter-frame declaration ----

    def add_weights(self, count: int) -> list[WeightRef]:
        """Reserve `count` trainable weights and return references to them.

        Pass a returned `WeightRef` wherever a gate takes an angle. The indices
        are local to this builder and start at `0`. When the block is combined
        with others, `assemble` shifts them to their positions in the whole
        circuit.

        Args:
            count: The number of weights to reserve.

        Returns:
            One `WeightRef` for each reserved weight, in order.
        """
        base = self._n_weights
        self._n_weights += count
        return [WeightRef(index=i) for i in range(base, base + count)]

    def add_inputs(self, count: int) -> list[InputRef]:
        """Reserve `count` input features and return references to them.

        This is the counterpart of `add_weights()` for data instead of trainable
        parameters.

        Args:
            count: The number of input features to reserve.

        Returns:
            One `InputRef` for each reserved feature, in order.
        """
        base = self._n_inputs
        self._n_inputs += count
        return [InputRef(index=i) for i in range(base, base + count)]

    # ---- gate methods ----

    def rx(self, angle: ParameterLike, qubit: int) -> "CircuitBuilder":
        """Add a rotation about the `X` axis.

        Args:
            angle: The rotation angle. It can be a number, or a `WeightRef`, an
                `InputRef` or an expression, which is evaluated when the circuit
                is called.
            qubit: The target qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        self._append("RX", (qubit,), (_coerce_parameter(angle),))
        return self

    def ry(self, angle: ParameterLike, qubit: int) -> "CircuitBuilder":
        """Add a rotation about the `Y` axis.

        Args:
            angle: The rotation angle. It can be a number, or a `WeightRef`, an
                `InputRef` or an expression.
            qubit: The target qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        self._append("RY", (qubit,), (_coerce_parameter(angle),))
        return self

    def rz(self, angle: ParameterLike, qubit: int) -> "CircuitBuilder":
        """Add a rotation about the `Z` axis.

        Args:
            angle: The rotation angle. It can be a number, or a `WeightRef`, an
                `InputRef` or an expression.
            qubit: The target qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        self._append("RZ", (qubit,), (_coerce_parameter(angle),))
        return self

    def h(self, qubit: int) -> "CircuitBuilder":
        """Add a Hadamard gate.

        Args:
            qubit: The target qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        self._append("H", (qubit,), ())
        return self

    def cnot(self, control: int, target: int) -> "CircuitBuilder":
        """Add a controlled-NOT (CNOT) gate.

        Args:
            control: The qubit that controls the flip.
            target: The qubit that is flipped.

        Returns:
            This builder, so that calls can be chained.

        Raises:
            ValueError: If `control` and `target` are the same qubit.
        """
        if control == target:
            raise ValueError("CNOT control and target must differ")
        self._append("CNOT", (control, target), ())
        return self

    def x(self, qubit: int) -> "CircuitBuilder":
        """Add a Pauli `X` gate, which is a bit flip.

        Args:
            qubit: The target qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        self._append("X", (qubit,), ())
        return self

    def s(self, qubit: int) -> "CircuitBuilder":
        """Add an `S` gate, which is a quarter turn about the `Z` axis.

        Its inverse is `sdg()`.

        Args:
            qubit: The target qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        self._append("S", (qubit,), ())
        return self

    def sdg(self, qubit: int) -> "CircuitBuilder":
        """Add an S-dagger gate (`Sdg`), which is the inverse of `s()`.

        Args:
            qubit: The target qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        self._append("Sdg", (qubit,), ())
        return self

    def z(self, qubit: int) -> "CircuitBuilder":
        """Add a Pauli `Z` gate, which is a phase flip.

        Args:
            qubit: The target qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        self._append("Z", (qubit,), ())
        return self

    def mcz(self, *qubits: int) -> "CircuitBuilder":
        """Add a multi-controlled `Z` gate.

        All the qubits except the last one are controls.

        Args:
            *qubits: Two or more distinct qubits. The controls come first, and the
                target is the last one.

        Returns:
            This builder, so that calls can be chained.

        Raises:
            ValueError: If fewer than two qubits are given, or if a qubit is
                repeated.
        """
        if len(qubits) < 2:
            raise ValueError("MCZ requires at least 2 qubits (1 control + 1 target)")
        if len(set(qubits)) != len(qubits):
            raise ValueError("MCZ qubits must be distinct")
        self._append("MCZ", qubits, ())
        return self

    def mcx(self, *qubits: int) -> "CircuitBuilder":
        """Add a multi-controlled `X` gate.

        All the qubits except the last one are controls.

        Args:
            *qubits: Two or more distinct qubits. The controls come first, and the
                target is the last one.

        Returns:
            This builder, so that calls can be chained.

        Raises:
            ValueError: If fewer than two qubits are given, or if a qubit is
                repeated.
        """
        if len(qubits) < 2:
            raise ValueError("MCX requires at least 2 qubits (1 control + 1 target)")
        if len(set(qubits)) != len(qubits):
            raise ValueError("MCX qubits must be distinct")
        self._append("MCX", qubits, ())
        return self

    def phase_shift(self, angle: ParameterLike, qubit: int) -> "CircuitBuilder":
        """Add a phase shift gate `R(φ)`.

        The gate multiplies `|1⟩` by `e^(iφ)` and leaves `|0⟩` unchanged.

        Args:
            angle: The phase `φ`. It can be a number or a symbolic reference.
            qubit: The target qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        self._append("PhaseShift", (qubit,), (_coerce_parameter(angle),))
        return self

    def swap(self, qubit0: int, qubit1: int) -> "CircuitBuilder":
        """Add a `SWAP` gate, which exchanges the states of two qubits.

        Args:
            qubit0: The first qubit.
            qubit1: The second qubit.

        Returns:
            This builder, so that calls can be chained.
        """
        if qubit0 == qubit1:
            raise ValueError("SWAP qubits must differ")
        self._append("SWAP", (qubit0, qubit1), ())
        return self

    def cry(self, angle: ParameterLike, control: int, target: int) -> "CircuitBuilder":
        """Add a controlled `RY` gate.

        Args:
            angle: The rotation angle. It can be a number or a symbolic reference.
            control: The qubit that controls the rotation.
            target: The qubit that is rotated.

        Returns:
            This builder, so that calls can be chained.
        """
        if control == target:
            raise ValueError("CRY control and target must differ")
        self._append("CRY", (control, target), (_coerce_parameter(angle),))
        return self

    def cphase(self, angle: ParameterLike, control: int, target: int) -> "CircuitBuilder":
        """Add a controlled phase shift gate.

        The gate is symmetric: the control and the target play the same role.

        Args:
            angle: The phase. It can be a number or a symbolic reference.
            control: One of the two qubits.
            target: The other qubit. The gate acts in the same way if the two
                qubits are exchanged.

        Returns:
            This builder, so that calls can be chained.
        """
        if control == target:
            raise ValueError("CPHASE control and target must differ")
        self._append("CPHASE", (control, target), (_coerce_parameter(angle),))
        return self

    def pauli_exponential(
        self,
        paulis: Sequence[tuple[str, int]],
        angle: ParameterLike,
        coefficient: float = 1.0,
    ) -> "CircuitBuilder":
        r"""Add the exponential of a product of Pauli operators.

        The gate is `exp(-i * coefficient * angle * P)`, where `P` is the product of
        `paulis`. Each element of `paulis` is a pair `(op, qubit)`, where `op` is
        `"X"`, `"Y"` or `"Z"`, and the qubits are distinct.

        This is the Trotter building block that the quantum approximate
        optimisation algorithm (QAOA) needs for its cost and mixer layers, and that
        the unitary coupled-cluster singles and doubles (UCCSD) ansatz needs for
        its excitation operators. It is built from the core gates only. First,
        `H` gates (for `X`) and `RX(π/2)` gates (for `Y`) change each factor to the
        `Z` basis. Then a ladder of controlled-NOT (CNOT) gates accumulates the
        parity on the last qubit, and a central `RZ(2 * coefficient * angle)` is
        applied (because `RZ(φ) = exp(-i φ Z / 2)`). Finally, the inverse of the
        first steps is applied.

        Args:
            paulis: The Pauli product, as `(operator, qubit)` pairs, where the
                operator is `"X"`, `"Y"` or `"Z"` and the qubits are distinct. An
                empty sequence is the identity, and nothing is added.
            angle: The rotation angle. It can be a number or a symbolic reference.
            coefficient: A number that multiplies the angle. This is where the
                weight of a Hamiltonian term goes. Default: `1.0`.

        Returns:
            This builder, so that calls can be chained.
        """
        terms = list(paulis)
        if not terms:
            return self
        qubits = [qubit for _, qubit in terms]
        # Rotate each factor into the Z basis: H for X, RX(+pi/2) for Y
        # (RX(-pi/2) Z RX(+pi/2) = Y, so this maps Y -> Z).
        for op, qubit in terms:
            if op == "X":
                self.h(qubit)
            elif op == "Y":
                self.rx(math.pi / 2, qubit)
            elif op != "Z":
                raise ValueError(f"Pauli op must be 'X', 'Y', or 'Z', got {op!r}")
        # Accumulate parity onto the last qubit with a CNOT ladder.
        pairs = list(zip(qubits, qubits[1:]))
        for control, target in pairs:
            self.cnot(control, target)
        self.rz(Scale(_coerce_parameter(angle), 2.0 * coefficient), qubits[-1])
        for control, target in reversed(pairs):
            self.cnot(control, target)
        # Rotate back out of the Z basis (inverse of the in-rotations).
        for op, qubit in terms:
            if op == "X":
                self.h(qubit)
            elif op == "Y":
                self.rx(-math.pi / 2, qubit)
        return self

    def _append(
        self,
        name: GateName,
        qubits: tuple[int, ...],
        params: tuple[Parameter, ...],
    ) -> None:
        for qubit in qubits:
            if not 0 <= qubit < self._n_qubits:
                raise ValueError(
                    f"gate {name}{qubits} uses qubit {qubit}, which is out of "
                    f"range for a {self._n_qubits}-qubit circuit"
                )
        self._gates.append(Gate(name=name, qubits=qubits, params=params))

    # ---- introspection ----

    @property
    def n_qubits(self) -> int:
        """The number of qubits of the circuit, as given to the constructor.

        Every qubit index that a gate uses must be smaller than this number, and the
        number never changes.
        """
        return self._n_qubits

    @property
    def n_weights(self) -> int:
        """The number of trainable weights that `add_weights()` has reserved so far.

        The built [`CircuitSpec`][eigenkit.core.ir.CircuitSpec] has the same number.
        """
        return self._n_weights

    @property
    def n_inputs(self) -> int:
        """The number of input features that `add_inputs()` has reserved so far.

        The built [`CircuitSpec`][eigenkit.core.ir.CircuitSpec] has the same number.
        """
        return self._n_inputs

    def __len__(self) -> int:
        return len(self._gates)

    # ---- finalize ----

    def build(self) -> CircuitSpec:
        """Return the current state of the builder as an immutable `CircuitSpec`."""
        return CircuitSpec(
            n_qubits=self._n_qubits,
            n_weights=self._n_weights,
            n_inputs=self._n_inputs,
            gates=tuple(self._gates),
        )

    def to_block(self):
        """Wrap the current state of the builder in a `CustomBlock`.

        This is useful to build a part of a circuit one gate at a time, and then to
        combine it with other blocks. When the blocks are assembled, the indices of
        the `WeightRef` and `InputRef` objects of this block are shifted to their
        positions in the whole circuit.
        """
        from .blocks import CustomBlock  # local import avoids a cycle

        return CustomBlock(
            n_qubits=self._n_qubits,
            gates=tuple(self._gates),
            n_weights=self._n_weights,
            n_inputs=self._n_inputs,
        )
