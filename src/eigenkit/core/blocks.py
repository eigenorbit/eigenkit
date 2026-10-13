"""Blocks: reusable parts of a circuit that do not depend on the provider.

Given the total number of qubits of the circuit, a block builds its gates and
states how many weights and inputs it uses. Blocks are combined with `assemble`:

```python
from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble

spec = assemble([AngleEncoding(in_features=3), BasicVariational(depth=2)])
```

`assemble` goes through the blocks in order and chooses the total number of
qubits. It shifts the local `WeightRef` and `InputRef` indices of each block so
that they point to the right place in the weight and input vectors of the whole
circuit, and it returns a single `CircuitSpec`. Blocks do not depend on a
provider or on PyTorch, and most of them do not use NumPy. Only the blocks that
check or compute a matrix (`QubitUnitary`, `ControlledUnitaryBlock` and
`QPEBlock`) use NumPy. The
blocks use `CircuitBuilder` internally, because it makes their code easier to
read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

from .builder import CircuitBuilder
from .ir import (
    CircuitSpec,
    Constant,
    Gate,
    GateName,
    InputRef,
    Multiply,
    Negate,
    Parameter,
    Scale,
    Sum,
    WeightRef,
    _array_to_data,
    _data_to_array,
    adjoint,
)


@dataclass(frozen=True)
class IRFragment:
    """The gates that one block adds to a circuit.

    The `WeightRef` and `InputRef` indices of a fragment are local to it, and they
    start at `0`. When the blocks are combined into a circuit, `assemble` shifts
    them so that they point to the right place in the weight and input vectors of
    the whole circuit.

    Attributes:
        gates: The gates of the block, in the order in which they are applied.
        n_weights: The number of trainable weights that the block uses.
        n_inputs: The number of input features that the block uses.
    """

    gates: tuple[Gate, ...]
    n_weights: int
    n_inputs: int


@runtime_checkable
class Block(Protocol):
    """The interface that every circuit block follows.

    A block is a reusable part of a circuit, such as an encoding, a variational
    layer or a fixed subroutine. `assemble` asks each block in turn how many qubits
    it needs, gives the circuit as many qubits as the widest block needs, and then
    asks every block to build its gates for that number of qubits. Any object with
    these two methods works as a block. It does not need to inherit from this
    class.
    """

    def required_qubits(self) -> int:
        """Return the minimum number of qubits that this block needs."""
        ...

    def build(self, n_qubits: int) -> IRFragment:
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        ...


_ROTATIONS = ("RX", "RY", "RZ")


def _rotation_method(builder: "CircuitBuilder", name: object):
    """Return the builder method that applies rotation `name`.

    The blocks are mutable dataclasses, so a rotation validated in
    `__post_init__` can still be reassigned before `build()` runs. This is
    that second check, in one place rather than once per block.

    Args:
        builder (CircuitBuilder): The builder to take the method from.
        name (object): `"RX"`, `"RY"` or `"RZ"`.

    Returns:
        The bound `rx`, `ry` or `rz` method.

    Raises:
        ValueError: If `name` is not a rotation axis.
    """
    method = {"RX": builder.rx, "RY": builder.ry, "RZ": builder.rz}.get(name)
    if method is None:
        raise ValueError(f"rotation must be RX/RY/RZ, got {name!r}")
    return method


def _check_rotation(argument: str, value: object) -> None:
    """Reject a rotation axis that is not one of RX, RY, RZ.

    Args:
        argument (str): Name of the constructor argument, used in the message.
        value (object): The value the caller passed.

    Raises:
        ValueError: If `value` is not `"RX"`, `"RY"` or `"RZ"`.
    """
    if value not in _ROTATIONS:
        raise ValueError(f"{argument} must be RX/RY/RZ, got {value!r}")


# ---- concrete blocks ----


@dataclass
class AngleEncoding:
    """Encode `in_features` data values as single-qubit rotations.

    Feature `i` sets the angle of one rotation on qubit `i`, so the block uses the
    qubits `0` to `in_features - 1`. This matters only when the block is combined
    with others. A circuit has as many qubits as its widest block, and when another
    block in the list needs more qubits, this block leaves the extra qubits
    unchanged.

    The block takes `in_features` inputs and no weights.

    Args:
        in_features: The number of features to encode, one per qubit. It must be
            positive, and a `ValueError` is raised otherwise.
        rotation: The rotation gate: `"RX"`, `"RY"` or `"RZ"`. Default: `"RY"`.
        scale: A constant factor applied to every input angle. It is useful when
            the data is normalised to `[0, 1]` and the circuit needs angles in
            `[0, 2π]`. It must be finite. Default: `1.0`.
    """

    in_features: int
    rotation: GateName = "RY"
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.in_features <= 0:
            raise ValueError(f"in_features must be positive, got {self.in_features}")
        _check_rotation("rotation", self.rotation)
        if not math.isfinite(self.scale):
            raise ValueError(f"scale must be finite, got {self.scale}")

    def required_qubits(self) -> int:
        """Return `in_features`, because the block uses one qubit per feature."""
        return self.in_features

    def build(self, n_qubits: int) -> IRFragment:
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        The qubits with an index of `in_features` or more are left unchanged.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        builder = CircuitBuilder(n_qubits=n_qubits)
        inputs = builder.add_inputs(self.in_features)
        rotate = _rotation_method(builder, self.rotation)
        for qubit in range(self.in_features):
            angle: Parameter = (
                Scale(expression=inputs[qubit], factor=self.scale)
                if self.scale != 1.0
                else inputs[qubit]
            )
            rotate(angle, qubit=qubit)
        circuit = builder.build()
        return IRFragment(
            gates=circuit.gates,
            n_weights=circuit.n_weights,
            n_inputs=circuit.n_inputs,
        )

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {
            "type": "AngleEncoding",
            "in_features": self.in_features,
            "rotation": self.rotation,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AngleEncoding":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `AngleEncoding`.
        """
        return cls(
            in_features=d["in_features"],
            rotation=d["rotation"],
            scale=d["scale"],
        )


@dataclass
class BasicVariational:
    """`depth` trainable layers of single-qubit rotations and entangling gates.

    Each layer applies one trainable rotation to every qubit, then a controlled-NOT
    (CNOT) gate from each qubit to the next one. The block takes
    `depth * n_qubits` weights and no inputs.

    Args:
        depth: The number of layers. It must be positive, and a `ValueError` is
            raised otherwise.
        rotation: The rotation gate of the trainable layer: `"RX"`, `"RY"` or
            `"RZ"`. Default: `"RY"`.
    """

    depth: int
    rotation: GateName = "RY"

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        _check_rotation("rotation", self.rotation)

    def required_qubits(self) -> int:
        """Return `1`, because the block extends to all the qubits of the circuit."""
        return 1  # fills whatever the circuit gives it

    def build(self, n_qubits: int) -> IRFragment:
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        The layers act on every qubit of the circuit.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        builder = CircuitBuilder(n_qubits=n_qubits)
        weights = builder.add_weights(self.depth * n_qubits)
        weight_iter = iter(weights)
        rotate = _rotation_method(builder, self.rotation)
        for _ in range(self.depth):
            for qubit in range(n_qubits):
                rotate(next(weight_iter), qubit=qubit)
            for qubit in range(n_qubits - 1):
                builder.cnot(qubit, qubit + 1)
        circuit = builder.build()
        return IRFragment(
            gates=circuit.gates,
            n_weights=circuit.n_weights,
            n_inputs=circuit.n_inputs,
        )

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {
            "type": "BasicVariational",
            "depth": self.depth,
            "rotation": self.rotation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BasicVariational":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `BasicVariational`.
        """
        return cls(depth=d["depth"], rotation=d["rotation"])


@dataclass
class StrongVariational:
    """`depth` trainable layers of `RZ`, `RY` and `RZ` rotations and entangling gates.

    Each layer applies the rotations `RZ`, `RY` and `RZ` to every qubit, then a
    controlled-NOT (CNOT) gate from each qubit to the next one. The block is
    modelled on PennyLane's `StronglyEntanglingLayers`.

    The gates are added one layer at a time: all the `RZ` rotations, then all the
    `RY` rotations, then all the `RZ` rotations, then the CNOT gates. They are not
    added as three rotations per qubit. The result is the same, because rotations
    on different qubits commute. The layered order lets the backends that support
    it apply each layer of rotations in one fast call, in place of `3 * n_qubits`
    calls on single qubits.

    The block takes `3 * depth * n_qubits` weights and no inputs.

    Args:
        depth: The number of layers. It must be positive, and a `ValueError` is
            raised otherwise.
    """

    depth: int

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")

    def required_qubits(self) -> int:
        """Return `1`, because the block extends to all the qubits of the circuit."""
        return 1

    def build(self, n_qubits: int) -> IRFragment:
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        The layers act on every qubit of the circuit.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        builder = CircuitBuilder(n_qubits=n_qubits)
        weights = builder.add_weights(3 * self.depth * n_qubits)
        weight_iter = iter(weights)
        for _ in range(self.depth):
            # First RZ layer over all qubits, then RY layer, then RZ layer.
            # Weight indices stay sequential: 0..n-1, n..2n-1, 2n..3n-1.
            for qubit in range(n_qubits):
                builder.rz(next(weight_iter), qubit=qubit)
            for qubit in range(n_qubits):
                builder.ry(next(weight_iter), qubit=qubit)
            for qubit in range(n_qubits):
                builder.rz(next(weight_iter), qubit=qubit)
            for qubit in range(n_qubits - 1):
                builder.cnot(qubit, qubit + 1)
        circuit = builder.build()
        return IRFragment(
            gates=circuit.gates,
            n_weights=circuit.n_weights,
            n_inputs=circuit.n_inputs,
        )

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {"type": "StrongVariational", "depth": self.depth}

    @classmethod
    def from_dict(cls, d: dict) -> "StrongVariational":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `StrongVariational`.
        """
        return cls(depth=d["depth"])


@dataclass
class Reuploading:
    """Data reuploading: the input is encoded several times, between trainable layers.

    The block repeats an encoding of the input followed by a variational layer,
    `depth` times. Each encoding can have a trainable scale for each feature and
    repetition.

    The arguments give two patterns:

    1. Standard reuploading (the default; Pérez-Salinas et al. 2020). The same
       input is encoded `depth` times, with new variational weights between the
       encodings.
    2. Trainable feature scaling (`trainable_scale=True`; Schuld et al. 2021).
       Each encoding rotation takes the angle `λ_{d,i} * x_i`, where each pair of
       repetition `d` and feature `i` has its own trainable `λ_{d,i}`.

    Args:
        in_features: The total number of input features. It must be positive.
        depth: The number of times the whole sequence is repeated. It must be
            positive.
        encoding_rotation: The rotation gate of the encoding: `"RX"`, `"RY"` or
            `"RZ"`. Default: `"RY"`.
        variational_rotation: The rotation gate of the variational layer: `"RX"`,
            `"RY"` or `"RZ"`. It is used only when `variational_type="basic"`.
            Default: `"RY"`.
        variational_type: `"basic"`, for one rotation per qubit followed by a ladder
            of controlled-NOT (CNOT) gates, or `"strong"`, for `RZ`, `RY` and `RZ`
            rotations on every qubit followed by a ladder of CNOT gates. Default:
            `"basic"`.
        trainable_scale: If `True`, each encoding rotation gets a trainable scale
            for each repetition and each feature. The CUDA Quantum backend does not
            support it, because it allows only limited arithmetic on parameters.
            Default: `False`.

    Weight count:
        - With `variational_type="basic"` and no trainable scale, the block has
          `depth * in_features` weights.
        - With `variational_type="strong"` and no trainable scale, it has
          `3 * depth * in_features` weights.
        - `trainable_scale=True` adds `depth * in_features` weights.
    """

    in_features: int
    depth: int
    encoding_rotation: GateName = "RY"
    variational_rotation: GateName = "RY"
    variational_type: str = "basic"  # "basic" | "strong"
    trainable_scale: bool = False

    def __post_init__(self):
        if self.in_features <= 0:
            raise ValueError(f"in_features must be positive, got {self.in_features}")
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        _check_rotation("encoding_rotation", self.encoding_rotation)
        _check_rotation("variational_rotation", self.variational_rotation)
        if self.variational_type not in {"basic", "strong"}:
            raise ValueError(
                f"variational_type must be 'basic' or 'strong', "
                f"got {self.variational_type!r}"
            )

    def required_qubits(self) -> int:
        """Return `in_features`: one qubit for each feature."""
        return self.in_features

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {
            "type": "Reuploading",
            "in_features": self.in_features,
            "depth": self.depth,
            "encoding_rotation": self.encoding_rotation,
            "variational_rotation": self.variational_rotation,
            "variational_type": self.variational_type,
            "trainable_scale": self.trainable_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reuploading":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `Reuploading`.
        """
        return cls(
            in_features=d["in_features"],
            depth=d["depth"],
            encoding_rotation=d["encoding_rotation"],
            variational_rotation=d["variational_rotation"],
            variational_type=d["variational_type"],
            trainable_scale=d.get("trainable_scale", False),
        )

    def build(self, n_qubits: int) -> IRFragment:
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        n = self.in_features
        builder = CircuitBuilder(n_qubits=n_qubits)
        inputs = builder.add_inputs(n)

        # Trainable scales: one weight per (depth, feature) pair, in
        # row-major (depth-major) order so resolution is straightforward.
        scale_weights: list = []
        if self.trainable_scale:
            scale_weights = builder.add_weights(self.depth * n)

        encoding_apply = _rotation_method(builder, self.encoding_rotation)
        variational_apply = _rotation_method(builder, self.variational_rotation)

        for layer_idx in range(self.depth):
            for qubit in range(n):
                feature: Parameter = inputs[qubit]
                if self.trainable_scale:
                    angle: Parameter = Multiply(
                        a=scale_weights[layer_idx * n + qubit],
                        b=feature,
                    )
                else:
                    angle = feature
                encoding_apply(angle, qubit=qubit)

            if self.variational_type == "basic":
                var_weights = builder.add_weights(n)
                for qubit in range(n):
                    variational_apply(var_weights[qubit], qubit=qubit)
            else:  # "strong"
                rz_weights = builder.add_weights(n)
                ry_weights = builder.add_weights(n)
                rz2_weights = builder.add_weights(n)
                for qubit in range(n):
                    builder.rz(rz_weights[qubit], qubit=qubit)
                for qubit in range(n):
                    builder.ry(ry_weights[qubit], qubit=qubit)
                for qubit in range(n):
                    builder.rz(rz2_weights[qubit], qubit=qubit)
            for qubit in range(n - 1):
                builder.cnot(qubit, qubit + 1)

        circuit = builder.build()
        return IRFragment(
            gates=circuit.gates,
            n_weights=circuit.n_weights,
            n_inputs=circuit.n_inputs,
        )


@dataclass
class CustomBlock:
    """A block made from the gates of a `CircuitBuilder`.

    `CircuitBuilder.to_block()` creates it. It is useful when you want to write a
    part of a circuit gate by gate and combine it with the standard blocks.

    Args:
        n_qubits: The number of qubits that the builder was given, which is also
            the number of qubits that this block needs. It must be positive.
        gates: The gates that the builder added, in the order in which they are
            applied.
        n_weights: The number of trainable weights that these gates use. It must
            not be negative.
        n_inputs: The number of input features that these gates use. It must not
            be negative.
    """

    n_qubits: int
    gates: tuple[Gate, ...]
    n_weights: int
    n_inputs: int

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {self.n_qubits}")
        if self.n_weights < 0:
            raise ValueError(f"n_weights must be non-negative, got {self.n_weights}")
        if self.n_inputs < 0:
            raise ValueError(f"n_inputs must be non-negative, got {self.n_inputs}")

    def required_qubits(self) -> int:
        """Return `n_qubits`, the number of qubits that the builder was given."""
        return self.n_qubits

    def build(self, n_qubits: int) -> IRFragment:
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        if n_qubits < self.n_qubits:
            raise ValueError(
                f"CustomBlock needs {self.n_qubits} qubits, circuit has {n_qubits}"
            )
        return IRFragment(
            gates=self.gates,
            n_weights=self.n_weights,
            n_inputs=self.n_inputs,
        )


# PennyLane refuses a state whose norm is off by more than this, and it is the
# strictest backend. Matching it means `AmplitudeEncoding` rejects exactly what
# PennyLane would and never sooner. An honestly normalised vector lands within
# 1.1e-16 of unit norm at every width from 2 to 4096, so the margin is ample.
_UNIT_NORM_TOLERANCE = 1e-10


# ---- QAE blocks ----


@dataclass
class AmplitudeEncoding:
    """Encode a classical vector as the amplitudes of a quantum state.

    The vector is padded with zeros to the next power of two. With
    `normalize=True`, the default, it is also rescaled to norm `1`. The number of
    qubits follows from the length of the vector: it is
    `max(1, ceil(log2(len(amplitudes))))`.

    The block adds a single `AmplitudeEncoding` gate. The `data` field of the gate
    holds the prepared amplitudes as a flat tuple, and each backend loads them with
    its own state preparation method.

    Args:
        amplitudes: The vector to load, as an array-like. Its length sets the
            number of qubits, and it is padded with zeros to the next power of
            two. A `ValueError` is raised if the vector is empty, if all its values
            are zero, or if it holds a not-a-number (NaN) or infinite value.
        normalize: If `True`, the vector is rescaled to norm `1` before it is
            loaded. Pass `False` when the vector is already a state. The block
            then stores it unchanged and checks it: a `ValueError` is raised if
            its norm differs from `1` by more than `1e-10`. This check exists
            because, without it, the same mistake would be caught or not
            depending on the backend: the NumPy simulator rescales the vector
            without a warning, and PennyLane refuses it. The same circuit would
            then behave differently on different backends. Default: `True`.

    Example:
        ```python
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        spec = assemble([AmplitudeEncoding([0.6, 0.8], normalize=False)])
        ```
    """

    amplitudes: object  # array-like of real or complex floats
    normalize: bool = True

    def __post_init__(self) -> None:
        raw = (
            list(self.amplitudes.tolist())
            if hasattr(self.amplitudes, "tolist")
            else list(self.amplitudes)
        )
        if not raw:
            raise ValueError("amplitudes must be non-empty")
        if any(not math.isfinite(abs(v)) for v in raw):
            raise ValueError("amplitudes must all be finite; got NaN or infinity")
        n_q = max(1, math.ceil(math.log2(len(raw)))) if len(raw) > 1 else 1
        target_len = 2**n_q
        raw = raw + [0.0] * (target_len - len(raw))
        norm = math.sqrt(sum(abs(v) ** 2 for v in raw))
        if self.normalize:
            if norm == 0.0:
                raise ValueError("amplitude vector is all-zero; cannot normalize")
            raw = [v / norm for v in raw]
        elif abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
            # `normalize=False` is the caller promising the vector is already a
            # state. Left to the backends, the promise would be kept or broken
            # by whichever one ran: the numpy simulator rescales silently,
            # PennyLane refuses. Checking here means the same answer
            # everywhere, and the complaint arrives where the mistake is.
            raise ValueError(
                f"normalize=False says the vector is already a state, but its "
                f"norm is {norm!r}, not 1. Pass normalize=True to have it "
                f"rescaled, or divide by the norm yourself."
            )
        object.__setattr__(self, "_amps", tuple(raw))
        object.__setattr__(self, "_n_qubits", n_q)

    def required_qubits(self) -> int:
        """Return the number of qubits that the padded vector needs.

        It is `ceil(log2(len(amplitudes)))`, and at least `1`.
        """
        return self._n_qubits  # type: ignore[attr-defined]

    def build(self, n_qubits: int) -> "IRFragment":
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        n_q: int = self._n_qubits  # type: ignore[attr-defined]
        if n_qubits < n_q:
            raise ValueError(
                f"AmplitudeEncoding needs {n_q} qubits, circuit has {n_qubits}"
            )
        gate = Gate(
            name="AmplitudeEncoding",
            qubits=tuple(range(n_q)),
            data=(_array_to_data(self._amps),),  # type: ignore[attr-defined]
        )
        return IRFragment(gates=(gate,), n_weights=0, n_inputs=0)

    def to_dict(self) -> dict:
        # Serialize _amps (the frozen, processed state used by build()) so that
        # from_dict() restores exactly what build() would produce, regardless of
        # any mutation to the original amplitudes array after construction.
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        raw = list(self._amps)  # type: ignore[attr-defined]
        # Encode complex values as [re, im] pairs; plain floats stay as-is.
        encoded = [[v.real, v.imag] if isinstance(v, complex) else float(v) for v in raw]
        return {
            "type": "AmplitudeEncoding",
            "amplitudes": encoded,
            "normalize": self.normalize,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AmplitudeEncoding":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `AmplitudeEncoding`.
        """
        raw = d["amplitudes"]
        amps = [complex(v[0], v[1]) if isinstance(v, list) else float(v) for v in raw]
        return cls(amplitudes=amps, normalize=d["normalize"])


@dataclass
class S0:
    """Reflection about the all-zero state `|0…0⟩`, built as `X^⊗n → MCZ → X^⊗n`.

    The block applies `X` to every qubit, then a multi-controlled `Z` gate (`MCZ`)
    on all of them, then `X` to every qubit again. It acts on the qubits `0` to
    `n_qubits - 1`, which are all the qubits of the circuit unless another block in
    the list needs more. This reflection is the `S₀` part of the Grover operator
    used by iterative quantum amplitude estimation (IQAE). The
    [`Grover`][eigenkit.core.blocks.Grover] block adds the same gates itself.

    Args:
        n_qubits: The number of qubits of the reflection. Every one of them takes
            part. It must be positive.

    Example:
        ```python
        from eigenkit.core.blocks import S0, assemble

        spec = assemble([S0(n_qubits=3)])
        ```
    """

    n_qubits: int

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {self.n_qubits}")

    def required_qubits(self) -> int:
        """Return `n_qubits`, the number of qubits that the reflection acts on."""
        return self.n_qubits

    def build(self, n_qubits: int) -> "IRFragment":
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        n = self.n_qubits
        if n_qubits < n:
            raise ValueError(f"S0 needs {n} qubits, circuit has {n_qubits}")
        qb = CircuitBuilder(n_qubits=n)
        for q in range(n):
            qb.x(q)
        if n == 1:
            # X → Z → X reflects |0⟩ → -|0⟩, |1⟩ → |1⟩ (same as MCZ with no controls)
            qb.z(0)
        else:
            qb.mcz(*range(n))
        for q in range(n):
            qb.x(q)
        spec = qb.build()
        return IRFragment(gates=spec.gates, n_weights=0, n_inputs=0)

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {"type": "S0", "n_qubits": self.n_qubits}

    @classmethod
    def from_dict(cls, d: dict) -> "S0":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `S0`.
        """
        return cls(n_qubits=d["n_qubits"])


@dataclass
class QubitUnitary:
    """Apply any unitary matrix to the given qubits.

    The matrix is stored in the `data` field of the gate as a nested tuple, so that
    the `Gate` stays hashable.

    Args:
        unitary: A `2^n × 2^n` matrix, as an array-like, where `n` is
            `len(wires)`. It is checked to be unitary when the block is created.
        wires: The target qubits, in the order that the matrix expects.

    Example:
        ```python
        import numpy as np
        from eigenkit.core.blocks import QubitUnitary, assemble

        U = np.array([[0, 1], [1, 0]])  # the Pauli X matrix
        spec = assemble([QubitUnitary(U, wires=(0,))])
        ```
    """

    unitary: object  # 2-D array-like
    wires: tuple[int, ...]

    def __post_init__(self) -> None:
        wires = tuple(self.wires)
        object.__setattr__(self, "wires", wires)
        if len(wires) == 0:
            raise ValueError("QubitUnitary: wires must not be empty")
        if any(q < 0 for q in wires):
            raise ValueError(
                f"QubitUnitary: wire indices must be non-negative, got {wires}"
            )
        if len(set(wires)) != len(wires):
            raise ValueError(f"QubitUnitary: wire indices must be distinct, got {wires}")
        object.__setattr__(
            self,
            "_matrix",
            _validate_unitary(self.unitary, len(wires), "QubitUnitary"),
        )

    def required_qubits(self) -> int:
        """Return `max(wires) + 1`, one more than the highest target qubit."""
        return max(self.wires) + 1

    def build(self, n_qubits: int) -> "IRFragment":
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        if n_qubits < self.required_qubits():
            raise ValueError(
                f"QubitUnitary on wires {self.wires} needs "
                f"{self.required_qubits()} qubits, circuit has {n_qubits}"
            )
        gate = Gate(
            name="QubitUnitary",
            qubits=self.wires,
            data=(self._matrix,),  # type: ignore[attr-defined]
        )
        return IRFragment(gates=(gate,), n_weights=0, n_inputs=0)

    def to_dict(self) -> dict:
        # _matrix is a nested tuple of complex numbers (stored by __post_init__
        # via _array_to_data). Encode each value as [re, im] for JSON safety.
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        mat = self._matrix  # type: ignore[attr-defined]
        encoded = [
            [[v.real, v.imag] if isinstance(v, complex) else [float(v), 0.0] for v in row]
            for row in mat
        ]
        return {
            "type": "QubitUnitary",
            "unitary": encoded,
            "wires": list(self.wires),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QubitUnitary":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `QubitUnitary`.
        """
        matrix = [[complex(v[0], v[1]) for v in row] for row in d["unitary"]]
        return cls(unitary=matrix, wires=tuple(d["wires"]))


@dataclass
class Grover:
    r"""The Grover operator `Q = A · S₀ · A† · S_ψ₁` used for amplitude estimation.

    This is the operator that iterative quantum amplitude estimation (IQAE) uses.
    The block builds it from a state-preparation oracle `A`, given as a
    `CircuitSpec`, and from these standard parts:

    - **`S_ψ₁`** flips the phase of the ancilla qubit, which is the last qubit. It
      is the gate `Z(n-1)`.
    - **`A†`** is the adjoint of the oracle. It uses the same weights and inputs
      as `A`.
    - **`S₀`** is the reflection about the all-zero state:
      `X^⊗n → MCZ → X^⊗n`, where `MCZ` is a multi-controlled `Z` gate.
    - **`A`** is the oracle.

    The gates are applied to the state in this order, from left to right:

    ```text
    Z(ancilla) → A† → S₀ → A
    ```

    `A` and `A†` share their weights and inputs, so
    `fragment.n_weights == a_circuit.n_weights`: the count is not doubled.

    Args:
        a_circuit: The state-preparation oracle `A`, as a `CircuitSpec` with at
            least one qubit. Its last qubit is used as the ancilla.

    Example:
        ```python
        from eigenkit.core.blocks import Grover, assemble
        from eigenkit.core.builder import CircuitBuilder

        qb = CircuitBuilder(n_qubits=2)
        qb.h(0).cnot(0, 1)
        spec = assemble([Grover(a_circuit=qb.build())])
        ```
    """

    a_circuit: CircuitSpec

    def __post_init__(self) -> None:
        if not isinstance(self.a_circuit, CircuitSpec):
            raise TypeError(
                f"Grover: a_circuit must be a CircuitSpec, "
                f"got {type(self.a_circuit).__name__}"
            )
        if self.a_circuit.n_qubits <= 0:
            raise ValueError(
                f"Grover: a_circuit must have at least one qubit, "
                f"got {self.a_circuit.n_qubits}"
            )

    def required_qubits(self) -> int:
        """Return the number of qubits of the oracle.

        The operator adds no qubits of its own.
        """
        return self.a_circuit.n_qubits

    def build(self, n_qubits: int) -> "IRFragment":
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        The oracle `A` and its adjoint `A†` share their weights and inputs, so these
        are not counted twice.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        a = self.a_circuit
        if n_qubits < a.n_qubits:
            raise ValueError(f"Grover needs {a.n_qubits} qubits, circuit has {n_qubits}")
        ancilla = a.n_qubits - 1

        # S_ψ₁: phase flip on the ancilla qubit
        s_psi1 = (Gate(name="Z", qubits=(ancilla,)),)

        # A†: reversed gates with negated rotation parameters
        a_dag = adjoint(a)

        # S₀: X^⊗n → MCZ → X^⊗n  (n=1: Z instead of MCZ)
        s0_qb = CircuitBuilder(n_qubits=a.n_qubits)
        for q in range(a.n_qubits):
            s0_qb.x(q)
        if a.n_qubits == 1:
            s0_qb.z(0)
        else:
            s0_qb.mcz(*range(a.n_qubits))
        for q in range(a.n_qubits):
            s0_qb.x(q)
        s0_gates = s0_qb.build().gates

        # Compose: S_ψ₁ → A† → S₀ → A
        all_gates = s_psi1 + a_dag.gates + s0_gates + a.gates
        return IRFragment(
            gates=all_gates,
            n_weights=a.n_weights,
            n_inputs=a.n_inputs,
        )

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {
            "type": "Grover",
            "a_circuit": _circuit_spec_to_json(self.a_circuit),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Grover":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `Grover`.
        """
        return cls(a_circuit=_circuit_spec_from_json(d["a_circuit"]))


# ---- HHL / QPE infrastructure blocks ----


def _unitary_tolerance(unitary: object) -> float:
    """Per-entry unitarity tolerance keyed on the *input* dtype.

    Detected before any conversion discards the dtype: float32/complex64
    matrices carry eps ~1.2e-7 and get 1e-5; 64-bit dtypes keep the strict
    1e-12.  Nested lists/tuples (e.g. ``U.tolist()``, which is exactly what
    oracle-conversion code produces) have lost their dtype, so they must get
    the float32 tolerance too — the values may originate from 32-bit data
    (measured complex64 U†U deviations reach ~1e-7), while genuinely
    non-unitary input is typically off by ≥1e-3.
    """
    import numpy as np  # lazy import — blocks otherwise stay numpy-free

    dtype = getattr(unitary, "dtype", None)
    if dtype is None:
        return 1e-5
    np_dtype = np.dtype(dtype)
    component_bits = np_dtype.itemsize * 8
    if np.issubdtype(np_dtype, np.complexfloating):
        component_bits //= 2
    return 1e-5 if component_bits <= 32 else 1e-12


def _validate_unitary(unitary: object, n_wires: int, context: str):
    """Validate that *unitary* is a 2^n_wires square unitary matrix.

    Shared by [`QubitUnitary`][eigenkit.core.blocks.QubitUnitary] and
    [`ControlledUnitaryBlock`][eigenkit.core.blocks.ControlledUnitaryBlock] so
    the two checks cannot drift apart, for example on the complex64 tolerance.

    Returns the matrix converted to nested-tuple gate data
    (via ``_array_to_data``), ready for the IR gate's ``data`` field.

    Raises:
        ValueError: On non-numeric input, wrong dimensionality, wrong shape,
            or U†U deviating from identity beyond the dtype-keyed tolerance.
    """
    import numpy as np  # lazy import — blocks otherwise stay numpy-free

    tol = _unitary_tolerance(unitary)
    try:
        arr = np.asarray(unitary, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: unitary must be a numeric 2-D matrix") from exc
    if arr.ndim != 2:
        raise ValueError(
            f"{context}: unitary must be a 2-D matrix, got a {arr.ndim}-D sequence"
        )
    expected = 2**n_wires
    if arr.shape != (expected, expected):
        raise ValueError(
            f"{context}: expected a {expected}×{expected} matrix "
            f"for {n_wires} qubit(s), "
            f"got shape {arr.shape[0]}×{arr.shape[1]}"
        )
    deviation = float(np.abs(arr.conj().T @ arr - np.eye(expected)).max())
    if deviation > expected * tol:
        raise ValueError(
            f"{context}: matrix is not unitary (U†U ≠ I); "
            f"max |U†U − I| = {deviation:.3e} exceeds tolerance {expected * tol:.1e}"
        )
    return _array_to_data(arr.tolist())


@dataclass
class ControlledUnitaryBlock:
    """Apply a unitary matrix `U` to `target_wires`, controlled by `control_wires`.

    Args:
        unitary: A `2^n_target × 2^n_target` matrix, as an array-like, where
            `n_target` is the number of target qubits.
        control_wires: The qubits that control the gate.
        target_wires: The qubits that the matrix acts on.
        control_values: The bit value that activates each control, one for each
            entry of `control_wires`. Default: `1` for every control.

    Example:
        ```python
        import numpy as np
        import scipy.linalg
        from eigenkit.core.blocks import ControlledUnitaryBlock, assemble

        A = np.diag([1.0, 2.0, 3.0, 4.0])  # a Hermitian matrix on two qubits
        t = 0.5
        U = scipy.linalg.expm(1j * A * t)
        block = ControlledUnitaryBlock(U, control_wires=(0,), target_wires=(1, 2))
        spec = assemble([block])
        ```
    """

    unitary: object
    control_wires: tuple[int, ...]
    target_wires: tuple[int, ...]
    control_values: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_wires", tuple(self.control_wires))
        object.__setattr__(self, "target_wires", tuple(self.target_wires))

        if len(self.control_wires) == 0:
            raise ValueError("ControlledUnitaryBlock: control_wires must not be empty")
        if len(self.target_wires) == 0:
            raise ValueError("ControlledUnitaryBlock: target_wires must not be empty")

        all_wires = self.control_wires + self.target_wires
        if any(q < 0 for q in all_wires):
            raise ValueError(
                f"ControlledUnitaryBlock: wire indices must be non-negative, "
                f"got controls {self.control_wires}, targets {self.target_wires}"
            )
        if len(set(self.control_wires)) != len(self.control_wires):
            raise ValueError(
                f"ControlledUnitaryBlock: control wires must be distinct, "
                f"got {self.control_wires}"
            )
        if len(set(self.target_wires)) != len(self.target_wires):
            raise ValueError(
                f"ControlledUnitaryBlock: target wires must be distinct, "
                f"got {self.target_wires}"
            )

        overlap = set(self.control_wires) & set(self.target_wires)
        if overlap:
            raise ValueError(
                f"ControlledUnitaryBlock: control_wires and target_wires overlap "
                f"on qubit(s) {sorted(overlap)}"
            )

        cv = tuple(self.control_values) if self.control_values is not None else None
        if cv is not None:
            if len(cv) != len(self.control_wires):
                raise ValueError(
                    f"ControlledUnitaryBlock: control_values has length {len(cv)} "
                    f"but control_wires has {len(self.control_wires)} element(s)"
                )
            bad = [v for v in cv if v not in (0, 1)]
            if bad:
                raise ValueError(
                    f"ControlledUnitaryBlock: control_values must contain only 0 or 1, "
                    f"got {bad}"
                )

        object.__setattr__(
            self,
            "_matrix",
            _validate_unitary(
                self.unitary, len(self.target_wires), "ControlledUnitaryBlock"
            ),
        )
        object.__setattr__(self, "control_values", cv)

    def required_qubits(self) -> int:
        """Return one more than the highest control or target qubit."""
        return max(*self.control_wires, *self.target_wires) + 1

    def build(self, n_qubits: int) -> "IRFragment":
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        if n_qubits < self.required_qubits():
            raise ValueError(
                f"ControlledUnitaryBlock on wires {self.control_wires}→{self.target_wires} "
                f"needs {self.required_qubits()} qubits, circuit has {n_qubits}"
            )
        gate = Gate(
            name="ControlledQubitUnitary",
            qubits=tuple(self.control_wires) + tuple(self.target_wires),
            data=(self._matrix, self.control_values),  # type: ignore[attr-defined]
        )
        return IRFragment(gates=(gate,), n_weights=0, n_inputs=0)

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        mat = self._matrix  # type: ignore[attr-defined]
        encoded = [
            [[v.real, v.imag] if isinstance(v, complex) else [float(v), 0.0] for v in row]
            for row in mat
        ]
        return {
            "type": "ControlledUnitaryBlock",
            "unitary": encoded,
            "control_wires": list(self.control_wires),
            "target_wires": list(self.target_wires),
            "control_values": list(self.control_values)
            if self.control_values is not None
            else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ControlledUnitaryBlock":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `ControlledUnitaryBlock`.
        """
        matrix = [[complex(v[0], v[1]) for v in row] for row in d["unitary"]]
        cv = tuple(d["control_values"]) if d.get("control_values") is not None else None
        return cls(
            unitary=matrix,
            control_wires=tuple(d["control_wires"]),
            target_wires=tuple(d["target_wires"]),
            control_values=cv,
        )


@dataclass
class QFTBlock:
    """Quantum Fourier transform (QFT) on `n_qubits` qubits.

    The transform is built from `H`, `CPHASE` and `SWAP` gates only, which are all
    gates that the intermediate representation (IR) of the circuit defines.

    Args:
        n_qubits: The number of qubits of the transform. It must be positive.
        inverse: If `True`, the block builds the inverse QFT, which quantum phase
            estimation (QPE) applies as its last step. Default: `False`.

    Example:
        ```python
        from eigenkit.core.blocks import AmplitudeEncoding, QFTBlock, assemble

        b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # eight values for three qubits
        spec = assemble([AmplitudeEncoding(b), QFTBlock(n_qubits=3, inverse=True)])
        ```
    """

    n_qubits: int
    inverse: bool = False

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {self.n_qubits}")

    def required_qubits(self) -> int:
        """Return `n_qubits`, the number of qubits that the transform acts on."""
        return self.n_qubits

    def build(self, n_qubits: int) -> IRFragment:
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        qb = CircuitBuilder(n_qubits=n_qubits)
        if self.inverse:
            _build_iqft(qb, self.n_qubits)
        else:
            _build_qft(qb, self.n_qubits)
        spec = qb.build()
        return IRFragment(gates=spec.gates, n_weights=0, n_inputs=0)

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {"type": "QFTBlock", "n_qubits": self.n_qubits, "inverse": self.inverse}

    @classmethod
    def from_dict(cls, d: dict) -> "QFTBlock":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `QFTBlock`.
        """
        return cls(n_qubits=d["n_qubits"], inverse=d.get("inverse", False))


@dataclass
class QPEBlock:
    """Quantum phase estimation (QPE) of a unitary oracle.

    The block applies `H` to every clock qubit, then the controlled powers
    `U^{2^k}` of the oracle, then the inverse quantum Fourier transform (`QFT†`):
    `H^⊗n_clock → controlled-U^{2^k} → QFT†`. Set `inverse=True` for the adjoint:
    `QFT → controlled-U†^{2^k} → H^⊗n_clock`.

    The oracle must be a `CircuitSpec` whose unitary can be read as a dense matrix.
    In practice, this means a circuit made of a single
    [`QubitUnitary`][eigenkit.core.blocks.QubitUnitary] gate. Parametrised
    oracles made of several gates are not supported. To use an
    [`AmplitudeEncoding`][eigenkit.core.blocks.AmplitudeEncoding] or a
    [`ControlledUnitaryBlock`][eigenkit.core.blocks.ControlledUnitaryBlock] as an
    oracle, first compute its unitary matrix and wrap it in a
    [`QubitUnitary`][eigenkit.core.blocks.QubitUnitary].

    The qubits are laid out as follows, with indices local to the block:

    ```text
    [clock(0 … n_clock-1) | state(n_clock … n_clock+n_state-1)]
    ```

    The caller is responsible for mapping these local indices to the qubits of
    the full circuit, for example to skip the ancilla qubit of the
    Harrow-Hassidim-Lloyd (HHL) algorithm.

    Args:
        oracle_circuit: A `CircuitSpec` whose single `QubitUnitary` gate defines
            the unitary `U` that acts on the state qubits. The single-gate
            requirement is checked when the block is built, and the type when the
            block is created.
        n_clock: The number of clock qubits, which sets the resolution of QPE. It
            must be at least `1`.
        inverse: If `True`, the block builds the inverse QPE, which undoes the
            phase estimation. Default: `False`.

    Example:
        ```python
        import numpy as np
        import scipy.linalg
        from eigenkit.core.blocks import QPEBlock, QubitUnitary, assemble

        A = np.array([[2, 0], [0, 1]], dtype=float)
        U = scipy.linalg.expm(1j * A * np.pi)
        oracle = assemble([QubitUnitary(U, wires=(0,))])
        spec = assemble([QPEBlock(oracle, n_clock=3)])
        ```
    """

    oracle_circuit: CircuitSpec
    n_clock: int
    inverse: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.oracle_circuit, CircuitSpec):
            raise TypeError(
                f"QPEBlock: oracle_circuit must be a CircuitSpec, "
                f"got {type(self.oracle_circuit).__name__}"
            )
        if self.n_clock < 1:
            raise ValueError(f"n_clock must be >= 1, got {self.n_clock}")

    def required_qubits(self) -> int:
        """Return `n_clock` plus the number of qubits of the oracle.

        The clock qubits come first, then the state qubits.
        """
        return self.n_clock + self.oracle_circuit.n_qubits

    def _extract_unitary(self):
        """Return the dense unitary matrix from a single-QubitUnitary oracle."""

        gates = self.oracle_circuit.gates
        if len(gates) != 1 or gates[0].name != "QubitUnitary":
            raise ValueError(
                "QPEBlock requires an oracle_circuit with exactly one "
                "QubitUnitary gate. Pre-compute the unitary matrix and wrap "
                "it with QubitUnitary before passing to QPEBlock."
            )
        return _data_to_array(gates[0].data[0])

    def build(self, n_qubits: int) -> "IRFragment":
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        The qubit indices are local to the block, and the caller maps them to the
        qubits of the full circuit.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. Its weight and input indices
                start at `0`, and they are shifted when the blocks are combined into a
                circuit.
        """
        import numpy as np

        if n_qubits < self.required_qubits():
            raise ValueError(
                f"QPEBlock needs {self.required_qubits()} qubits "
                f"(n_clock={self.n_clock} + oracle={self.oracle_circuit.n_qubits}), "
                f"circuit has {n_qubits}"
            )

        U = self._extract_unitary()
        n_state = self.oracle_circuit.n_qubits
        n_cl = self.n_clock
        clock = tuple(range(n_cl))
        state = tuple(range(n_cl, n_cl + n_state))

        # Precompute U^{2^k} for k = 0 … n_clock-1.
        # Clock qubit k controls U^{2^(n_clock-1-k)} (big-endian QPE convention).
        # Re-orthogonalise each power via the polar decomposition (U = W·P, take W)
        # to suppress floating-point drift that accumulates when k is large (e.g.
        # k=13 → U^8192 via repeated squaring can exceed the unitarity tolerance).
        def _nearest_unitary(M: np.ndarray) -> np.ndarray:
            W, _, Vh = np.linalg.svd(M)
            return W @ Vh

        U_pows = [_nearest_unitary(np.linalg.matrix_power(U, 2**k)) for k in range(n_cl)]

        gates: list[Gate] = []

        if not self.inverse:
            # Forward QPE: H⊗n → C-U^{2^k} → IQFT
            for w in clock:
                gates.append(Gate(name="H", qubits=(w,)))

            for k, ctrl_w in enumerate(clock):
                U_k = U_pows[n_cl - 1 - k]
                frag = ControlledUnitaryBlock(
                    unitary=U_k,
                    control_wires=(ctrl_w,),
                    target_wires=state,
                ).build(n_qubits)
                gates.extend(frag.gates)

            # IQFT on clock qubits — local indices, no remapping needed here
            # because clock occupies 0..n_cl-1 in this fragment.
            iqft_frag = QFTBlock(n_qubits=n_cl, inverse=True).build(n_cl)
            # Gates are frozen, so the fragment's can be shared rather than
            # rebuilt field by field — a rebuild that names only three of the
            # five fields drops `data` and `adjoint`.
            gates.extend(iqft_frag.gates)

        else:
            # Inverse QPE: QFT → C-U†^{2^k} → H⊗n
            qft_frag = QFTBlock(n_qubits=n_cl).build(n_cl)
            gates.extend(qft_frag.gates)

            for k, ctrl_w in enumerate(clock):
                U_k_dag = U_pows[n_cl - 1 - k].conj().T
                frag = ControlledUnitaryBlock(
                    unitary=U_k_dag,
                    control_wires=(ctrl_w,),
                    target_wires=state,
                ).build(n_qubits)
                gates.extend(frag.gates)

            for w in clock:
                gates.append(Gate(name="H", qubits=(w,)))

        return IRFragment(gates=tuple(gates), n_weights=0, n_inputs=0)

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {
            "type": "QPEBlock",
            "oracle_circuit": _circuit_spec_to_json(self.oracle_circuit),
            "n_clock": self.n_clock,
            "inverse": self.inverse,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QPEBlock":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `QPEBlock`.
        """
        return cls(
            oracle_circuit=_circuit_spec_from_json(d["oracle_circuit"]),
            n_clock=d["n_clock"],
            inverse=d.get("inverse", False),
        )


@dataclass
class TrotterBlock:
    """First-order or second-order Trotter approximation of `e^{iAt}`.

    The block builds the time evolution under a Hermitian matrix
    `A = Σ_k c_k P_k`, given as `(coeff, pauli_string)` terms (for example from
    `pauli_decompose()`). It uses elementary gates only (`H`, `RX`, controlled-NOT
    (CNOT) and `RZ`), and no dense `QubitUnitary`. It replaces applying
    `scipy.linalg.expm(1j*A*t)` as a single opaque unitary, and it builds the
    evolution as hardware would run it.

    Each term `e^{i·c·P·τ}` is built as follows. Every factor of `P` that is not
    `Z` is rotated into the eigenbasis of `Z` (with `H` for `X`, and `RX(+π/2)` for
    `Y`). A ladder of CNOT gates collects the parity of the active qubits on the
    last one, a single `RZ(-2cτ)` is applied there, and then the ladder and the
    change of basis are undone. With the convention `RZ(θ) = e^{-iθZ/2}` of the
    library, this gives `e^{+iAt}` up to a global phase, which matches the sign of
    the reference Harrow-Hassidim-Lloyd (HHL) solver. The identity term is only a
    global phase, so it is skipped: it has no observable effect here, and it
    becomes a phase on the clock qubit in the controlled version used for quantum
    phase estimation (QPE).

    Args:
        pauli_terms: A sequence of `(coeff, pauli_string)` pairs. All the strings
            have the same length `n`, which is the number of qubits. Each `coeff`
            must be real up to numerical tolerance, because `A` is Hermitian.
        time: The evolution time `t`.
        steps: The number of Trotter steps. The approximation error decreases as
            `steps` increases. Default: `1`.
        order: `1` (Lie-Trotter) or `2` (symmetric Strang splitting, which has a
            smaller error for the same `steps`). Default: `1`.

    Example:
        ```python
        import numpy as np
        from eigenkit.algorithms._hamiltonian_sim import pauli_decompose
        from eigenkit.core.blocks import TrotterBlock, assemble

        A = np.array([[1.0, 0.5], [0.5, -1.0]])  # a Hermitian matrix on one qubit
        terms = pauli_decompose(A)
        spec = assemble([TrotterBlock(terms, time=1.0, steps=20, order=2)])
        ```
    """

    pauli_terms: tuple[tuple[float, str], ...]
    time: float
    steps: int = 1
    order: int = 1

    def __post_init__(self) -> None:
        terms = tuple(self.pauli_terms)
        if not terms:
            raise ValueError("TrotterBlock: pauli_terms must be non-empty")
        n = len(terms[0][1])
        if n == 0:
            raise ValueError("TrotterBlock: Pauli strings must be non-empty")
        coefficients: list[complex] = []
        for coeff, pauli_string in terms:
            if len(pauli_string) != n:
                raise ValueError(
                    "TrotterBlock: all Pauli strings must have the same length; "
                    f"got {len(pauli_string)} and {n}"
                )
            if any(ch not in "IXYZ" for ch in pauli_string):
                raise ValueError(
                    f"TrotterBlock: invalid Pauli string {pauli_string!r}; "
                    "expected characters from I, X, Y, Z"
                )
            coefficients.append(complex(coeff))

        # Relative to the whole operator, and to nothing else.  A Hermitian A of
        # magnitude 1e10 leaves float residue ~1e-6 in the imaginary parts, so an
        # absolute test would reject it — but that residue comes from the largest term,
        # not from the one carrying it, so a per-coefficient test would demand ~1e-17 of
        # a 1e-9 term sitting in that same operator.  The largest magnitude is what sets
        # the noise floor.  Anchoring the tolerance at 1.0 instead — `max(1.0, …)` —
        # would make the test vacuous for an operator whose every coefficient is at or
        # below 1e-8: a blatantly non-Hermitian one would pass and have its imaginary
        # part dropped a few lines further down.
        scale = max((abs(c) for c in coefficients), default=0.0)
        for c, (coeff, pauli_string) in zip(coefficients, terms):
            if abs(c.imag) > 1e-8 * scale:
                raise ValueError(
                    f"TrotterBlock: coefficient {coeff} for {pauli_string!r} is "
                    "not real; A must be Hermitian."
                )
        clean = [
            (c.real, pauli_string) for c, (_, pauli_string) in zip(coefficients, terms)
        ]
        if self.steps < 1:
            raise ValueError(f"TrotterBlock: steps must be >= 1, got {self.steps}")
        if self.order not in (1, 2):
            raise ValueError(f"TrotterBlock: order must be 1 or 2, got {self.order}")
        object.__setattr__(self, "pauli_terms", tuple(clean))
        object.__setattr__(self, "_n_qubits", n)

    def required_qubits(self) -> int:
        """Return the length of the Pauli strings, one qubit for each factor."""
        return self._n_qubits  # type: ignore[attr-defined]

    def build(self, n_qubits: int) -> IRFragment:
        """Build the Trotter product for a circuit with `n_qubits` qubits.

        The qubit indices are local to the block, and the caller maps them to the
        qubits of the full circuit.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. It holds elementary gates only,
                and no weights.
        """
        n = self._n_qubits  # type: ignore[attr-defined]
        if n_qubits < n:
            raise ValueError(f"TrotterBlock needs {n} qubits, circuit has {n_qubits}")
        qb = CircuitBuilder(n_qubits=n_qubits)
        tau = self.time / self.steps
        for _ in range(self.steps):
            if self.order == 1:
                for coeff, pauli_string in self.pauli_terms:
                    _append_pauli_exp(qb, coeff, pauli_string, tau)
            else:  # symmetric second-order: forward half-steps, then reversed
                for coeff, pauli_string in self.pauli_terms:
                    _append_pauli_exp(qb, coeff, pauli_string, tau / 2.0)
                for coeff, pauli_string in reversed(self.pauli_terms):
                    _append_pauli_exp(qb, coeff, pauli_string, tau / 2.0)
        spec = qb.build()
        return IRFragment(gates=spec.gates, n_weights=0, n_inputs=0)

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {
            "type": "TrotterBlock",
            "pauli_terms": [
                [coeff, pauli_string] for coeff, pauli_string in self.pauli_terms
            ],
            "time": self.time,
            "steps": self.steps,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrotterBlock":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `TrotterBlock`.
        """
        return cls(
            pauli_terms=tuple((c, s) for c, s in d["pauli_terms"]),
            time=d["time"],
            steps=d.get("steps", 1),
            order=d.get("order", 1),
        )


def _append_pauli_exp(
    qb: CircuitBuilder,
    coeff: float,
    pauli_string: str,
    dt: float,
    *,
    control: int | None = None,
    state_wires: tuple[int, ...] | None = None,
) -> None:
    """Append ``e^{i·coeff·P·dt}`` for one Pauli string, optionally controlled.

    Non-Z factors are rotated into the Z basis (H for X, ``RX(+pi/2)`` for Y, the
    same convention as `CircuitBuilder.pauli_exponential()`, so no caller needs
    the discrete S/Sdg gates), a CNOT ladder accumulates their parity onto the last
    active qubit, and the rotation goes there.  ``RZ(θ) = e^{-iθZ/2}``, so the angle
    is ``-2·coeff·dt`` to realise ``e^{+i·coeff·Z·dt}``.

    With ``control`` given, only that rotation is controlled: the basis change and
    the ladder conjugate it and cancel to identity when the control is |0⟩, so the
    controlled RZ expands to ``RZ(θ/2)·CNOT·RZ(-θ/2)·CNOT`` and no CRZ primitive is
    needed.  ``state_wires`` maps Pauli positions onto circuit qubits.

    An all-identity string is a global phase: unobservable on its own, and a
    ``PhaseShift(coeff·dt)`` on the control qubit when controlled.
    """
    positions = state_wires if state_wires is not None else range(len(pauli_string))
    active = [(positions[i], ch) for i, ch in enumerate(pauli_string) if ch != "I"]
    if not active:
        if control is not None:
            qb.phase_shift(coeff * dt, qubit=control)
        return
    for w, ch in active:
        if ch == "X":
            qb.h(w)
        elif ch == "Y":
            qb.rx(math.pi / 2, w)
    wires = [w for w, _ in active]
    for a, b in zip(wires, wires[1:]):
        qb.cnot(a, b)
    target = wires[-1]
    theta = -2.0 * coeff * dt
    if control is None:
        qb.rz(theta, target)
    else:
        qb.rz(theta / 2.0, qubit=target)
        qb.cnot(control, target)
        qb.rz(-theta / 2.0, qubit=target)
        qb.cnot(control, target)
    for a, b in zip(wires[:-1][::-1], wires[1:][::-1]):
        qb.cnot(a, b)
    for w, ch in active:
        if ch == "X":
            qb.h(w)
        elif ch == "Y":
            qb.rx(-math.pi / 2, w)


def _append_controlled_trotter(
    qb: CircuitBuilder,
    control: int,
    pauli_terms: tuple[tuple[float, str], ...],
    tau: float,
    n_steps: int,
    order: int,
    state_wires: tuple[int, ...],
    reverse: bool = False,
) -> None:
    """Append ``controlled-e^{iA·(n_steps·tau)}`` as a Trotter product.

    ``reverse=True`` emits the factors in reverse order, which is what makes the
    result the *adjoint* and not merely the negative-time product.  At first order
    the step is ``W(τ) = e^{X_L}···e^{X_1}``, so its inverse is ``e^{−X_1}···e^{−X_L}``;
    negating ``tau`` alone gives ``e^{−X_L}···e^{−X_1}``, a different unitary.  The
    second-order product is a palindrome whose negative-time build is already its own
    adjoint, so ``reverse`` leaves it alone: reversing it would yield
    ``G_L…G_1G_1…G_L`` instead of ``G_1…G_LG_L…G_1`` and break an exact uncompute.
    """

    def emit(coeff: float, pauli_string: str, dt: float) -> None:
        _append_pauli_exp(
            qb, coeff, pauli_string, dt, control=control, state_wires=state_wires
        )

    first_order_terms = tuple(reversed(pauli_terms)) if reverse else pauli_terms
    for _ in range(n_steps):
        if order == 1:
            for coeff, pauli_string in first_order_terms:
                emit(coeff, pauli_string, tau)
        else:  # symmetric second order — already self-adjoint at −tau
            for coeff, pauli_string in pauli_terms:
                emit(coeff, pauli_string, tau / 2.0)
            for coeff, pauli_string in reversed(pauli_terms):
                emit(coeff, pauli_string, tau / 2.0)


@dataclass
class HamiltonianQPEBlock:
    """Quantum phase estimation (QPE) of `e^{iAt}`, built with Trotter products.

    This block replaces [`QPEBlock`][eigenkit.core.blocks.QPEBlock] in the clock
    register of the Harrow-Hassidim-Lloyd (HHL) solver with `mode="honest"`, where
    the circuit is built as hardware would run it. It turns the Hermitian matrix
    `A = Σ_k c_k P_k` into elementary gates with Trotter products. It does not
    exponentiate a dense oracle and raise it to powers with `matrix_power`: both of
    these cost `O(N³)` for an `N × N` matrix, and neither gives a circuit of
    elementary gates. The controlled evolution `controlled-e^{iA·2^m·t}` that a
    clock qubit needs is a Trotter product with a total time of `2^m·t`, and never
    a matrix power. The step size `t/steps` is fixed, so the power `2^m` uses
    `steps·2^m` steps.

    Only the single `RZ` of each Trotter term is controlled. The controlled `RZ` is
    built from `RZ` and controlled-NOT (CNOT) gates, as `CRZ = RZ·CNOT·RZ·CNOT`.
    The change of basis and the ladder gates cancel out when the control is `|0⟩`.
    The identity term of `A` becomes a `PhaseShift` on the clock qubit, which is a
    controlled global phase. As a result, the estimated phase reflects the full
    eigenvalue and not only its traceless part, which matches what
    [`QPEBlock`][eigenkit.core.blocks.QPEBlock] gives from `expm(1j·A·t)`.

    The qubits are laid out as follows, with indices local to the block:

    ```text
    [clock(0 … n_clock-1) | state(n_clock … n_clock+n_state-1)]
    ```

    Clock qubit `k` controls `e^{iA·2^(n_clock-1-k)·t}`. This is the big-endian
    convention of QPE, the same as in [`QPEBlock`][eigenkit.core.blocks.QPEBlock].
    The caller maps these local indices to the qubits of the full circuit, for
    example to skip the HHL ancilla qubit.

    Args:
        pauli_terms: The `(coeff, pauli_string)` terms of the Hermitian matrix `A`,
            for example from `pauli_decompose`. Every string has length `n_state`.
        n_clock: The number of clock qubits, which sets the resolution of QPE.
        time: The base evolution time `t`.
        steps: The number of Trotter steps for the base time `t`. Default: `1`.
        order: The Trotter order, `1` or `2`. Default: `1`.
        inverse: If `True`, the block builds the inverse QPE, which undoes the
            phase estimation. Default: `False`.

    Example:
        ```python
        import numpy as np
        from eigenkit.algorithms._hamiltonian_sim import pauli_decompose
        from eigenkit.core.blocks import HamiltonianQPEBlock, assemble

        A = np.array([[1.0, 0.5], [0.5, -1.0]])  # a Hermitian matrix on one qubit
        terms = pauli_decompose(A)
        spec = assemble([HamiltonianQPEBlock(terms, n_clock=4, time=1.0, steps=8)])
        ```
    """

    pauli_terms: tuple[tuple[float, str], ...]
    n_clock: int
    time: float
    steps: int = 1
    order: int = 1
    inverse: bool = False

    def __post_init__(self) -> None:
        # Reuse TrotterBlock's validation/normalisation for the term list so the
        # two blocks cannot disagree on what a valid Hamiltonian looks like.
        probe = TrotterBlock(self.pauli_terms, time=self.time, steps=1, order=1)
        object.__setattr__(self, "pauli_terms", probe.pauli_terms)
        object.__setattr__(self, "_n_state", probe.required_qubits())
        if self.n_clock < 1:
            raise ValueError(f"n_clock must be >= 1, got {self.n_clock}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {self.order}")

    def required_qubits(self) -> int:
        """Return `n_clock` plus the number of state qubits.

        The clock qubits come first, then the state qubits.
        """
        return self.n_clock + self._n_state  # type: ignore[attr-defined]

    def build(self, n_qubits: int) -> IRFragment:
        """Build the phase estimation, with Trotter products, for `n_qubits` qubits.

        The qubit indices are local to the block, and the caller maps them to the
        qubits of the full circuit.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled, which
                is the number of qubits of the widest block in the list.

        Returns:
            The gates of the block, as an `IRFragment`. It holds elementary gates only,
                and no weights.
        """
        n_state: int = self._n_state  # type: ignore[attr-defined]
        need = self.n_clock + n_state
        if n_qubits < need:
            raise ValueError(
                f"HamiltonianQPEBlock needs {need} qubits "
                f"(n_clock={self.n_clock} + state={n_state}), circuit has {n_qubits}"
            )
        qb = CircuitBuilder(n_qubits=n_qubits)
        clock = tuple(range(self.n_clock))
        state = tuple(range(self.n_clock, self.n_clock + n_state))
        tau = self.time / self.steps

        if not self.inverse:
            for w in clock:
                qb.h(w)
            for k, ctrl in enumerate(clock):
                power = self.n_clock - 1 - k
                _append_controlled_trotter(
                    qb,
                    ctrl,
                    self.pauli_terms,
                    tau,
                    self.steps * (2**power),
                    self.order,
                    state,
                )
            # IQFT on the clock (qubits 0..n_clock-1).
            _build_iqft(qb, self.n_clock)
        else:
            # QFT on the clock, controlled-U†, then Hadamards.
            _build_qft(qb, self.n_clock)
            for k, ctrl in enumerate(clock):
                power = self.n_clock - 1 - k
                _append_controlled_trotter(
                    qb,
                    ctrl,
                    self.pauli_terms,
                    -tau,
                    self.steps * (2**power),
                    self.order,
                    state,
                    reverse=True,
                )
            for w in clock:
                qb.h(w)

        spec = qb.build()
        return IRFragment(gates=spec.gates, n_weights=0, n_inputs=0)

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        When a saved model is loaded, the `type` key selects the block class whose
        `from_dict()` rebuilds the block.

        Returns:
            The fields of the block, and its `type`.
        """
        return {
            "type": "HamiltonianQPEBlock",
            "pauli_terms": [[c, s] for c, s in self.pauli_terms],
            "n_clock": self.n_clock,
            "time": self.time,
            "steps": self.steps,
            "order": self.order,
            "inverse": self.inverse,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HamiltonianQPEBlock":
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            The rebuilt `HamiltonianQPEBlock`.
        """
        return cls(
            pauli_terms=tuple((c, s) for c, s in d["pauli_terms"]),
            n_clock=d["n_clock"],
            time=d["time"],
            steps=d.get("steps", 1),
            order=d.get("order", 1),
            inverse=d.get("inverse", False),
        )


def _build_qft(qb: CircuitBuilder, n: int) -> None:
    """Build the standard QFT in-place on qubits 0..n-1."""
    for k in range(n):
        qb.h(k)
        for j in range(k + 1, n):
            angle = 2.0 * math.pi / (2 ** (j - k + 1))
            qb.cphase(angle, control=j, target=k)
    for k in range(n // 2):
        qb.swap(k, n - 1 - k)


def _build_iqft(qb: CircuitBuilder, n: int) -> None:
    """Build the inverse QFT in-place on qubits 0..n-1."""
    for k in range(n // 2):
        qb.swap(k, n - 1 - k)
    for k in range(n - 1, -1, -1):
        for j in range(n - 1, k, -1):
            angle = -2.0 * math.pi / (2 ** (j - k + 1))
            qb.cphase(angle, control=j, target=k)
        qb.h(k)


# ---- CircuitSpec JSON helpers (used by Grover.to_dict / from_dict) ----
# These helpers live in blocks.py to avoid a circular import with
# algorithms/_serialization.py (which imports blocks). The format for
# parameter nodes matches _serialization.py exactly so that bundles saved
# via either path are interchangeable.


def _param_to_json(p: Parameter) -> dict:
    """Recursively encode a Parameter tree as a JSON-safe dict."""
    if isinstance(p, Constant):
        return {"t": "Constant", "v": p.value}
    if isinstance(p, WeightRef):
        return {"t": "WeightRef", "i": p.index}
    if isinstance(p, InputRef):
        return {"t": "InputRef", "i": p.index}
    if isinstance(p, Scale):
        return {"t": "Scale", "e": _param_to_json(p.expression), "f": p.factor}
    if isinstance(p, Negate):
        return {"t": "Negate", "e": _param_to_json(p.expression)}
    if isinstance(p, Sum):
        return {"t": "Sum", "es": [_param_to_json(e) for e in p.expressions]}
    if isinstance(p, Multiply):
        return {"t": "Multiply", "a": _param_to_json(p.a), "b": _param_to_json(p.b)}
    raise TypeError(f"unknown parameter: {type(p)}")


def _param_from_json(d: dict) -> Parameter:
    """Reconstruct a Parameter tree from a JSON-loaded dict."""
    t = d["t"]
    if t == "Constant":
        return Constant(d["v"])
    if t == "WeightRef":
        return WeightRef(d["i"])
    if t == "InputRef":
        return InputRef(d["i"])
    if t == "Scale":
        return Scale(_param_from_json(d["e"]), d["f"])
    if t == "Negate":
        return Negate(_param_from_json(d["e"]))
    if t == "Sum":
        return Sum(tuple(_param_from_json(e) for e in d["es"]))
    if t == "Multiply":
        return Multiply(_param_from_json(d["a"]), _param_from_json(d["b"]))
    raise ValueError(f"unknown param type in JSON: {t!r}")


def _data_item_to_json(item) -> dict:
    """Encode one element of Gate.data as a JSON-safe dict.

    Uses an explicit ``"kind"`` tag to distinguish flat (1-D) from
    matrix (2-D) payloads, avoiding ambiguity when encoding complex
    numbers as ``[re, im]`` pairs inside a 1-D array. ``None`` is encoded as
    ``{"kind": "null"}`` to support ControlledQubitUnitary gates whose
    ``control_values`` (data[1]) is absent.
    """
    if item is None:
        return {"kind": "null"}

    def _enc(v) -> object:
        if isinstance(v, complex):
            return [v.real, v.imag]
        # Preserve integer control_values as ints, not floats.
        if not isinstance(v, float) and hasattr(v, "__index__"):
            return int(v)
        return float(v)

    if item and isinstance(item[0], tuple):
        # 2-D: nested tuples (QubitUnitary matrix)
        return {
            "kind": "matrix",
            "rows": [[_enc(v) for v in row] for row in item],
        }
    # 1-D: flat tuple (AmplitudeEncoding amplitudes or control_values ints)
    return {"kind": "flat", "values": [_enc(v) for v in item]}


def _data_item_from_json(d: dict):
    """Reconstruct one Gate.data element from a JSON-loaded dict."""
    if d["kind"] == "null":
        return None

    def _dec(v) -> object:
        return complex(v[0], v[1]) if isinstance(v, list) else v

    if d["kind"] == "flat":
        return tuple(_dec(v) for v in d["values"])
    # "matrix"
    return tuple(tuple(_dec(v) for v in row) for row in d["rows"])


def _gate_to_json(g: "Gate") -> dict:
    result: dict = {
        "n": g.name,
        "q": list(g.qubits),
        "p": [_param_to_json(p) for p in g.params],
    }
    if g.data:
        result["d"] = [_data_item_to_json(item) for item in g.data]
    if g.adjoint:
        result["adj"] = True
    return result


def _gate_from_json(d: dict) -> "Gate":
    data = tuple(_data_item_from_json(item) for item in d.get("d", []))
    return Gate(
        name=d["n"],
        qubits=tuple(d["q"]),
        params=tuple(_param_from_json(p) for p in d["p"]),
        data=data,
        adjoint=d.get("adj", False),
    )


def _circuit_spec_to_json(spec: CircuitSpec) -> dict:
    return {
        "n_qubits": spec.n_qubits,
        "n_weights": spec.n_weights,
        "n_inputs": spec.n_inputs,
        "gates": [_gate_to_json(g) for g in spec.gates],
    }


def _circuit_spec_from_json(d: dict) -> CircuitSpec:
    return CircuitSpec(
        n_qubits=d["n_qubits"],
        n_weights=d["n_weights"],
        n_inputs=d["n_inputs"],
        gates=tuple(_gate_from_json(g) for g in d["gates"]),
    )


# ---- assembly ----


def _relocate_parameter(
    parameter: Parameter,
    weight_offset: int,
    input_offset: int,
) -> Parameter:
    """Shift WeightRef/InputRef indices into the global frame, descending
    recursively through Scale, Negate, Sum, and Multiply nodes."""
    if isinstance(parameter, WeightRef):
        return WeightRef(index=parameter.index + weight_offset)
    if isinstance(parameter, InputRef):
        return InputRef(index=parameter.index + input_offset)
    if isinstance(parameter, Constant):
        return parameter
    if isinstance(parameter, Scale):
        return Scale(
            expression=_relocate_parameter(
                parameter.expression, weight_offset, input_offset
            ),
            factor=parameter.factor,
        )
    if isinstance(parameter, Negate):
        return Negate(
            expression=_relocate_parameter(
                parameter.expression, weight_offset, input_offset
            ),
        )
    if isinstance(parameter, Sum):
        return Sum(
            expressions=tuple(
                _relocate_parameter(e, weight_offset, input_offset)
                for e in parameter.expressions
            )
        )
    if isinstance(parameter, Multiply):
        return Multiply(
            a=_relocate_parameter(parameter.a, weight_offset, input_offset),
            b=_relocate_parameter(parameter.b, weight_offset, input_offset),
        )
    raise TypeError(f"unknown parameter kind: {type(parameter)}")


def assemble(blocks: Iterable[Block | CircuitBuilder]) -> CircuitSpec:
    """Combine blocks into a single `CircuitSpec`.

    A `CircuitBuilder` can take the place of a block. It is replaced by the result
    of its `to_block()`, so a part of a circuit written gate by gate can go
    directly in the list.

    The blocks are combined as follows:

    - `n_qubits` is the largest `required_qubits()` of the blocks.
    - The weights and the inputs of the blocks are placed one after the other, in
      the order of the blocks.
    - The local `WeightRef` and `InputRef` indices of each block are shifted by
      the number of weights and inputs of the blocks before it.

    Args:
        blocks: The blocks to combine, in the order in which they are applied. Any
            of them can be a `CircuitBuilder`, which is taken as it is when
            `assemble` is called.

    Returns:
        One `CircuitSpec` that contains all the blocks.

    Raises:
        ValueError: If `blocks` is empty.

    Example:
        ```python
        from eigenkit import Encoding, Variational
        from eigenkit.core import assemble

        spec = assemble([Encoding(in_features=4), Variational(depth=2)])
        print(spec.n_qubits, spec.n_weights, spec.n_inputs)  # 4 8 4
        ```
    """
    block_list = [
        block.to_block() if isinstance(block, CircuitBuilder) else block
        for block in blocks
    ]
    if not block_list:
        raise ValueError("at least one block required")
    n_qubits = max(block.required_qubits() for block in block_list)

    all_gates: list[Gate] = []
    weight_offset = 0
    input_offset = 0
    for block in block_list:
        fragment = block.build(n_qubits)
        for gate in fragment.gates:
            relocated_params = tuple(
                _relocate_parameter(p, weight_offset, input_offset) for p in gate.params
            )
            all_gates.append(
                Gate(
                    name=gate.name,
                    qubits=gate.qubits,
                    params=relocated_params,
                    data=gate.data,
                    adjoint=gate.adjoint,
                )
            )
        weight_offset += fragment.n_weights
        input_offset += fragment.n_inputs

    return CircuitSpec(
        n_qubits=n_qubits,
        n_weights=weight_offset,
        n_inputs=input_offset,
        gates=tuple(all_gates),
    )
