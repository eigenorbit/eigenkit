"""The intermediate representation (IR) of quantum circuits.

The IR is plain data and does not depend on any third-party package. Every object
is a frozen dataclass, so circuits can be hashed and pickled, and they can be
shared safely between threads or processes.

The parameter expressions (`Scale`, `Negate`, `Sum` and `Multiply`) let encoding
blocks write angles such as `RY(2π * x)` or `RY(λ * x)`, so the user does not have
to scale the inputs beforehand. They also let transforms such as `adjoint` work on
symbolic references instead of fixed numbers.

Example:
    ```python
    from eigenkit import Encoding, Variational
    from eigenkit.core import assemble
    from eigenkit.core.ir import Observable, adjoint

    spec = assemble([Encoding(in_features=2), Variational(depth=1)])

    print(spec.n_qubits, len(spec.gates))       # 2 5
    print(spec.gates[0].name, spec.gates[0].qubits)  # RY (0,)

    # adjoint reverses the gate order and negates every angle
    print(adjoint(spec).gates[0].name)          # CNOT

    # an observable can be parsed from a sum of Pauli terms written as a string
    print(Observable.from_string("0.5 Z0 X1").terms[0].coeff)  # 0.5
    ```
"""

from __future__ import annotations

import numbers
import re
from dataclasses import dataclass, replace
from typing import Literal, Union


# ---- gate vocabulary ----

GateName = Literal[
    "RX",
    "RY",
    "RZ",
    "H",
    "CNOT",
    "X",
    "Z",
    "S",
    "Sdg",
    "MCZ",
    "MCX",
    "PhaseShift",
    "SWAP",
    "CRY",
    "CPHASE",
    "AmplitudeEncoding",
    "QubitUnitary",
    "ControlledQubitUnitary",
]

SINGLE_QUBIT_ROTATIONS: frozenset[str] = frozenset({"RX", "RY", "RZ", "PhaseShift"})
TWO_QUBIT_ROTATIONS: frozenset[str] = frozenset({"CRY", "CPHASE"})
NON_PARAMETRIC_GATES: frozenset[str] = frozenset(
    {"H", "CNOT", "X", "Z", "S", "Sdg", "MCZ", "MCX", "SWAP"}
)
# Data-carrying IR primitives emitted by high-level blocks (AmplitudeEncoding,
# QubitUnitary, ControlledQubitUnitary). They are not gates in the
# circuit-algebra sense and must not appear in NON_PARAMETRIC_GATES or be mixed
# with rotation/structural gates.
DATA_GATES: frozenset[str] = frozenset(
    {"AmplitudeEncoding", "QubitUnitary", "ControlledQubitUnitary"}
)
ALL_GATES: frozenset[str] = (
    SINGLE_QUBIT_ROTATIONS | TWO_QUBIT_ROTATIONS | NON_PARAMETRIC_GATES | DATA_GATES
)

# Non-parametric gates that are not their own inverse: the `adjoint` transform
# maps each to its conjugate. Everything else (H, X, CNOT) is self-inverse.
_ADJOINT_NAME: dict[str, str] = {"S": "Sdg", "Sdg": "S"}

# Gates whose expectation carries more than one frequency in their angle, so
# the two-term parameter-shift rule is not exact for them. A controlled
# rotation's generator |1><1| (x) P/2 has eigenvalues {0, 0, +1/2, -1/2} and so
# two distinct gaps, {1/2, 1}. Every other parametric gate in the vocabulary
# has a single gap -- PhaseShift and CPHASE included, whose {0, 1} generators
# give the gap 1 alone -- and the two-term rule is exact for those.
MULTI_FREQUENCY_GATES: frozenset[str] = frozenset({"CRY"})


# ---- parameter expressions ----


class _ParamOps:
    """Python arithmetic on parameter expressions, shared by every node type.

    Lets an angle be written the way it reads on paper --
    `2.0 * (pi - x[0]) * (pi - x[1])` -- instead of nesting
    `Scale(Multiply(Sum((Constant(PI), Negate(x[0]))), ...), 2.0)` by hand.
    Each operator builds exactly the node its explicit constructor would, so
    the two spellings produce identical IR and nothing downstream can tell
    them apart.

    Two conveniences are worth knowing:

    - **Multiplication picks the node for you.** A plain number becomes
      `Scale`, so a backend sees the constant directly; only symbol times
      symbol becomes `Multiply`, which is the case cudaq cannot lower. Hand
      construction gets this wrong easily, and nothing warns.
    - **Addition flattens.** `a + b + c` is one `Sum` of three terms rather
      than a `Sum` nesting a `Sum`, which is what the same expression written
      by hand would be.

    Division is by a number only: there is no division node in the IR, so
    `x / 2` is `Scale(x, 0.5)` and `x / y` has nothing to lower to.
    """

    __slots__ = ()

    def __neg__(self) -> "Parameter":
        return negate_parameter(self)

    def __pos__(self) -> "Parameter":
        return self

    def __add__(self, other: "ParameterLike") -> "Parameter":
        return _sum_of(self, _coerce_parameter(other))

    def __radd__(self, other: "ParameterLike") -> "Parameter":
        return _sum_of(_coerce_parameter(other), self)

    def __sub__(self, other: "ParameterLike") -> "Parameter":
        return _sum_of(self, negate_parameter(_coerce_parameter(other)))

    def __rsub__(self, other: "ParameterLike") -> "Parameter":
        return _sum_of(_coerce_parameter(other), negate_parameter(self))

    def __mul__(self, other: "ParameterLike") -> "Parameter":
        return _product_of(self, _coerce_parameter(other))

    def __rmul__(self, other: "ParameterLike") -> "Parameter":
        return _product_of(_coerce_parameter(other), self)

    def __truediv__(self, other: "ParameterLike") -> "Parameter":
        # No division node exists in the IR, so only a constant divisor can be
        # represented -- as its reciprocal factor.
        divisor = _constant_value(other)
        if divisor is None:
            raise TypeError(
                "a parameter expression can only be divided by a number, not "
                f"by {type(other).__name__}; the IR has no division node. "
                "Multiply by a precomputed reciprocal instead."
            )
        return Scale(self, 1.0 / divisor)


def _constant_value(value: "ParameterLike") -> float | None:
    """The scalar `value` stands for, or `None` if it is genuinely symbolic.

    A `Constant` node counts: `Constant(2.0) * x` should fold to `Scale` for
    the same reason `2.0 * x` does, rather than becoming a `Multiply` that
    backends have to treat as two symbolic operands.
    """
    if isinstance(value, Constant):
        return value.value
    if isinstance(value, numbers.Real):
        return float(value)
    return None


def _product_of(left: "Parameter", right: "Parameter") -> "Parameter":
    """Multiply two expressions, choosing `Scale` whenever one side is constant.

    `Multiply` is for the genuinely bilinear case only -- it is the node cudaq
    cannot lower, and the one a backend cannot fold into a gate angle -- so a
    constant on either side (or both) collapses instead.
    """
    left_value = _constant_value(left)
    right_value = _constant_value(right)
    if left_value is not None and right_value is not None:
        return Constant(left_value * right_value)
    if right_value is not None:
        return Scale(left, right_value)
    if left_value is not None:
        return Scale(right, left_value)
    return Multiply(left, right)


def _sum_of(left: "Parameter", right: "Parameter") -> "Sum":
    """Add two expressions, flattening a `Sum` that is already on either side.

    Keeps `a + b + c` a single three-term `Sum` instead of nesting one `Sum`
    inside another. Only top-level operands are flattened; a `Sum` buried
    under a `Scale` or `Negate` is left alone, because those change what its
    terms mean.
    """
    terms: list["Parameter"] = []
    for part in (left, right):
        if isinstance(part, Sum):
            terms.extend(part.expressions)
        else:
            terms.append(part)
    return Sum(tuple(terms))


@dataclass(frozen=True)
class Constant(_ParamOps):
    """A fixed angle, used where a block needs a rotation by a set amount.

    Attributes:
        value: The angle, in radians.
    """

    value: float


@dataclass(frozen=True)
class WeightRef(_ParamOps):
    """A placeholder for the trainable weight at `index` in the weight vector.

    Attributes:
        index: The position in the weight vector of the circuit. Each block numbers
            the weights of its own fragment from `0`, and `assemble` shifts these
            indices to their positions in the whole circuit.
    """

    index: int


@dataclass(frozen=True)
class InputRef(_ParamOps):
    """A placeholder for the input feature at `index` in the input vector.

    Attributes:
        index: The position in the input vector of the circuit. `assemble` shifts
            it in the same way as the indices of `WeightRef`.
    """

    index: int


@dataclass(frozen=True)
class Scale(_ParamOps):
    """A constant multiple of another expression: `factor * expression`.

    Use it instead of `Multiply` when one of the two factors is a plain number. A
    backend can then include the constant factor in the gate angle, instead of
    treating it as a second symbolic operand.

    Attributes:
        expression: The expression to scale.
        factor: The constant factor.
    """

    expression: "Parameter"
    factor: float


@dataclass(frozen=True)
class Negate(_ParamOps):
    """The negation of another expression: `-expression`.

    The `adjoint` transform builds this node when it reverses a rotation. When the
    angle is a `Constant` or a `Scale`, it changes the sign of the number instead,
    and when the angle is already a `Negate`, it removes the negation.

    Attributes:
        expression: The expression to negate.
    """

    expression: "Parameter"


@dataclass(frozen=True)
class Sum(_ParamOps):
    """The sum of several expressions.

    Attributes:
        expressions: The terms to add. There must be at least one. An empty `Sum`
            is not evaluated as `0`: `resolve` and the backends start from the
            first term, so they raise an `IndexError` on it.
    """

    expressions: tuple["Parameter", ...]


@dataclass(frozen=True)
class Multiply(_ParamOps):
    """The product of two parameter expressions.

    It is used when a gate angle is the product of two symbolic terms. The most
    common case is a trainable scaling of a feature (Schuld et al. 2021):
    `RY(λ * x)`, where `λ` is a `WeightRef` and `x` is an `InputRef`. When one of
    the two factors is a constant number, use `Scale` instead, so that backends can
    see the number directly.

    Backends that do not support the product of two symbolic terms raise
    `NotImplementedError` when this node appears. Currently, this is the case of
    the CUDA Quantum backend.

    Attributes:
        a: The left factor.
        b: The right factor.
    """

    a: "Parameter"
    b: "Parameter"


Parameter = Union[Constant, WeightRef, InputRef, Scale, Negate, Sum, Multiply]

# What a caller may pass anywhere an angle is accepted: an expression, or a
# plain number that `_coerce_parameter` wraps as a `Constant`.
ParameterLike = Union[Parameter, float, int]


def _coerce_parameter(value: ParameterLike) -> Parameter:
    """Wrap plain numbers as `Constant` so downstream code only ever sees
    `Parameter`.

    `numbers.Real` rather than `(int, float)`, so a numpy scalar -- what
    indexing a numpy array of angles hands you -- is wrapped rather than
    passed through as a non-`Parameter` that fails much later, somewhere
    with no clue about where it came from.
    """
    if isinstance(value, _ParamOps):
        return value
    if isinstance(value, numbers.Real):
        return Constant(float(value))
    raise TypeError(
        f"expected a parameter expression or a number, got {type(value).__name__}"
    )


def resolve(parameter: Parameter, weights, inputs) -> float:
    """Compute the numeric value of a parameter expression, recursively.

    `weights` and `inputs` can be any container that supports indexing (a NumPy
    array, a PyTorch tensor or a list). The type of the result follows the type of
    these containers, so the same function serves both the NumPy and the PyTorch
    evaluation paths.

    Args:
        parameter (Parameter): The angle expression to evaluate.
        weights (np.ndarray | Sequence[float]): The weight vector. It can be any
            container that supports indexing: a NumPy array, a PyTorch tensor or a
            list.
        inputs (np.ndarray | Sequence[float]): The input vector. It can be any
            container that `weights` can be.

    Returns:
        The numeric angle.
    """
    if isinstance(parameter, Constant):
        return parameter.value
    if isinstance(parameter, WeightRef):
        return weights[parameter.index]
    if isinstance(parameter, InputRef):
        return inputs[parameter.index]
    if isinstance(parameter, Scale):
        return parameter.factor * resolve(parameter.expression, weights, inputs)
    if isinstance(parameter, Negate):
        return -resolve(parameter.expression, weights, inputs)
    if isinstance(parameter, Sum):
        total = resolve(parameter.expressions[0], weights, inputs)
        for expression in parameter.expressions[1:]:
            total = total + resolve(expression, weights, inputs)
        return total
    if isinstance(parameter, Multiply):
        return resolve(parameter.a, weights, inputs) * resolve(
            parameter.b, weights, inputs
        )
    raise TypeError(f"unknown parameter kind: {type(parameter).__name__}")


def negate_parameter(parameter: Parameter) -> Parameter:
    """Negate a parameter expression of any kind, symbolically.

    Args:
        parameter (Parameter): The angle expression to negate.

    Returns:
        The negated expression.
    """
    if isinstance(parameter, Constant):
        return Constant(-parameter.value)
    if isinstance(parameter, Negate):
        # double negation cancels
        return parameter.expression
    if isinstance(parameter, Scale):
        return Scale(parameter.expression, -parameter.factor)
    return Negate(parameter)


# ---- gates, observables, circuits ----


def _array_to_data(arr) -> tuple:
    """Convert a 1-D or 2-D array-like to a nested tuple for Gate.data.

    Nested tuples are hashable and picklable, keeping Gate a valid frozen
    dataclass. Accepts lists, tuples, or numpy arrays.
    """
    raw = arr.tolist() if hasattr(arr, "tolist") else list(arr)
    if raw and hasattr(raw[0], "__iter__") and not isinstance(raw[0], str):
        return tuple(
            tuple(row.tolist() if hasattr(row, "tolist") else row) for row in raw
        )
    return tuple(raw)


def _data_to_array(data: tuple):
    """Reconstruct a complex128 numpy array from a nested tuple stored in
    Gate.data. Backends use this to get a usable matrix / vector."""
    import numpy as np

    return np.array(data, dtype=np.complex128)


@dataclass(frozen=True)
class Gate:
    """One gate of a circuit: which gate it is, where it acts, and its angles.

    Attributes:
        name: The name of the gate, from the fixed list in `GateName`.
        qubits: The indices of the qubits that the gate acts on, in the order that
            the gate expects (the controls before the targets).
        params: The angle expressions, one for each rotation parameter. Each one is
            a `Constant`, a `WeightRef`, an `InputRef`, or an arithmetic
            combination of them.
        data: The constant arrays of the gates that carry data
            (`AmplitudeEncoding`, `QubitUnitary` and `ControlledQubitUnitary`).
            They are stored as nested tuples so that the gate can still be hashed.
            The field is empty for every other gate. `_array_to_data` and
            `_data_to_array` convert between arrays and this form.
        adjoint: If True, the conjugate transpose of the gate is applied instead
            of the gate itself. It is used only for `AmplitudeEncoding` gates,
            inside Grover iterates. The adjoint of a `QubitUnitary` is computed in
            the intermediate representation instead, by taking the conjugate
            transpose of the stored matrix, so this flag is never set for it.
    """

    name: GateName
    qubits: tuple[int, ...]
    params: tuple[Parameter, ...] = ()
    data: tuple = ()
    adjoint: bool = False


_PAULI_LETTERS: frozenset[str] = frozenset({"X", "Y", "Z"})


@dataclass(frozen=True)
class PauliTerm:
    """A coefficient times a product of Pauli operators on distinct qubits.

    `PauliTerm(0.5, (("Z", 0), ("X", 1)))` is the term `0.5 Z0 X1`.

    Attributes:
        coeff: The real coefficient. Complex values are rejected, because an
            `Observable` represents only Hermitian operators.
        paulis: Pairs of a Pauli letter (`"X"`, `"Y"` or `"Z"`) and a qubit index.
            It must not be empty, because a term without Pauli operators is a
            constant, which an `Observable` does not hold. Each qubit can appear
            at most once, so reduce products such as `Z * Z = I` before building
            the term.

    Raises:
        TypeError: If `coeff` is complex.
        ValueError: If `paulis` is empty, if a Pauli letter is not `X`, `Y` or
            `Z`, or if a qubit appears more than once.
    """

    coeff: float
    paulis: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if isinstance(self.coeff, complex):
            raise TypeError(
                f"PauliTerm.coeff must be real, got complex {self.coeff!r}. "
                f"Our Observable models Hermitian operators only."
            )
        if not self.paulis:
            # A term with no Paulis is a constant, and the backends disagree
            # about it: six would compute it, pennylane would raise, and braket
            # would drop it and return a number short by exactly the constant.
            # Rather than pick one and hope every future backend follows, the
            # term cannot be built — which is also the convention chemistry
            # uses, where `PauliSum.to_observable` returns the constant
            # separately.
            raise ValueError(
                f"PauliTerm with no Paulis is the constant {self.coeff:g}, and an "
                f"Observable holds Pauli terms only. Keep the constant beside the "
                f"observable — the chemistry entry points take it as `offset`."
            )
        seen_qubits: set[int] = set()
        for op, qubit in self.paulis:
            if op not in _PAULI_LETTERS:
                raise ValueError(f"Pauli operator must be X, Y, or Z, got {op!r}")
            if qubit in seen_qubits:
                raise ValueError(
                    f"qubit {qubit} appears more than once in Pauli term "
                    f"{self.paulis!r}; reduce the product first (e.g. Z*Z=I)"
                )
            seen_qubits.add(qubit)


@dataclass(frozen=True)
class Observable:
    """A Hermitian operator to measure, written as a sum of Pauli terms.

    You can build one with the `z`, `x` and `y` methods for a single Pauli
    operator on one qubit, with `from_string` for a whole sum, or with the `X`,
    `Y` and `Z` functions of `eigenkit.core.observables`, which support
    arithmetic.

    Attributes:
        terms: The Pauli terms of the sum. It can be empty: `jordan_wigner`
            returns an empty observable, together with an offset, when the
            operator reduces to a multiple of the identity. However, `Circuit`
            refuses to measure an empty observable, because there is nothing to
            measure.
    """

    terms: tuple[PauliTerm, ...]

    @classmethod
    def z(cls, qubit: int) -> "Observable":
        """Return the observable `Z` on the single qubit `qubit`.

        Args:
            qubit (int): The index of the qubit.
        """
        return cls(terms=(PauliTerm(coeff=1.0, paulis=(("Z", qubit),)),))

    @classmethod
    def x(cls, qubit: int) -> "Observable":
        """Return the observable `X` on the single qubit `qubit`.

        Args:
            qubit (int): The index of the qubit.
        """
        return cls(terms=(PauliTerm(coeff=1.0, paulis=(("X", qubit),)),))

    @classmethod
    def y(cls, qubit: int) -> "Observable":
        """Return the observable `Y` on the single qubit `qubit`.

        Args:
            qubit (int): The index of the qubit.
        """
        return cls(terms=(PauliTerm(coeff=1.0, paulis=(("Y", qubit),)),))

    @classmethod
    def from_string(cls, text: str) -> "Observable":
        """Parse a sum of Pauli terms written as a string.

        Examples of such strings are `"0.5 Z0 X1 + 0.3 Y2"` and `"-Z0 + Z1"`.

        The grammar is as follows. The terms are joined by `+` or `-`. Each term
        starts with an optional numeric coefficient, in plain or scientific
        notation (`1e-3` is accepted). The coefficient is followed by an optional
        `*` and by one or more `[XYZ]<qubit>` operators on distinct qubits. Any
        amount of whitespace is accepted, including none.

        Every part of the string must be a valid term, and every term must contain
        Pauli operators. A constant term (`"0.5 + Z0"`, `"0.5 I"`) raises an error
        instead of being dropped from the sum: an `Observable` holds only Pauli
        terms, and the constant must be kept next to it as an offset.

        Args:
            text (str): The string to parse.

        Returns:
            The parsed observable.

        Raises:
            ValueError: If `text` is empty, if it contains a constant term, if a
                part of it is not a valid term, or if a term uses the same qubit
                twice.
        """
        return _parse_observable(text)


@dataclass(frozen=True)
class CircuitSpec:
    """A whole circuit as data.

    It holds the list of gates, the number of qubits, and the lengths of the weight
    and input vectors that the gates refer to. Every provider compiles this object,
    and `assemble` builds it from a list of blocks. It holds no weights and no
    observables: the weights are passed at each call, and the observables are set
    by the `Circuit` that wraps the spec.

    Attributes:
        n_qubits: The number of qubits of the circuit.
        n_weights: The length of the weight vector that the gates refer to
            through `WeightRef`.
        n_inputs: The length of the input vector that the gates refer to through
            `InputRef`.
        gates: The gates, in the order in which they are applied.
    """

    n_qubits: int
    n_weights: int
    n_inputs: int
    gates: tuple[Gate, ...]


# ---- observable parsing ----


# A coefficient is a plain or scientific-notation real: 2, 0.5, .5, 1e-3, 1.5E+2.
# Spelling the exponent out matters — a pattern that stops at the mantissa leaves
# the exponent behind as unconsumed text rather than failing, and "1e-3 Z0" then
# parses as the "-3" term.
_COEFFICIENT = r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_TERM_PATTERN = re.compile(
    rf"\s*([+-])\s*"  # sign, explicit on every term after normalisation
    rf"({_COEFFICIENT})?\s*"  # coefficient, 1.0 when omitted
    rf"\*?\s*"  # optional multiplication sign: "0.5 * Z0"
    rf"((?:[XYZ]\d+\s*)+|I\b\s*)?"  # Pauli product; I is matched only to reject it
)
_PAULI_PATTERN = re.compile(r"([XYZ])(\d+)")


def _parse_observable(text: str) -> Observable:
    normalized = text.strip()
    if not normalized:
        raise ValueError("observable string is empty")
    if not normalized.startswith(("+", "-")):
        normalized = "+" + normalized
    terms: list[PauliTerm] = []
    # Match anchored at the cursor rather than scanning for matches anywhere.
    # `finditer` skips whatever it cannot match, so a term the grammar does not
    # cover would be dropped from the sum instead of rejected, and only text
    # trailing the last match would ever be reported.
    position = 0
    while position < len(normalized):
        match = _TERM_PATTERN.match(normalized, position)
        coeff_str = match.group(2) if match else None
        pauli_block = match.group(3) if match else None
        if match is None or (coeff_str is None and pauli_block is None):
            raise ValueError(
                f"cannot parse observable {text!r}: expected a term such as "
                f"'0.5 Z0 X1' or 'Y2' at {normalized[position:].strip()!r}"
            )
        coefficient = 1.0 if coeff_str is None else float(coeff_str)
        if match.group(1) == "-":
            coefficient = -coefficient
        paulis = tuple(
            (op, int(qubit)) for op, qubit in _PAULI_PATTERN.findall(pauli_block or "")
        )
        if not paulis:
            # Nothing but a coefficient. Two different faults look alike here,
            # so separate them by what follows: another term means the user
            # wrote a real constant, anything else means the operator itself
            # did not parse ("0.5 Z" is a truncated term, not a constant).
            rest = normalized[match.end() :].lstrip()
            if rest and not rest.startswith(("+", "-")):
                raise ValueError(
                    f"cannot parse observable {text!r}: expected a term such "
                    f"as '0.5 Z0 X1' or 'Y2' at {rest!r}"
                )
            # An `Observable` carries Pauli terms only, and most backends
            # reject a term with no Paulis outright. The grammar recognises one
            # so it can be named here: dropping it in silence would turn
            # "0.5 + Z0" into Z0 and move the whole spectrum by 0.5.
            raise ValueError(
                f"constant term {coefficient:g} in observable {text!r}: an "
                f"Observable holds Pauli terms only. Keep the constant "
                f"separate — the chemistry entry points take it as `offset`."
            )
        sorted_paulis = tuple(sorted(paulis, key=lambda pair: pair[1]))
        terms.append(PauliTerm(coeff=coefficient, paulis=sorted_paulis))
        position = match.end()
    if not terms:
        raise ValueError(f"no Pauli terms parsed from {text!r}")
    return Observable(terms=tuple(terms))


# ---- gradient-safety analysis ----


# The two analyses below each come in a weight-facing and an input-facing
# form. They differ in one thing only — which reference type is the variable
# being differentiated against, and which is therefore a constant — so both are
# written once over a `ref_type` and specialised by the four public wrappers.
# Written out per side, a new `Parameter` node would have to be added in seven
# places, and omitting one of the input-side twins would produce a silently wrong
# input jacobian rather than an error.
_LEAVES = (Constant, WeightRef, InputRef)


def _collect_dependencies(parameter: Parameter, ref_type: type, deps: set[int]) -> None:
    """Accumulate the indices of every `ref_type` reference in `parameter`."""
    if isinstance(parameter, ref_type):
        deps.add(parameter.index)
        return
    if isinstance(parameter, _LEAVES):
        # A literal, or a reference of the kind we are not collecting.
        return
    if isinstance(parameter, (Scale, Negate)):
        _collect_dependencies(parameter.expression, ref_type, deps)
        return
    if isinstance(parameter, Sum):
        for expression in parameter.expressions:
            _collect_dependencies(expression, ref_type, deps)
        return
    if isinstance(parameter, Multiply):
        _collect_dependencies(parameter.a, ref_type, deps)
        _collect_dependencies(parameter.b, ref_type, deps)
        return
    raise TypeError(f"unknown parameter kind: {type(parameter).__name__}")


def _chain_factor(
    parameter: Parameter, ref_type: type, index: int, weights, inputs
) -> float:
    """Partial derivative of `parameter` against `ref_type(index)`."""
    if isinstance(parameter, ref_type):
        return 1.0 if parameter.index == index else 0.0
    if isinstance(parameter, _LEAVES):
        return 0.0
    if isinstance(parameter, Scale):
        return parameter.factor * _chain_factor(
            parameter.expression, ref_type, index, weights, inputs
        )
    if isinstance(parameter, Negate):
        return -_chain_factor(parameter.expression, ref_type, index, weights, inputs)
    if isinstance(parameter, Sum):
        total = 0.0
        for expression in parameter.expressions:
            total += _chain_factor(expression, ref_type, index, weights, inputs)
        return total
    if isinstance(parameter, Multiply):
        # Product rule: d(a*b)/dx = (da/dx)*b + a*(db/dx). No `!= 0.0`
        # short-circuits: on a batched (transposed) inputs vector a nested
        # factor can be an array, where a scalar truth test raises "ambiguous".
        da = _chain_factor(parameter.a, ref_type, index, weights, inputs)
        db = _chain_factor(parameter.b, ref_type, index, weights, inputs)
        return da * resolve(parameter.b, weights, inputs) + (
            resolve(parameter.a, weights, inputs) * db
        )
    raise TypeError(f"unknown parameter kind: {type(parameter).__name__}")


def _is_naive_shift_safe(parameter: Parameter, ref_type: type) -> bool:
    """Whether this one angle allows naive shifting against `ref_type`.

    It does only when the angle is either the bare reference or independent of
    every reference of that kind. Any Scale, Multiply, Sum or Negate wrapping
    one introduces a chain-rule factor that the naive parameter-shift jacobian
    ignores. A bare reference can still be unsafe when it feeds another angle
    too; `_find_shift_unsafe_gates` checks that across the circuit.
    """
    if isinstance(parameter, ref_type):
        return True
    deps: set[int] = set()
    _collect_dependencies(parameter, ref_type, deps)
    return not deps


def _find_shift_unsafe_gates(circuit: "CircuitSpec", ref_type: type) -> list["Gate"]:
    """Gates the naive shift of a `ref_type` vector entry cannot differentiate.

    That is a gate whose angle wraps such a reference in an arithmetic node, or
    whose bare reference also feeds another angle. Shifting a shared entry moves
    every angle it feeds at once, which adds frequencies the two- and four-term
    rules do not cover.
    """
    uses: dict[int, int] = {}
    for gate in circuit.gates:
        for parameter in gate.params:
            deps: set[int] = set()
            _collect_dependencies(parameter, ref_type, deps)
            for index in deps:
                uses[index] = uses.get(index, 0) + 1
    return [
        gate
        for gate in circuit.gates
        if any(
            not _is_naive_shift_safe(p, ref_type)
            or (isinstance(p, ref_type) and uses[p.index] > 1)
            for p in gate.params
        )
    ]


def find_parameter_shift_unsafe_gates(circuit: "CircuitSpec") -> list["Gate"]:
    """Return the gates whose weight angle needs the chain-rule path.

    A gate is included when its angle wraps a `WeightRef` in an arithmetic node,
    which is a `Scale`, `Multiply`, `Sum` or `Negate`, or when it uses a weight
    that another angle also uses. `Circuit` reads this list to choose how to
    compute gradients. A circuit without such gates takes the fast path, which
    applies the parameter-shift rule directly to each weight. A circuit with such
    gates takes the chain-rule path. Two examples are
    `Reuploading(trainable_scale=True)`, which builds `Multiply(weight, input)`,
    and a builder circuit that uses one weight in two gates.

    Args:
        circuit (CircuitSpec): The circuit to inspect.

    Returns:
        The gates that need the chain-rule path.
    """
    return _find_shift_unsafe_gates(circuit, WeightRef)


def find_input_shift_unsafe_gates(circuit: "CircuitSpec") -> list["Gate"]:
    """Return the gates whose input angle needs the chain-rule path.

    A gate is included when its angle wraps an `InputRef` in an arithmetic node,
    or when it uses an input that another angle also uses. `Reuploading` does this
    when it encodes each feature once per layer.

    An arithmetic node is a `Scale`, `Multiply`, `Sum` or `Negate`. This function
    is the counterpart of `find_parameter_shift_unsafe_gates` for the inputs. When
    the list is empty, the fast path, which shifts each input directly, can be
    used. Otherwise, the gradients are correct only with the chain-rule path,
    which shifts single gate angles with `shift_gate_angle`.

    Args:
        circuit (CircuitSpec): The circuit to inspect.

    Returns:
        The gates that need the chain-rule path.
    """
    return _find_shift_unsafe_gates(circuit, InputRef)


def _refs_under_multi_frequency_gates(
    circuit: "CircuitSpec", ref_type: type
) -> frozenset[int]:
    """Indices of every `ref_type` reference feeding a multi-frequency gate."""
    found: set[int] = set()
    for gate in circuit.gates:
        if gate.name not in MULTI_FREQUENCY_GATES:
            continue
        for parameter in gate.params:
            _collect_dependencies(parameter, ref_type, found)
    return frozenset(found)


def weights_needing_four_term_shift(circuit: "CircuitSpec") -> frozenset[int]:
    """Return the weights that the two-term parameter-shift rule cannot differentiate.

    A weight is included when one of the gates that it feeds is in
    `MULTI_FREQUENCY_GATES`. A weight that feeds more than one gate never uses
    this direct shift: the circuit takes the chain-rule path, which shifts each
    gate on its own (see `find_parameter_shift_unsafe_gates`).

    Args:
        circuit (CircuitSpec): The circuit to analyse.

    Returns:
        The indices of the weights that need the four-term rule.
    """
    return _refs_under_multi_frequency_gates(circuit, WeightRef)


def inputs_needing_four_term_shift(circuit: "CircuitSpec") -> frozenset[int]:
    """Return the inputs that the two-term parameter-shift rule cannot differentiate.

    This function is the counterpart of `weights_needing_four_term_shift` for the
    inputs.

    Args:
        circuit (CircuitSpec): The circuit to analyse.

    Returns:
        The indices of the inputs that need the four-term rule.
    """
    return _refs_under_multi_frequency_gates(circuit, InputRef)


def collect_weight_dependencies(parameter: Parameter) -> set[int]:
    """Return the set of weight indices that a parameter expression depends on.

    Args:
        parameter (Parameter): The angle expression to inspect.

    Returns:
        The indices of the weights that the expression reads.
    """
    deps: set[int] = set()
    _collect_dependencies(parameter, WeightRef, deps)
    return deps


def collect_input_dependencies(parameter: Parameter) -> set[int]:
    """Return the set of input indices that a parameter expression depends on.

    This function is the counterpart of `collect_weight_dependencies` for the
    inputs. The parameter-shift Jacobian that applies the chain rule uses it to
    find the gates that contribute to the gradient with respect to a given input.

    Args:
        parameter (Parameter): The angle expression to inspect.

    Returns:
        The indices of the inputs that the expression reads.
    """
    deps: set[int] = set()
    _collect_dependencies(parameter, InputRef, deps)
    return deps


def chain_factor(
    parameter: Parameter,
    weight_index: int,
    weights,
    inputs,
) -> float:
    """Compute the partial derivative of `parameter` with respect to one weight.

    The derivative with respect to `weights[weight_index]` is evaluated
    numerically at the given `weights` and `inputs`. The parameter-shift Jacobian
    that applies the chain rule uses it to turn the gradient with respect to a gate
    angle into its contribution to the gradient with respect to one weight.

    Args:
        parameter (Parameter): The angle expression to differentiate.
        weight_index (int): The index of the weight to differentiate with respect
            to.
        weights (np.ndarray | Sequence[float]): The weight vector. It can be any
            container that supports indexing: a NumPy array, a PyTorch tensor or a
            list.
        inputs (np.ndarray | Sequence[float]): The input vector. It can be any
            container that `weights` can be.

    Returns:
        The partial derivative. For a single sample, it is a float. When `inputs`
            is a batch, transposed so that `inputs[j]` holds feature `j` of every
            sample, and the expression contains a `Multiply`, it can be an array
            with one value per sample.
    """
    return _chain_factor(parameter, WeightRef, weight_index, weights, inputs)


def chain_factor_input(
    parameter: Parameter,
    input_index: int,
    weights,
    inputs,
) -> float:
    """Compute the partial derivative of `parameter` with respect to one input.

    The derivative with respect to `inputs[input_index]` is evaluated numerically
    at the given `weights` and `inputs`. This function is the counterpart of
    `chain_factor` for the inputs.

    Args:
        parameter (Parameter): The angle expression to differentiate.
        input_index (int): The index of the input to differentiate with respect
            to.
        weights (np.ndarray | Sequence[float]): The weight vector. It can be any
            container that supports indexing: a NumPy array, a PyTorch tensor or a
            list.
        inputs (np.ndarray | Sequence[float]): The input vector. It can be any
            container that `weights` can be.

    Returns:
        The partial derivative. For a single sample, it is a float. When `inputs`
            is a batch, transposed so that `inputs[j]` holds feature `j` of every
            sample, and the expression contains a `Multiply`, it can be an array
            with one value per sample.
    """
    return _chain_factor(parameter, InputRef, input_index, weights, inputs)


def shift_gate_angle(
    circuit: "CircuitSpec",
    gate_index: int,
    shift_value: float,
) -> "CircuitSpec":
    """Return a copy of the circuit in which the angle of one gate is shifted.

    In the new `CircuitSpec`, the first parameter of the gate at `gate_index` is
    replaced by `Sum((original, Constant(shift_value)))`. The parameter-shift
    Jacobian that applies the chain rule uses it to shift the angle of one gate,
    without changing the weights or the other gates. The shifts are `±π/2`, and
    also `±3π/2` for the gates in `MULTI_FREQUENCY_GATES`.

    Args:
        circuit (CircuitSpec): The circuit to shift.
        gate_index (int): The position of the gate in `circuit.gates`.
        shift_value (float): The amount added to the first angle of that gate.

    Returns:
        A new `CircuitSpec` in which only that gate is shifted.
    """
    if not 0 <= gate_index < len(circuit.gates):
        raise IndexError(
            f"gate_index {gate_index} out of range for circuit with "
            f"{len(circuit.gates)} gates"
        )
    gate = circuit.gates[gate_index]
    if not gate.params:
        raise ValueError(
            f"gate {gate.name} at index {gate_index} has no parametric angle"
        )
    shifted = Sum((gate.params[0], Constant(shift_value)))
    new_params = (shifted,) + gate.params[1:]
    new_gates = list(circuit.gates)
    # `replace` rather than a fresh `Gate`: naming only the three fields the
    # shift touches would silently reset `data` and `adjoint` to their defaults,
    # so a shifted QubitUnitary would lose its matrix and a daggered gate would
    # come back undaggered — a wrong gradient rather than an error.
    new_gates[gate_index] = replace(gate, params=new_params)
    return CircuitSpec(
        n_qubits=circuit.n_qubits,
        n_weights=circuit.n_weights,
        n_inputs=circuit.n_inputs,
        gates=tuple(new_gates),
    )


# ---- transforms ----


def _conj_transpose_nested(mat: tuple) -> tuple:
    """Conjugate transpose of a nested-tuple matrix without numpy.

    Used by `adjoint()` to compute QubitUnitary† in the IR layer,
    keeping `ir.py` free of third-party dependencies.
    """
    n = len(mat)
    return tuple(
        tuple(
            mat[j][i].conjugate() if isinstance(mat[j][i], complex) else mat[j][i]
            for j in range(n)
        )
        for i in range(n)
    )


def adjoint(circuit: CircuitSpec) -> CircuitSpec:
    """Reverse the order of the gates and invert each gate.

    The transform does not depend on the provider. It returns another `CircuitSpec`
    that any backend can compile, so the transform does not have to be written
    again for each backend.

    Each type of gate is inverted as follows:

    * **Rotations with parameters** (`RX`, `RY`, `RZ`, `PhaseShift`, `CRY`,
      `CPHASE`): the angle parameters are negated.
    * **Gates without parameters** (`H`, `X`, `Z`, `CNOT`, `SWAP`, `MCZ`, `MCX`):
      they are copied unchanged, because applying them twice gives the identity.
      `S` and `Sdg` are not their own inverse, so each one is replaced by the
      other.
    * **`QubitUnitary(U)`**: the stored matrix is replaced by its conjugate
      transpose `U†`. It is computed without NumPy, so that the intermediate
      representation stays pure Python.
    * **`ControlledQubitUnitary(U)`**: the target unitary (`data[0]`) is replaced
      by its conjugate transpose, and the control values (`data[1]`) are kept
      unchanged.
    * **`AmplitudeEncoding`**: the transform flips `gate.adjoint` and leaves the
      inversion to the backend. PennyLane, for example, applies the conjugate
      transpose of the state-preparation unitary. The Qiskit backend does not
      support this adjoint: `Circuit` rejects such a circuit with a `ValueError`,
      and the backend itself raises `NotImplementedError`.

    Args:
        circuit (CircuitSpec): The circuit to invert.

    Returns:
        The inverted circuit, with the gates in reverse order.
    """
    _parametric = SINGLE_QUBIT_ROTATIONS | TWO_QUBIT_ROTATIONS
    flipped_gates: list[Gate] = []
    for gate in reversed(circuit.gates):
        if gate.name in _parametric:
            negated = tuple(negate_parameter(p) for p in gate.params)
            flipped_gates.append(Gate(gate.name, gate.qubits, negated))
        elif gate.name == "QubitUnitary":
            # Compute U† in the IR: no backend dependency, no numpy.
            conj_T = _conj_transpose_nested(gate.data[0])
            flipped_gates.append(
                Gate(name="QubitUnitary", qubits=gate.qubits, data=(conj_T,))
            )
        elif gate.name == "ControlledQubitUnitary":
            # U† of a controlled-U: conjugate-transpose the target unitary
            # (data[0]); control_values (data[1]) are preserved unchanged.
            conj_T = _conj_transpose_nested(gate.data[0])
            flipped_gates.append(
                Gate(
                    name="ControlledQubitUnitary",
                    qubits=gate.qubits,
                    data=(conj_T, gate.data[1]),
                )
            )
        elif gate.name == "AmplitudeEncoding":
            # Cannot invert state-prep symbolically; signal the backend.
            flipped_gates.append(
                Gate(
                    name="AmplitudeEncoding",
                    qubits=gate.qubits,
                    data=gate.data,
                    adjoint=not gate.adjoint,
                )
            )
        elif gate.name in NON_PARAMETRIC_GATES:
            adjoint_name = _ADJOINT_NAME.get(gate.name, gate.name)
            flipped_gates.append(Gate(adjoint_name, gate.qubits, gate.params))
        else:
            raise NotImplementedError(f"adjoint for gate '{gate.name}' not defined")
    return CircuitSpec(
        n_qubits=circuit.n_qubits,
        n_weights=circuit.n_weights,
        n_inputs=circuit.n_inputs,
        gates=tuple(flipped_gates),
    )
