"""Functions and classes that make observables easy to write.

There are three ways to choose what a circuit measures:

```python
# Pauli algebra
observable = Z(0) * X(1) + 0.5 * Y(2)

# class methods, or a string
observable = Observable.from_string("Z0 X1 + 0.5 Y2")
observable = Observable.z(0)

# no observable at all: by default, Circuit measures Z on every qubit
```

`PauliExpr` is a simple builder. It collects terms, each made of a coefficient
and a tensor product of Pauli operators, and it converts them to an `Observable`
when needed. A product of two operators on the same qubit is simplified when the
two operators are equal (`P * P = I`). A product of different Pauli operators on
the same qubit (`X * Y = iZ` and so on) needs complex arithmetic and is not
supported.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Iterable

from .ir import Observable, PauliTerm


@dataclass(frozen=True)
class _Term:
    coeff: float
    paulis: tuple[tuple[str, int], ...]  # sorted by qubit, one Pauli per qubit


def _normalize_paulis(
    paulis: Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    """Canonicalize a product of Pauli factors.

    Same Pauli twice on a qubit cancels (P*P = I). Different Paulis on
    the same qubit require complex coefficients (X*Y = iZ, etc.) and are
    rejected here.
    """
    by_qubit: dict[int, str] = {}
    for operator, qubit in paulis:
        if qubit not in by_qubit:
            by_qubit[qubit] = operator
            continue
        existing = by_qubit[qubit]
        if existing == operator:
            # P * P = I
            del by_qubit[qubit]
        else:
            raise ValueError(
                f"product of different Paulis on qubit {qubit}: "
                f"{existing}*{operator} requires complex arithmetic (out of scope)"
            )
    return tuple(
        sorted(
            ((operator, qubit) for qubit, operator in by_qubit.items()),
            key=lambda pair: pair[1],
        )
    )


def _is_scalar(value: object) -> bool:
    """Whether `value` may multiply or offset an expression.

    `numbers.Real` rather than `(int, float)`, so a numpy scalar — what a user
    gets from indexing an array of coefficients — counts as a number instead of
    falling through to the expression branch.

    `bool` is excluded even though it is a `numbers.Real`: `expr + True` is a
    type confusion, not a request for a `1.0` offset, and treating it as a
    coefficient would build that term in silence.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


class PauliExpr:
    """A sum of terms, each a coefficient times a product of Pauli operators.

    In each term, the Pauli operators act on distinct qubits, so the product is a
    tensor product.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: Iterable[_Term]):
        """Build an expression from raw terms.

        Args:
            terms: Internal records of the terms. Prefer the `X`, `Y` and `Z`
                functions and ordinary arithmetic, such as `0.5 * Z(0) + X(1)`.
                They build the same expression without creating these records
                directly.
        """
        self._terms = tuple(terms)

    @classmethod
    def single(cls, operator: str, qubit: int, coeff: float = 1.0) -> "PauliExpr":
        """Return `coeff` times a single Pauli `operator` on `qubit`.

        Args:
            operator (str): The Pauli operator: `"X"`, `"Y"` or `"Z"`.
            qubit (int): The index of the qubit.
            coeff (float): The coefficient. Default: `1.0`.

        Returns:
            The expression, which has a single term.
        """
        return cls((_Term(coeff, ((operator, qubit),)),))

    # ---- arithmetic ----

    def __add__(self, other: "PauliExpr | float | int") -> "PauliExpr":
        if _is_scalar(other):
            if other == 0:
                return self
            return PauliExpr(self._terms + (_Term(float(other), ()),))
        if not isinstance(other, PauliExpr):
            # Not `raise`: returning NotImplemented lets Python try the other
            # operand's reflected method and, failing that, raise the ordinary
            # TypeError. Reaching straight for `other._terms` would surface an
            # AttributeError naming a private slot instead.
            return NotImplemented
        return PauliExpr(self._terms + other._terms)

    __radd__ = __add__

    def __sub__(self, other) -> "PauliExpr":
        if not _is_scalar(other) and not isinstance(other, PauliExpr):
            return NotImplemented
        return self + (-1.0) * other

    def __rsub__(self, other) -> "PauliExpr":
        if not _is_scalar(other) and not isinstance(other, PauliExpr):
            return NotImplemented
        return (-1.0) * self + other

    def __neg__(self) -> "PauliExpr":
        return PauliExpr(_Term(-t.coeff, t.paulis) for t in self._terms)

    def __mul__(self, other: "PauliExpr | float | int") -> "PauliExpr":
        if _is_scalar(other):
            return PauliExpr(_Term(t.coeff * float(other), t.paulis) for t in self._terms)
        if not isinstance(other, PauliExpr):
            return NotImplemented
        product: list[_Term] = []
        for left in self._terms:
            for right in other._terms:
                product.append(
                    _Term(
                        coeff=left.coeff * right.coeff,
                        paulis=_normalize_paulis(left.paulis + right.paulis),
                    )
                )
        return PauliExpr(product)

    __rmul__ = __mul__

    # ---- conversion ----

    @property
    def constant(self) -> float:
        """The coefficient of the identity term, or `0.0` when there is none."""
        return sum(term.coeff for term in self._terms if not term.paulis)

    def to_observable(self) -> Observable:
        """Convert the expression to an `Observable`.

        The terms with the same Pauli string are combined, and the terms whose
        coefficient is `0` (at most `1e-15` in magnitude) are dropped.

        An `Observable` holds only Pauli terms, so a non-zero identity term raises
        an error. Such a term comes from a scalar offset, or from `P * P = I`.
        Dropping it without an error would shift every expectation value. To
        measure such an expression, measure `expr - expr.constant` and add
        `expr.constant` to the result. The chemistry entry points take this
        constant as `offset`.

        Returns:
            The observable.

        Raises:
            ValueError: If the expression has a constant term that is not `0`, or
                if no term that can be measured is left.
        """
        combined: dict[tuple[tuple[str, int], ...], float] = {}
        for term in self._terms:
            key = term.paulis
            combined[key] = combined.get(key, 0.0) + term.coeff
        constant = combined.get((), 0.0)
        if abs(constant) > 1e-12:
            raise ValueError(
                f"constant term {constant:g} in the expression: an Observable holds "
                "Pauli terms only, and dropping the constant would shift every "
                "expectation value. Measure `expr - expr.constant` and add "
                "`expr.constant` to the result."
            )
        terms = tuple(
            PauliTerm(coeff=coeff, paulis=paulis)
            for paulis, coeff in combined.items()
            if paulis and abs(coeff) > 1e-15
        )
        if not terms:
            raise ValueError(
                "observable reduced to zero or identity-only; nothing to measure"
            )
        return Observable(terms=terms)


# ---- public Pauli constructors ----


def X(qubit: int) -> PauliExpr:  # noqa: N802 — conventional quantum notation
    """Return Pauli `X` on `qubit`, as an expression that can be added and scaled.

    Args:
        qubit (int): The index of the qubit.

    Returns:
        The expression.
    """
    return PauliExpr.single("X", qubit)


def Y(qubit: int) -> PauliExpr:  # noqa: N802
    """Return Pauli `Y` on `qubit`, as an expression that can be added and scaled.

    Args:
        qubit (int): The index of the qubit.

    Returns:
        The expression.
    """
    return PauliExpr.single("Y", qubit)


def Z(qubit: int) -> PauliExpr:  # noqa: N802
    """Return Pauli `Z` on `qubit`, as an expression that can be added and scaled.

    Args:
        qubit (int): The index of the qubit.

    Returns:
        The expression.
    """
    return PauliExpr.single("Z", qubit)


def all_z(n_qubits: int) -> list[Observable]:
    """Return one `Z` observable for each qubit.

    This is useful as a default measurement.

    Args:
        n_qubits (int): The number of qubits of the circuit.

    Returns:
        One `Observable` for each qubit.
    """
    return [Observable.z(q) for q in range(n_qubits)]


def from_string(text: str) -> Observable:
    """Parse an observable from a string.

    This function is a shortcut for `Observable.from_string`.

    Args:
        text (str): The string to parse.

    Returns:
        The parsed observable.
    """
    return Observable.from_string(text)
