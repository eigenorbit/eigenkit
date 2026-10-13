r"""Minimal Pauli-string algebra used to build qubit Hamiltonians.

A `PauliSum` is a linear combination of Pauli strings with complex
coefficients. A Pauli string is a map ``{qubit: 'X'|'Y'|'Z'}`` (identity on
omitted qubits). This is just enough algebra (multiply, add, scale) for the
fermion-to-qubit mappings to expand ladder operators into Pauli operators;
it is not a general-purpose operator class.

Single-qubit products follow the standard relations, e.g. \(X \cdot Y = iZ\),
\(Y \cdot X = -iZ\), \(X \cdot X = I\).
"""

from __future__ import annotations

from ..core.ir import Observable, PauliTerm


# sigma_a · sigma_b = phase · sigma_c  (None == identity), for a, b in X/Y/Z.
_PRODUCT: dict[tuple[str, str], tuple[complex, str | None]] = {
    ("X", "X"): (1, None),
    ("Y", "Y"): (1, None),
    ("Z", "Z"): (1, None),
    ("X", "Y"): (1j, "Z"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("X", "Z"): (-1j, "Y"),
}


def _multiply_strings(
    left: dict[int, str], right: dict[int, str]
) -> tuple[complex, dict[int, str]]:
    """Return ``(phase, product_string)`` for ``left · right`` (left first)."""
    phase: complex = 1 + 0j
    result = dict(left)
    for qubit, op in right.items():
        if qubit in result:
            factor, combined = _PRODUCT[(result[qubit], op)]
            phase *= factor
            if combined is None:
                del result[qubit]
            else:
                result[qubit] = combined
        else:
            result[qubit] = op
    return phase, result


def _key(string: dict[int, str]) -> tuple[tuple[int, str], ...]:
    return tuple(sorted(string.items()))


class PauliSum:
    """A complex linear combination of Pauli strings, keyed by sorted
    ``(qubit, op)`` tuples (the empty key is the identity)."""

    def __init__(self) -> None:
        self.terms: dict[tuple[tuple[int, str], ...], complex] = {}

    @classmethod
    def identity(cls, coefficient: complex = 1.0) -> "PauliSum":
        result = cls()
        if coefficient != 0:
            result.terms[()] = complex(coefficient)
        return result

    def _add_term(self, key: tuple[tuple[int, str], ...], coefficient: complex) -> None:
        total = self.terms.get(key, 0j) + coefficient
        if total == 0:
            self.terms.pop(key, None)
        else:
            self.terms[key] = total

    def __add__(self, other: "PauliSum") -> "PauliSum":
        result = PauliSum()
        result.terms = dict(self.terms)
        for key, coefficient in other.terms.items():
            result._add_term(key, coefficient)
        return result

    def __mul__(self, other: "PauliSum") -> "PauliSum":
        result = PauliSum()
        for key_l, coeff_l in self.terms.items():
            left = dict(key_l)
            for key_r, coeff_r in other.terms.items():
                phase, combined = _multiply_strings(left, dict(key_r))
                result._add_term(_key(combined), coeff_l * coeff_r * phase)
        return result

    def to_observable(self, *, atol: float = 1e-10) -> tuple[Observable, float]:
        """Return ``(Observable, offset)``.

        The identity term becomes the real ``offset`` (the Pauli backends
        reject identity-only terms); the remaining terms become an
        `Observable`. Coefficients must be real to numerical tolerance — a
        Hermitian operator always maps to real Pauli coefficients; a non-real
        coefficient signals a non-Hermitian input and raises.
        """
        offset = 0.0
        pauli_terms: list[PauliTerm] = []
        for key, coefficient in self.terms.items():
            if abs(coefficient.imag) > atol:
                raise ValueError(
                    f"non-real Pauli coefficient {coefficient} for term {key}; "
                    "the operator is not Hermitian"
                )
            value = coefficient.real
            if abs(value) <= atol:
                continue
            if not key:
                offset += value
                continue
            paulis = tuple((op, qubit) for qubit, op in key)
            pauli_terms.append(PauliTerm(coeff=value, paulis=paulis))
        return Observable(terms=tuple(pauli_terms)), offset
