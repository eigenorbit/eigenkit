r"""Fermionic operators: linear combinations of products of creation /
annihilation operators on spin-orbital modes.

Conventions (OpenFermion-compatible):
- A *mode* is a spin-orbital, indexed ``0 .. M-1``.
- A *ladder operator* is ``(mode, action)`` with ``action = 1`` for creation
  \(a_{\text{mode}}^\dagger\) and ``action = 0`` for annihilation ``a_mode``.
- A *term* is a tuple of ladder operators applied left-to-right, e.g.
  \(a_2^\dagger a_0\) is ``((2, 1), (0, 0))``.

Operators are NOT normal-ordered (the Jordan-Wigner map handles raw products
directly), so \(a_p^\dagger a_q\) and \(a_q a_p^\dagger\) are stored as written.
"""

from __future__ import annotations

from typing import Union


Action = int  # 1 = creation, 0 = annihilation
Term = tuple[tuple[int, Action], ...]
Scalar = Union[int, float, complex]


class FermionOperator:
    """A complex linear combination of fermionic terms, stored as `{term: coeff}`."""

    def __init__(self, terms: dict[Term, complex] | None = None) -> None:
        """Build an operator from a mapping of terms to coefficients.

        Args:
            terms: Maps each term to its coefficient. A term is a tuple of
                `(mode, action)` pairs, where `action` is `1` for creation and `0` for
                annihilation, applied from left to right. The empty tuple is the
                identity. The coefficients of repeated terms are added together. Leave
                it out for the zero operator. The default is `None`.
        """
        self.terms: dict[Term, complex] = {}
        if terms:
            for term, coefficient in terms.items():
                self._add(term, complex(coefficient))

    # ---- constructors ----

    @classmethod
    def zero(cls) -> "FermionOperator":
        """The zero operator, which has no terms.

        Returns:
            The zero `FermionOperator`.
        """
        return cls()

    @classmethod
    def identity(cls, coefficient: Scalar = 1.0) -> "FermionOperator":
        """The identity operator, optionally scaled.

        Args:
            coefficient: The factor that multiplies the identity. The default is `1.0`.

        Returns:
            The scaled identity, as a `FermionOperator`.
        """
        return cls({(): complex(coefficient)})

    @classmethod
    def creation(cls, mode: int) -> "FermionOperator":
        """The creation operator `a†_mode`.

        Args:
            mode: The index of the spin-orbital in which an electron is created.

        Returns:
            The operator, as a `FermionOperator`.
        """
        return cls({((mode, 1),): 1.0 + 0j})

    @classmethod
    def annihilation(cls, mode: int) -> "FermionOperator":
        """The annihilation operator `a_mode`.

        Args:
            mode: The index of the spin-orbital from which an electron is removed.

        Returns:
            The operator, as a `FermionOperator`.
        """
        return cls({((mode, 0),): 1.0 + 0j})

    @classmethod
    def number(cls, mode: int) -> "FermionOperator":
        """The number operator `a†_mode a_mode`.

        It counts the electrons in one spin-orbital.

        Args:
            mode (int): The index of the spin-orbital.

        Returns:
            The operator, as a `FermionOperator`.
        """
        return cls({((mode, 1), (mode, 0)): 1.0 + 0j})

    # ---- algebra ----

    def _add(self, term: Term, coefficient: complex) -> None:
        total = self.terms.get(term, 0j) + coefficient
        if total == 0:
            self.terms.pop(term, None)
        else:
            self.terms[term] = total

    def __add__(self, other: "FermionOperator") -> "FermionOperator":
        result = FermionOperator(dict(self.terms))
        for term, coefficient in other.terms.items():
            result._add(term, coefficient)
        return result

    def __sub__(self, other: "FermionOperator") -> "FermionOperator":
        return self + (-1.0) * other

    def __mul__(self, other: Union["FermionOperator", Scalar]) -> "FermionOperator":
        if isinstance(other, (int, float, complex)):
            return FermionOperator(
                {term: coeff * other for term, coeff in self.terms.items()}
            )
        result = FermionOperator()
        for term_l, coeff_l in self.terms.items():
            for term_r, coeff_r in other.terms.items():
                result._add(term_l + term_r, coeff_l * coeff_r)
        return result

    def __rmul__(self, scalar: Scalar) -> "FermionOperator":
        return self * scalar

    def __neg__(self) -> "FermionOperator":
        return self * (-1.0)

    def hermitian_conjugate(self) -> "FermionOperator":
        """The adjoint of the operator.

        The order of each term is reversed, creation and annihilation are swapped, and
        each coefficient is replaced by its complex conjugate.

        Returns:
            The adjoint, as a `FermionOperator`.
        """
        result = FermionOperator()
        for term, coefficient in self.terms.items():
            dagger_term = tuple((mode, 1 - action) for mode, action in reversed(term))
            result._add(dagger_term, coefficient.conjugate())
        return result

    def __repr__(self) -> str:
        return f"FermionOperator({self.terms!r})"
