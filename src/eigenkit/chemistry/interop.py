"""OpenFermion interop (the optional ``chemistry`` extra).

OpenFermion's `FermionOperator` and `QubitOperator` share our term structure
exactly — a `FermionOperator` term is a tuple of ``(mode, action)`` with
``action`` 1=creation / 0=annihilation, and a `QubitOperator` term is a tuple
of ``(qubit, 'X'|'Y'|'Z')`` — so the conversions are direct. These adapters let
you (a) bring an OpenFermion-built Hamiltonian onto the spine, and (b) reach
mappings OpenFermion provides (e.g. Bravyi-Kitaev) via our `Observable`.

The ``->`` conversions read only the ``.terms`` mapping (no OpenFermion import
needed); the ``to_openfermion`` direction imports OpenFermion lazily.
"""

from __future__ import annotations

from ..core.ir import Observable
from ._pauli import PauliSum
from .fermion import FermionOperator


def fermion_operator_from_openfermion(operator) -> FermionOperator:
    """Convert an `openfermion.FermionOperator` to an eigenkit `FermionOperator`.

    Args:
        operator (openfermion.FermionOperator): The fermionic operator to convert.

    Returns:
        The same operator, as an eigenkit `FermionOperator`.
    """
    return FermionOperator(
        {term: complex(coeff) for term, coeff in operator.terms.items()}
    )


def to_openfermion_fermion_operator(operator: FermionOperator):
    """Convert an eigenkit `FermionOperator` to an `openfermion.FermionOperator`.

    Args:
        operator (FermionOperator): The fermionic operator to convert.

    Returns:
        (openfermion.FermionOperator): The same operator, in OpenFermion.
    """
    from openfermion import FermionOperator as OpenFermionOperator

    result = OpenFermionOperator()
    for term, coefficient in operator.terms.items():
        result += OpenFermionOperator(term, coefficient)
    return result


def observable_from_openfermion(
    qubit_operator, *, atol: float = 1e-10
) -> tuple[Observable, float]:
    """Convert an `openfermion.QubitOperator` to an `(Observable, offset)` pair.

    The identity term becomes the real `offset`. The coefficients must be real,
    because the operator must be Hermitian, and a `ValueError` is raised otherwise.

    Args:
        qubit_operator (openfermion.QubitOperator): The operator to convert.
        atol (float): Terms whose coefficient is smaller than this value in
            magnitude are dropped.

    Returns:
        observable: The operator, without its identity term.
        offset: The coefficient of the identity term.
    """
    total = PauliSum()
    for term, coefficient in qubit_operator.terms.items():
        string = {qubit: pauli for qubit, pauli in term}
        total._add_term(tuple(sorted(string.items())), complex(coefficient))
    return total.to_observable(atol=atol)
