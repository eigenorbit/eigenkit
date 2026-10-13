r"""Fermion-to-qubit mappings: build a qubit `Observable` from a
`FermionOperator`.

Jordan-Wigner convention (OpenFermion-compatible):

\[
a_j = \Big(\prod_{k<j} Z_k\Big)\,\frac{X_j + i Y_j}{2}
\qquad
a_j^\dagger = \Big(\prod_{k<j} Z_k\Big)\,\frac{X_j - i Y_j}{2}
\]

so the number operator maps to \(a_j^\dagger a_j = (I - Z_j)/2\), with an
occupied orbital being qubit \(|1\rangle\). The parity string
\(\prod_{k<j} Z_k\) is what enforces fermionic antisymmetry.

`jordan_wigner` returns ``(Observable, offset)``: the constant (identity)
term is returned separately as ``offset`` because the Pauli backends reject
identity-only terms, and a variational solver adds it back to ``<H>``.
"""

from __future__ import annotations

from ._pauli import PauliSum
from .fermion import FermionOperator


def _jordan_wigner_ladder(mode: int, action: int) -> PauliSum:
    """JW image of a single ladder operator on ``mode``.

    ``action = 1`` (creation) uses ``(X - iY)/2``; ``action = 0`` (annihilation)
    uses ``(X + iY)/2``; both carry the parity string ``Z_0 ... Z_{mode-1}``.
    """
    parity = {k: "Z" for k in range(mode)}
    x_string = dict(parity)
    x_string[mode] = "X"
    y_string = dict(parity)
    y_string[mode] = "Y"

    y_sign = -1j if action == 1 else 1j  # creation: -iY, annihilation: +iY
    result = PauliSum()
    result._add_term(tuple(sorted(x_string.items())), 0.5)
    result._add_term(tuple(sorted(y_string.items())), 0.5 * y_sign)
    return result


def jordan_wigner(operator: FermionOperator, *, atol: float = 1e-10):
    """Map a `FermionOperator` to qubits with the Jordan-Wigner transformation.

    The operator must be Hermitian, as a Hamiltonian is. If a resulting
    coefficient is not real, a `ValueError` is raised.

    Args:
        operator (FermionOperator): The fermionic operator to map.
        atol (float): Terms whose coefficient is smaller than this value in
            magnitude are dropped.

    Returns:
        observable (Observable): The operator on qubits, without its identity part.
        offset (float): The coefficient of the identity part, which the
            `Observable` does not contain.
    """
    total = PauliSum()
    for term, coefficient in operator.terms.items():
        product = PauliSum.identity(coefficient)
        for mode, action in term:
            product = product * _jordan_wigner_ladder(mode, action)
        total = total + product
    return total.to_observable(atol=atol)


def bravyi_kitaev(operator: FermionOperator, n_qubits: int | None = None):
    """Map a `FermionOperator` to qubits with the Bravyi-Kitaev transformation.

    The transformation itself is done by OpenFermion (the `chemistry` extra), and
    the result is converted to an eigenkit `Observable`. The Jordan-Wigner mapping,
    in contrast, is implemented in eigenkit. `n_qubits` defaults to the largest
    mode index plus one.

    Args:
        operator (FermionOperator): The fermionic operator to map.
        n_qubits (int): The number of qubits.

    Returns:
        observable (Observable): The operator on qubits, without its identity part.
        offset (float): The coefficient of the identity part, which the
            `Observable` does not contain.
    """
    from openfermion import bravyi_kitaev as openfermion_bravyi_kitaev

    from .interop import observable_from_openfermion, to_openfermion_fermion_operator

    qubit_operator = openfermion_bravyi_kitaev(
        to_openfermion_fermion_operator(operator), n_qubits=n_qubits
    )
    return observable_from_openfermion(qubit_operator)
