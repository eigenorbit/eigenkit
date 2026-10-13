"""OpenFermion interop adapters. Skipped unless OpenFermion (the chemistry
extra) is installed; runs in CI where the extra is present."""

from __future__ import annotations

import pytest

from eigenkit.chemistry import (
    FermionOperator,
    fermion_operator_from_openfermion,
    observable_from_openfermion,
    to_openfermion_fermion_operator,
)


pytest.importorskip("openfermion")


def test_observable_from_qubit_operator():
    from openfermion import QubitOperator

    qubit_operator = QubitOperator("X0 Z1", 0.5) + QubitOperator("", 1.2)
    observable, offset = observable_from_openfermion(qubit_operator)
    assert offset == pytest.approx(1.2)  # identity term -> offset
    terms = {term.paulis: term.coeff for term in observable.terms}
    assert terms == {(("X", 0), ("Z", 1)): pytest.approx(0.5)}


def test_fermion_operator_round_trip():
    operator = FermionOperator.creation(2) * FermionOperator.annihilation(0)
    operator = operator + FermionOperator.number(1) * 0.5
    round_tripped = fermion_operator_from_openfermion(
        to_openfermion_fermion_operator(operator)
    )
    assert round_tripped.terms == operator.terms


def test_non_hermitian_qubit_operator_rejected():
    from openfermion import QubitOperator

    with pytest.raises(ValueError, match="not Hermitian"):
        observable_from_openfermion(QubitOperator("X0", 0.5j))
