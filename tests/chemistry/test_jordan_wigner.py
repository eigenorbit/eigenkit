"""Jordan-Wigner correctness against analytic identities.

These pin the convention down with operators whose Pauli image is known by
hand — no fabricated constants:
  - number operator  n_j = a_j^† a_j = (I - Z_j)/2,
  - canonical anticommutators {a_i, a_j^†} = delta_ij I,
  - non-Hermitian inputs are rejected.
"""

from __future__ import annotations

import pytest

from eigenkit.chemistry import FermionOperator, jordan_wigner


def _terms(observable):
    """Map an Observable to {paulis: coeff} for order-independent comparison."""
    return {term.paulis: pytest.approx(term.coeff) for term in observable.terms}


class TestNumberOperator:
    def test_mode_0(self):
        # n_0 = (I - Z_0)/2  ->  offset 0.5, term -0.5 Z_0.
        observable, offset = jordan_wigner(FermionOperator.number(0))
        assert offset == pytest.approx(0.5)
        assert _terms(observable) == {(("Z", 0),): -0.5}

    def test_mode_2_parity_cancels(self):
        # n_2 maps to (I - Z_2)/2: the JW parity string Z_0 Z_1 cancels.
        observable, offset = jordan_wigner(FermionOperator.number(2))
        assert offset == pytest.approx(0.5)
        assert _terms(observable) == {(("Z", 2),): -0.5}


class TestAnticommutators:
    def test_same_mode_is_identity(self):
        # {a_0, a_0^†} = a_0 a_0^† + a_0^† a_0 = I.
        op = FermionOperator.annihilation(0) * FermionOperator.creation(0)
        op = op + FermionOperator.creation(0) * FermionOperator.annihilation(0)
        observable, offset = jordan_wigner(op)
        assert offset == pytest.approx(1.0)
        assert observable.terms == ()

    def test_different_modes_vanish(self):
        # {a_0, a_1^†} = 0.
        op = FermionOperator.annihilation(0) * FermionOperator.creation(1)
        op = op + FermionOperator.creation(1) * FermionOperator.annihilation(0)
        observable, offset = jordan_wigner(op)
        assert offset == pytest.approx(0.0)
        assert observable.terms == ()


class TestHermiticity:
    def test_non_hermitian_input_rejected(self):
        # A bare creation operator maps to (X - iY)/2 — complex coefficients.
        with pytest.raises(ValueError, match="not Hermitian"):
            jordan_wigner(FermionOperator.creation(0))

    def test_hopping_term_is_hermitian(self):
        # a_0^† a_1 + a_1^† a_0 is Hermitian -> real Pauli coefficients.
        op = FermionOperator.creation(0) * FermionOperator.annihilation(1)
        op = op + FermionOperator.creation(1) * FermionOperator.annihilation(0)
        observable, offset = jordan_wigner(op)
        assert offset == pytest.approx(0.0)
        # = (X_0 X_1 + Y_0 Y_1) / 2.
        assert _terms(observable) == {
            (("X", 0), ("X", 1)): 0.5,
            (("Y", 0), ("Y", 1)): 0.5,
        }
