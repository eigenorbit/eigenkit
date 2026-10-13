"""Tests for Pauli algebra, same-qubit cancellation, and observable building."""

from __future__ import annotations

import pytest

from eigenkit.core.observables import X, Y, Z, all_z, from_string


class TestSingleConstructors:
    def test_z_to_observable(self):
        obs = Z(0).to_observable()
        assert obs.terms[0].coeff == 1.0
        assert obs.terms[0].paulis == (("Z", 0),)

    def test_x_y_z_differ(self):
        assert X(0).to_observable().terms[0].paulis == (("X", 0),)
        assert Y(0).to_observable().terms[0].paulis == (("Y", 0),)
        assert Z(0).to_observable().terms[0].paulis == (("Z", 0),)


class TestAlgebra:
    def test_scalar_times_pauli(self):
        obs = (0.5 * Z(0)).to_observable()
        assert obs.terms[0].coeff == 0.5

    def test_pauli_times_scalar(self):
        obs = (Z(0) * 0.5).to_observable()
        assert obs.terms[0].coeff == 0.5

    def test_sum_of_paulis(self):
        obs = (Z(0) + X(1)).to_observable()
        assert len(obs.terms) == 2

    def test_difference(self):
        obs = (Z(0) - 0.5 * X(1)).to_observable()
        coeffs = sorted([t.coeff for t in obs.terms])
        assert coeffs == [-0.5, 1.0]

    def test_tensor_product_distinct_qubits(self):
        obs = (Z(0) * X(1)).to_observable()
        assert obs.terms[0].paulis == (("Z", 0), ("X", 1))

    def test_scalar_addition_is_refused(self):
        """Dropping the identity term would shift every expectation value."""
        expr = Z(0) + 1.0
        assert expr.constant == 1.0
        with pytest.raises(ValueError, match="constant term 1"):
            expr.to_observable()

    def test_subtracting_the_constant_leaves_the_pauli_part(self):
        expr = 0.5 * (1 - Z(0) * Z(1))
        obs = (expr - expr.constant).to_observable()
        assert [(t.coeff, t.paulis) for t in obs.terms] == [(-0.5, (("Z", 0), ("Z", 1)))]

    def test_negation(self):
        obs = (-Z(0)).to_observable()
        assert obs.terms[0].coeff == -1.0


class TestSameQubitCancellation:
    """P * P = I — both cancel; different Paulis on same qubit raise."""

    def test_z_times_z_is_identity(self):
        with pytest.raises(ValueError, match="constant term 1"):
            (Z(0) * Z(0)).to_observable()

    def test_z_times_z_in_mixed_sum_keeps_the_constant(self):
        expr = Z(0) * Z(0) + X(1)
        assert expr.constant == 1.0
        obs = (expr - expr.constant).to_observable()
        assert [t.paulis for t in obs.terms] == [(("X", 1),)]

    def test_x_times_y_on_same_qubit_rejected(self):
        with pytest.raises(ValueError, match="requires complex arithmetic"):
            (X(0) * Y(0)).to_observable()

    def test_three_paulis_distinct_qubits(self):
        obs = (X(0) * Y(1) * Z(2)).to_observable()
        assert obs.terms[0].paulis == (("X", 0), ("Y", 1), ("Z", 2))


class TestCombining:
    def test_duplicate_paulis_combine_coefficients(self):
        obs = (Z(0) + 0.5 * Z(0)).to_observable()
        assert len(obs.terms) == 1
        assert obs.terms[0].coeff == 1.5

    def test_canceling_terms_drop(self):
        with pytest.raises(ValueError, match="zero or identity-only"):
            (Z(0) - Z(0)).to_observable()


class TestAllZ:
    def test_returns_list_of_observables(self):
        observables = all_z(3)
        assert len(observables) == 3
        for index, obs in enumerate(observables):
            assert obs.terms[0].paulis == (("Z", index),)


class TestFromStringAlias:
    def test_module_alias_matches_classmethod(self):
        # Both should produce the same Observable for the same input.
        from eigenkit.core.ir import Observable as Obs

        a = from_string("0.5 Z0 X1")
        b = Obs.from_string("0.5 Z0 X1")
        assert a.terms == b.terms
