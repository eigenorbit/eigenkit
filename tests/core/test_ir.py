"""Tests for the IR module: parameters, resolve, adjoint, Observable, parsing."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from eigenkit.core.ir import (
    ALL_GATES,
    CircuitSpec,
    Constant,
    Gate,
    InputRef,
    Multiply,
    Negate,
    Observable,
    PauliTerm,
    Scale,
    Sum,
    WeightRef,
    adjoint,
    chain_factor,
    collect_weight_dependencies,
    negate_parameter,
    resolve,
    shift_gate_angle,
)


# ---- parameter kinds and resolve ----


class TestResolveLeaves:
    def test_constant(self):
        assert resolve(Constant(0.5), [], []) == 0.5

    def test_weight_ref(self):
        assert resolve(WeightRef(2), [10, 20, 30], []) == 30

    def test_input_ref(self):
        assert resolve(InputRef(0), [], [7.5]) == 7.5


class TestResolveExpressions:
    def test_scale(self):
        assert resolve(Scale(WeightRef(0), 2.0), [3.0], []) == 6.0

    def test_negate(self):
        assert resolve(Negate(InputRef(0)), [], [4.0]) == -4.0

    def test_sum_two(self):
        assert resolve(Sum((WeightRef(0), InputRef(0))), [1.0], [2.0]) == 3.0

    def test_sum_many(self):
        expression = Sum((Constant(1.0), Constant(2.0), Constant(3.0)))
        assert resolve(expression, [], []) == 6.0

    def test_nested_scale_negate(self):
        expression = Negate(Scale(WeightRef(0), 2.0))
        assert resolve(expression, [3.0], []) == -6.0

    def test_nested_sum_scale(self):
        expression = Sum((Scale(WeightRef(0), 2.0), InputRef(0)))
        assert resolve(expression, [1.0], [0.5]) == 2.5

    def test_works_with_numpy_arrays(self):
        weights = np.array([1.0, 2.0, 3.0])
        assert resolve(WeightRef(1), weights, np.array([])) == 2.0

    def test_multiply_two_refs(self):
        # lambda * x where lambda = weights[0], x = inputs[0]
        assert resolve(Multiply(WeightRef(0), InputRef(0)), [0.5], [4.0]) == 2.0

    def test_multiply_with_constant(self):
        assert resolve(Multiply(WeightRef(0), Constant(3.0)), [2.0], []) == 6.0

    def test_multiply_nested(self):
        # (lambda * x) + b   where b = weights[1]
        expression = Sum((Multiply(WeightRef(0), InputRef(0)), WeightRef(1)))
        assert resolve(expression, [2.0, 1.5], [3.0]) == 7.5

    def test_multiply_with_torch_tensors(self):
        import torch

        weights = torch.tensor([2.0])
        inputs = torch.tensor([3.0])
        result = resolve(Multiply(WeightRef(0), InputRef(0)), weights, inputs)
        assert result.item() == 6.0
        # Multiply preserves grad-fn — important for trainable_scale in
        # native-autograd backends.
        weights.requires_grad_()
        result = resolve(Multiply(WeightRef(0), InputRef(0)), weights, inputs)
        assert result.requires_grad

    def test_rejects_unknown_parameter(self):
        with pytest.raises(TypeError, match="unknown parameter kind"):
            resolve("not a parameter", [], [])


# ---- negate_parameter ----


class TestNegateParameter:
    def test_constant(self):
        assert negate_parameter(Constant(1.5)) == Constant(-1.5)

    def test_double_negation_cancels(self):
        inner = WeightRef(0)
        assert negate_parameter(Negate(inner)) == inner

    def test_scale_flips_factor(self):
        assert negate_parameter(Scale(WeightRef(0), 2.0)) == Scale(WeightRef(0), -2.0)

    def test_symbolic_wraps_in_negate(self):
        reference = WeightRef(0)
        assert negate_parameter(reference) == Negate(reference)

    def test_input_ref_wraps(self):
        reference = InputRef(3)
        assert negate_parameter(reference) == Negate(reference)

    def test_multiply_wraps_in_negate(self):
        product = Multiply(WeightRef(0), InputRef(0))
        assert negate_parameter(product) == Negate(product)

    def test_negate_of_multiply_evaluates_correctly(self):
        # Adjoint of RY(lambda * x) needs to negate the angle in the IR.
        product = Multiply(WeightRef(0), InputRef(0))
        negated = negate_parameter(product)
        assert resolve(negated, [0.5], [4.0]) == -2.0


# ---- PauliTerm validation ----


class TestPauliTerm:
    def test_constructs_simple(self):
        term = PauliTerm(coeff=0.5, paulis=(("Z", 0), ("X", 1)))
        assert term.coeff == 0.5
        assert term.paulis == (("Z", 0), ("X", 1))

    def test_rejects_complex_coefficient(self):
        with pytest.raises(TypeError, match="must be real"):
            PauliTerm(coeff=1j, paulis=(("Z", 0),))

    def test_rejects_non_pauli_operator(self):
        with pytest.raises(ValueError, match="Pauli operator"):
            PauliTerm(coeff=1.0, paulis=(("S", 0),))

    def test_rejects_duplicate_qubit(self):
        with pytest.raises(ValueError, match="more than once"):
            PauliTerm(coeff=1.0, paulis=(("Z", 0), ("X", 0)))


# ---- Observable constructors ----


class TestObservableConstructors:
    def test_z(self):
        obs = Observable.z(2)
        assert obs.terms[0].paulis == (("Z", 2),)
        assert obs.terms[0].coeff == 1.0

    def test_x(self):
        assert Observable.x(0).terms[0].paulis == (("X", 0),)

    def test_y(self):
        assert Observable.y(1).terms[0].paulis == (("Y", 1),)


class TestObservableFromString:
    def test_single_term(self):
        obs = Observable.from_string("Z0")
        assert len(obs.terms) == 1
        assert obs.terms[0].coeff == 1.0
        assert obs.terms[0].paulis == (("Z", 0),)

    def test_coefficient(self):
        obs = Observable.from_string("0.5 Z0")
        assert obs.terms[0].coeff == 0.5

    def test_negative_term(self):
        obs = Observable.from_string("-Z0")
        assert obs.terms[0].coeff == -1.0

    def test_negative_with_coeff(self):
        obs = Observable.from_string("-0.3 Z0")
        assert obs.terms[0].coeff == -0.3

    def test_multiple_paulis(self):
        obs = Observable.from_string("Z0 X1")
        assert obs.terms[0].paulis == (("Z", 0), ("X", 1))

    def test_multiple_terms(self):
        obs = Observable.from_string("Z0 + X1")
        assert len(obs.terms) == 2
        assert obs.terms[0].paulis == (("Z", 0),)
        assert obs.terms[1].paulis == (("X", 1),)

    def test_mixed_signs(self):
        obs = Observable.from_string("Z0 - 0.5 X1 + Y2")
        coeffs = [t.coeff for t in obs.terms]
        assert coeffs == [1.0, -0.5, 1.0]

    def test_sorts_paulis_by_qubit(self):
        obs = Observable.from_string("X2 Z0")
        assert obs.terms[0].paulis == (("Z", 0), ("X", 2))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            Observable.from_string("")

    @pytest.mark.parametrize("text", ["0.5", "2 I", "0.5 + Z0", "0.5 I + Z0"])
    def test_a_constant_term_is_rejected_not_dropped(self, text):
        # An Observable holds Pauli terms only — chemistry keeps the constant
        # beside it as `offset`. Skipping the constant and returning the rest
        # would move the whole spectrum without saying so.
        with pytest.raises(ValueError, match="constant term"):
            Observable.from_string(text)

    def test_scientific_notation_is_read_as_one_number(self):
        # A grammar that matches "1", skips the "e", and reads "-3" as the
        # coefficient of "1e-3 Z0" yields a silently wrong Hamiltonian.
        assert Observable.from_string("1e-3 Z0").terms[0].coeff == 0.001
        assert Observable.from_string("1.5E+2 Z0").terms[0].coeff == 150.0

    def test_an_explicit_multiplication_sign_is_allowed(self):
        assert Observable.from_string("0.5 * Z0") == Observable.from_string("0.5 Z0")

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            Observable.from_string("!!!")

    @pytest.mark.parametrize("text", ["Z0 Q1", "0.5 I0", "Z0 +", "1e", "0.5 Z"])
    def test_a_term_the_grammar_misses_is_rejected_not_skipped(self, text):
        # Dropping whatever the grammar cannot match would let a typo remove
        # a term from the operator instead of failing.
        with pytest.raises(ValueError, match="cannot parse observable"):
            Observable.from_string(text)


class TestConstantTermsAreRejected:
    """A constant is not part of an Observable.

    A `PauliTerm` with no Paulis is the identity times its coefficient. The
    backends would not agree on it: numpy, qiskit, qiskit-aer, qsim, qulacs and
    tensorcircuit would compute it, pennylane would raise, and braket would
    silently drop it -- returning a number short by exactly the constant, which
    for a chemistry Hamiltonian is the nuclear-repulsion offset. Rather than
    pick a winner and rely on every future backend following, the term cannot
    be constructed; the constant travels beside the observable, which is what
    `jordan_wigner` returns as `offset`.
    """

    def test_a_term_with_no_paulis_cannot_be_built(self):
        with pytest.raises(ValueError, match="holds Pauli terms only"):
            PauliTerm(2.0, ())

    def test_the_message_names_the_constant_and_the_way_out(self):
        with pytest.raises(ValueError, match=r"constant 2\.5"):
            PauliTerm(2.5, ())
        with pytest.raises(ValueError, match="offset"):
            PauliTerm(2.5, ())

    def test_an_ordinary_term_is_unaffected(self):
        assert PauliTerm(2.0, (("Z", 0),)).coeff == 2.0

    def test_an_empty_observable_is_still_a_valid_value(self):
        # `jordan_wigner` returns one, paired with an offset, when the operator
        # reduces to a multiple of the identity. It is measuring it that fails.
        assert Observable(terms=()).terms == ()


# ---- Gate and CircuitSpec basics ----


class TestGate:
    def test_constructs(self):
        gate = Gate(name="RX", qubits=(0,), params=(Constant(0.5),))
        assert gate.name == "RX"
        assert gate.qubits == (0,)

    def test_frozen(self):
        gate = Gate(name="H", qubits=(0,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            gate.name = "X"  # type: ignore[misc]


class TestCircuitSpec:
    def test_constructs(self):
        ir = CircuitSpec(n_qubits=2, n_weights=1, n_inputs=0, gates=(Gate("H", (0,)),))
        assert ir.n_qubits == 2
        assert len(ir.gates) == 1

    def test_all_gates_catalog(self):
        assert "RX" in ALL_GATES
        assert "CNOT" in ALL_GATES
        assert "H" in ALL_GATES


# ---- adjoint transform ----


class TestAdjoint:
    def test_reverses_order(self):
        ir = CircuitSpec(
            n_qubits=2,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("H", (0,)), Gate("CNOT", (0, 1))),
        )
        result = adjoint(ir)
        assert [g.name for g in result.gates] == ["CNOT", "H"]

    def test_negates_rotation_constants(self):
        ir = CircuitSpec(
            n_qubits=1,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("RX", (0,), (Constant(0.5),)),),
        )
        result = adjoint(ir)
        assert result.gates[0].params[0] == Constant(-0.5)

    def test_negates_symbolic_ref(self):
        """Expression nodes let us adjoint circuits with trainable params."""
        ir = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=0,
            gates=(Gate("RY", (0,), (WeightRef(0),)),),
        )
        result = adjoint(ir)
        assert result.gates[0].params[0] == Negate(WeightRef(0))

    def test_evaluates_adjoint_parameter_correctly(self):
        """Adjoint parameter, when resolved, equals the negated original."""
        adjoint_angle = negate_parameter(WeightRef(0))
        assert resolve(adjoint_angle, [math.pi / 3], []) == -math.pi / 3

    def test_double_adjoint_is_identity(self):
        ir = CircuitSpec(
            n_qubits=2,
            n_weights=1,
            n_inputs=0,
            gates=(
                Gate("RY", (0,), (WeightRef(0),)),
                Gate("CNOT", (0, 1)),
                Gate("RX", (1,), (Constant(0.3),)),
            ),
        )
        twice = adjoint(adjoint(ir))
        assert [g.name for g in twice.gates] == [g.name for g in ir.gates]
        # Rotation params equal in value after double negation
        for original, back in zip(ir.gates, twice.gates):
            if original.name in {"RX", "RY", "RZ"}:
                lhs = resolve(original.params[0], [0.5], [])
                rhs = resolve(back.params[0], [0.5], [])
                assert lhs == rhs


# ---- chain-rule helpers (used by chain-rule-aware parameter-shift) ----


class TestCollectWeightDependencies:
    def test_constant(self):
        assert collect_weight_dependencies(Constant(0.5)) == set()

    def test_input_ref(self):
        assert collect_weight_dependencies(InputRef(0)) == set()

    def test_weight_ref(self):
        assert collect_weight_dependencies(WeightRef(3)) == {3}

    def test_scale(self):
        assert collect_weight_dependencies(Scale(WeightRef(2), 2.0)) == {2}

    def test_negate(self):
        assert collect_weight_dependencies(Negate(WeightRef(1))) == {1}

    def test_sum(self):
        expr = Sum((WeightRef(0), InputRef(0), WeightRef(2)))
        assert collect_weight_dependencies(expr) == {0, 2}

    def test_multiply(self):
        expr = Multiply(WeightRef(0), WeightRef(1))
        assert collect_weight_dependencies(expr) == {0, 1}

    def test_multiply_with_input(self):
        expr = Multiply(WeightRef(0), InputRef(0))
        assert collect_weight_dependencies(expr) == {0}

    def test_nested(self):
        expr = Sum((Scale(WeightRef(0), 2.0), Multiply(WeightRef(3), InputRef(0))))
        assert collect_weight_dependencies(expr) == {0, 3}


class TestChainFactor:
    """Symbolic ∂(parameter)/∂(weights[w]) evaluated numerically."""

    def test_constant_is_zero(self):
        assert chain_factor(Constant(7.0), 0, [0.5], []) == 0.0

    def test_input_ref_is_zero(self):
        assert chain_factor(InputRef(0), 0, [], [4.0]) == 0.0

    def test_weight_ref_one_for_match(self):
        assert chain_factor(WeightRef(0), 0, [0.5], []) == 1.0

    def test_weight_ref_zero_for_mismatch(self):
        assert chain_factor(WeightRef(1), 0, [0.5, 0.7], []) == 0.0

    def test_scale_pulls_out_factor(self):
        assert chain_factor(Scale(WeightRef(0), 3.0), 0, [0.5], []) == 3.0

    def test_negate(self):
        assert chain_factor(Negate(WeightRef(0)), 0, [0.5], []) == -1.0

    def test_sum_of_weights(self):
        expr = Sum((WeightRef(0), WeightRef(0), WeightRef(1)))
        # d/dw_0 = 1 + 1 + 0 = 2
        assert chain_factor(expr, 0, [0.5, 1.0], []) == 2.0

    def test_multiply_product_rule_with_input(self):
        # d(w * x)/dw = x
        assert chain_factor(Multiply(WeightRef(0), InputRef(0)), 0, [0.5], [4.0]) == 4.0

    def test_multiply_product_rule_two_weights(self):
        # d(w0 * w1)/dw_0 = w_1; d(w_0 * w_1)/dw_1 = w_0
        expr = Multiply(WeightRef(0), WeightRef(1))
        assert chain_factor(expr, 0, [0.5, 0.7], []) == 0.7
        assert chain_factor(expr, 1, [0.5, 0.7], []) == 0.5

    def test_multiply_with_unrelated_weight(self):
        # d(w_0 * x)/dw_2 = 0
        assert (
            chain_factor(Multiply(WeightRef(0), InputRef(0)), 2, [0.5, 0, 0], [4.0])
            == 0.0
        )

    def test_nested_scale_inside_multiply(self):
        # angle = (2 * w_0) * x  -> d/dw_0 = 2 * x
        expr = Multiply(Scale(WeightRef(0), 2.0), InputRef(0))
        assert chain_factor(expr, 0, [0.5], [3.0]) == 6.0

    def test_negate_of_multiply(self):
        # -(w * x)  -> d/dw = -x
        expr = Negate(Multiply(WeightRef(0), InputRef(0)))
        assert chain_factor(expr, 0, [0.5], [4.0]) == -4.0


class TestShiftGateAngle:
    def test_wraps_first_param_in_sum_with_constant(self):
        ir = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=0,
            gates=(Gate("RY", (0,), (WeightRef(0),)),),
        )
        shifted = shift_gate_angle(ir, gate_index=0, shift_value=0.5)
        new_angle = shifted.gates[0].params[0]
        # Original gate is unchanged; only the slot at gate_index is wrapped.
        assert new_angle == Sum((WeightRef(0), Constant(0.5)))

    def test_resolves_to_original_plus_shift(self):
        ir = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=0,
            gates=(Gate("RX", (0,), (WeightRef(0),)),),
        )
        shifted = shift_gate_angle(ir, gate_index=0, shift_value=0.7)
        original_value = resolve(ir.gates[0].params[0], [0.3], [])
        shifted_value = resolve(shifted.gates[0].params[0], [0.3], [])
        assert shifted_value == pytest.approx(original_value + 0.7)

    def test_other_gates_unchanged(self):
        ir = CircuitSpec(
            n_qubits=2,
            n_weights=2,
            n_inputs=0,
            gates=(
                Gate("RY", (0,), (WeightRef(0),)),
                Gate("RY", (1,), (WeightRef(1),)),
            ),
        )
        shifted = shift_gate_angle(ir, gate_index=1, shift_value=0.1)
        assert shifted.gates[0] == ir.gates[0]  # untouched
        assert shifted.gates[1] != ir.gates[1]  # the perturbed one

    def test_rejects_out_of_range(self):
        ir = CircuitSpec(
            n_qubits=1,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("H", (0,)),),
        )
        with pytest.raises(IndexError):
            shift_gate_angle(ir, gate_index=5, shift_value=0.0)

    def test_rejects_non_parametric_gate(self):
        ir = CircuitSpec(
            n_qubits=2,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("CNOT", (0, 1)),),
        )
        with pytest.raises(ValueError, match="no parametric angle"):
            shift_gate_angle(ir, gate_index=0, shift_value=0.0)


# ---- operator syntax for parameter expressions ----


class TestParameterOperators:
    """`x + y`, `2 * x`, `pi - x` build exactly the nodes the explicit
    constructors do, so the two spellings are interchangeable."""

    def test_add_builds_sum(self):
        assert InputRef(0) + WeightRef(1) == Sum((InputRef(0), WeightRef(1)))

    def test_number_on_the_left_is_wrapped(self):
        assert 0.5 + InputRef(0) == Sum((Constant(0.5), InputRef(0)))

    def test_sub_negates_the_right_operand(self):
        assert InputRef(0) - 0.25 == Sum((InputRef(0), Constant(-0.25)))

    def test_number_minus_expression(self):
        assert math.pi - InputRef(0) == Sum((Constant(math.pi), Negate(InputRef(0))))

    def test_neg_delegates_to_negate_parameter(self):
        assert -InputRef(0) == Negate(InputRef(0))
        # negate_parameter collapses a double negation; the operator inherits that
        negated = -InputRef(0)
        assert -negated == InputRef(0)
        assert -Scale(InputRef(0), 2.0) == Scale(InputRef(0), -2.0)

    def test_pos_is_identity(self):
        assert +InputRef(0) == InputRef(0)

    def test_number_times_expression_is_scale(self):
        """A constant factor must become `Scale`, not `Multiply`: backends fold
        a `Scale` into the gate angle, and cudaq cannot lower `Multiply` at all."""
        assert 2.0 * InputRef(0) == Scale(InputRef(0), 2.0)
        assert InputRef(0) * 2.0 == Scale(InputRef(0), 2.0)

    def test_expression_times_expression_is_multiply(self):
        assert WeightRef(0) * InputRef(1) == Multiply(WeightRef(0), InputRef(1))

    def test_a_constant_node_folds_like_a_plain_number(self):
        """`Constant(2.0) * x` is still a constant factor, so it must reach
        `Scale` rather than becoming a two-symbol `Multiply`."""
        assert Constant(2.0) * InputRef(0) == Scale(InputRef(0), 2.0)
        assert InputRef(0) * Constant(2.0) == Scale(InputRef(0), 2.0)
        assert InputRef(0) / Constant(4.0) == Scale(InputRef(0), 0.25)

    def test_truediv_becomes_reciprocal_scale(self):
        assert InputRef(0) / 4 == Scale(InputRef(0), 0.25)

    def test_truediv_by_expression_is_rejected(self):
        with pytest.raises(TypeError, match="only be divided by a number"):
            InputRef(0) / WeightRef(0)

    def test_numpy_scalars_are_accepted(self):
        """Indexing a numpy array of angles yields np.float64 / np.int64, which
        are `numbers.Real` but not `float` / `int`."""
        assert np.float64(2.0) * InputRef(0) == Scale(InputRef(0), 2.0)
        assert InputRef(0) * np.int64(3) == Scale(InputRef(0), 3.0)
        assert np.float64(0.5) + InputRef(0) == Sum((Constant(0.5), InputRef(0)))

    def test_addition_flattens_nested_sums(self):
        assert InputRef(0) + InputRef(1) + InputRef(2) == Sum(
            (InputRef(0), InputRef(1), InputRef(2))
        )

    def test_flattening_stops_at_a_wrapped_sum(self):
        """A `Sum` under a `Scale` means something different from its terms, so
        it must not be spliced into the outer sum."""
        inner = Scale(Sum((InputRef(0), InputRef(1))), 2.0)
        assert inner + InputRef(2) == Sum((inner, InputRef(2)))

    def test_every_node_type_supports_the_operators(self):
        nodes = [
            Constant(1.0),
            WeightRef(0),
            InputRef(0),
            Scale(InputRef(0), 2.0),
            Negate(InputRef(0)),
            Sum((InputRef(0),)),
            Multiply(WeightRef(0), InputRef(0)),
        ]
        for node in nodes:
            assert isinstance(node + 1.0, Sum)
            # Constant * number folds to a Constant; everything else scales.
            expected = Constant if isinstance(node, Constant) else Scale
            assert isinstance(node * 2.0, expected)
            assert resolve(-node, [0.3], [0.7]) == -resolve(node, [0.3], [0.7])

    def test_two_constants_fold_to_one(self):
        assert Constant(1.5) * 2.0 == Constant(3.0)
        assert Constant(1.5) * Constant(2.0) == Constant(3.0)


class TestZZFeatureMapAngle:
    """The angle a ZZ feature map needs, both ways round."""

    ANGLE_EXPLICIT = Scale(
        Multiply(
            Sum((Constant(math.pi), Negate(InputRef(0)))),
            Sum((Constant(math.pi), Negate(InputRef(1)))),
        ),
        2.0,
    )

    def test_operator_form_matches_explicit_construction(self):
        written = 2.0 * ((math.pi - InputRef(0)) * (math.pi - InputRef(1)))
        assert written == self.ANGLE_EXPLICIT

    def test_left_to_right_form_differs_in_shape_but_not_in_value(self):
        """Without the inner parentheses, `2 * a * b` associates left-to-right
        into `Multiply(Scale(a, 2), b)`. A different tree, the same angle."""
        written = 2.0 * (math.pi - InputRef(0)) * (math.pi - InputRef(1))
        assert written != self.ANGLE_EXPLICIT
        inputs = [0.3, 0.7]
        assert resolve(written, [], inputs) == pytest.approx(
            resolve(self.ANGLE_EXPLICIT, [], inputs)
        )

    def test_resolves_to_the_textbook_value(self):
        inputs = [0.3, 0.7]
        expected = 2.0 * (math.pi - inputs[0]) * (math.pi - inputs[1])
        assert resolve(self.ANGLE_EXPLICIT, [], inputs) == pytest.approx(expected)
