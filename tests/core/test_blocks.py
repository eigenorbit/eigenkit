"""Tests for the block system: concrete blocks, protocol, assembly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eigenkit.core.blocks import (
    S0,
    AmplitudeEncoding,
    AngleEncoding,
    BasicVariational,
    Block,
    CustomBlock,
    Grover,
    QubitUnitary,
    Reuploading,
    StrongVariational,
    _data_item_from_json,
    _data_item_to_json,
    _gate_to_json,
    _param_from_json,
    _param_to_json,
    _relocate_parameter,
    assemble,
)
from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.ir import (
    Constant,
    Gate,
    InputRef,
    Multiply,
    Negate,
    Scale,
    Sum,
    WeightRef,
    _array_to_data,
    _data_to_array,
)


# ---- protocol ----


class TestBlockProtocol:
    def test_runtime_checkable(self):
        assert isinstance(AngleEncoding(in_features=2), Block)
        assert isinstance(BasicVariational(depth=1), Block)

    def test_non_block_fails_instance_check(self):
        assert not isinstance("just a string", Block)


# ---- AngleEncoding ----


class TestAngleEncoding:
    def test_required_qubits(self):
        assert AngleEncoding(in_features=3).required_qubits() == 3

    def test_builds_ry_per_feature(self):
        fragment = AngleEncoding(in_features=3).build(n_qubits=3)
        assert len(fragment.gates) == 3
        for gate in fragment.gates:
            assert gate.name == "RY"
        assert fragment.n_inputs == 3
        assert fragment.n_weights == 0

    def test_unscaled_inputs_are_plain_refs(self):
        fragment = AngleEncoding(in_features=2).build(n_qubits=2)
        assert fragment.gates[0].params[0] == InputRef(0)

    def test_scaled_inputs_wrap_in_scale_expression(self):
        fragment = AngleEncoding(in_features=2, scale=2.0).build(n_qubits=2)
        assert fragment.gates[0].params[0] == Scale(InputRef(0), 2.0)

    def test_invalid_rotation_raises(self):
        with pytest.raises(ValueError, match="rotation must be RX/RY/RZ"):
            AngleEncoding(in_features=2, rotation="BAD")  # type: ignore[arg-type]

    def test_rejects_non_finite_scale(self):
        with pytest.raises(ValueError, match="scale must be finite"):
            AngleEncoding(in_features=2, scale=float("nan"))
        with pytest.raises(ValueError, match="scale must be finite"):
            AngleEncoding(in_features=2, scale=float("inf"))

    def test_rejects_zero_in_features(self):
        with pytest.raises(ValueError, match="in_features must be positive"):
            AngleEncoding(in_features=0)

    def test_rejects_negative_in_features(self):
        with pytest.raises(ValueError, match="in_features must be positive"):
            AngleEncoding(in_features=-1)


# ---- BasicVariational ----


class TestBasicVariational:
    def test_weight_count(self):
        fragment = BasicVariational(depth=3).build(n_qubits=4)
        assert fragment.n_weights == 3 * 4

    def test_gate_structure(self):
        # depth=1 → 1 rotation layer + 1 CNOT ladder on 3 qubits = 3 + 2 = 5 gates
        fragment = BasicVariational(depth=1, rotation="RY").build(n_qubits=3)
        assert len(fragment.gates) == 5
        gate_names = [g.name for g in fragment.gates]
        assert gate_names == ["RY", "RY", "RY", "CNOT", "CNOT"]

    def test_rejects_invalid_rotation(self):
        with pytest.raises(ValueError, match="rotation must be RX/RY/RZ"):
            BasicVariational(depth=1, rotation="BAD")  # type: ignore[arg-type]

    def test_weights_are_sequential_refs(self):
        fragment = BasicVariational(depth=1, rotation="RX").build(n_qubits=3)
        rotations = [g for g in fragment.gates if g.name == "RX"]
        assert rotations[0].params[0] == WeightRef(0)
        assert rotations[1].params[0] == WeightRef(1)
        assert rotations[2].params[0] == WeightRef(2)

    def test_rejects_zero_depth(self):
        with pytest.raises(ValueError, match="depth must be positive"):
            BasicVariational(depth=0)

    def test_rejects_negative_depth(self):
        with pytest.raises(ValueError, match="depth must be positive"):
            BasicVariational(depth=-2)


# ---- CustomBlock (from builder) ----


class TestCustomBlock:
    def test_preserves_gates(self):
        qc = CircuitBuilder(n_qubits=2)
        w = qc.add_weights(1)
        qc.ry(w[0], 0).cnot(0, 1)
        block = qc.to_block()
        fragment = block.build(n_qubits=2)
        assert len(fragment.gates) == 2
        assert fragment.n_weights == 1

    def test_rejects_too_few_qubits(self):
        qc = CircuitBuilder(n_qubits=3)
        qc.h(0)
        block = qc.to_block()
        with pytest.raises(ValueError, match="needs 3 qubits"):
            block.build(n_qubits=2)

    def test_rejects_non_positive_n_qubits(self):
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            CustomBlock(n_qubits=0, gates=(), n_weights=0, n_inputs=0)

    def test_rejects_negative_n_weights(self):
        with pytest.raises(ValueError, match="n_weights must be non-negative"):
            CustomBlock(n_qubits=1, gates=(), n_weights=-1, n_inputs=0)

    def test_rejects_negative_n_inputs(self):
        with pytest.raises(ValueError, match="n_inputs must be non-negative"):
            CustomBlock(n_qubits=1, gates=(), n_weights=0, n_inputs=-1)


# ---- _relocate_parameter ----


class TestRelocation:
    def test_weight_ref(self):
        assert _relocate_parameter(
            WeightRef(2), weight_offset=5, input_offset=0
        ) == WeightRef(7)

    def test_input_ref(self):
        assert _relocate_parameter(
            InputRef(0), weight_offset=0, input_offset=3
        ) == InputRef(3)

    def test_constant_unchanged(self):
        assert _relocate_parameter(Constant(0.5), 10, 10) == Constant(0.5)

    def test_descends_into_scale(self):
        result = _relocate_parameter(Scale(WeightRef(1), 2.0), 5, 0)
        assert result == Scale(WeightRef(6), 2.0)

    def test_descends_into_negate(self):
        result = _relocate_parameter(Negate(InputRef(0)), 0, 3)
        assert result == Negate(InputRef(3))

    def test_descends_into_sum(self):
        original = Sum((WeightRef(0), InputRef(0)))
        result = _relocate_parameter(original, 2, 5)
        assert result == Sum((WeightRef(2), InputRef(5)))

    def test_descends_into_multiply(self):
        original = Multiply(a=WeightRef(0), b=InputRef(0))
        result = _relocate_parameter(original, 4, 3)
        assert result == Multiply(a=WeightRef(4), b=InputRef(3))

    def test_nested_expression(self):
        original = Scale(Negate(Sum((WeightRef(0), InputRef(0)))), 3.0)
        result = _relocate_parameter(original, 4, 6)
        assert result == Scale(Negate(Sum((WeightRef(4), InputRef(6)))), 3.0)


# ---- assemble ----


class TestAssemble:
    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            assemble([])

    def test_single_block(self):
        ir = assemble([AngleEncoding(in_features=2)])
        assert ir.n_qubits == 2
        assert ir.n_inputs == 2
        assert ir.n_weights == 0

    def test_two_blocks_concatenate_weights(self):
        ir = assemble(
            [
                BasicVariational(depth=1),
                BasicVariational(depth=1),
            ]
        )
        assert ir.n_weights == 2  # 1 qubit x (1 + 1) layers (required_qubits=1)

    def test_encoding_plus_variational(self):
        ir = assemble(
            [
                AngleEncoding(in_features=3),
                BasicVariational(depth=2),
            ]
        )
        assert ir.n_qubits == 3
        assert ir.n_inputs == 3
        assert ir.n_weights == 6

    def test_weight_indices_relocated_globally(self):
        """Both variational blocks use WeightRef(0..) locally; assembler
        offsets the second block's indices."""
        ir = assemble(
            [
                BasicVariational(depth=1, rotation="RY"),
                BasicVariational(depth=1, rotation="RX"),
            ]
        )
        rotation_gates = [g for g in ir.gates if g.name in {"RX", "RY"}]
        weight_indices = [g.params[0].index for g in rotation_gates]
        assert weight_indices == sorted(weight_indices)
        assert max(weight_indices) == ir.n_weights - 1

    def test_input_indices_relocated_globally(self):
        ir = assemble(
            [
                AngleEncoding(in_features=2),
                AngleEncoding(in_features=2),
            ]
        )
        input_refs = [g.params[0].index for g in ir.gates if g.name == "RY"]
        assert input_refs == [0, 1, 2, 3]

    def test_compose_custom_block_with_encoding(self):
        qc = CircuitBuilder(n_qubits=2)
        w = qc.add_weights(1)
        qc.ry(w[0], 0).cnot(0, 1)
        ir = assemble([AngleEncoding(in_features=2), qc.to_block()])
        assert ir.n_qubits == 2
        assert ir.n_inputs == 2
        assert ir.n_weights == 1
        # The custom block's gates come after the encoding's
        assert [g.name for g in ir.gates] == ["RY", "RY", "RY", "CNOT"]

    def test_builder_stands_in_for_its_block(self):
        qc = CircuitBuilder(n_qubits=2)
        w = qc.add_weights(1)
        qc.ry(w[0], 0).cnot(0, 1)
        expected = assemble([AngleEncoding(in_features=2), qc.to_block()])
        assert assemble([AngleEncoding(in_features=2), qc]) == expected

    def test_scale_expression_gets_relocated(self):
        """A block emitting Scale(InputRef(0), 2.0) locally should have
        InputRef.index offset after assembly."""
        ir = assemble(
            [
                AngleEncoding(in_features=2),
                AngleEncoding(in_features=2, scale=2.0),
            ]
        )
        # Third RY gate is the first input of the scaled block → InputRef(2)
        scaled_gate = ir.gates[2]
        assert scaled_gate.params[0] == Scale(InputRef(2), 2.0)


# ---- AmplitudeEncoding ----


class TestAmplitudeEncoding:
    def test_required_qubits_power_of_two(self):
        assert AmplitudeEncoding([0.5, 0.5, 0.5, 0.5]).required_qubits() == 2

    def test_required_qubits_single(self):
        assert AmplitudeEncoding([1.0, 0.0]).required_qubits() == 1

    def test_required_qubits_non_power_pads_up(self):
        # 3 elements → padded to 4 → 2 qubits
        assert AmplitudeEncoding([1.0, 0.0, 0.0]).required_qubits() == 2

    def test_normalizes_by_default(self):
        block = AmplitudeEncoding([1.0, 2.0, 3.0, 4.0])
        fragment = block.build(n_qubits=2)
        gate = fragment.gates[0]
        amps = _data_to_array(gate.data[0])
        assert abs(np.linalg.norm(amps) - 1.0) < 1e-12

    def test_normalize_false_preserves_amplitudes(self):
        vals = [0.5, 0.5, 0.5, 0.5]
        block = AmplitudeEncoding(vals, normalize=False)
        fragment = block.build(n_qubits=2)
        amps = _data_to_array(fragment.gates[0].data[0])
        np.testing.assert_allclose(amps, vals, atol=1e-14)

    def test_pads_to_power_of_two(self):
        block = AmplitudeEncoding([1.0, 0.0, 0.0], normalize=False)
        fragment = block.build(n_qubits=2)
        amps = _data_to_array(fragment.gates[0].data[0])
        assert len(amps) == 4
        assert amps[3] == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            AmplitudeEncoding([])

    def test_all_zero_normalize_raises(self):
        with pytest.raises(ValueError, match="all-zero"):
            AmplitudeEncoding([0.0, 0.0])

    def test_nan_amplitude_raises(self):
        with pytest.raises(ValueError, match="finite"):
            AmplitudeEncoding([1.0, float("nan")])

    def test_infinite_amplitude_raises(self):
        with pytest.raises(ValueError, match="finite"):
            AmplitudeEncoding([1.0, float("inf")], normalize=False)

    def test_builds_single_gate(self):
        fragment = AmplitudeEncoding([0.5, 0.5, 0.5, 0.5]).build(n_qubits=2)
        assert len(fragment.gates) == 1
        assert fragment.gates[0].name == "AmplitudeEncoding"

    def test_gate_targets_all_qubits(self):
        fragment = AmplitudeEncoding([0.5] * 4).build(n_qubits=2)
        assert fragment.gates[0].qubits == (0, 1)

    def test_no_weights_or_inputs(self):
        fragment = AmplitudeEncoding([1.0, 0.0]).build(n_qubits=1)
        assert fragment.n_weights == 0
        assert fragment.n_inputs == 0

    def test_too_few_qubits_raises(self):
        with pytest.raises(ValueError, match="2 qubits"):
            AmplitudeEncoding([0.5, 0.5, 0.5, 0.5]).build(n_qubits=1)

    def test_numpy_array_input(self):
        arr = np.array([0.5, 0.5, 0.5, 0.5])
        block = AmplitudeEncoding(arr, normalize=False)
        fragment = block.build(n_qubits=2)
        amps = _data_to_array(fragment.gates[0].data[0])
        np.testing.assert_allclose(amps.real, arr, atol=1e-14)

    def test_data_survives_assemble(self):
        # gate.data must be forwarded by assemble's relocation step
        amps = [0.5, 0.5, 0.5, 0.5]
        spec = assemble([AmplitudeEncoding(amps, normalize=False)])
        gate = spec.gates[0]
        assert gate.name == "AmplitudeEncoding"
        recovered = _data_to_array(gate.data[0])
        np.testing.assert_allclose(recovered.real, amps, atol=1e-14)


# ---- S0 ----


class TestS0:
    def test_required_qubits(self):
        assert S0(n_qubits=3).required_qubits() == 3

    def test_gate_count(self):
        # n X gates + MCZ + n X gates = 2n+1
        fragment = S0(n_qubits=3).build(n_qubits=3)
        assert len(fragment.gates) == 2 * 3 + 1

    def test_gate_structure(self):
        fragment = S0(n_qubits=2).build(n_qubits=2)
        names = [g.name for g in fragment.gates]
        assert names == ["X", "X", "MCZ", "X", "X"]

    def test_mcz_spans_all_qubits(self):
        fragment = S0(n_qubits=3).build(n_qubits=3)
        mcz = next(g for g in fragment.gates if g.name == "MCZ")
        assert set(mcz.qubits) == {0, 1, 2}

    def test_no_weights_or_inputs(self):
        fragment = S0(n_qubits=2).build(n_qubits=2)
        assert fragment.n_weights == 0
        assert fragment.n_inputs == 0

    def test_rejects_zero_n_qubits(self):
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            S0(n_qubits=0)

    def test_rejects_negative_n_qubits(self):
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            S0(n_qubits=-2)


# ---- QubitUnitary ----


class TestQubitUnitary:
    def _hadamard(self):
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)

    def _cnot(self):
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )

    def test_required_qubits_single_wire(self):
        assert QubitUnitary(self._hadamard(), wires=(0,)).required_qubits() == 1

    def test_required_qubits_uses_max_wire(self):
        assert QubitUnitary(self._hadamard(), wires=(3,)).required_qubits() == 4

    def test_required_qubits_two_wire(self):
        assert QubitUnitary(self._cnot(), wires=(0, 1)).required_qubits() == 2

    def test_wrong_matrix_size_raises(self):
        bad = np.eye(3)  # 3x3 is not a power of 2
        with pytest.raises(ValueError, match="2×2"):
            QubitUnitary(bad, wires=(0,))

    def test_non_square_raises(self):
        bad = np.ones((2, 4))
        with pytest.raises(ValueError):
            QubitUnitary(bad, wires=(0,))

    def test_builds_single_gate(self):
        fragment = QubitUnitary(self._hadamard(), wires=(0,)).build(n_qubits=1)
        assert len(fragment.gates) == 1
        assert fragment.gates[0].name == "QubitUnitary"

    def test_gate_targets_correct_wires(self):
        fragment = QubitUnitary(self._cnot(), wires=(1, 2)).build(n_qubits=3)
        assert fragment.gates[0].qubits == (1, 2)

    def test_matrix_stored_in_data(self):
        H = self._hadamard()
        fragment = QubitUnitary(H, wires=(0,)).build(n_qubits=1)
        recovered = _data_to_array(fragment.gates[0].data[0])
        np.testing.assert_allclose(recovered, H, atol=1e-14)

    def test_2qubit_matrix_stored_correctly(self):
        C = self._cnot()
        fragment = QubitUnitary(C, wires=(0, 1)).build(n_qubits=2)
        recovered = _data_to_array(fragment.gates[0].data[0])
        np.testing.assert_allclose(recovered, C, atol=1e-14)

    def test_no_weights_or_inputs(self):
        fragment = QubitUnitary(self._hadamard(), wires=(0,)).build(n_qubits=1)
        assert fragment.n_weights == 0
        assert fragment.n_inputs == 0

    def test_too_few_qubits_raises(self):
        with pytest.raises(ValueError, match="needs"):
            QubitUnitary(self._cnot(), wires=(2, 3)).build(n_qubits=2)

    def test_data_survives_assemble(self):
        H = self._hadamard()
        spec = assemble([QubitUnitary(H, wires=(0,))])
        recovered = _data_to_array(spec.gates[0].data[0])
        np.testing.assert_allclose(recovered, H, atol=1e-14)

    def test_data_survives_assemble_after_other_block(self):
        # gate.data must survive relocation even when offset > 0
        H = self._hadamard()
        spec = assemble([AngleEncoding(in_features=2), QubitUnitary(H, wires=(0,))])
        qu_gate = next(g for g in spec.gates if g.name == "QubitUnitary")
        recovered = _data_to_array(qu_gate.data[0])
        np.testing.assert_allclose(recovered, H, atol=1e-14)


# ---- Grover ----


class TestGrover:
    def _bell_oracle(self):
        qb = CircuitBuilder(n_qubits=2)
        qb.h(0).cnot(0, 1)
        return qb.build()

    def _parametric_oracle(self):
        qb = CircuitBuilder(n_qubits=2)
        w = qb.add_weights(1)
        qb.ry(w[0], 0).cnot(0, 1)
        return qb.build()

    def test_required_qubits(self):
        assert Grover(self._bell_oracle()).required_qubits() == 2

    def test_no_weight_doubling(self):
        # Weights shared between A and A†: n_weights equals oracle's count
        fragment = Grover(self._parametric_oracle()).build(n_qubits=2)
        assert fragment.n_weights == 1

    def test_no_input_doubling(self):
        qb = CircuitBuilder(n_qubits=2)
        x = qb.add_inputs(1)
        qb.ry(x[0], 0).cnot(0, 1)
        oracle = qb.build()
        fragment = Grover(oracle).build(n_qubits=2)
        assert fragment.n_inputs == 1

    def test_gate_order(self):
        # Gate sequence: Z(ancilla) | A† | S₀ | A
        fragment = Grover(self._bell_oracle()).build(n_qubits=2)
        names = [g.name for g in fragment.gates]
        # First gate: S_ψ₁ = Z on ancilla
        assert names[0] == "Z"
        # A† = reverse of (H, CNOT) → (CNOT, H)
        assert names[1] == "CNOT"
        assert names[2] == "H"
        # S₀ = X, X, MCZ, X, X for 2 qubits
        assert names[3:8] == ["X", "X", "MCZ", "X", "X"]
        # A = H, CNOT
        assert names[8] == "H"
        assert names[9] == "CNOT"

    def test_z_targets_ancilla(self):
        fragment = Grover(self._bell_oracle()).build(n_qubits=2)
        z_gate = fragment.gates[0]
        assert z_gate.name == "Z"
        assert z_gate.qubits == (1,)  # ancilla = last qubit

    def test_too_few_qubits_raises(self):
        with pytest.raises(ValueError, match="qubits"):
            Grover(self._bell_oracle()).build(n_qubits=1)

    def test_rejects_non_spec_oracle(self):
        with pytest.raises(TypeError, match="must be a CircuitSpec"):
            Grover(a_circuit="not a spec")  # type: ignore[arg-type]


# ---- serialization roundtrips ----


class TestAmplitudeEncodingRoundtrip:
    def test_real_amplitudes(self):
        block = AmplitudeEncoding([1.0, 2.0, 3.0, 4.0], normalize=True)
        restored = AmplitudeEncoding.from_dict(block.to_dict())
        np.testing.assert_allclose(
            _data_to_array(restored.build(restored.required_qubits()).gates[0].data[0]),
            _data_to_array(block.build(block.required_qubits()).gates[0].data[0]),
            atol=1e-14,
        )

    def test_complex_amplitudes(self):
        block = AmplitudeEncoding([1 + 1j, 1 - 1j, 0.5 + 0j, 0j], normalize=True)
        restored = AmplitudeEncoding.from_dict(block.to_dict())
        np.testing.assert_allclose(
            _data_to_array(restored.build(restored.required_qubits()).gates[0].data[0]),
            _data_to_array(block.build(block.required_qubits()).gates[0].data[0]),
            atol=1e-14,
        )

    def test_normalize_flag_preserved(self):
        for flag in (True, False):
            block = AmplitudeEncoding([0.5, 0.5, 0.5, 0.5], normalize=flag)
            assert AmplitudeEncoding.from_dict(block.to_dict()).normalize == flag

    def test_type_key(self):
        assert AmplitudeEncoding([1.0, 0.0]).to_dict()["type"] == "AmplitudeEncoding"


class TestS0Roundtrip:
    def test_roundtrip(self):
        block = S0(n_qubits=3)
        restored = S0.from_dict(block.to_dict())
        assert restored.n_qubits == 3

    def test_type_key(self):
        assert S0(n_qubits=2).to_dict()["type"] == "S0"

    def test_gate_structure_preserved(self):
        block = S0(n_qubits=2)
        restored = S0.from_dict(block.to_dict())
        assert [g.name for g in restored.build(2).gates] == [
            g.name for g in block.build(2).gates
        ]


class TestQubitUnitaryRoundtrip:
    def _hadamard(self):
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)

    def test_1qubit_roundtrip(self):
        H = self._hadamard()
        block = QubitUnitary(H, wires=(0,))
        restored = QubitUnitary.from_dict(block.to_dict())
        recovered = _data_to_array(restored.build(1).gates[0].data[0])
        np.testing.assert_allclose(recovered, H, atol=1e-14)

    def test_2qubit_roundtrip(self):
        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        block = QubitUnitary(CNOT, wires=(0, 1))
        restored = QubitUnitary.from_dict(block.to_dict())
        recovered = _data_to_array(restored.build(2).gates[0].data[0])
        np.testing.assert_allclose(recovered, CNOT, atol=1e-14)

    def test_wires_preserved(self):
        H = self._hadamard()
        block = QubitUnitary(H, wires=(2,))
        assert QubitUnitary.from_dict(block.to_dict()).wires == (2,)

    def test_type_key(self):
        assert (
            QubitUnitary(self._hadamard(), wires=(0,)).to_dict()["type"] == "QubitUnitary"
        )


class TestGroverRoundtrip:
    def _bell_oracle(self):
        qb = CircuitBuilder(n_qubits=2)
        qb.h(0).cnot(0, 1)
        return qb.build()

    def _parametric_oracle(self):
        qb = CircuitBuilder(n_qubits=2)
        w = qb.add_weights(1)
        qb.ry(w[0], 0).cnot(0, 1)
        return qb.build()

    def test_static_oracle_roundtrip(self):
        block = Grover(a_circuit=self._bell_oracle())
        restored = Grover.from_dict(block.to_dict())
        orig_gates = block.build(2).gates
        rest_gates = restored.build(2).gates
        assert len(orig_gates) == len(rest_gates)
        assert [g.name for g in orig_gates] == [g.name for g in rest_gates]

    def test_parametric_oracle_roundtrip(self):
        block = Grover(a_circuit=self._parametric_oracle())
        restored = Grover.from_dict(block.to_dict())
        assert restored.a_circuit.n_weights == 1
        assert len(restored.build(2).gates) == len(block.build(2).gates)

    def test_type_key(self):
        block = Grover(a_circuit=self._bell_oracle())
        assert block.to_dict()["type"] == "Grover"


# ---- AngleEncoding to_dict / from_dict ----


class TestAngleEncodingSerialize:
    def test_to_dict(self):
        d = AngleEncoding(in_features=3, rotation="RX", scale=2.0).to_dict()
        assert d == {
            "type": "AngleEncoding",
            "in_features": 3,
            "rotation": "RX",
            "scale": 2.0,
        }

    def test_from_dict_roundtrip(self):
        block = AngleEncoding(in_features=2, rotation="RZ", scale=0.5)
        restored = AngleEncoding.from_dict(block.to_dict())
        assert restored.in_features == 2
        assert restored.rotation == "RZ"
        assert restored.scale == 0.5


# ---- BasicVariational to_dict / from_dict ----


class TestBasicVariationalSerialize:
    def test_to_dict(self):
        d = BasicVariational(depth=2, rotation="RX").to_dict()
        assert d == {"type": "BasicVariational", "depth": 2, "rotation": "RX"}

    def test_from_dict_roundtrip(self):
        block = BasicVariational(depth=3, rotation="RZ")
        restored = BasicVariational.from_dict(block.to_dict())
        assert restored.depth == 3
        assert restored.rotation == "RZ"


# ---- StrongVariational ----


class TestStrongVariational:
    def test_required_qubits(self):
        assert StrongVariational(depth=1).required_qubits() == 1

    def test_weight_count(self):
        fragment = StrongVariational(depth=2).build(n_qubits=3)
        assert fragment.n_weights == 3 * 2 * 3

    def test_gate_structure_single_qubit(self):
        fragment = StrongVariational(depth=1).build(n_qubits=1)
        names = [g.name for g in fragment.gates]
        assert names == ["RZ", "RY", "RZ"]

    def test_gate_structure_two_qubits(self):
        fragment = StrongVariational(depth=1).build(n_qubits=2)
        names = [g.name for g in fragment.gates]
        assert names == ["RZ", "RZ", "RY", "RY", "RZ", "RZ", "CNOT"]

    def test_no_inputs(self):
        fragment = StrongVariational(depth=1).build(n_qubits=2)
        assert fragment.n_inputs == 0

    def test_to_dict(self):
        assert StrongVariational(depth=3).to_dict() == {
            "type": "StrongVariational",
            "depth": 3,
        }

    def test_from_dict(self):
        block = StrongVariational.from_dict({"type": "StrongVariational", "depth": 2})
        assert block.depth == 2

    def test_roundtrip(self):
        block = StrongVariational(depth=2)
        restored = StrongVariational.from_dict(block.to_dict())
        orig_names = [g.name for g in block.build(2).gates]
        rest_names = [g.name for g in restored.build(2).gates]
        assert orig_names == rest_names

    def test_rejects_zero_depth(self):
        with pytest.raises(ValueError, match="depth must be positive"):
            StrongVariational(depth=0)

    def test_rejects_negative_depth(self):
        with pytest.raises(ValueError, match="depth must be positive"):
            StrongVariational(depth=-1)


# ---- Reuploading ----


class TestReuploading:
    def test_invalid_in_features_raises(self):
        with pytest.raises(ValueError, match="in_features"):
            Reuploading(in_features=0, depth=1)

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError, match="depth"):
            Reuploading(in_features=2, depth=0)

    def test_invalid_variational_type_raises(self):
        with pytest.raises(ValueError, match="variational_type"):
            Reuploading(in_features=2, depth=1, variational_type="bad")

    def test_required_qubits(self):
        assert Reuploading(in_features=4, depth=1).required_qubits() == 4

    def test_basic_build_counts(self):
        fragment = Reuploading(in_features=2, depth=1).build(n_qubits=2)
        assert fragment.n_inputs == 2
        assert fragment.n_weights == 2

    def test_strong_build_weight_count(self):
        fragment = Reuploading(in_features=2, depth=1, variational_type="strong").build(
            n_qubits=2
        )
        assert fragment.n_weights == 3 * 2

    def test_trainable_scale_weight_count(self):
        fragment = Reuploading(in_features=2, depth=2, trainable_scale=True).build(
            n_qubits=2
        )
        # 2*2 scale + 2*2 basic variational
        assert fragment.n_weights == 2 * 2 + 2 * 2

    def test_trainable_scale_emits_multiply(self):
        fragment = Reuploading(in_features=2, depth=1, trainable_scale=True).build(
            n_qubits=2
        )
        encoding_gate = fragment.gates[0]
        assert isinstance(encoding_gate.params[0], Multiply)

    def test_invalid_encoding_rotation_raises(self):
        with pytest.raises(ValueError, match="encoding_rotation must be RX/RY/RZ"):
            Reuploading(in_features=2, depth=1, encoding_rotation="BAD")  # type: ignore

    def test_invalid_variational_rotation_raises(self):
        with pytest.raises(ValueError, match="variational_rotation must be RX/RY/RZ"):
            Reuploading(in_features=2, depth=1, variational_rotation="BAD")  # type: ignore

    def test_depth_doubles_weight_count(self):
        f1 = Reuploading(in_features=2, depth=1).build(n_qubits=2)
        f2 = Reuploading(in_features=2, depth=2).build(n_qubits=2)
        assert f2.n_weights == 2 * f1.n_weights

    def test_to_dict_keys(self):
        d = Reuploading(in_features=3, depth=2, variational_type="strong").to_dict()
        assert d["type"] == "Reuploading"
        assert d["in_features"] == 3
        assert d["depth"] == 2
        assert d["variational_type"] == "strong"
        assert d["trainable_scale"] is False

    def test_from_dict_roundtrip(self):
        block = Reuploading(in_features=4, depth=2, trainable_scale=True)
        restored = Reuploading.from_dict(block.to_dict())
        assert restored == block

    def test_from_dict_defaults(self):
        d = {
            "type": "Reuploading",
            "in_features": 2,
            "depth": 1,
            "encoding_rotation": "RY",
            "variational_rotation": "RY",
            "variational_type": "basic",
        }
        block = Reuploading.from_dict(d)
        assert block.trainable_scale is False


# ---- _relocate_parameter TypeError ----


class TestRelocateParameterUnknown:
    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            _relocate_parameter("not_a_param", 0, 0)  # type: ignore


# ---- Parameter JSON serialization ----


class TestParamSerialization:
    def test_constant_roundtrip(self):
        p = Constant(3.14)
        assert _param_from_json(_param_to_json(p)) == p

    def test_weight_ref_roundtrip(self):
        assert _param_from_json(_param_to_json(WeightRef(5))) == WeightRef(5)

    def test_input_ref_roundtrip(self):
        assert _param_from_json(_param_to_json(InputRef(2))) == InputRef(2)

    def test_scale_roundtrip(self):
        p = Scale(WeightRef(1), 2.5)
        assert _param_from_json(_param_to_json(p)) == p

    def test_negate_roundtrip(self):
        p = Negate(InputRef(0))
        assert _param_from_json(_param_to_json(p)) == p

    def test_sum_roundtrip(self):
        p = Sum((WeightRef(0), InputRef(1), Constant(0.5)))
        assert _param_from_json(_param_to_json(p)) == p

    def test_multiply_roundtrip(self):
        p = Multiply(WeightRef(0), InputRef(1))
        assert _param_from_json(_param_to_json(p)) == p

    def test_nested_roundtrip(self):
        p = Scale(Negate(Sum((WeightRef(0), InputRef(0)))), 3.0)
        assert _param_from_json(_param_to_json(p)) == p

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            _param_to_json("not_a_param")  # type: ignore

    def test_unknown_json_tag_raises(self):
        with pytest.raises(ValueError):
            _param_from_json({"t": "Unknown"})


# ---- Data item JSON serialization ----


class TestDataItemSerialization:
    def test_flat_real_roundtrip(self):
        item = (0.5, 0.25, 0.75, 1.0)
        restored = _data_item_from_json(_data_item_to_json(item))
        assert tuple(float(v) for v in restored) == pytest.approx(item)

    def test_flat_complex_roundtrip(self):
        item = (complex(1, 0.5), complex(0, -1))
        restored = _data_item_from_json(_data_item_to_json(item))
        for a, b in zip(restored, item):
            assert abs(a - b) < 1e-14

    def test_matrix_roundtrip(self):
        h = 1 / math.sqrt(2)
        item = (
            (complex(h, 0), complex(h, 0)),
            (complex(h, 0), complex(-h, 0)),
        )
        restored = _data_item_from_json(_data_item_to_json(item))
        for row_r, row_o in zip(restored, item):
            for a, b in zip(row_r, row_o):
                assert abs(a - b) < 1e-14


# ---- _gate_to_json with data and adjoint ----


class TestGateToJson:
    def test_gate_with_data_serialized(self):
        mat = np.array([[1, 0], [0, -1]], dtype=complex)
        data = (_array_to_data(mat),)
        gate = Gate(name="QubitUnitary", qubits=(0,), params=(), data=data)
        d = _gate_to_json(gate)
        assert "d" in d
        assert d["n"] == "QubitUnitary"

    def test_adjoint_flag_serialized(self):
        amps = (0.5, 0.5, 0.5, 0.5)
        gate = Gate(
            name="AmplitudeEncoding", qubits=(0, 1), params=(), data=(amps,), adjoint=True
        )
        d = _gate_to_json(gate)
        assert d.get("adj") is True

    def test_no_adjoint_flag_when_false(self):
        gate = Gate(name="H", qubits=(0,))
        d = _gate_to_json(gate)
        assert "adj" not in d
        assert "d" not in d


class TestAmplitudeEncodingUnitNorm:
    """`normalize=False` is a claim, and the block checks it.

    Without the check the claim would be settled by whichever backend ran:
    numpy rescales an unnormalised vector silently, PennyLane refuses it. The
    same circuit would therefore behave differently on different backends,
    which is the one thing the provider layer promises it will not do. The
    tolerance matches PennyLane's, the strictest of them, so nothing is
    rejected here that would run there.
    """

    def test_an_unnormalised_vector_is_refused(self):
        with pytest.raises(ValueError, match="norm is 1.4142135623730951, not 1"):
            AmplitudeEncoding([1.0, 1.0], normalize=False)

    def test_the_message_says_how_to_fix_it(self):
        with pytest.raises(ValueError, match="Pass normalize=True"):
            AmplitudeEncoding([3.0, 4.0], normalize=False)

    def test_a_normalised_vector_is_accepted(self):
        block = AmplitudeEncoding([0.6, 0.8], normalize=False)
        assert np.allclose(np.asarray(block._amps, dtype=complex), [0.6, 0.8])

    @pytest.mark.parametrize("n", [2, 4, 16, 256, 4096])
    def test_no_false_rejection_at_any_width(self, n):
        # An honestly normalised vector lands within 1.1e-16 of unit norm, so
        # the 1e-10 tolerance has ample margin.
        rng = np.random.default_rng(0)
        vector = rng.normal(size=n)
        AmplitudeEncoding(vector / np.linalg.norm(vector), normalize=False)

    def test_a_complex_state_is_accepted(self):
        state = np.array([1 + 1j, 1 - 1j]) / 2.0
        AmplitudeEncoding(state, normalize=False)

    def test_padding_does_not_break_the_check(self):
        # Three entries pad to four with a zero, which leaves the norm alone.
        AmplitudeEncoding([0.6, 0.8, 0.0], normalize=False)

    def test_an_all_zero_vector_is_refused_under_either_flag(self):
        with pytest.raises(ValueError, match="all-zero"):
            AmplitudeEncoding([0.0, 0.0], normalize=True)
        with pytest.raises(ValueError, match="norm is 0.0, not 1"):
            AmplitudeEncoding([0.0, 0.0], normalize=False)

    def test_normalize_true_still_rescales(self):
        block = AmplitudeEncoding([1.0, 1.0], normalize=True)
        assert np.allclose(np.asarray(block._amps, dtype=complex), [2**-0.5, 2**-0.5])

    @pytest.mark.parametrize("backend", ["numpy", "pennylane"])
    def test_the_backends_that_split_agree(self, backend):
        # Named rather than read from the registry: tests elsewhere register
        # extra providers globally, some with no AmplitudeEncoding at all.
        # Reading `registered_names()` here would make this test depend on
        # which other files had run first.
        from eigenkit.core.circuit import Circuit
        from eigenkit.core.observables import all_z
        from eigenkit.core.provider import registered_names

        if backend not in registered_names():
            pytest.skip(f"{backend} is not installed")
        spec = assemble([AmplitudeEncoding([0.6, 0.8], normalize=False)])
        empty = np.zeros(0)
        compiled = Circuit(spec, all_z(1), backend, "numpy").compiled
        np.testing.assert_allclose(
            compiled.statevector(empty, empty), [0.6, 0.8], atol=1e-8
        )
