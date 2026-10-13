"""Tests for the CircuitBuilder imperative API."""

from __future__ import annotations

import pytest

from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.ir import Constant, InputRef, WeightRef


class TestConstruction:
    def test_basic(self):
        qc = CircuitBuilder(n_qubits=3)
        assert qc.n_qubits == 3
        assert qc.n_weights == 0
        assert qc.n_inputs == 0
        assert len(qc) == 0

    def test_rejects_zero_qubits(self):
        with pytest.raises(ValueError, match="positive"):
            CircuitBuilder(n_qubits=0)

    def test_rejects_negative_qubits(self):
        with pytest.raises(ValueError, match="positive"):
            CircuitBuilder(n_qubits=-1)


class TestParameterFrame:
    def test_add_weights_returns_refs(self):
        qc = CircuitBuilder(n_qubits=1)
        weights = qc.add_weights(3)
        assert weights == [WeightRef(0), WeightRef(1), WeightRef(2)]
        assert qc.n_weights == 3

    def test_add_weights_increments(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.add_weights(2)
        next_batch = qc.add_weights(3)
        assert next_batch == [WeightRef(2), WeightRef(3), WeightRef(4)]
        assert qc.n_weights == 5

    def test_add_inputs_returns_refs(self):
        qc = CircuitBuilder(n_qubits=1)
        inputs = qc.add_inputs(2)
        assert inputs == [InputRef(0), InputRef(1)]

    def test_weights_and_inputs_are_independent(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.add_weights(3)
        qc.add_inputs(2)
        # Adding more weights doesn't affect input indices
        more = qc.add_weights(1)
        assert more == [WeightRef(3)]


class TestGateMethods:
    def test_rx_with_constant(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.rx(0.5, qubit=0)
        ir = qc.build()
        assert ir.gates[0].name == "RX"
        assert ir.gates[0].params[0] == Constant(0.5)

    def test_ry_with_weight(self):
        qc = CircuitBuilder(n_qubits=1)
        w = qc.add_weights(1)
        qc.ry(w[0], qubit=0)
        ir = qc.build()
        assert ir.gates[0].params[0] == WeightRef(0)

    def test_rz_with_input(self):
        qc = CircuitBuilder(n_qubits=1)
        x = qc.add_inputs(1)
        qc.rz(x[0], qubit=0)
        ir = qc.build()
        assert ir.gates[0].params[0] == InputRef(0)

    def test_hadamard(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.h(0)
        ir = qc.build()
        assert ir.gates[0].name == "H"
        assert ir.gates[0].params == ()

    def test_cnot(self):
        qc = CircuitBuilder(n_qubits=2)
        qc.cnot(0, 1)
        ir = qc.build()
        assert ir.gates[0].name == "CNOT"
        assert ir.gates[0].qubits == (0, 1)

    def test_chaining(self):
        qc = CircuitBuilder(n_qubits=2)
        qc.h(0).cnot(0, 1).h(1)
        assert len(qc) == 3


class TestValidation:
    def test_rejects_out_of_range_qubit(self):
        qc = CircuitBuilder(n_qubits=2)
        with pytest.raises(ValueError, match="out of range"):
            qc.rx(0.5, qubit=5)

    def test_error_mentions_gate_name(self):
        qc = CircuitBuilder(n_qubits=2)
        with pytest.raises(ValueError, match="RX"):
            qc.rx(0.5, qubit=5)

    def test_cnot_rejects_same_qubit(self):
        qc = CircuitBuilder(n_qubits=2)
        with pytest.raises(ValueError, match="must differ"):
            qc.cnot(0, 0)

    def test_cnot_rejects_out_of_range(self):
        qc = CircuitBuilder(n_qubits=2)
        with pytest.raises(ValueError, match="out of range"):
            qc.cnot(0, 5)


class TestBuild:
    def test_produces_immutable_ircircuit(self):
        qc = CircuitBuilder(n_qubits=2)
        qc.h(0)
        ir = qc.build()
        # CircuitSpec is frozen dataclass; gates is a tuple, not list
        assert isinstance(ir.gates, tuple)

    def test_reflects_state(self):
        qc = CircuitBuilder(n_qubits=3)
        w = qc.add_weights(2)
        x = qc.add_inputs(1)
        qc.ry(w[0], 0).rx(x[0], 1).rz(w[1], 2)
        ir = qc.build()
        assert ir.n_qubits == 3
        assert ir.n_weights == 2
        assert ir.n_inputs == 1
        assert len(ir.gates) == 3

    def test_build_does_not_consume(self):
        """Build returns a snapshot; the builder stays usable."""
        qc = CircuitBuilder(n_qubits=1)
        qc.h(0)
        ir1 = qc.build()
        qc.h(0)
        ir2 = qc.build()
        assert len(ir1.gates) == 1
        assert len(ir2.gates) == 2


class TestToBlock:
    def test_produces_block(self):
        from eigenkit.core.blocks import Block, CustomBlock

        qc = CircuitBuilder(n_qubits=2)
        qc.add_weights(1)
        qc.h(0).cnot(0, 1)
        block = qc.to_block()
        assert isinstance(block, CustomBlock)
        assert isinstance(block, Block)  # runtime_checkable Protocol

    def test_block_preserves_counts(self):
        qc = CircuitBuilder(n_qubits=2)
        qc.add_weights(3)
        qc.add_inputs(1)
        block = qc.to_block()
        assert block.n_weights == 3
        assert block.n_inputs == 1
        assert block.n_qubits == 2
