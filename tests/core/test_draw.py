"""Text drawer for CircuitSpec."""

from __future__ import annotations

import pytest

from eigenkit import Circuit, CircuitBuilder, Encoding, Variational, all_z
from eigenkit.core.draw import draw_text
from eigenkit.core.ir import (
    CircuitSpec,
    Multiply,
    Negate,
    Scale,
    Sum,
)


def _spec_from_builder(builder: CircuitBuilder) -> CircuitSpec:
    return builder.build()


class TestSingleWire:
    def test_one_rx_gate(self):
        qc = CircuitBuilder(n_qubits=1)
        weights = qc.add_weights(1)
        qc.rx(weights[0], qubit=0)

        text = draw_text(qc.build())
        assert text == "q_0: ──RX(w_0)──"

    def test_constant_angle(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.rx(0.5, qubit=0)
        text = draw_text(qc.build())
        assert "RX(0.5)" in text

    def test_h_gate_no_params(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.h(qubit=0)
        text = draw_text(qc.build())
        assert text == "q_0: ──H──"

    def test_empty_wire(self):
        qc = CircuitBuilder(n_qubits=1)
        text = draw_text(qc.build())
        assert text == "q_0: "


class TestMultipleQubits:
    def test_independent_gates(self):
        qc = CircuitBuilder(n_qubits=2)
        weights = qc.add_weights(2)
        qc.rx(weights[0], qubit=0)
        qc.ry(weights[1], qubit=1)

        # Three lines: q_0 row, connector, q_1 row.
        lines = draw_text(qc.build()).splitlines()
        assert lines[0] == "q_0: ──RX(w_0)──"
        assert lines[2] == "q_1: ──RY(w_1)──"

    def test_cnot_adjacent_qubits(self):
        qc = CircuitBuilder(n_qubits=2)
        weights = qc.add_weights(2)
        qc.rx(weights[0], qubit=0)
        qc.ry(weights[1], qubit=1)
        qc.cnot(0, 1)

        text = draw_text(qc.build())
        lines = text.splitlines()
        # 3 lines: q_0, connector, q_1.
        assert len(lines) == 3
        # Control on q_0, target on q_1.
        assert "●" in lines[0]
        assert "X" in lines[2]
        # Vertical connector between them.
        assert "│" in lines[1]

    def test_cnot_skipping_qubit(self):
        # CNOT(0, 2) should place a │ on both connector rows.
        qc = CircuitBuilder(n_qubits=3)
        qc.cnot(0, 2)
        text = draw_text(qc.build())
        lines = text.splitlines()
        assert len(lines) == 5
        assert "●" in lines[0]
        assert "│" in lines[1]
        assert "│" in lines[3]
        assert "X" in lines[4]


class TestColumnPacking:
    def test_parallel_single_qubit_gates_share_column(self):
        # RX on q_0 and RY on q_1 should occupy the same column.
        qc = CircuitBuilder(n_qubits=2)
        qc.rx(0.5, qubit=0)
        qc.ry(0.3, qubit=1)

        text = draw_text(qc.build())
        lines = text.splitlines()
        # Both wires (lines[0] and lines[2]; lines[1] is the connector row)
        # share the same length when their gates occupy the same column.
        assert len(lines[0]) == len(lines[2])

    def test_cnot_forces_alignment(self):
        # Gates after a CNOT cannot squeeze into earlier columns.
        qc = CircuitBuilder(n_qubits=2)
        qc.rx(0.5, qubit=0)
        qc.cnot(0, 1)
        qc.ry(0.3, qubit=1)

        text = draw_text(qc.build())
        # The RY appears after the CNOT, not in its column.
        # We just check that all 3 distinct gate labels render.
        assert "RX(0.5)" in text
        assert "RY(0.3)" in text
        assert "●" in text and "X" in text


class TestParameterExpressions:
    def test_scale(self):
        qc = CircuitBuilder(n_qubits=1)
        inputs = qc.add_inputs(1)
        qc.rx(Scale(inputs[0], factor=2.0), qubit=0)
        assert "RX(2*x_0)" in draw_text(qc.build())

    def test_negate(self):
        qc = CircuitBuilder(n_qubits=1)
        weights = qc.add_weights(1)
        qc.rz(Negate(weights[0]), qubit=0)
        assert "RZ(-w_0)" in draw_text(qc.build())

    def test_sum(self):
        qc = CircuitBuilder(n_qubits=1)
        w = qc.add_weights(1)
        x = qc.add_inputs(1)
        qc.ry(Sum((w[0], x[0])), qubit=0)
        assert "RY(w_0+x_0)" in draw_text(qc.build())

    def test_multiply(self):
        qc = CircuitBuilder(n_qubits=1)
        w = qc.add_weights(1)
        x = qc.add_inputs(1)
        qc.rx(Multiply(w[0], x[0]), qubit=0)
        assert "RX(w_0*x_0)" in draw_text(qc.build())

    def test_negate_of_sum_uses_parens(self):
        qc = CircuitBuilder(n_qubits=1)
        w = qc.add_weights(1)
        x = qc.add_inputs(1)
        qc.rz(Negate(Sum((w[0], x[0]))), qubit=0)
        assert "RZ(-(w_0+x_0))" in draw_text(qc.build())


@pytest.mark.pennylane
class TestCircuitDrawMethod:
    def test_top_level_circuit_draw(self):
        circuit = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1, rotation="RY")],
            observables=all_z(2),
            provider="pennylane",
        )
        text = circuit.draw()
        assert "q_0:" in text
        assert "q_1:" in text
        # Encoding emits RY(x_*) by default; Variational emits RY(w_*) here.
        assert "x_0" in text and "x_1" in text
        assert "w_0" in text and "w_1" in text


class TestDecimals:
    def test_decimals_arg(self):
        qc = CircuitBuilder(n_qubits=1)
        qc.rx(0.123456789, qubit=0)
        assert "RX(0.123)" in draw_text(qc.build(), decimals=3)
        assert "RX(0.12346)" in draw_text(qc.build(), decimals=5)


class TestEmptyCircuit:
    def test_zero_qubits(self):
        spec = CircuitSpec(n_qubits=0, n_weights=0, n_inputs=0, gates=())
        text = draw_text(spec)
        assert "empty" in text.lower()
