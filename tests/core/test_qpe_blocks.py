"""QPE / QFT / ControlledUnitary IR-block tests (the blocks HHL/IQAE build on)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eigenkit.core.blocks import (
    ControlledUnitaryBlock,
    QFTBlock,
    QPEBlock,
    QubitUnitary,
    _circuit_spec_from_json,
    _circuit_spec_to_json,
    _data_item_from_json,
    _data_item_to_json,
    assemble,
)
from eigenkit.core.builder import CircuitBuilder


def _ry_unitary(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _qpe_oracle(theta: float):
    """1-qubit QubitUnitary oracle for QPEBlock tests."""
    U = _ry_unitary(theta)
    return assemble([QubitUnitary(U, wires=(0,))])


class TestQPEBlock:
    def test_required_qubits(self):
        oracle = _qpe_oracle(0.5)
        block = QPEBlock(oracle, n_clock=3)
        assert block.required_qubits() == 4  # 3 clock + 1 state

    def test_n_clock_zero_raises(self):
        oracle = _qpe_oracle(0.5)
        with pytest.raises(ValueError, match="n_clock"):
            QPEBlock(oracle, n_clock=0)

    def test_n_clock_negative_raises(self):
        oracle = _qpe_oracle(0.5)
        with pytest.raises(ValueError, match="n_clock"):
            QPEBlock(oracle, n_clock=-1)

    def test_non_spec_oracle_raises(self):
        with pytest.raises(TypeError, match="must be a CircuitSpec"):
            QPEBlock("not a spec", n_clock=2)  # type: ignore[arg-type]

    def test_too_few_qubits_raises(self):
        oracle = _qpe_oracle(0.5)
        with pytest.raises(ValueError, match="qubits"):
            QPEBlock(oracle, n_clock=3).build(n_qubits=3)

    def test_forward_starts_with_hadamards(self):
        oracle = _qpe_oracle(0.5)
        fragment = QPEBlock(oracle, n_clock=2).build(n_qubits=3)
        names = [g.name for g in fragment.gates]
        assert names[0] == "H"
        assert names[1] == "H"

    def test_forward_ends_with_iqft_gates(self):
        # IQFT on 2 clock qubits: SWAP + CPHASE + H gates
        oracle = _qpe_oracle(0.5)
        fragment = QPEBlock(oracle, n_clock=2).build(n_qubits=3)
        names = [g.name for g in fragment.gates]
        # Last gates come from IQFT (ends with H on qubit 0)
        assert "H" in names[-4:]

    def test_inverse_gate_sequence_differs_from_forward(self):
        oracle = _qpe_oracle(0.5)
        fwd = QPEBlock(oracle, n_clock=2).build(n_qubits=3)
        inv = QPEBlock(oracle, n_clock=2, inverse=True).build(n_qubits=3)
        # Same gate multiset but different order — sequences must not be equal.
        assert [g.name for g in fwd.gates] != [g.name for g in inv.gates]

    def test_inverse_ends_with_hadamards(self):
        oracle = _qpe_oracle(0.5)
        fragment = QPEBlock(oracle, n_clock=2, inverse=True).build(n_qubits=3)
        names = [g.name for g in fragment.gates]
        assert names[-1] == "H"
        assert names[-2] == "H"

    def test_no_weights_or_inputs(self):
        oracle = _qpe_oracle(0.5)
        fragment = QPEBlock(oracle, n_clock=2).build(n_qubits=3)
        assert fragment.n_weights == 0
        assert fragment.n_inputs == 0

    def test_controlled_unitaries_present(self):
        oracle = _qpe_oracle(0.5)
        fragment = QPEBlock(oracle, n_clock=2).build(n_qubits=3)
        names = [g.name for g in fragment.gates]
        assert "ControlledQubitUnitary" in names

    def test_non_qubit_unitary_oracle_raises(self):
        qb = CircuitBuilder(n_qubits=1)
        qb.h(0)
        oracle = qb.build()
        with pytest.raises(ValueError, match="QubitUnitary"):
            QPEBlock(oracle, n_clock=2).build(n_qubits=3)

    def test_serialization_roundtrip(self):
        oracle = _qpe_oracle(math.pi / 4)
        block = QPEBlock(oracle, n_clock=3, inverse=False)
        restored = QPEBlock.from_dict(block.to_dict())
        assert restored.n_clock == block.n_clock
        assert restored.inverse == block.inverse
        # Check oracle matrix preserved
        orig_gates = block.build(4).gates
        rest_gates = restored.build(4).gates
        assert len(orig_gates) == len(rest_gates)

    def test_inverse_serialization_roundtrip(self):
        oracle = _qpe_oracle(math.pi / 3)
        block = QPEBlock(oracle, n_clock=2, inverse=True)
        restored = QPEBlock.from_dict(block.to_dict())
        assert restored.inverse is True


# ---------------------------------------------------------------------------
# ControlledUnitaryBlock — validation
# ---------------------------------------------------------------------------

_U2 = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X, unitary


class TestQFTBlockValidation:
    def test_rejects_zero_n_qubits(self):
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            QFTBlock(n_qubits=0)

    def test_rejects_negative_n_qubits(self):
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            QFTBlock(n_qubits=-2, inverse=True)

    def test_accepts_single_qubit(self):
        assert QFTBlock(n_qubits=1).required_qubits() == 1


class TestControlledUnitaryBlockValidation:
    def test_valid_construction(self):
        block = ControlledUnitaryBlock(unitary=_U2, control_wires=(0,), target_wires=(1,))
        assert block.control_wires == (0,)
        assert block.target_wires == (1,)

    def test_empty_control_wires_raises(self):
        with pytest.raises(ValueError, match="control_wires must not be empty"):
            ControlledUnitaryBlock(unitary=_U2, control_wires=(), target_wires=(1,))

    def test_control_target_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            ControlledUnitaryBlock(unitary=_U2, control_wires=(1,), target_wires=(1,))

    def test_control_values_wrong_length_raises(self):
        with pytest.raises(ValueError, match="control_values has length"):
            ControlledUnitaryBlock(
                unitary=_U2,
                control_wires=(0, 1),
                target_wires=(2,),
                control_values=(1,),  # length 1, expected 2
            )

    def test_control_values_non_bit_raises(self):
        with pytest.raises(ValueError, match="only 0 or 1"):
            ControlledUnitaryBlock(
                unitary=_U2,
                control_wires=(0,),
                target_wires=(1,),
                control_values=(2,),
            )

    def test_control_values_none_is_valid(self):
        block = ControlledUnitaryBlock(
            unitary=_U2, control_wires=(0,), target_wires=(1,), control_values=None
        )
        assert block.control_values is None

    def test_control_values_zero_accepted(self):
        block = ControlledUnitaryBlock(
            unitary=_U2,
            control_wires=(0,),
            target_wires=(1,),
            control_values=(0,),
        )
        assert block.control_values == (0,)

    def test_build_emits_controlled_gate(self):
        block = ControlledUnitaryBlock(unitary=_U2, control_wires=(0,), target_wires=(1,))
        fragment = block.build(n_qubits=2)
        assert len(fragment.gates) == 1
        assert fragment.gates[0].name == "ControlledQubitUnitary"

    def test_non_unitary_matrix_raises(self):
        not_unitary = np.array([[2, 0], [0, 1]], dtype=complex)
        with pytest.raises(ValueError, match="not unitary"):
            ControlledUnitaryBlock(
                unitary=not_unitary, control_wires=(0,), target_wires=(1,)
            )

    def test_nearly_unitary_float32_tolerant(self):
        U = np.array([[0, 1], [1, 0]], dtype=np.float32)
        # float32 tolerance is 1e-5; this should pass without error
        block = ControlledUnitaryBlock(unitary=U, control_wires=(0,), target_wires=(1,))
        assert block is not None

    def test_non_unitary_2qubit_target_raises(self):
        bad = np.eye(4, dtype=complex)
        bad[0, 0] = 2.0  # breaks unitarity
        with pytest.raises(ValueError, match="not unitary"):
            ControlledUnitaryBlock(unitary=bad, control_wires=(0,), target_wires=(1, 2))

    def test_duplicate_control_wires_raise(self):
        # Accepted, (0, 0) would emit a gate with qubits=(0, 0, 1) that only
        # fails deep inside backend dispatch.
        with pytest.raises(ValueError, match="distinct"):
            ControlledUnitaryBlock(unitary=_U2, control_wires=(0, 0), target_wires=(1,))

    def test_duplicate_target_wires_raise(self):
        U4 = np.eye(4, dtype=complex)
        with pytest.raises(ValueError, match="distinct"):
            ControlledUnitaryBlock(unitary=U4, control_wires=(0,), target_wires=(1, 1))

    def test_empty_target_wires_raise(self):
        # Refused up front rather than surfacing later in required_qubits()
        # as TypeError: 'int' object is not iterable.
        with pytest.raises(ValueError, match="target_wires must not be empty"):
            ControlledUnitaryBlock(unitary=[[1.0]], control_wires=(0,), target_wires=())

    def test_negative_wire_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ControlledUnitaryBlock(unitary=_U2, control_wires=(-1,), target_wires=(1,))

    def test_complex64_tolerance_matches_qubitunitary(self):
        # ControlledUnitaryBlock gives complex64 the same single-precision
        # tolerance QubitUnitary does, so a matrix that passes one passes
        # the other.
        scipy_linalg = pytest.importorskip("scipy.linalg")
        U32 = scipy_linalg.expm(1j * np.array([[1.0, 0.3], [0.3, 0.7]])).astype(
            np.complex64
        )
        QubitUnitary(U32, wires=(0,))  # accepted
        block = ControlledUnitaryBlock(unitary=U32, control_wires=(1,), target_wires=(0,))
        assert block is not None

    def test_tolist_input_accepted(self):
        # Nested lists (U.tolist()) lose the dtype; ControlledUnitaryBlock's
        # _validate_unitary then falls back to the float32 tolerance (1e-5), so
        # valid float32-derived unitaries are accepted instead of being rejected
        # by the strict 1e-12 check.
        scipy_linalg = pytest.importorskip("scipy.linalg")
        U32 = scipy_linalg.expm(1j * np.array([[1.0, 0.3], [0.3, 0.7]])).astype(
            np.complex64
        )
        ControlledUnitaryBlock(
            unitary=U32.tolist(), control_wires=(1,), target_wires=(0,)
        )

    def test_genuinely_non_unitary_list_still_rejected(self):
        with pytest.raises(ValueError, match="not unitary"):
            QubitUnitary([[1.0, 0.0], [0.0, 1.001]], wires=(0,))


# ---------------------------------------------------------------------------
# ControlledUnitaryBlock serialization — control_values=None
# ---------------------------------------------------------------------------


class TestControlledUnitaryBlockSerialization:
    def test_data_item_to_json_none(self):
        result = _data_item_to_json(None)
        assert result == {"kind": "null"}

    def test_data_item_from_json_null(self):
        result = _data_item_from_json({"kind": "null"})
        assert result is None

    def test_circuit_spec_roundtrip_no_control_values(self):

        block = ControlledUnitaryBlock(
            unitary=_U2, control_wires=(0,), target_wires=(1,), control_values=None
        )
        spec = assemble([block])
        d = _circuit_spec_to_json(spec)
        restored = _circuit_spec_from_json(d)
        assert len(restored.gates) == 1
        assert restored.gates[0].name == "ControlledQubitUnitary"
        assert restored.gates[0].data[1] is None

    def test_circuit_spec_roundtrip_with_control_values(self):

        block = ControlledUnitaryBlock(
            unitary=_U2, control_wires=(0,), target_wires=(1,), control_values=(1,)
        )
        spec = assemble([block])
        d = _circuit_spec_to_json(spec)
        restored = _circuit_spec_from_json(d)
        assert restored.gates[0].data[1] == (1.0,)
