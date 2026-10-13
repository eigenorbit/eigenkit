"""Tests for the reference numpy statevector simulator.

Three things are checked, each cross-validated against an independently
trusted source:

1. Forward values + zero-state probability vs PennyLane (machine precision).
2. Gradients through the parameter-shift dispatch vs PennyLane's
   native autograd.
3. The analytic `jacobian` vs the parameter-shift / chain-rule
   jacobians on `_ir.Circuit` (which item 2 ties back to PennyLane).

Together these triangulate: PennyLane-native <-> numpy-parameter-shift <->
numpy-adjoint must all agree.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from eigenkit.core.backends.numpy_backend import NumpyConfig, NumpySimulator
from eigenkit.core.blocks import (
    AmplitudeEncoding,
    AngleEncoding,
    BasicVariational,
    QubitUnitary,
    assemble,
)
from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import (
    CircuitSpec,
    Constant,
    Gate,
    InputRef,
    Multiply,
    Negate,
    Scale,
    Sum,
    WeightRef,
    _array_to_data,
    adjoint,
)
from eigenkit.core.observables import Z, all_z, from_string
from eigenkit.core.provider import (
    parameter_shift_input_jacobian,
    parameter_shift_jacobian,
    resolve_provider,
)


# This module is part of the framework-free core-only CI run, so it must import
# with no optional extras installed. The pure-numpy tests always run; the
# cross-framework parity classes below are gated on their extra being present.
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None


@pytest.fixture
def ir():
    return assemble(
        [
            AngleEncoding(in_features=3),
            BasicVariational(depth=2, rotation="RY"),
        ]
    )


@pytest.fixture
def weights():
    return np.array([0.4, 1.1, -0.3, 0.9, 0.2, -0.7], dtype=np.float64)


@pytest.fixture
def inputs():
    return np.array([0.5, -0.2, 0.8], dtype=np.float64)


@pytest.fixture
def unsafe_spec():
    """A circuit whose angles wrap refs in Scale / Multiply / Sum / Negate,
    so the naive parameter-shift is wrong and the chain-rule path kicks in.
    The adjoint jacobian must agree with that path."""
    return CircuitSpec(
        n_qubits=2,
        n_weights=2,
        n_inputs=2,
        gates=(
            Gate("RY", (0,), (Scale(InputRef(0), 2.0),)),
            Gate("RX", (1,), (Multiply(WeightRef(0), InputRef(1)),)),
            Gate("CNOT", (0, 1)),
            Gate("RZ", (0,), (Sum((WeightRef(1), Constant(0.3))),)),
            Gate("RY", (1,), (Negate(WeightRef(0)),)),
        ),
    )


# ---- forward parity vs PennyLane ----


@pytest.mark.skipif(not _HAS_PENNYLANE, reason="pennylane extra not installed")
class TestForwardParity:
    def test_single_observable(self, ir, weights, inputs):
        reference = Circuit(ir, Z(1).to_observable(), "pennylane", "numpy")
        candidate = Circuit(ir, Z(1).to_observable(), "numpy", "numpy")
        assert abs(reference(weights, inputs) - candidate(weights, inputs)) < 1e-10

    def test_all_z(self, ir, weights, inputs):
        reference = Circuit(ir, all_z(3), "pennylane", "numpy")
        candidate = Circuit(ir, all_z(3), "numpy", "numpy")
        assert np.allclose(
            reference(weights, inputs), candidate(weights, inputs), atol=1e-10
        )

    def test_hamiltonian(self, ir, weights, inputs):
        observable = from_string("0.5 Z0 X1 - 0.3 Y2 + 0.7 Z0 Z2")
        reference = Circuit(ir, observable, "pennylane", "numpy")
        candidate = Circuit(ir, observable, "numpy", "numpy")
        assert abs(reference(weights, inputs) - candidate(weights, inputs)) < 1e-10

    def test_zero_state_probability(self, ir, weights, inputs):
        reference = Circuit(ir, all_z(3), "pennylane", "numpy")
        candidate = Circuit(ir, all_z(3), "numpy", "numpy")
        assert (
            abs(
                reference.zero_state_probability(weights, inputs)
                - candidate.zero_state_probability(weights, inputs)
            )
            < 1e-10
        )


# ---- gradients through the parameter-shift dispatch ----


@pytest.mark.skipif(
    not (_HAS_TORCH and _HAS_PENNYLANE),
    reason="torch + pennylane extras not installed",
)
class TestGradientParity:
    def test_grad_matches_pennylane(self, ir, weights, inputs):
        import torch

        observables = [from_string("0.7 Z0 + 0.3 X1 Z2"), Z(0).to_observable()]
        pl_w = torch.tensor(weights, requires_grad=True)
        np_w = torch.tensor(weights, requires_grad=True)
        inputs_t = torch.tensor(inputs)

        pennylane = Circuit(ir, observables, "pennylane", "torch")
        numpy_sim = Circuit(ir, observables, "numpy", "torch")

        (pennylane(pl_w, inputs_t) ** 2).sum().backward()
        (numpy_sim(np_w, inputs_t) ** 2).sum().backward()

        assert torch.allclose(pl_w.grad, np_w.grad, atol=1e-6)

    def test_input_grad_matches_pennylane(self, ir, weights, inputs):
        import torch

        observables = [Z(0).to_observable(), Z(1).to_observable()]
        pl_x = torch.tensor(inputs, requires_grad=True)
        np_x = torch.tensor(inputs, requires_grad=True)
        weights_t = torch.tensor(weights)

        pennylane = Circuit(ir, observables, "pennylane", "torch")
        numpy_sim = Circuit(ir, observables, "numpy", "torch")

        (pennylane(weights_t, pl_x) ** 2).sum().backward()
        (numpy_sim(weights_t, np_x) ** 2).sum().backward()

        assert torch.allclose(pl_x.grad, np_x.grad, atol=1e-6)


# ---- adjoint jacobian ----


class TestAdjointJacobian:
    # The numpy sim advertises gradient_method="adjoint", so
    # `_gradient_jacobians` returns the adjoint result — these tests
    # therefore compare against INDEPENDENT parameter-shift references
    # (the standalone helpers / the chain-rule method) to keep the
    # cross-validation honest.
    def test_weights_match_parameter_shift_safe(self, ir, weights, inputs):
        observables = [from_string("0.7 Z0 + 0.3 X1 Z2"), Z(0).to_observable()]
        circuit = Circuit(ir, observables, "numpy", "numpy")
        reference_w = parameter_shift_jacobian(circuit.compiled, weights, inputs)
        adjoint_w, _ = circuit.compiled.jacobian(weights, inputs)
        assert np.allclose(reference_w, adjoint_w, atol=1e-8)

    def test_inputs_match_parameter_shift_safe(self, ir, weights, inputs):
        observables = [Z(0).to_observable(), Z(1).to_observable()]
        circuit = Circuit(ir, observables, "numpy", "numpy")
        ref_w = parameter_shift_jacobian(circuit.compiled, weights, inputs)
        ref_x = parameter_shift_input_jacobian(circuit.compiled, weights, inputs)
        adj_w, adj_x = circuit.compiled.jacobian(weights, inputs, need_input_grad=True)
        assert np.allclose(ref_w, adj_w, atol=1e-8)
        assert np.allclose(ref_x, adj_x, atol=1e-8)

    def test_matches_chain_rule_unsafe(self, unsafe_spec):
        observables = [from_string("0.7 Z0 + 0.3 X1"), Z(0).to_observable()]
        circuit = Circuit(unsafe_spec, observables, "numpy", "numpy")
        weights = np.array([0.6, -0.4], dtype=np.float64)
        inputs = np.array([0.3, 1.2], dtype=np.float64)
        # _chain_rule_jacobians always uses parameter-shift, independent of
        # the backend's advertised gradient_method.
        ref_w, ref_x = circuit._chain_rule_jacobians(
            weights, inputs, need_input_grad=True
        )
        adj_w, adj_x = circuit.compiled.jacobian(weights, inputs, need_input_grad=True)
        assert np.allclose(ref_w, adj_w, atol=1e-8)
        assert np.allclose(ref_x, adj_x, atol=1e-8)

    def test_dispatch_routes_to_adjoint(self, ir, weights, inputs):
        # With gradient_method="adjoint", _gradient_jacobians must return the
        # adjoint result (not parameter-shift).
        observables = [from_string("0.7 Z0 + 0.3 X1 Z2"), Z(0).to_observable()]
        circuit = Circuit(ir, observables, "numpy", "numpy")
        dispatched_w, _ = circuit._gradient_jacobians(weights, inputs)
        adjoint_w, _ = circuit.compiled.jacobian(weights, inputs)
        assert np.array_equal(dispatched_w, adjoint_w)

    def test_jacobian_shapes(self, ir, weights, inputs):
        observables = [Z(0).to_observable(), Z(1).to_observable(), Z(2).to_observable()]
        compiled = NumpySimulator().compile(ir, [o for o in observables])
        weight_jac, input_jac = compiled.jacobian(weights, inputs, need_input_grad=True)
        assert weight_jac.shape == (3, ir.n_weights)
        assert input_jac.shape == (3, ir.n_inputs)

    def test_batched_jacobian_matches_per_sample(self, unsafe_spec):
        # A 2-D inputs batch -> batched adjoint sweep; must equal the
        # per-sample jacobians stacked, including the Multiply/CNOT chain-rule.
        observables = [from_string("0.7 Z0 + 0.3 X1"), Z(0).to_observable()]
        compiled = NumpySimulator().compile(unsafe_spec, observables)
        weights = np.array([0.6, -0.4], dtype=np.float64)
        batch = np.array([[0.3, 1.2], [-0.5, 0.8], [0.1, -0.9]], dtype=np.float64)
        bw, bx = compiled.jacobian(weights, batch, need_input_grad=True)
        assert bw.shape == (3, 2, unsafe_spec.n_weights)  # (B, n_obs, n_weights)
        assert bx.shape == (3, 2, unsafe_spec.n_inputs)
        ref_w = np.stack(
            [compiled.jacobian(weights, x, need_input_grad=True)[0] for x in batch]
        )
        ref_x = np.stack(
            [compiled.jacobian(weights, x, need_input_grad=True)[1] for x in batch]
        )
        assert np.allclose(bw, ref_w, atol=1e-12)
        assert np.allclose(bx, ref_x, atol=1e-12)


# ---- batching ----


class TestBatching:
    def test_batch_shape(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "numpy", "numpy")
        batch = np.stack([inputs, inputs * 0.5, inputs * -1.0])
        assert circuit(weights, batch).shape == (3, 3)

    def test_batch_matches_per_sample(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "numpy", "numpy")
        batch = np.stack([inputs, inputs * 0.5])
        batched = circuit(weights, batch)
        per_sample = np.stack([circuit(weights, batch[i]) for i in range(len(batch))])
        assert np.allclose(batched, per_sample)

    def test_batch_input_dependent_angles_match_per_sample(self, unsafe_spec):
        # Angles wrapping InputRef in Scale/Multiply exercise the per-sample
        # (B, 2, 2) rotation path; the CNOT exercises the batched swap.
        circuit = Circuit(unsafe_spec, all_z(2), "numpy", "numpy")
        w = np.array([0.7, -0.4], dtype=np.float64)
        batch = np.array([[0.5, -0.2], [0.1, 0.9], [-0.3, 0.6]], dtype=np.float64)
        batched = circuit(w, batch)
        per_sample = np.stack([circuit(w, batch[i]) for i in range(len(batch))])
        assert np.allclose(batched, per_sample, atol=1e-12)

    def test_batch_hamiltonian_scalar_observable(self, ir, weights):
        H = 0.5 * Z(0) - 0.3 * Z(1) * Z(2)
        circuit = Circuit(ir, H, "numpy", "numpy")
        batch = np.array([[0.5, -0.2, 0.8], [0.1, 0.3, -0.4]], dtype=np.float64)
        out = circuit(weights, batch)
        assert out.shape == (2,)  # single (scalar) observable -> (B,)
        per_sample = np.array([circuit(weights, batch[i]) for i in range(len(batch))])
        assert np.allclose(out, per_sample, atol=1e-12)


# ---- CNOT applied as an amplitude swap (np.flip), not a matrix multiply ----


class TestCnotSwap:
    @pytest.mark.parametrize(
        "control_bit,expected_index",
        [(0, 0), (1, 3)],  # control=0 -> |00>; control=1 -> |11> (q0 is MSB)
    )
    def test_cnot_truth_table(self, control_bit, expected_index):
        from eigenkit.core.builder import CircuitBuilder

        qc = CircuitBuilder(n_qubits=2)
        if control_bit:
            qc.x(0)  # prepare |10>
        qc.cnot(0, 1)
        circuit = Circuit(qc.build(), provider="numpy", interface="numpy")
        probs = circuit.probabilities(np.empty(0))
        assert probs[expected_index] == pytest.approx(1.0)

    def test_cnot_leaves_control_zero_untouched(self):
        from eigenkit.core.builder import CircuitBuilder

        qc = CircuitBuilder(n_qubits=2)
        qc.x(1)  # |01>: control qubit 0 is |0>, so CNOT must not flip the target
        qc.cnot(0, 1)
        circuit = Circuit(qc.build(), provider="numpy", interface="numpy")
        probs = circuit.probabilities(np.empty(0))
        assert probs[1] == pytest.approx(1.0)  # still |01>


# ---- state management: readout + injection ----


class TestStateManagement:
    """`statevector()` readout and `initial_state` injection — the primitive
    behind quantum reservoir carry-over and amplitude encoding."""

    def test_statevector_matches_probabilities(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "numpy", "numpy")
        sv = circuit.statevector(weights, inputs)
        assert sv.shape == (8,)
        assert sv.dtype == np.complex128
        assert np.allclose(np.abs(sv) ** 2, circuit.probabilities(weights, inputs))

    def test_statevector_batch_shape_and_rows(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "numpy", "numpy")
        batch = np.stack([inputs, 0.5 * inputs, -inputs])
        sv = circuit.statevector(weights, batch)
        assert sv.shape == (3, 8)
        assert np.allclose(sv[0], circuit.statevector(weights, inputs))
        assert np.allclose(sv[2], circuit.statevector(weights, -inputs))

    def test_injected_state_is_identity_through_empty_circuit(self):
        prep = CircuitSpec(2, 0, 0, (Gate("RY", (0,), (Constant(0.7),)),))
        psi = Circuit(prep, all_z(2), "numpy", "numpy").statevector(np.empty(0))
        empty = Circuit(CircuitSpec(2, 0, 0, ()), all_z(2), "numpy", "numpy")
        assert np.allclose(empty.statevector(np.empty(0), initial_state=psi), psi)

    def test_injection_equals_in_circuit_preparation(self):
        # Preparing |psi> with gates then running the rest must equal injecting
        # |psi> as the initial state of the rest.
        prep = CircuitSpec(2, 0, 0, (Gate("RY", (0,), (Constant(0.7),)),))
        rest = CircuitSpec(
            2, 0, 0, (Gate("RX", (1,), (Constant(0.3),)), Gate("CNOT", (0, 1)))
        )
        full = CircuitSpec(2, 0, 0, prep.gates + rest.gates)

        psi = Circuit(prep, all_z(2), "numpy", "numpy").statevector(np.empty(0))
        direct = Circuit(full, all_z(2), "numpy", "numpy").statevector(np.empty(0))
        injected = Circuit(rest, all_z(2), "numpy", "numpy").statevector(
            np.empty(0), initial_state=psi
        )
        assert np.allclose(direct, injected)

    def test_reservoir_carryover(self):
        # Carry the state across three identical ticks; must equal one circuit
        # with the tick applied three times. (First tick starts from |0...0>.)
        tick = CircuitSpec(
            2, 0, 0, (Gate("RY", (0,), (Constant(0.5),)), Gate("CNOT", (0, 1)))
        )
        c = Circuit(tick, all_z(2), "numpy", "numpy")
        state = None
        for _ in range(3):
            state = c.statevector(np.empty(0), initial_state=state)
        combined = CircuitSpec(2, 0, 0, tick.gates * 3)
        expected = Circuit(combined, all_z(2), "numpy", "numpy").statevector(np.empty(0))
        assert np.allclose(state, expected)

    def test_batch_injection_shared_and_per_sample(self):
        spec = CircuitSpec(2, 0, 1, (Gate("RY", (0,), (InputRef(0),)),))
        circuit = Circuit(spec, all_z(2), "numpy", "numpy")
        batch_inputs = np.array([[0.2], [0.5], [-0.4]])

        shared = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)  # |00>
        sv_shared = circuit.statevector(np.empty(0), batch_inputs, initial_state=shared)
        assert sv_shared.shape == (3, 4)
        # |00> is the default start, so injecting it changes nothing.
        assert np.allclose(sv_shared, circuit.statevector(np.empty(0), batch_inputs))

        per_sample = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.complex128,
        )  # row 1 starts from |10> (q0 is MSB), rows 0 and 2 from |00>
        sv = circuit.statevector(np.empty(0), batch_inputs, initial_state=per_sample)
        assert sv.shape == (3, 4)
        # Rows 0 and 2 start from |00>, matching the default-start batch.
        assert np.allclose(sv[0], sv_shared[0])
        assert np.allclose(sv[2], sv_shared[2])

    def test_injection_wrong_length_raises(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "numpy", "numpy")  # 3 qubits -> needs len 8
        with pytest.raises(ValueError, match="length 8"):
            circuit.statevector(weights, inputs, initial_state=np.ones(4))

    def test_injection_unnormalized_raises(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "numpy", "numpy")
        with pytest.raises(ValueError, match="normalized"):
            circuit.statevector(weights, inputs, initial_state=np.ones(8))


# ---- registration + handle ----


class TestRegistrationAndHandle:
    def test_registered_under_numpy(self):
        assert resolve_provider("numpy").name == "numpy"

    def test_config_routes(self, ir, weights, inputs):
        via_name = Circuit(ir, all_z(3), "numpy", "numpy")
        via_config = Circuit(ir, all_z(3), NumpyConfig(), "numpy")
        assert np.allclose(via_name(weights, inputs), via_config(weights, inputs))

    def test_gradient_method_is_adjoint(self):
        assert NumpySimulator().gradient_method == "adjoint"

    def test_handle_exposes_run_and_adjoint(self, ir):
        compiled = NumpySimulator().compile(ir, all_z(3))
        assert hasattr(compiled, "run")
        assert hasattr(compiled, "zero_state_probability")
        assert hasattr(compiled, "jacobian")

    def test_no_native_torch_path(self, ir):
        # The reference sim is numpy-only; it should not advertise run_torch,
        # so the torch dispatch wraps its numpy jacobian (adjoint) instead.
        compiled = NumpySimulator().compile(ir, all_z(3))
        assert not hasattr(compiled, "run_torch")


# ---- provider-native gradient dispatch ----


class TestProviderNativeJacobian:
    """The gradient dispatch prefers a backend's own `jacobian`; the numpy sim
    provides adjoint, and PennyLane provides its own engine — they must agree."""

    def test_numpy_sim_exposes_jacobian(self, ir):
        compiled = NumpySimulator().compile(ir, all_z(3))
        assert hasattr(compiled, "jacobian")

    def test_pennylane_exposes_jacobian_and_matches_numpy(self, ir, weights, inputs):
        pytest.importorskip("pennylane")
        from eigenkit.core.backends.pennylane_backend import PennyLaneBackend

        observables = [from_string("0.7 Z0 + 0.3 X1 Z2"), Z(0).to_observable()]
        numpy_compiled = NumpySimulator().compile(ir, observables)
        pennylane_compiled = PennyLaneBackend().compile(ir, observables)
        # PennyLane advertises its own gradient engine via `jacobian`...
        assert hasattr(pennylane_compiled, "jacobian")
        # ...and it agrees with the numpy sim's adjoint (both are exact).
        np_w, np_x = numpy_compiled.jacobian(weights, inputs, need_input_grad=True)
        pl_w, pl_x = pennylane_compiled.jacobian(weights, inputs, need_input_grad=True)
        assert np.allclose(np_w, pl_w, atol=1e-6)
        assert np.allclose(np_x, pl_x, atol=1e-6)

    def test_braket_has_no_jacobian_falls_back(self, ir):
        # A backend without a native gradient must NOT expose `jacobian`, so
        # the core parameter-shift fallback is used.
        pytest.importorskip("braket")
        from eigenkit.core.backends.braket_backend import BraketBackend

        compiled = BraketBackend().compile(ir, all_z(3))
        assert not hasattr(compiled, "jacobian")


# ---- data-carrying gates: QubitUnitary + AmplitudeEncoding ----


class TestDataGates:
    """The numpy reference sim implements the data-carrying gates (QubitUnitary,
    AmplitudeEncoding) end to end — forward, state readout, adjoint round-trip,
    and the native adjoint jacobian — so the default backend actually supports
    the QAE blocks instead of passing validation and failing at run time."""

    _H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    _CNOT = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
    )

    def _circuit(self, spec):
        return Circuit(spec, all_z(spec.n_qubits), "numpy", "numpy")

    def test_data_gate_runs_on_default_backend(self):
        # A data gate must run on the default numpy provider, not
        # pass validate_circuit and then raise NotImplementedError at run time.
        spec = assemble([QubitUnitary(self._H, wires=(0,))])
        out = self._circuit(spec)(np.empty(0), np.empty(0))
        assert out.shape == (1,)
        assert out[0] == pytest.approx(0.0, abs=1e-12)  # H|0> -> <Z> = 0

    def test_qubit_unitary_single_qubit_matches_hadamard(self):
        spec = assemble([QubitUnitary(self._H, wires=(0,))])
        probs = self._circuit(spec).probabilities(np.empty(0))
        assert np.allclose(probs, [0.5, 0.5])

    def test_qubit_unitary_two_qubit_ordering(self):
        # X on q0 then CNOT-as-a-unitary(0,1) must yield |11>: this pins the
        # qubit-0-MSB matrix ordering (a flipped ordering would give |10>).
        spec = CircuitSpec(
            2,
            0,
            0,
            (
                Gate("X", (0,)),
                Gate("QubitUnitary", (0, 1), data=(_array_to_data(self._CNOT),)),
            ),
        )
        probs = self._circuit(spec).probabilities(np.empty(0))
        expected = np.zeros(4)
        expected[0b11] = 1.0
        assert np.allclose(probs, expected)

    def test_amplitude_encoding_prepares_state(self):
        amps = [0.1, 0.2, 0.3, 0.4]
        spec = assemble([AmplitudeEncoding(amps, normalize=True)])
        sv = self._circuit(spec).statevector(np.empty(0))
        expected = np.array(amps, dtype=np.complex128)
        expected /= np.linalg.norm(expected)
        assert np.allclose(sv, expected)

    def test_amplitude_encoding_adjoint_round_trips(self):
        # A†·A|0> = |0>: the state-prep unitary and its adjoint compose to I.
        fwd = assemble([AmplitudeEncoding([1.0, 2.0, 3.0, 4.0], normalize=True)])
        composed = CircuitSpec(fwd.n_qubits, 0, 0, fwd.gates + adjoint(fwd).gates)
        sv = self._circuit(composed).statevector(np.empty(0))
        zero = np.zeros(2**fwd.n_qubits, dtype=np.complex128)
        zero[0] = 1.0
        assert np.allclose(sv, zero)

    def test_native_jacobian_back_propagates_through_qubit_unitary(self):
        # The adjoint sweep must invert a (non-parametric) data gate, so
        # .jacobian works for circuits mixing data gates with trainable gates.
        spec = CircuitSpec(
            1,
            1,
            0,
            (
                Gate("RY", (0,), (WeightRef(0),)),
                Gate("QubitUnitary", (0,), data=(_array_to_data(self._H),)),
            ),
        )
        circuit = self._circuit(spec)
        jac = circuit.jacobian(np.array([0.3]), np.empty(0))

        def f(t):
            return float(np.asarray(circuit(np.array([t]), np.empty(0)))[0])

        fd = (f(0.3 + 1e-6) - f(0.3 - 1e-6)) / 2e-6
        assert jac[0, 0] == pytest.approx(fd, abs=1e-5)

    def test_native_jacobian_back_propagates_through_amplitude_encoding(self):
        spec = CircuitSpec(
            1,
            1,
            0,
            (
                Gate("AmplitudeEncoding", (0,), data=(_array_to_data([0.6, 0.8]),)),
                Gate("RY", (0,), (WeightRef(0),)),
            ),
        )
        circuit = self._circuit(spec)
        jac = circuit.jacobian(np.array([0.5]), np.empty(0))

        def f(t):
            return float(np.asarray(circuit(np.array([t]), np.empty(0)))[0])

        fd = (f(0.5 + 1e-6) - f(0.5 - 1e-6)) / 2e-6
        assert jac[0, 0] == pytest.approx(fd, abs=1e-5)
