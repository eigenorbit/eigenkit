"""qulacs backend: parity tests, skipped when qulacs is not installed.

The numpy reference simulator is the ground truth (adjoint-exact, always
available), so these compare qulacs against ``provider="numpy"`` — forward
values, native adjoint gradients, the fidelity-kernel primitive, and
statevector I/O — plus the full gate set qulacs advertises."""

from __future__ import annotations

import math

import numpy as np
import pytest


qulacs = pytest.importorskip("qulacs")  # skips entire module when missing

from eigenkit.core.backends.qulacs_backend import (  # noqa: E402
    QulacsBackend,
    QulacsConfig,
)
from eigenkit.core.blocks import (  # noqa: E402
    AmplitudeEncoding,
    AngleEncoding,
    BasicVariational,
    QubitUnitary,
    Reuploading,
    StrongVariational,
    assemble,
)
from eigenkit.core.builder import CircuitBuilder  # noqa: E402
from eigenkit.core.circuit import Circuit  # noqa: E402
from eigenkit.core.ir import (  # noqa: E402
    ALL_GATES,
    CircuitSpec,
    Gate,
    adjoint,
)
from eigenkit.core.observables import Z, all_z, from_string  # noqa: E402
from eigenkit.core.provider import register  # noqa: E402


@pytest.fixture(autouse=True)
def _register_qulacs():
    register("qulacs", QulacsBackend())


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


class TestForwardParity:
    def test_single_observable(self, ir, weights, inputs):
        ql = Circuit(ir, Z(1).to_observable(), "qulacs", "numpy")
        ref = Circuit(ir, Z(1).to_observable(), "numpy", "numpy")
        assert abs(ql(weights, inputs) - ref(weights, inputs)) < 1e-10

    def test_all_z(self, ir, weights, inputs):
        ql = Circuit(ir, all_z(3), "qulacs", "numpy")
        ref = Circuit(ir, all_z(3), "numpy", "numpy")
        assert np.allclose(ql(weights, inputs), ref(weights, inputs), atol=1e-10)

    def test_hamiltonian(self, ir, weights, inputs):
        observable = from_string("0.5 Z0 X1 - 0.3 Y2 + 0.7 Z0 Z2")
        ql = Circuit(ir, observable, "qulacs", "numpy")
        ref = Circuit(ir, observable, "numpy", "numpy")
        assert abs(ql(weights, inputs) - ref(weights, inputs)) < 1e-10

    def test_strong_variational(self):
        ir = assemble([AngleEncoding(in_features=2), StrongVariational(depth=2)])
        weights = np.random.default_rng(0).standard_normal(ir.n_weights)
        inputs = np.array([0.3, -0.5], dtype=np.float64)
        ql = Circuit(ir, all_z(2), "qulacs", "numpy")(weights, inputs)
        ref = Circuit(ir, all_z(2), "numpy", "numpy")(weights, inputs)
        assert np.allclose(ql, ref, atol=1e-10)

    def test_batch_shape_and_values(self, ir, weights, inputs):
        batch = np.stack([inputs, inputs * 0.5, inputs * -1.0])
        ql = Circuit(ir, all_z(3), "qulacs", "numpy")(weights, batch)
        ref = Circuit(ir, all_z(3), "numpy", "numpy")(weights, batch)
        assert ql.shape == (3, 3)
        assert np.allclose(ql, ref, atol=1e-10)


class TestAdjointGradientParity:
    """qulacs `backprop` must match the numpy adjoint jacobian exactly."""

    def test_weight_jacobian(self, ir, weights, inputs):
        observable = from_string("0.7 Z0 + 0.3 X1 Z2 - 0.5 Y0 X2")
        ql = Circuit(ir, observable, "qulacs", "numpy").jacobian(weights, inputs)
        ref = Circuit(ir, observable, "numpy", "numpy").jacobian(weights, inputs)
        assert np.allclose(ql, ref, atol=1e-9)

    def test_input_jacobian(self, ir, weights, inputs):
        ql = Circuit(ir, all_z(3), "qulacs", "numpy").jacobian(
            weights, inputs, input_grad=True
        )
        ref = Circuit(ir, all_z(3), "numpy", "numpy").jacobian(
            weights, inputs, input_grad=True
        )
        assert np.allclose(ql[0], ref[0], atol=1e-9)  # weight jac
        assert np.allclose(ql[1], ref[1], atol=1e-9)  # input jac

    def test_trainable_scale_chain_rule(self):
        """`Reuploading(trainable_scale=True)` emits `Multiply(weight, input)`
        angles; the adjoint scatter must apply the chain-rule factor."""
        ir = assemble([Reuploading(in_features=2, depth=2, trainable_scale=True)])
        weights = np.random.default_rng(1).standard_normal(ir.n_weights)
        inputs = np.array([0.6, -0.4], dtype=np.float64)
        ql = Circuit(ir, all_z(2), "qulacs", "numpy").jacobian(weights, inputs)
        ref = Circuit(ir, all_z(2), "numpy", "numpy").jacobian(weights, inputs)
        assert np.allclose(ql, ref, atol=1e-9)

    def test_batched_jacobian(self, ir, weights, inputs):
        observables = [from_string("0.7 Z0 + 0.3 X1 Z2"), Z(0).to_observable()]
        batch = np.stack([inputs, inputs * 0.5, -inputs])
        ql = Circuit(ir, observables, "qulacs", "numpy").compiled.jacobian(weights, batch)
        ref = Circuit(ir, observables, "numpy", "numpy").compiled.jacobian(weights, batch)
        assert ql[0].shape == (3, 2, ir.n_weights)
        assert np.allclose(ql[0], ref[0], atol=1e-9)

    def test_torch_interface_matches_numpy(self, ir, weights, inputs):
        """qulacs' adjoint jacobian flows through the torch autograd wrapper
        the same way the numpy backend's does."""
        torch = pytest.importorskip("torch")
        observables = [from_string("0.7 Z0 + 0.3 X1 Z2"), Z(0).to_observable()]
        ql_w = torch.tensor(weights, requires_grad=True)
        ref_w = torch.tensor(weights, requires_grad=True)
        x_t = torch.tensor(inputs)

        ql = Circuit(ir, observables, "qulacs", "torch")
        ref = Circuit(ir, observables, "numpy", "torch")
        (ql(ql_w, x_t) ** 2).sum().backward()
        (ref(ref_w, x_t) ** 2).sum().backward()
        assert torch.allclose(ql_w.grad, ref_w.grad, atol=1e-8)


class TestCapabilities:
    def test_adjoint_gradient_method(self):
        assert QulacsBackend().gradient_method == "adjoint"

    def test_supports_full_gate_set(self):
        assert QulacsBackend().capabilities.supported_gates == ALL_GATES

    def test_amplitude_encoding_adjoint_unsupported(self):
        caps = QulacsBackend().capabilities
        assert "AmplitudeEncoding" not in caps.supported_adjoint_gates


class TestCompile:
    def test_compile_exposes_run_and_jacobian(self, ir):
        compiled = QulacsBackend().compile(ir, all_z(3))
        assert hasattr(compiled, "run")
        assert hasattr(compiled, "jacobian")
        assert compiled.supports_batched_jacobian is True

    def test_native_jacobian_is_preferred(self, ir, weights, inputs):
        """A compiled circuit exposing `jacobian` is used instead of the
        parameter-shift fallback — assert the handle carries the method."""
        circuit = Circuit(ir, all_z(3), "qulacs", "numpy")
        assert getattr(circuit.compiled, "jacobian", None) is not None

    def test_compiled_circuit_reusable(self, ir, weights, inputs):
        compiled = QulacsBackend().compile(ir, all_z(3))
        a = compiled.run(weights, inputs)
        b = compiled.run(weights, inputs)
        assert np.array_equal(a, b)

    def test_no_native_torch(self, ir):
        """qulacs is not a differentiable-tensor library: no `run_torch`."""
        compiled = QulacsBackend().compile(ir, all_z(3))
        assert not hasattr(compiled, "run_torch")


class TestZeroStateProbability:
    """The fidelity-kernel primitive behind QuantumKernel / QSVC / QSVR."""

    def test_matches_numpy(self, ir, weights, inputs):
        ql = Circuit(ir, all_z(3), "qulacs", "numpy")
        ref = Circuit(ir, all_z(3), "numpy", "numpy")
        assert np.isclose(
            ql.compiled.zero_state_probability(weights, inputs),
            ref.compiled.zero_state_probability(weights, inputs),
            atol=1e-10,
        )

    def test_hadamard_gives_half(self):
        spec = CircuitSpec(n_qubits=2, n_weights=0, n_inputs=0, gates=(Gate("H", (0,)),))
        compiled = QulacsBackend().compile(spec, all_z(2))
        prob = compiled.zero_state_probability(np.empty(0), np.empty(0))
        assert abs(prob - 0.5) < 1e-10


class TestStateVector:
    def test_readout_matches_numpy(self, ir, weights, inputs):
        ql = Circuit(ir, all_z(3), "qulacs", "numpy").statevector(weights, inputs)
        ref = Circuit(ir, all_z(3), "numpy", "numpy").statevector(weights, inputs)
        assert ql.shape == (8,)
        assert np.allclose(ql, ref, atol=1e-10)

    def test_endianness_qubit0_is_msb(self):
        """X on qubit 0 must land at flat index 4 (|100>), qubit-0-MSB order."""
        spec = CircuitSpec(n_qubits=3, n_weights=0, n_inputs=0, gates=(Gate("X", (0,)),))
        sv = Circuit(spec, all_z(3), "qulacs", "numpy").statevector(np.empty(0))
        assert int(np.argmax(np.abs(sv))) == 4

    def test_injection_matches_readout(self):
        prep = CircuitSpec(
            3, 0, 0, (Gate("RY", (0,), (_const(0.7),)), Gate("CNOT", (0, 1)))
        )
        rest = CircuitSpec(
            3, 0, 0, (Gate("RY", (2,), (_const(0.3),)), Gate("CNOT", (1, 2)))
        )
        full = CircuitSpec(3, 0, 0, prep.gates + rest.gates)
        psi = Circuit(prep, all_z(3), "qulacs", "numpy").statevector(np.empty(0))
        direct = Circuit(full, all_z(3), "qulacs", "numpy").statevector(np.empty(0))
        injected = Circuit(rest, all_z(3), "qulacs", "numpy").statevector(
            np.empty(0), initial_state=psi
        )
        assert np.allclose(direct, injected, atol=1e-10)

    def test_unnormalized_initial_state_raises(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "qulacs", "numpy")
        with pytest.raises(ValueError, match="normalized"):
            circuit.statevector(weights, inputs, initial_state=np.ones(8))


def _const(value):
    from eigenkit.core.ir import Constant

    return Constant(value)


def _gate_spec_parity(build_fn, n_qubits, obs_qubit=0):
    qc = CircuitBuilder(n_qubits=n_qubits)
    build_fn(qc)
    spec = qc.build()
    obs = Z(obs_qubit).to_observable()
    w = np.zeros(spec.n_weights)
    x = np.zeros(spec.n_inputs)
    ql = Circuit(spec, obs, "qulacs", "numpy")(w, x)
    ref = Circuit(spec, obs, "numpy", "numpy")(w, x)
    return ql, ref


class TestGateCoverage:
    """Every gate qulacs advertises must match numpy (constant-angle paths)."""

    @pytest.mark.parametrize(
        "name, build, n, obs_q",
        [
            ("S/Sdg", lambda qc: qc.h(0).s(0).sdg(0).h(0), 2, 0),
            ("X", lambda qc: qc.x(0), 2, 0),
            ("Z", lambda qc: qc.h(0).z(0).h(0), 2, 0),
            ("PhaseShift", lambda qc: qc.h(0).phase_shift(math.pi / 3, 0).h(0), 2, 0),
            ("SWAP", lambda qc: qc.x(0).swap(0, 1), 2, 1),
            ("CRY", lambda qc: qc.x(0).cry(math.pi / 3, 0, 1), 2, 1),
            ("CPHASE", lambda qc: qc.h(0).h(1).cphase(math.pi / 3, 0, 1), 2, 0),
            ("MCX", lambda qc: qc.x(0).x(1).mcx(0, 1, 2), 3, 2),
            ("MCZ", lambda qc: qc.x(0).x(1).h(2).mcz(0, 1, 2).h(2), 3, 2),
        ],
    )
    def test_gate_matches_numpy(self, name, build, n, obs_q):
        ql, ref = _gate_spec_parity(build, n, obs_q)
        assert abs(ql - ref) < 1e-10, name


class TestDataGates:
    def test_qubit_unitary_2qubit(self):
        cnot = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        qb = CircuitBuilder(n_qubits=2)
        qb.x(0)
        prep = qb.build()
        qu = assemble([QubitUnitary(cnot, wires=(0, 1))]).gates
        spec = CircuitSpec(2, 0, 0, prep.gates + qu)
        ql = Circuit(spec, all_z(2), "qulacs", "numpy")(np.empty(0), np.empty(0))
        ref = Circuit(spec, all_z(2), "numpy", "numpy")(np.empty(0), np.empty(0))
        assert np.allclose(ql, ref, atol=1e-10)

    def test_qubit_unitary_3qubit_bit_reversal(self):
        rng = np.random.default_rng(99)
        m = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        u, _ = np.linalg.qr(m)
        spec = assemble([QubitUnitary(u, wires=(0, 1, 2))])
        ql = Circuit(spec, all_z(3), "qulacs", "numpy")(np.empty(0), np.empty(0))
        ref = Circuit(spec, all_z(3), "numpy", "numpy")(np.empty(0), np.empty(0))
        assert np.allclose(ql, ref, atol=1e-10)

    def test_amplitude_encoding_nonuniform(self):
        spec = assemble([AmplitudeEncoding([1.0, 2.0, 3.0, 4.0], normalize=True)])
        ql = Circuit(spec, all_z(2), "qulacs", "numpy")(np.empty(0), np.empty(0))
        ref = Circuit(spec, all_z(2), "numpy", "numpy")(np.empty(0), np.empty(0))
        assert np.allclose(ql, ref, atol=1e-10)

    def test_amplitude_encoding_adjoint_raises(self):
        spec = adjoint(
            assemble([AmplitudeEncoding([1.0, 0.0, 0.0, 0.0], normalize=False)])
        )
        with pytest.raises(ValueError, match="adjoint"):
            Circuit(spec, all_z(2), "qulacs", "numpy")(np.empty(0), np.empty(0))


class TestVariableControlledRotations:
    """Trainable-angle CRY / CPHASE / PhaseShift go through the parametric
    decomposition; their gradients must still match numpy."""

    def test_cry_variable_gradient(self):
        qc = CircuitBuilder(n_qubits=2)
        w = qc.add_weights(1)
        qc.h(0)
        qc.cry(w[0], 0, 1)
        spec = qc.build()
        wv = np.array([0.8])
        obs = from_string("0.4 Z1 + 0.6 X0")
        ql = Circuit(spec, obs, "qulacs", "numpy").jacobian(wv, np.empty(0))
        ref = Circuit(spec, obs, "numpy", "numpy").jacobian(wv, np.empty(0))
        assert np.allclose(ql, ref, atol=1e-9)

    def test_cphase_variable_gradient(self):
        qc = CircuitBuilder(n_qubits=2)
        w = qc.add_weights(1)
        qc.h(0)
        qc.h(1)
        qc.cphase(w[0], 0, 1)
        spec = qc.build()
        wv = np.array([0.8])
        obs = from_string("0.5 X0 X1")
        ql = Circuit(spec, obs, "qulacs", "numpy").jacobian(wv, np.empty(0))
        ref = Circuit(spec, obs, "numpy", "numpy").jacobian(wv, np.empty(0))
        assert np.allclose(ql, ref, atol=1e-9)


class TestPolymorphicProvider:
    def test_config_as_provider(self, ir, weights, inputs):
        """Pass QulacsConfig directly as provider — no register call needed."""
        circuit = Circuit(ir, all_z(3), provider=QulacsConfig(), interface="numpy")
        assert circuit(weights, inputs).shape == (3,)
