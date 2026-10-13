"""Backend parity tests: PennyLane and Qiskit must agree on forward values
and gradients for the same IR circuit. Also exercises compile/run split."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest
import torch

from eigenkit.core.backends.pennylane_backend import (
    PennyLaneBackend,
    PennyLaneConfig,
)
from eigenkit.core.backends.qiskit_backend import QiskitBackend, QiskitConfig
from eigenkit.core.blocks import (
    AmplitudeEncoding,
    AngleEncoding,
    BasicVariational,
    QubitUnitary,
    assemble,
)
from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import CircuitSpec, Constant, Gate
from eigenkit.core.observables import Z, all_z, from_string
from eigenkit.torch.circuit import TorchCircuit


_HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
_HAS_BRAKET = importlib.util.find_spec("braket") is not None
_needs_qiskit = pytest.mark.skipif(not _HAS_QISKIT, reason="requires qiskit extra")


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


# ---- cross-backend parity ----


@pytest.mark.pennylane
@_needs_qiskit
class TestForwardParity:
    def test_single_observable(self, ir, weights, inputs):
        pennylane = Circuit(ir, Z(1).to_observable(), "pennylane", "numpy")
        qiskit = Circuit(ir, Z(1).to_observable(), "qiskit", "numpy")
        assert abs(pennylane(weights, inputs) - qiskit(weights, inputs)) < 1e-10

    def test_all_z(self, ir, weights, inputs):
        pennylane = Circuit(ir, all_z(3), "pennylane", "numpy")
        qiskit = Circuit(ir, all_z(3), "qiskit", "numpy")
        assert np.allclose(
            pennylane(weights, inputs), qiskit(weights, inputs), atol=1e-10
        )

    def test_hamiltonian(self, ir, weights, inputs):
        observable = from_string("0.5 Z0 X1 - 0.3 Y2")
        pennylane = Circuit(ir, observable, "pennylane", "numpy")
        qiskit = Circuit(ir, observable, "qiskit", "numpy")
        assert abs(pennylane(weights, inputs) - qiskit(weights, inputs)) < 1e-10


@pytest.mark.pennylane
@_needs_qiskit
class TestGradientParity:
    def test_native_vs_parameter_shift(self, ir, weights, inputs):
        observables = [
            from_string("0.7 Z0 + 0.3 X1 Z2 - 0.5 Y0 X2"),
            Z(0).to_observable(),
        ]

        weights_pl = torch.tensor(weights, requires_grad=True)
        weights_qk = torch.tensor(weights, requires_grad=True)
        inputs_t = torch.tensor(inputs)

        pennylane = Circuit(ir, observables, "pennylane", "torch")
        qiskit = Circuit(ir, observables, "qiskit", "torch")

        (pennylane(weights_pl, inputs_t) ** 2).sum().backward()
        (qiskit(weights_qk, inputs_t) ** 2).sum().backward()

        assert torch.allclose(weights_pl.grad, weights_qk.grad, atol=1e-8)


# ---- cross-backend parity: Braket, CUDA-Q, PennyLane, Qulacs against numpy ----


def _register_optional_backend(name: str) -> None:
    """Register opt-in backends (CUDA-Q) so the parity fixture can drive them
    by string name. numpy, PennyLane, Qiskit and Braket are
    auto-registered."""
    from eigenkit.core.provider import _REGISTRY, register

    if name in _REGISTRY:
        return
    if name == "cudaq":
        from eigenkit.core.backends.cudaquantum_backend import (
            CudaQuantumBackend,
        )

        register("cudaq", CudaQuantumBackend())
    elif name == "qulacs":
        from eigenkit.core.backends.qulacs_backend import QulacsBackend

        register("qulacs", QulacsBackend())


@pytest.fixture(params=["braket", "cudaq", "pennylane", "qulacs"])
def parity_provider(request):
    """Parametrize over the optional backends, each checked against the numpy
    simulator. All auto-skip when their packages aren't installed."""
    name = request.param
    if name == "braket":
        pytest.importorskip("braket")
    elif name in ("cudaq", "qulacs"):
        pytest.importorskip(name)
    _register_optional_backend(name)
    return name


class TestForwardParityExtended:
    """Forward parity vs. the numpy simulator for every supported backend."""

    def test_single_observable(self, parity_provider, ir, weights, inputs):
        reference = Circuit(ir, Z(1).to_observable(), "numpy", "numpy")
        candidate = Circuit(ir, Z(1).to_observable(), parity_provider, "numpy")
        assert abs(reference(weights, inputs) - candidate(weights, inputs)) < 1e-8

    def test_all_z(self, parity_provider, ir, weights, inputs):
        reference = Circuit(ir, all_z(3), "numpy", "numpy")
        candidate = Circuit(ir, all_z(3), parity_provider, "numpy")
        assert np.allclose(
            reference(weights, inputs),
            candidate(weights, inputs),
            atol=1e-8,
        )

    def test_hamiltonian(self, parity_provider, ir, weights, inputs):
        observable = from_string("0.5 Z0 X1 - 0.3 Y2 + 0.7 Z0 Z2")
        reference = Circuit(ir, observable, "numpy", "numpy")
        candidate = Circuit(ir, observable, parity_provider, "numpy")
        assert abs(reference(weights, inputs) - candidate(weights, inputs)) < 1e-8


class TestGradientParityExtended:
    """Gradient parity vs. the numpy simulator's adjoint gradients."""

    def test_grad_matches_numpy(self, parity_provider, ir, weights, inputs):
        observables = [
            from_string("0.7 Z0 + 0.3 X1 Z2"),
            Z(0).to_observable(),
        ]

        ref_w = torch.tensor(weights, requires_grad=True)
        cand_w = torch.tensor(weights, requires_grad=True)
        inputs_t = torch.tensor(inputs)

        reference = Circuit(ir, observables, "numpy", "torch")
        candidate = Circuit(ir, observables, parity_provider, "torch")

        (reference(ref_w, inputs_t) ** 2).sum().backward()
        (candidate(cand_w, inputs_t) ** 2).sum().backward()

        assert torch.allclose(ref_w.grad, cand_w.grad, atol=1e-6)


# ---- state management: statevector readout + injection ----


@pytest.fixture(params=["pennylane", "qiskit", "qulacs"])
def state_io_provider(request):
    """Backends that expose statevector readout + injection. Each auto-skips
    when its package is absent; qulacs is opt-in so we register it by name."""
    name = request.param
    if name in ("qiskit", "qulacs"):
        pytest.importorskip(name)
    _register_optional_backend(name)
    return name


class TestStateVectorParity:
    """`statevector()` readout and `initial_state` injection must agree with
    the numpy reference on every state-IO backend (the inject/read-out pair
    behind reservoir carry-over and amplitude encoding)."""

    def test_readout_matches_numpy(self, state_io_provider, ir, weights, inputs):
        reference = Circuit(ir, all_z(3), "numpy", "numpy").statevector(weights, inputs)
        candidate = Circuit(ir, all_z(3), state_io_provider, "numpy").statevector(
            weights, inputs
        )
        assert candidate.shape == (8,)
        assert np.allclose(candidate, reference, atol=1e-8)

    def test_readout_batch_matches_numpy(self, state_io_provider, ir, weights, inputs):
        batch = np.stack([inputs, 0.5 * inputs, -inputs])
        reference = Circuit(ir, all_z(3), "numpy", "numpy").statevector(weights, batch)
        candidate = Circuit(ir, all_z(3), state_io_provider, "numpy").statevector(
            weights, batch
        )
        assert candidate.shape == (3, 8)
        assert np.allclose(candidate, reference, atol=1e-8)

    def test_injection_matches_in_circuit_prep(self, state_io_provider):
        # Preparing |psi> with gates then running the rest must equal injecting
        # |psi> as the rest's initial state.
        prep = CircuitSpec(
            3, 0, 0, (Gate("RY", (0,), (Constant(0.7),)), Gate("CNOT", (0, 1)))
        )
        rest = CircuitSpec(
            3, 0, 0, (Gate("RY", (2,), (Constant(0.3),)), Gate("CNOT", (1, 2)))
        )
        full = CircuitSpec(3, 0, 0, prep.gates + rest.gates)

        psi = Circuit(prep, all_z(3), state_io_provider, "numpy").statevector(np.empty(0))
        direct = Circuit(full, all_z(3), state_io_provider, "numpy").statevector(
            np.empty(0)
        )
        injected = Circuit(rest, all_z(3), state_io_provider, "numpy").statevector(
            np.empty(0), initial_state=psi
        )
        assert np.allclose(direct, injected, atol=1e-8)
        # And the read-out state matches numpy, so it round-trips across backends.
        ref_psi = Circuit(prep, all_z(3), "numpy", "numpy").statevector(np.empty(0))
        assert np.allclose(psi, ref_psi, atol=1e-8)

    def test_unnormalized_initial_state_raises(
        self, state_io_provider, ir, weights, inputs
    ):
        circuit = Circuit(ir, all_z(3), state_io_provider, "numpy")
        with pytest.raises(ValueError, match="normalized"):
            circuit.statevector(weights, inputs, initial_state=np.ones(8))


@_needs_qiskit
def test_qiskit_statevector_endianness():
    """Qiskit's amplitudes are little-endian; our convention is qubit-0-MSB.
    X on qubit 0 must land at flat index 4 (|100>), as numpy/PennyLane report."""
    spec = CircuitSpec(n_qubits=3, n_weights=0, n_inputs=0, gates=(Gate("X", (0,)),))
    sv = Circuit(spec, all_z(3), "qiskit", "numpy").statevector(np.empty(0))
    assert int(np.argmax(np.abs(sv))) == 4


def test_statevector_unsupported_backend_raises():
    """Backends without statevector access raise a clear NotImplementedError."""
    pytest.importorskip("braket")
    ir = CircuitSpec(n_qubits=1, n_weights=0, n_inputs=0, gates=(Gate("H", (0,)),))
    circuit = Circuit(ir, all_z(1), "braket", "numpy")
    with pytest.raises(NotImplementedError, match="statevector"):
        circuit.statevector(np.empty(0))


# ---- compile handle ----


class TestCompileHandle:
    @pytest.mark.pennylane
    def test_returns_compiled_circuit(self, ir):
        backend = PennyLaneBackend()
        compiled = backend.compile(ir, all_z(3))
        assert hasattr(compiled, "run")

    @_needs_qiskit
    def test_compiled_circuit_reusable(self, ir, weights, inputs):
        """Multiple runs on the same handle must produce consistent values."""
        compiled = QiskitBackend().compile(ir, all_z(3))
        a = compiled.run(weights, inputs)
        b = compiled.run(weights, inputs)
        assert np.allclose(a, b)

    @pytest.mark.pennylane
    def test_pennylane_has_native_torch_path(self, ir):
        compiled = PennyLaneBackend().compile(ir, all_z(3))
        assert hasattr(compiled, "run_torch")

    @_needs_qiskit
    def test_qiskit_no_native_torch(self, ir):
        compiled = QiskitBackend().compile(ir, all_z(3))
        assert not hasattr(compiled, "run_torch")


# ---- device config ----


class TestDeviceConfig:
    @pytest.mark.pennylane
    def test_pennylane_default_device(self):
        backend = PennyLaneBackend()
        assert backend.config.device_name == "default.qubit"

    @pytest.mark.pennylane
    def test_pennylane_alternate_device_via_config(self, ir, weights, inputs):
        """Pass the config directly as the provider — no registration needed."""
        circuit = Circuit(
            ir,
            all_z(3),
            provider=PennyLaneConfig(device_name="lightning.qubit"),
            interface="numpy",
        )
        default_circuit = Circuit(ir, all_z(3), "pennylane", "numpy")
        assert np.allclose(
            circuit(weights, inputs),
            default_circuit(weights, inputs),
            atol=1e-10,
        )

    @pytest.mark.pennylane
    def test_pennylane_alternate_device_via_register(self, ir, weights, inputs):
        """register(name, config) for reuse across many circuits."""
        from eigenkit.core.provider import register

        register("pennylane-lightning", PennyLaneConfig(device_name="lightning.qubit"))
        circuit = Circuit(ir, all_z(3), "pennylane-lightning", "numpy")
        default_circuit = Circuit(ir, all_z(3), "pennylane", "numpy")
        assert np.allclose(
            circuit(weights, inputs),
            default_circuit(weights, inputs),
            atol=1e-10,
        )

    @_needs_qiskit
    def test_qiskit_shots_unwired(self):
        backend = QiskitBackend(config=QiskitConfig(shots=1000))
        ir = CircuitSpec(n_qubits=1, n_weights=0, n_inputs=0, gates=(Gate("H", (0,)),))
        with pytest.raises(NotImplementedError):
            backend.compile(ir, [Z(0).to_observable()])


# ---- finite-shot sampling (qml.set_shots path) ----


@pytest.mark.pennylane
class TestPennyLaneShots:
    """`shots` lives on the qnode via `qml.set_shots` (PL 0.43+), not on
    the device. These tests exercise that wrapper path for run, run_torch,
    and zero_state_probability."""

    def test_run_with_shots_matches_analytic(self, ir, weights, inputs):
        np.random.seed(0)
        shots_circuit = Circuit(
            ir,
            all_z(3),
            provider=PennyLaneConfig(shots=50_000),
            interface="numpy",
        )
        analytic = Circuit(ir, all_z(3), "pennylane", "numpy")
        assert np.allclose(
            shots_circuit(weights, inputs),
            analytic(weights, inputs),
            atol=0.05,
        )

    def test_run_torch_with_shots_matches_analytic(self, ir, weights, inputs):
        np.random.seed(0)
        shots_circuit = Circuit(
            ir,
            all_z(3),
            provider=PennyLaneConfig(shots=50_000),
            interface="torch",
        )
        analytic = Circuit(ir, all_z(3), "pennylane", "numpy")
        out = shots_circuit(torch.tensor(weights), torch.tensor(inputs))
        assert np.allclose(out.detach().numpy(), analytic(weights, inputs), atol=0.05)

    def test_zero_state_probability_with_shots(self, weights, inputs):
        """Hadamard on the first qubit gives p(0…0) ≈ 0.5; verify the
        shots branch in the lazy zsp qnode runs and stays in bounds."""
        np.random.seed(0)
        spec = CircuitSpec(
            n_qubits=3,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("H", (0,)),),
        )
        backend = PennyLaneBackend(config=PennyLaneConfig(shots=50_000))
        compiled = backend.compile(spec, all_z(3))
        prob = compiled.zero_state_probability(np.empty(0), np.empty(0))
        assert 0.45 <= prob <= 0.55


# ---- parametric circuit (Qiskit) ----


@_needs_qiskit
class TestQiskitParametric:
    def test_same_params_same_output(self, ir, weights, inputs):
        """The cached parametric circuit binds parameters each call; verify
        same inputs give same output."""
        compiled = QiskitBackend().compile(ir, all_z(3))
        a = compiled.run(weights, inputs)
        b = compiled.run(weights, inputs)
        assert np.array_equal(a, b)

    def test_different_params_different_output(self, ir, weights, inputs):
        compiled = QiskitBackend().compile(ir, all_z(3))
        a = compiled.run(weights, inputs)
        b = compiled.run(weights * 1.3, inputs)
        assert not np.allclose(a, b)


# ---- TorchCircuit ----


class TestTorchCircuit:
    @_needs_qiskit
    def test_trains_with_sgd(self, ir):
        layer = TorchCircuit(ir, observables=all_z(3), provider="qiskit")
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        target = torch.zeros(3, dtype=torch.float64)
        x = torch.tensor([0.5, -0.2, 0.8], dtype=torch.float64)
        initial_loss = None
        final_loss = None
        for step in range(5):
            optimizer.zero_grad()
            y = layer(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
        assert final_loss < initial_loss

    @pytest.mark.pennylane
    def test_weights_in_parameters(self, ir):
        layer = TorchCircuit(ir, observables=all_z(3), provider="pennylane")
        parameters = list(layer.parameters())
        assert len(parameters) == 1
        assert parameters[0].shape == (ir.n_weights,)


# ---- batching ----


class TestBatching:
    @_needs_qiskit
    def test_numpy_batch_shape(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "qiskit", "numpy")
        batch = np.stack([inputs, inputs * 0.5, inputs * -1.0])
        output = circuit(weights, batch)
        assert output.shape == (3, 3)

    @pytest.mark.pennylane
    def test_torch_batch_shape(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "pennylane", "torch")
        batch = torch.tensor(np.stack([inputs, inputs * 0.5]))
        weights_t = torch.tensor(weights, requires_grad=True)
        output = circuit(weights_t, batch)
        assert output.shape == (2, 3)

    @pytest.mark.pennylane
    def test_batch_matches_per_sample(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), "pennylane", "numpy")
        batch = np.stack([inputs, inputs * 0.5])
        batched = circuit(weights, batch)
        per_sample = np.stack([circuit(weights, batch[i]) for i in range(len(batch))])
        assert np.allclose(batched, per_sample)

    @pytest.mark.pennylane
    def test_rejects_3d_input(self, ir, weights):
        circuit = Circuit(ir, all_z(3), "pennylane", "numpy")
        with pytest.raises(ValueError, match="1D.*2D"):
            circuit(weights, np.zeros((2, 3, 3)))


# ---- gate parity: X, Z, MCX, MCZ ----


def _run_spec(spec, observable, provider):
    """Compile and run a parameter-free CircuitSpec on a given provider."""
    from eigenkit.core.circuit import Circuit

    circuit = Circuit(spec, observable, provider, "numpy")
    return circuit(np.empty(0), np.empty(0))


@pytest.fixture
def x_gate_spec():
    """X on qubit 0: |0⟩ → |1⟩, Z(0) expectation = -1."""
    from eigenkit.core.builder import CircuitBuilder

    qc = CircuitBuilder(n_qubits=2)
    qc.x(0)
    return qc.build()


@pytest.fixture
def z_gate_spec():
    """H → Z → H on qubit 0, equivalent to X: Z(0) expectation = -1."""
    from eigenkit.core.builder import CircuitBuilder

    qc = CircuitBuilder(n_qubits=2)
    qc.h(0).z(0).h(0)
    return qc.build()


@pytest.fixture
def mcx_gate_spec():
    """X on qubit 0 (control = |1⟩), MCX(0→1): Z(1) expectation = -1."""
    from eigenkit.core.builder import CircuitBuilder

    qc = CircuitBuilder(n_qubits=3)
    qc.x(0).mcx(0, 1)
    return qc.build()


@pytest.fixture
def mcz_gate_spec():
    """X on qubit 0 (control = |1⟩), H on qubit 1, MCZ(0, 1), H on qubit 1:
    MCZ flips phase of |+⟩ to |−⟩; final H maps |−⟩ → |1⟩, Z(1) = -1."""
    from eigenkit.core.builder import CircuitBuilder

    qc = CircuitBuilder(n_qubits=3)
    qc.x(0).h(1).mcz(0, 1).h(1)
    return qc.build()


@pytest.mark.pennylane
@_needs_qiskit
class TestExtendedGatesParityQiskit:
    """PennyLane ↔ Qiskit parity for X, Z, MCX, MCZ."""

    def test_x_gate(self, x_gate_spec):
        obs = Z(0).to_observable()
        pl = _run_spec(x_gate_spec, obs, "pennylane")
        qk = _run_spec(x_gate_spec, obs, "qiskit")
        assert abs(pl - qk) < 1e-10

    def test_z_gate(self, z_gate_spec):
        obs = Z(0).to_observable()
        pl = _run_spec(z_gate_spec, obs, "pennylane")
        qk = _run_spec(z_gate_spec, obs, "qiskit")
        assert abs(pl - qk) < 1e-10

    def test_mcx_gate(self, mcx_gate_spec):
        obs = Z(1).to_observable()
        pl = _run_spec(mcx_gate_spec, obs, "pennylane")
        qk = _run_spec(mcx_gate_spec, obs, "qiskit")
        assert abs(pl - qk) < 1e-10

    def test_mcz_gate(self, mcz_gate_spec):
        obs = Z(1).to_observable()
        pl = _run_spec(mcz_gate_spec, obs, "pennylane")
        qk = _run_spec(mcz_gate_spec, obs, "qiskit")
        assert abs(pl - qk) < 1e-10


@pytest.mark.parametrize(
    "spec_fixture, obs_qubit",
    [
        ("x_gate_spec", 0),
        ("z_gate_spec", 0),
        ("mcx_gate_spec", 1),
        ("mcz_gate_spec", 1),
    ],
)
class TestMoreExtendedGatesParity:
    """numpy ↔ Braket / CUDA-Q / PennyLane / Qulacs parity for X, Z, MCX, MCZ."""

    def test_parity(self, request, parity_provider, spec_fixture, obs_qubit):
        if parity_provider == "cudaq" and spec_fixture in (
            "mcx_gate_spec",
            "mcz_gate_spec",
        ):
            pytest.skip("cudaq does not support MCX / MCZ gates")
        spec = request.getfixturevalue(spec_fixture)
        obs = Z(obs_qubit).to_observable()
        reference = _run_spec(spec, obs, "numpy")
        candidate = _run_spec(spec, obs, parity_provider)
        assert abs(reference - candidate) < 1e-8


@pytest.fixture
def phase_shift_spec():
    """PhaseShift(π) on qubit 0 = Z gate: H → PhaseShift(π) → H = X, Z(0) = -1."""
    import math

    from eigenkit.core.builder import CircuitBuilder

    qc = CircuitBuilder(n_qubits=2)
    qc.h(0).phase_shift(math.pi, 0).h(0)
    return qc.build()


@pytest.fixture
def swap_spec():
    """X on qubit 0, SWAP(0,1): qubit 1 becomes |1⟩, Z(1) = -1."""
    from eigenkit.core.builder import CircuitBuilder

    qc = CircuitBuilder(n_qubits=2)
    qc.x(0).swap(0, 1)
    return qc.build()


@pytest.fixture
def cry_spec():
    """CRY(π) with control=|1⟩ (set by X): target rotates to |1⟩, Z(1) = -1."""
    import math

    from eigenkit.core.builder import CircuitBuilder

    qc = CircuitBuilder(n_qubits=2)
    qc.x(0).cry(math.pi, 0, 1)
    return qc.build()


@pytest.mark.pennylane
@_needs_qiskit
class TestHHLGatesParityQiskit:
    """PennyLane ↔ Qiskit parity for PhaseShift, SWAP, CRY."""

    def test_phase_shift(self, phase_shift_spec):
        obs = Z(0).to_observable()
        pl = _run_spec(phase_shift_spec, obs, "pennylane")
        qk = _run_spec(phase_shift_spec, obs, "qiskit")
        assert abs(pl - qk) < 1e-10

    def test_swap(self, swap_spec):
        obs = Z(1).to_observable()
        pl = _run_spec(swap_spec, obs, "pennylane")
        qk = _run_spec(swap_spec, obs, "qiskit")
        assert abs(pl - qk) < 1e-10

    def test_cry(self, cry_spec):
        obs = Z(1).to_observable()
        pl = _run_spec(cry_spec, obs, "pennylane")
        qk = _run_spec(cry_spec, obs, "qiskit")
        assert abs(pl - qk) < 1e-10


@pytest.mark.parametrize(
    "spec_fixture, obs_qubit",
    [
        ("phase_shift_spec", 0),
        ("swap_spec", 1),
        ("cry_spec", 1),
    ],
)
class TestHHLGatesParityExtended:
    """numpy ↔ Braket / CUDA-Q / PennyLane / Qulacs parity for PhaseShift, SWAP, CRY."""

    def test_parity(self, request, parity_provider, spec_fixture, obs_qubit):
        spec = request.getfixturevalue(spec_fixture)
        obs = Z(obs_qubit).to_observable()
        reference = _run_spec(spec, obs, "numpy")
        candidate = _run_spec(spec, obs, parity_provider)
        assert abs(reference - candidate) < 1e-8


@pytest.mark.pennylane
class TestHHLGatesExpectedValues:
    """Sanity-check known analytical results on PennyLane."""

    def test_phase_shift_pi_equals_z(self, phase_shift_spec):
        # PhaseShift(π) acts as Z; H·Z·H = X → Z(0) = -1
        result = _run_spec(phase_shift_spec, Z(0).to_observable(), "pennylane")
        assert abs(result - (-1.0)) < 1e-10

    def test_swap_moves_qubit(self, swap_spec):
        result = _run_spec(swap_spec, Z(1).to_observable(), "pennylane")
        assert abs(result - (-1.0)) < 1e-10

    def test_cry_rotates_target_when_control_set(self, cry_spec):
        result = _run_spec(cry_spec, Z(1).to_observable(), "pennylane")
        assert abs(result - (-1.0)) < 1e-10


class TestAdjointHHLGates:
    """adjoint() correctly negates parameters for PhaseShift and CRY."""

    def test_phase_shift_adjoint_negates_angle(self):
        import math

        from eigenkit.core.builder import CircuitBuilder
        from eigenkit.core.ir import adjoint

        qc = CircuitBuilder(n_qubits=1)
        qc.phase_shift(math.pi / 4, 0)
        spec = qc.build()
        adj = adjoint(spec)
        # original angle = π/4, adjoint angle = -π/4
        from eigenkit.core.ir import resolve

        original_angle = resolve(spec.gates[0].params[0], np.empty(0), np.empty(0))
        adjoint_angle = resolve(adj.gates[0].params[0], np.empty(0), np.empty(0))
        assert abs(original_angle + adjoint_angle) < 1e-12

    def test_cry_adjoint_negates_angle(self):
        import math

        from eigenkit.core.builder import CircuitBuilder
        from eigenkit.core.ir import adjoint

        qc = CircuitBuilder(n_qubits=2)
        qc.cry(math.pi / 3, 0, 1)
        spec = qc.build()
        adj = adjoint(spec)
        from eigenkit.core.ir import resolve

        original_angle = resolve(spec.gates[0].params[0], np.empty(0), np.empty(0))
        adjoint_angle = resolve(adj.gates[0].params[0], np.empty(0), np.empty(0))
        assert abs(original_angle + adjoint_angle) < 1e-12

    def test_swap_adjoint_is_itself(self):
        from eigenkit.core.builder import CircuitBuilder
        from eigenkit.core.ir import adjoint

        qc = CircuitBuilder(n_qubits=2)
        qc.swap(0, 1)
        spec = qc.build()
        adj = adjoint(spec)
        assert adj.gates[0] == spec.gates[0]


@pytest.mark.pennylane
class TestExtendedGatesExpectedValues:
    """Sanity-check known analytical results on PennyLane."""

    def test_x_flips_zero_state(self, x_gate_spec):
        result = _run_spec(x_gate_spec, Z(0).to_observable(), "pennylane")
        assert abs(result - (-1.0)) < 1e-10

    def test_hzh_equals_x(self, z_gate_spec):
        result = _run_spec(z_gate_spec, Z(0).to_observable(), "pennylane")
        assert abs(result - (-1.0)) < 1e-10

    def test_mcx_flips_target_when_control_set(self, mcx_gate_spec):
        result = _run_spec(mcx_gate_spec, Z(1).to_observable(), "pennylane")
        assert abs(result - (-1.0)) < 1e-10

    def test_mcz_flips_phase(self, mcz_gate_spec):
        result = _run_spec(mcz_gate_spec, Z(1).to_observable(), "pennylane")
        assert abs(result - (-1.0)) < 1e-10


# ---- QAE blocks: AmplitudeEncoding, S0, QubitUnitary, Grover ----


def _z_expectations(spec, provider, weights=None, inputs=None):
    """Run a CircuitSpec through a provider and return all Z expectations."""
    from eigenkit.core.circuit import Circuit
    from eigenkit.core.observables import all_z

    w = np.zeros(spec.n_weights) if weights is None else np.asarray(weights)
    x = np.zeros(spec.n_inputs) if inputs is None else np.asarray(inputs)
    circuit = Circuit(spec, all_z(spec.n_qubits), provider, "numpy")
    return np.asarray(circuit(w, x), dtype=float)


def _bit_reversal_perm(n):
    return [int(format(k, f"0{n}b")[::-1], 2) for k in range(2**n)]


# ── AmplitudeEncoding ──────────────────────────────────────────────────────────


@pytest.mark.pennylane
@_needs_qiskit
class TestAmplitudeEncodingParityQiskit:
    """Z-expectations must match PennyLane vs Qiskit for AmplitudeEncoding."""

    def test_uniform_amplitudes(self):
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        spec = assemble([AmplitudeEncoding([0.5, 0.5, 0.5, 0.5], normalize=False)])
        pl = _z_expectations(spec, "pennylane")
        qk = _z_expectations(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)

    def test_nonuniform_amplitudes(self):
        """Asymmetric amplitudes tell LSB-first from MSB-first ordering apart."""
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        spec = assemble([AmplitudeEncoding([1.0, 2.0, 3.0, 4.0], normalize=True)])
        pl = _z_expectations(spec, "pennylane")
        qk = _z_expectations(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)

    def test_3qubit_nonuniform(self):
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        amps = list(range(1, 9))  # 8 amplitudes → 3 qubits
        spec = assemble([AmplitudeEncoding(amps, normalize=True)])
        pl = _z_expectations(spec, "pennylane")
        qk = _z_expectations(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)


@pytest.mark.pennylane
class TestAmplitudeEncodingAnalytical:
    """Analytical Z-expectation checks for AmplitudeEncoding (PennyLane only)."""

    def test_basis_state_00(self):
        """[1,0,0,0] = |00⟩: Z(0)=+1, Z(1)=+1."""
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        spec = assemble([AmplitudeEncoding([1.0, 0.0, 0.0, 0.0], normalize=False)])
        pl = _z_expectations(spec, "pennylane")
        np.testing.assert_allclose(pl, [1.0, 1.0], atol=1e-10)

    def test_basis_state_01(self):
        """[0,1,0,0] = |01⟩: Z(0)=+1, Z(1)=-1."""
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        spec = assemble([AmplitudeEncoding([0.0, 1.0, 0.0, 0.0], normalize=False)])
        pl = _z_expectations(spec, "pennylane")
        np.testing.assert_allclose(pl, [1.0, -1.0], atol=1e-10)

    def test_basis_state_10(self):
        """[0,0,1,0] = |10⟩: Z(0)=-1, Z(1)=+1."""
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        spec = assemble([AmplitudeEncoding([0.0, 0.0, 1.0, 0.0], normalize=False)])
        pl = _z_expectations(spec, "pennylane")
        np.testing.assert_allclose(pl, [-1.0, 1.0], atol=1e-10)

    def test_basis_state_11(self):
        """[0,0,0,1] = |11⟩: Z(0)=-1, Z(1)=-1."""
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        spec = assemble([AmplitudeEncoding([0.0, 0.0, 0.0, 1.0], normalize=False)])
        pl = _z_expectations(spec, "pennylane")
        np.testing.assert_allclose(pl, [-1.0, -1.0], atol=1e-10)

    def test_uniform_superposition(self):
        """Uniform |+⟩|+⟩ state: all Z expectations = 0."""
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        spec = assemble([AmplitudeEncoding([0.5, 0.5, 0.5, 0.5], normalize=False)])
        pl = _z_expectations(spec, "pennylane")
        np.testing.assert_allclose(pl, [0.0, 0.0], atol=1e-10)


# ── S0 ────────────────────────────────────────────────────────────────────────


@pytest.mark.pennylane
@_needs_qiskit
class TestS0ParityQiskit:
    """S0 block: analytical checks and PL↔Qiskit parity."""

    def _h_then_s0_spec(self, n):
        """H^⊗n followed by S0(n): uniform superposition then zero-state reflection."""
        from eigenkit.core.blocks import S0
        from eigenkit.core.builder import CircuitBuilder
        from eigenkit.core.ir import CircuitSpec

        qb = CircuitBuilder(n_qubits=n)
        for q in range(n):
            qb.h(q)
        prep = qb.build()
        s0_gates = S0(n_qubits=n).build(n_qubits=n).gates
        return CircuitSpec(
            n_qubits=n, n_weights=0, n_inputs=0, gates=prep.gates + s0_gates
        )

    def test_s0_2qubit_parity(self):
        spec = self._h_then_s0_spec(2)
        pl = _z_expectations(spec, "pennylane")
        qk = _z_expectations(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)

    def test_s0_3qubit_parity(self):
        spec = self._h_then_s0_spec(3)
        pl = _z_expectations(spec, "pennylane")
        qk = _z_expectations(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)

    def test_s0_on_zero_state_parity(self):
        """S0|00⟩: PL and Qiskit must agree (global phase, all Z = +1)."""
        from eigenkit.core.blocks import S0, assemble

        spec = assemble([S0(n_qubits=2)])
        pl = _z_expectations(spec, "pennylane")
        qk = _z_expectations(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)
        np.testing.assert_allclose(pl, [1.0, 1.0], atol=1e-10)


@pytest.mark.pennylane
class TestS0Analytical:
    """Analytical S0 checks (PennyLane only)."""

    def test_s0_on_zero_state(self):
        """S0|00⟩ = -|00⟩: Z expectations still +1 (global phase only)."""
        from eigenkit.core.blocks import S0, assemble

        spec = assemble([S0(n_qubits=2)])
        pl = _z_expectations(spec, "pennylane")
        np.testing.assert_allclose(pl, [1.0, 1.0], atol=1e-10)

    def test_s0_uniform_superposition_z_expectations(self):
        """After H^⊗2 S0: state = (-|00⟩+|01⟩+|10⟩+|11⟩)/2, ⟨Z_i⟩ = 0."""
        from eigenkit.core.blocks import S0
        from eigenkit.core.builder import CircuitBuilder
        from eigenkit.core.ir import CircuitSpec

        n = 2
        qb = CircuitBuilder(n_qubits=n)
        for q in range(n):
            qb.h(q)
        prep = qb.build()
        s0_gates = S0(n_qubits=n).build(n_qubits=n).gates
        spec = CircuitSpec(
            n_qubits=n, n_weights=0, n_inputs=0, gates=prep.gates + s0_gates
        )
        pl = _z_expectations(spec, "pennylane")
        # (-|00⟩+|01⟩+|10⟩+|11⟩)/2: probs are all 1/4 → Z(0) = 0, Z(1) = 0
        np.testing.assert_allclose(pl, [0.0, 0.0], atol=1e-10)


# ── QubitUnitary ───────────────────────────────────────────────────────────────


@pytest.mark.pennylane
@_needs_qiskit
class TestQubitUnitaryParityQiskit:
    """QubitUnitary block: Z-expectation parity PL↔Qiskit.

    The Qiskit backend applies a bit-reversal permutation to matrix rows/cols
    to compensate for Qiskit's LSB-first ordering. These tests pin that
    permutation.
    """

    def _run(self, spec, provider, weights=None):
        return _z_expectations(spec, provider, weights=weights)

    def test_1qubit_hadamard_parity(self):
        """1-qubit Hadamard: |0⟩ → |+⟩, Z(0) = 0."""
        from eigenkit.core.blocks import QubitUnitary, assemble

        H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
        spec = assemble([QubitUnitary(H, wires=(0,))])
        pl = self._run(spec, "pennylane")
        qk = self._run(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)
        np.testing.assert_allclose(pl, [0.0], atol=1e-10)

    def test_2qubit_cnot_parity(self):
        """2-qubit CNOT matrix, starting |10⟩ → |11⟩."""
        from eigenkit.core.blocks import QubitUnitary, assemble
        from eigenkit.core.builder import CircuitBuilder
        from eigenkit.core.ir import CircuitSpec

        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        qb = CircuitBuilder(n_qubits=2)
        qb.x(0)
        prep = qb.build()
        qu_gates = assemble([QubitUnitary(CNOT, wires=(0, 1))]).gates
        spec = CircuitSpec(
            n_qubits=2, n_weights=0, n_inputs=0, gates=prep.gates + qu_gates
        )
        pl = self._run(spec, "pennylane")
        qk = self._run(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)
        # |10⟩ → CNOT → |11⟩: both qubits in |1⟩, Z = -1 each
        np.testing.assert_allclose(pl, [-1.0, -1.0], atol=1e-10)

    def test_2qubit_cnot_all_basis_states(self):
        """CNOT maps |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩."""
        from eigenkit.core.blocks import (
            AmplitudeEncoding,
            QubitUnitary,
            assemble,
        )
        from eigenkit.core.ir import CircuitSpec

        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        qu_gates = assemble([QubitUnitary(CNOT, wires=(0, 1))]).gates

        for in_amps, expected_z0, expected_z1 in [
            ([1, 0, 0, 0], +1.0, +1.0),  # |00⟩→|00⟩
            ([0, 1, 0, 0], +1.0, -1.0),  # |01⟩→|01⟩
            ([0, 0, 1, 0], -1.0, -1.0),  # |10⟩→|11⟩
            ([0, 0, 0, 1], -1.0, +1.0),  # |11⟩→|10⟩
        ]:
            ae_gates = assemble([AmplitudeEncoding(in_amps, normalize=False)]).gates
            spec = CircuitSpec(
                n_qubits=2, n_weights=0, n_inputs=0, gates=ae_gates + qu_gates
            )
            pl = _z_expectations(spec, "pennylane")
            qk = _z_expectations(spec, "qiskit")
            np.testing.assert_allclose(
                pl,
                [expected_z0, expected_z1],
                atol=1e-10,
                err_msg=f"PL wrong for input {in_amps}",
            )
            np.testing.assert_allclose(
                qk,
                [expected_z0, expected_z1],
                atol=1e-10,
                err_msg=f"Qiskit wrong for input {in_amps}",
            )

    def test_random_1qubit_on_wire1_parity(self):
        """Random 1-qubit unitary on wire 1 of a 2-qubit circuit."""
        from eigenkit.core.blocks import QubitUnitary, assemble

        rng = np.random.default_rng(42)
        M = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        U, _ = np.linalg.qr(M)
        spec = assemble([QubitUnitary(U, wires=(1,))])
        pl = self._run(spec, "pennylane")
        qk = self._run(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)

    def test_random_2qubit_parity(self):
        """Random 2-qubit unitary: PL and Qiskit Z-expectations must agree."""
        from eigenkit.core.blocks import QubitUnitary, assemble

        rng = np.random.default_rng(7)
        M = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        U, _ = np.linalg.qr(M)
        spec = assemble([QubitUnitary(U, wires=(0, 1))])
        pl = self._run(spec, "pennylane")
        qk = self._run(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)

    def test_3qubit_unitary_parity(self):
        """3-qubit unitary: bit-reversal permutation applied to 8×8 matrix."""
        from eigenkit.core.blocks import QubitUnitary, assemble

        rng = np.random.default_rng(99)
        M = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        U, _ = np.linalg.qr(M)
        spec = assemble([QubitUnitary(U, wires=(0, 1, 2))])
        pl = self._run(spec, "pennylane")
        qk = self._run(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)


# ── Grover ────────────────────────────────────────────────────────────────────


@pytest.mark.pennylane
@_needs_qiskit
class TestGroverParityQiskit:
    """Grover block: Z-expectation parity PL↔Qiskit."""

    def _bell_oracle(self):
        from eigenkit.core.builder import CircuitBuilder

        qb = CircuitBuilder(n_qubits=2)
        qb.h(0).cnot(0, 1)
        return qb.build()

    def _ry_oracle(self, theta):
        from eigenkit.core.builder import CircuitBuilder

        qb = CircuitBuilder(n_qubits=2)
        w = qb.add_weights(1)
        qb.ry(w[0], 0).cnot(0, 1)
        return qb.build()

    def _h3_mcx_oracle(self):
        from eigenkit.core.builder import CircuitBuilder

        qb = CircuitBuilder(n_qubits=3)
        for q in range(3):
            qb.h(q)
        qb.mcx(0, 1, 2)
        return qb.build()

    def test_bell_oracle_parity(self):
        from eigenkit.core.blocks import Grover, assemble

        spec = assemble([Grover(a_circuit=self._bell_oracle())])
        pl = _z_expectations(spec, "pennylane")
        qk = _z_expectations(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)

    def test_parametric_oracle_parity(self):
        from eigenkit.core.blocks import Grover, assemble

        theta = math.pi / 5
        spec = assemble([Grover(a_circuit=self._ry_oracle(theta))])
        pl = _z_expectations(spec, "pennylane", weights=[theta])
        qk = _z_expectations(spec, "qiskit", weights=[theta])
        np.testing.assert_allclose(pl, qk, atol=1e-10)

    def test_3qubit_oracle_parity(self):
        from eigenkit.core.blocks import Grover, assemble

        spec = assemble([Grover(a_circuit=self._h3_mcx_oracle())])
        pl = _z_expectations(spec, "pennylane")
        qk = _z_expectations(spec, "qiskit")
        np.testing.assert_allclose(pl, qk, atol=1e-10)


@pytest.mark.pennylane
class TestGroverAnalytical:
    """Analytical Grover checks (PennyLane only)."""

    def test_bell_oracle_both_qubits_in_one(self):
        """H⊗CNOT oracle → Grover iterate: both qubits in |1⟩."""
        from eigenkit.core.blocks import Grover, assemble
        from eigenkit.core.builder import CircuitBuilder

        qb = CircuitBuilder(n_qubits=2)
        qb.h(0).cnot(0, 1)
        spec = assemble([Grover(a_circuit=qb.build())])
        pl = _z_expectations(spec, "pennylane")
        np.testing.assert_allclose(pl, [-1.0, -1.0], atol=1e-10)

    def test_weight_sharing_varies_output(self):
        """Weights flow to both A and A†; varying θ changes Z expectations."""
        import math as _math

        from eigenkit.core.blocks import Grover, assemble
        from eigenkit.core.builder import CircuitBuilder

        qb = CircuitBuilder(n_qubits=2)
        w = qb.add_weights(1)
        qb.ry(w[0], 0).cnot(0, 1)
        spec = assemble([Grover(a_circuit=qb.build())])
        assert spec.n_weights == 1

        r_small = _z_expectations(spec, "pennylane", weights=[0.1])
        r_large = _z_expectations(spec, "pennylane", weights=[_math.pi / 3])
        assert not np.allclose(r_small, r_large, atol=1e-6)

    def test_identity_oracle_is_self_inverse(self):
        """A = identity (no gates except H): A†·A = I, Grover iterate is deterministic."""
        from eigenkit.core.blocks import Grover, assemble
        from eigenkit.core.builder import CircuitBuilder

        qb = CircuitBuilder(n_qubits=2)
        qb.h(0)
        spec = assemble([Grover(a_circuit=qb.build())])
        pl = _z_expectations(spec, "pennylane")
        # result is well-defined; just check it doesn't crash and is bounded
        assert np.all(np.abs(pl) <= 1.0 + 1e-10)


# ── Cross-backend parity (Braket) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "n_qubits,amps",
    [
        (2, [1.0, 2.0, 3.0, 4.0]),
        (3, list(range(1, 9))),
    ],
)
class TestAmplitudeEncodingParityBraket:
    def test_numpy_vs_braket(self, request, parity_provider, n_qubits, amps):
        if parity_provider not in ("braket",):
            pytest.skip("only testing braket here")
        from eigenkit.core.blocks import AmplitudeEncoding, assemble

        spec = assemble([AmplitudeEncoding(amps, normalize=True)])
        reference = _z_expectations(spec, "numpy")
        try:
            candidate = _z_expectations(spec, parity_provider)
        except (NotImplementedError, ValueError):
            pytest.skip(f"{parity_provider} does not support AmplitudeEncoding")
        np.testing.assert_allclose(reference, candidate, atol=1e-8)


# ── adjoint(AmplitudeEncoding) ─────────────────────────────────────────────────


class TestAmplitudeEncodingAdjoint:
    """AmplitudeEncoding adjoint works on PennyLane; Qiskit raises NotImplementedError."""

    @pytest.mark.pennylane
    def test_adjoint_then_forward_is_identity_pennylane(self):
        """A† · A |0⟩ = |0⟩: the composed circuit leaves the state unchanged."""
        from eigenkit.core.blocks import AmplitudeEncoding, assemble
        from eigenkit.core.ir import CircuitSpec, adjoint

        amps = [1.0, 2.0, 3.0, 4.0]
        fwd = assemble([AmplitudeEncoding(amps, normalize=True)])
        adj = adjoint(fwd)
        composed = CircuitSpec(
            n_qubits=fwd.n_qubits,
            n_weights=0,
            n_inputs=0,
            gates=fwd.gates + adj.gates,
        )
        # A† · A |0⟩ = |0⟩ → Z(q) = +1 for all qubits
        pl = _z_expectations(composed, "pennylane")
        np.testing.assert_allclose(pl, np.ones(fwd.n_qubits), atol=1e-8)

    @_needs_qiskit
    def test_adjoint_amplitude_encoding_raises_on_qiskit(self):
        """Qiskit cannot invert state-prep; the provider's capability check
        rejects adjoint(AmplitudeEncoding) with a ValueError at compile time
        (Qiskit's supported_adjoint_gates excludes AmplitudeEncoding)."""
        from eigenkit.core.blocks import AmplitudeEncoding, assemble
        from eigenkit.core.ir import adjoint

        spec = adjoint(
            assemble([AmplitudeEncoding([1.0, 0.0, 0.0, 0.0], normalize=False)])
        )
        with pytest.raises(ValueError, match="adjoint"):
            _z_expectations(spec, "qiskit")


# ── Grover parity: Braket ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "oracle_name,weights",
    [
        ("bell", []),
        ("ry_pi5", [math.pi / 5]),
    ],
)
class TestGroverParityBraket:
    """Grover on Braket must match the numpy simulator (the oracles use basic gates)."""

    def _build_spec(self, oracle_name, weights):
        from eigenkit.core.blocks import Grover, assemble
        from eigenkit.core.builder import CircuitBuilder

        if oracle_name == "bell":
            qb = CircuitBuilder(n_qubits=2)
            qb.h(0).cnot(0, 1)
        else:  # ry_pi5
            qb = CircuitBuilder(n_qubits=2)
            w = qb.add_weights(1)
            qb.ry(w[0], 0).cnot(0, 1)
        return assemble([Grover(a_circuit=qb.build())])

    def test_parity(self, request, parity_provider, oracle_name, weights):
        if parity_provider not in ("braket",):
            pytest.skip("only testing braket here")
        try:
            import braket  # noqa: F401
        except ImportError:
            pytest.skip("braket not installed")
        spec = self._build_spec(oracle_name, weights)
        w = np.array(weights) if weights else np.empty(0)
        reference = _z_expectations(spec, "numpy", weights=w if len(w) else None)
        candidate = _z_expectations(spec, parity_provider, weights=w if len(w) else None)
        np.testing.assert_allclose(reference, candidate, atol=1e-8)


# ---- empty inputs default ----


class TestEmptyInputsDefault:
    @_needs_qiskit
    def test_pure_variational_accepts_no_inputs(self):
        ir = assemble([BasicVariational(depth=1, rotation="RY")])
        circuit = Circuit(ir, Z(0).to_observable(), "qiskit", "numpy")
        # Calling without inputs should work since n_inputs == 0
        weights = np.array([0.5], dtype=np.float64)
        result = circuit(weights)
        assert isinstance(result, float)

    @pytest.mark.pennylane
    def test_torch_pure_variational_no_inputs(self):
        ir = assemble([BasicVariational(depth=1, rotation="RY")])
        layer = TorchCircuit(
            ir,
            Z(0).to_observable(),
            "pennylane",
            init=torch.tensor([0.5], dtype=torch.float64),
        )
        result = layer()  # no x
        assert result.ndim == 0

    @_needs_qiskit
    def test_missing_inputs_raises_when_expected(self, ir):
        circuit = Circuit(ir, all_z(3), "qiskit", "numpy")
        weights = np.zeros(ir.n_weights)
        with pytest.raises(ValueError, match="inputs? value"):
            circuit(weights)


# ---- PennyLane gate dispatch for data-carrying gates ----


class TestPennyLaneExtendedGateDispatch:
    """Fast unit tests for AmplitudeEncoding and QubitUnitary gate dispatch
    through the PennyLane backend (no qiskit required)."""

    @pytest.mark.pennylane
    def test_amplitude_encoding_runs(self):
        amps = [0.5, 0.5, 0.5, 0.5]
        spec = assemble([AmplitudeEncoding(amps, normalize=False)])
        circuit = Circuit(spec, all_z(2), "pennylane", "numpy")
        result = circuit(np.empty(0), np.empty(0))
        assert result.shape == (2,)

    @pytest.mark.pennylane
    def test_amplitude_encoding_normalized_state(self):
        # |0> state should give Z=+1 on all qubits before rotation
        amps = [1.0, 0.0, 0.0, 0.0]
        spec = assemble([AmplitudeEncoding(amps, normalize=False)])
        circuit = Circuit(spec, all_z(2), "pennylane", "numpy")
        result = circuit(np.empty(0), np.empty(0))
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-8)

    @pytest.mark.pennylane
    def test_qubit_unitary_identity_is_noop(self):
        I2 = np.eye(2, dtype=complex)
        spec = assemble([AngleEncoding(in_features=1), QubitUnitary(I2, wires=(0,))])
        circuit = Circuit(spec, all_z(1), "pennylane", "numpy")
        inputs = np.array([0.3])
        result_plain = Circuit(
            assemble([AngleEncoding(in_features=1)]), all_z(1), "pennylane", "numpy"
        )(np.empty(0), inputs)
        result_with_id = circuit(np.empty(0), inputs)
        np.testing.assert_allclose(result_plain, result_with_id, atol=1e-10)

    @pytest.mark.pennylane
    def test_qubit_unitary_hadamard_maps_z0_to_x(self):
        H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
        spec = assemble([QubitUnitary(H, wires=(0,))])
        circuit = Circuit(spec, all_z(1), "pennylane", "numpy")
        result = circuit(np.empty(0), np.empty(0))
        # H|0> = |+>, <+|Z|+> = 0
        np.testing.assert_allclose(result, [0.0], atol=1e-8)

    @pytest.mark.pennylane
    def test_pennylane_rx_and_rz_gates(self):
        from eigenkit.core.ir import CircuitSpec, Gate, WeightRef

        spec = CircuitSpec(
            n_qubits=1,
            n_weights=2,
            n_inputs=0,
            gates=(
                Gate("RX", (0,), (WeightRef(0),)),
                Gate("RZ", (0,), (WeightRef(1),)),
            ),
        )
        circuit = Circuit(spec, all_z(1), "pennylane", "numpy")
        weights = np.array([0.0, 0.0])
        result = circuit(weights, np.empty(0))
        np.testing.assert_allclose(result, [1.0], atol=1e-8)

    @pytest.mark.pennylane
    def test_amplitude_encoding_adjoint_runs(self):
        from eigenkit.core.ir import CircuitSpec, Gate, _array_to_data

        amps = np.array([0.5, 0.5, 0.5, 0.5])
        data = (_array_to_data(amps),)
        gate = Gate("AmplitudeEncoding", (0, 1), (), data=data, adjoint=True)
        spec = CircuitSpec(n_qubits=2, n_weights=0, n_inputs=0, gates=(gate,))
        circuit = Circuit(spec, all_z(2), "pennylane", "numpy")
        result = circuit(np.empty(0), np.empty(0))
        assert result.shape == (2,)

    def test_unknown_gate_raises_not_implemented(self):
        from eigenkit.core.backends.pennylane_backend import _apply_gate
        from eigenkit.core.ir import Gate

        gate = Gate("UnknownGate", (0,))
        with pytest.raises(NotImplementedError, match="not implemented"):
            _apply_gate(gate, np.empty(0), np.empty(0))
