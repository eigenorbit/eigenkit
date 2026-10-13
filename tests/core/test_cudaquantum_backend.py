"""CUDA-Q backend: parity tests, skipped when cudaq is not installed."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch


cudaq = pytest.importorskip("cudaq")  # skips entire module when missing

from eigenkit.core.backends.cudaquantum_backend import (  # noqa: E402
    CudaQuantumBackend,
    CudaQuantumConfig,
)
from eigenkit.core.blocks import (  # noqa: E402
    AngleEncoding,
    BasicVariational,
    assemble,
)
from eigenkit.core.builder import CircuitBuilder  # noqa: E402
from eigenkit.core.circuit import Circuit  # noqa: E402
from eigenkit.core.ir import (  # noqa: E402
    Constant,
    Gate,
    InputRef,
    Multiply,
    Sum,
    WeightRef,
)
from eigenkit.core.observables import Z, all_z, from_string  # noqa: E402
from eigenkit.core.provider import register  # noqa: E402


@pytest.fixture(autouse=True)
def _register_cudaq():
    register("cudaq", CudaQuantumBackend())


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
        reference = Circuit(ir, Z(1).to_observable(), "numpy", "numpy")
        cq = Circuit(ir, Z(1).to_observable(), "cudaq", "numpy")
        assert abs(reference(weights, inputs) - cq(weights, inputs)) < 1e-8

    def test_all_z(self, ir, weights, inputs):
        reference = Circuit(ir, all_z(3), "numpy", "numpy")
        cq = Circuit(ir, all_z(3), "cudaq", "numpy")
        assert np.allclose(reference(weights, inputs), cq(weights, inputs), atol=1e-8)

    def test_hamiltonian(self, ir, weights, inputs):
        observable = from_string("0.5 Z0 X1 - 0.3 Y2")
        reference = Circuit(ir, observable, "numpy", "numpy")
        cq = Circuit(ir, observable, "cudaq", "numpy")
        assert abs(reference(weights, inputs) - cq(weights, inputs)) < 1e-8


class TestGradientParity:
    def test_param_shift_matches_native(self, ir, weights, inputs):
        observables = [Z(0).to_observable(), Z(1).to_observable()]

        ref_w = torch.tensor(weights, requires_grad=True)
        cq_w = torch.tensor(weights, requires_grad=True)
        x_t = torch.tensor(inputs)

        reference = Circuit(ir, observables, "numpy", "torch")
        cq = Circuit(ir, observables, "cudaq", "torch")

        (reference(ref_w, x_t) ** 2).sum().backward()
        (cq(cq_w, x_t) ** 2).sum().backward()

        assert torch.allclose(ref_w.grad, cq_w.grad, atol=1e-6)


class TestNotAutoRegistered:
    """`cudaq` is opt-in: importing the package must not register it.

    The autouse fixture above registers cudaq, so `_REGISTRY` in this session
    cannot answer that either way. Checking the real question means looking at
    a registry this module's fixture has not touched.
    """

    def test_importing_the_package_does_not_register_cudaq(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import eigenkit\n"
                "from eigenkit.core.provider import _LAZY, _REGISTRY\n"
                "assert 'cudaq' not in _REGISTRY, 'registered eagerly'\n"
                "assert 'cudaq' not in _LAZY, 'registered lazily'\n",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_the_fixture_is_what_makes_it_resolvable_here(self):
        from eigenkit.core.provider import _REGISTRY

        assert "cudaq" in _REGISTRY

    def test_config_polymorphic(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), provider=CudaQuantumConfig(), interface="numpy")
        assert circuit(weights, inputs).shape == (3,)


class TestZeroStateProbability:
    """Parity check for `zero_state_probability` against the numpy simulator."""

    def test_inversion_returns_to_zero(self):
        # U(x) U^dag(x) leaves the state at |0...0>.
        qc = CircuitBuilder(n_qubits=2)
        x = qc.add_inputs(2)
        for q in range(2):
            qc.rx(x[q], qubit=q)
        for q in reversed(range(2)):
            qc.rx(_negate_via_scale(x[q]), qubit=q)
        spec = qc.build()

        circuit = Circuit(spec, all_z(2), "cudaq", "numpy")
        prob = circuit.zero_state_probability(
            weights=np.array([], dtype=np.float64),
            inputs=np.array([0.4, -0.3], dtype=np.float64),
        )
        assert abs(prob - 1.0) < 1e-9

    def test_matches_numpy(self, ir, weights, inputs):
        ref = Circuit(ir, all_z(3), "numpy", "numpy")
        cq = Circuit(ir, all_z(3), "cudaq", "numpy")
        assert (
            abs(
                ref.zero_state_probability(weights, inputs)
                - cq.zero_state_probability(weights, inputs)
            )
            < 1e-9
        )


def _negate_via_scale(parameter):
    """Cudaq supports `Scale` but not `Negate` of a symbolic ref. Use
    `Scale(expr, factor=-1.0)` to invert an angle in tests."""
    from eigenkit.core.ir import Scale

    return Scale(expression=parameter, factor=-1.0)


class TestUnsupportedExpressions:
    """`Sum` and `Multiply` over symbolic refs raise NotImplementedError
    on cudaq because QuakeValue arithmetic is restricted."""

    def test_sum_raises(self):
        spec = _spec_with_first_gate_param(
            n_weights=2,
            n_inputs=0,
            param=Sum((WeightRef(0), WeightRef(1))),
        )
        with pytest.raises(NotImplementedError, match="Sum"):
            Circuit(spec, all_z(1), "cudaq", "numpy")

    def test_multiply_raises(self):
        spec = _spec_with_first_gate_param(
            n_weights=1,
            n_inputs=1,
            param=Multiply(a=WeightRef(0), b=InputRef(0)),
        )
        with pytest.raises(NotImplementedError, match="Multiply"):
            Circuit(spec, all_z(1), "cudaq", "numpy")

    def test_weightref_without_weights_raises(self):
        # n_weights=0 but the gate references weights[0] — make_kernel
        # produces no weight_arg, so resolution should fail loudly.
        from eigenkit.core.ir import CircuitSpec

        spec = CircuitSpec(
            n_qubits=1,
            n_weights=0,
            n_inputs=1,
            gates=(Gate("RX", (0,), (WeightRef(0),)),),
        )
        with pytest.raises(ValueError, match="no weights"):
            Circuit(spec, all_z(1), "cudaq", "numpy")

    def test_inputref_without_inputs_raises(self):
        from eigenkit.core.ir import CircuitSpec

        spec = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=0,
            gates=(Gate("RX", (0,), (InputRef(0),)),),
        )
        with pytest.raises(ValueError, match="no inputs"):
            Circuit(spec, all_z(1), "cudaq", "numpy")


def _spec_with_first_gate_param(n_weights: int, n_inputs: int, param):
    from eigenkit.core.ir import CircuitSpec

    return CircuitSpec(
        n_qubits=1,
        n_weights=n_weights,
        n_inputs=n_inputs,
        gates=(Gate("RX", (0,), (param,)),),
    )


class TestKernelConstructionBranches:
    """`make_kernel(...)` is called with different signatures depending on
    whether the circuit has weights, inputs, both, or neither."""

    def test_weights_only(self):
        ir = assemble([BasicVariational(depth=1, rotation="RY")])
        circuit = Circuit(ir, all_z(1), "cudaq", "numpy")
        weights = np.array([0.4], dtype=np.float64)
        # No inputs needed when n_inputs == 0; the dispatch supplies an empty array.
        result = circuit(weights)
        assert result.shape == (1,)

    def test_inputs_only(self):
        ir = assemble([AngleEncoding(in_features=1, rotation="RY")])
        circuit = Circuit(ir, all_z(1), "cudaq", "numpy")
        weights = np.array([], dtype=np.float64)
        inputs = np.array([0.6], dtype=np.float64)
        result = circuit(weights, inputs)
        assert result.shape == (1,)

    def test_neither(self):
        # Build a constant-angle circuit by hand: no weights, no inputs.
        from eigenkit.core.ir import CircuitSpec

        spec = CircuitSpec(
            n_qubits=1,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("RX", (0,), (Constant(0.5),)),),
        )
        circuit = Circuit(spec, all_z(1), "cudaq", "numpy")
        weights = np.array([], dtype=np.float64)
        result = circuit(weights)
        assert result.shape == (1,)
        # cos(0.5) since RX(0.5) on |0> gives <Z> = cos(0.5).
        assert abs(result[0] - np.cos(0.5)) < 1e-8


class TestGateCoverage:
    """Exercise H and CNOT paths in `_apply_gate`."""

    def test_h_and_cnot(self):
        # H on q0, CNOT(0, 1) -> Bell state. Z(0) and Z(1) both have
        # expectation 0 on a Bell state.
        from eigenkit.core.ir import CircuitSpec

        spec = CircuitSpec(
            n_qubits=2,
            n_weights=0,
            n_inputs=0,
            gates=(
                Gate("H", (0,), ()),
                Gate("CNOT", (0, 1), ()),
            ),
        )
        circuit = Circuit(spec, all_z(2), "cudaq", "numpy")
        weights = np.array([], dtype=np.float64)
        result = circuit(weights)
        assert np.allclose(result, [0.0, 0.0], atol=1e-8)

    def test_cnot_lowers_to_the_right_state(self):
        """CNOT on |00> leaves it alone; on |10> it flips the target.

        Gate rejection is covered by `test_declared_gates_are_lowerable.py`;
        this checks that CNOT computes the right answer.
        """
        from eigenkit.core.ir import CircuitSpec

        no_weights = np.array([], dtype=np.float64)
        both_zero = CircuitSpec(
            n_qubits=2, n_weights=0, n_inputs=0, gates=(Gate("CNOT", (0, 1), ()),)
        )
        np.testing.assert_allclose(
            Circuit(both_zero, all_z(2), "cudaq", "numpy")(no_weights),
            [1.0, 1.0],
            atol=1e-9,
        )
        control_set = CircuitSpec(
            n_qubits=2,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("X", (0,), ()), Gate("CNOT", (0, 1), ())),
        )
        np.testing.assert_allclose(
            Circuit(control_set, all_z(2), "cudaq", "numpy")(no_weights),
            [-1.0, -1.0],
            atol=1e-9,
        )


class TestShotsBranch:
    """`shots=N` switches `cudaq.observe` to the finite-shot signature."""

    def test_finite_shots_runs(self, ir, weights, inputs):
        register("cudaq-shots", CudaQuantumBackend(config=CudaQuantumConfig(shots=2048)))
        circuit = Circuit(ir, Z(0).to_observable(), "cudaq-shots", "numpy")
        analytic = Circuit(ir, Z(0).to_observable(), "cudaq", "numpy")
        # Finite-shots should agree with analytic to within shot noise.
        # 2048 shots gives roughly +/- 0.05 std on a single Pauli expval.
        sampled = circuit(weights, inputs)
        exact = analytic(weights, inputs)
        assert abs(sampled - exact) < 0.1


class TestTorchStillCompilesAfterCudaq:
    """CUDA-Q loads its LLVM/MLIR libraries into the global symbol scope, and
    triton (torch's compiler on Linux, which bundles another LLVM) crashes in
    `dlopen` if it loads after them: unless the load order is handled, the
    first `torch.optim` call after a cudaq circuit segfaults. A fresh process,
    because this session already has triton loaded by the root conftest."""

    def test_an_optimizer_after_a_cudaq_circuit_does_not_crash(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch\n"
                "from eigenkit import Circuit, Encoding, Variational, all_z\n"
                "from eigenkit.configs import CudaQuantumConfig\n"
                "Circuit(blocks=[Encoding(in_features=2), Variational(depth=1)],\n"
                "        observables=all_z(2), provider=CudaQuantumConfig())\n"
                "torch.optim.Adam([torch.zeros(1, requires_grad=True)])\n",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr[-2000:]
