"""tensorcircuit backend: parity tests vs the numpy oracle. Skipped when
tensorcircuit-ng is not installed."""

from __future__ import annotations

import numpy as np
import pytest


tc = pytest.importorskip("tensorcircuit")  # skips entire module when missing

from eigenkit.core.backends.tensorcircuit_backend import (  # noqa: E402
    TensorCircuitBackend,
    TensorCircuitConfig,
)
from eigenkit.core.blocks import (  # noqa: E402
    AngleEncoding,
    BasicVariational,
    StrongVariational,
    assemble,
)
from eigenkit.core.circuit import Circuit  # noqa: E402
from eigenkit.core.observables import Z, all_z, from_string  # noqa: E402
from eigenkit.core.provider import register  # noqa: E402


@pytest.fixture(autouse=True)
def _register_tc():
    register("tensorcircuit", TensorCircuitBackend())


@pytest.fixture
def ir():
    return assemble(
        [AngleEncoding(in_features=3), BasicVariational(depth=2, rotation="RY")]
    )


@pytest.fixture
def weights():
    return np.array([0.4, 1.1, -0.3, 0.9, 0.2, -0.7], dtype=np.float64)


@pytest.fixture
def inputs():
    return np.array([0.5, -0.2, 0.8], dtype=np.float64)


class TestForwardParity:
    def test_single_observable(self, ir, weights, inputs):
        ref = Circuit(ir, Z(1).to_observable(), "numpy", "numpy")
        tcc = Circuit(ir, Z(1).to_observable(), "tensorcircuit", "numpy")
        assert abs(ref(weights, inputs) - tcc(weights, inputs)) < 1e-9

    def test_all_z(self, ir, weights, inputs):
        ref = Circuit(ir, all_z(3), "numpy", "numpy")
        tcc = Circuit(ir, all_z(3), "tensorcircuit", "numpy")
        assert np.allclose(ref(weights, inputs), tcc(weights, inputs), atol=1e-9)

    def test_hamiltonian(self, ir, weights, inputs):
        obs = from_string("0.5 Z0 X1 - 0.3 Y2 + 0.7 Z0 Z2")
        ref = Circuit(ir, obs, "numpy", "numpy")
        tcc = Circuit(ir, obs, "tensorcircuit", "numpy")
        assert abs(ref(weights, inputs) - tcc(weights, inputs)) < 1e-9

    def test_strong_variational(self):
        ir = assemble([AngleEncoding(in_features=2), StrongVariational(depth=2)])
        w = np.random.default_rng(0).standard_normal(ir.n_weights)
        x = np.array([0.3, -0.5], dtype=np.float64)
        ref = Circuit(ir, all_z(2), "numpy", "numpy")(w, x)
        tcc = Circuit(ir, all_z(2), "tensorcircuit", "numpy")(w, x)
        assert np.allclose(ref, tcc, atol=1e-9)

    def test_batched_inputs(self, ir, weights):
        x = np.random.default_rng(1).standard_normal((5, 3))
        ref = Circuit(ir, all_z(3), "numpy", "numpy")(weights, x)
        tcc = Circuit(ir, all_z(3), "tensorcircuit", "numpy")(weights, x)
        assert ref.shape == tcc.shape == (5, 3)
        assert np.allclose(ref, tcc, atol=1e-9)


class TestGradientParity:
    def test_weight_jacobian_matches_numpy(self, ir, weights, inputs):
        obs = from_string("0.7 Z0 + 0.3 X1 Z2")
        ref = Circuit(ir, obs, "numpy", "numpy").jacobian(weights, inputs)
        tcj = Circuit(ir, obs, "tensorcircuit", "numpy").jacobian(weights, inputs)
        assert np.allclose(ref, tcj, atol=1e-8)

    def test_input_jacobian_matches_numpy(self, ir, weights, inputs):
        obs = all_z(3)
        rw, rx = Circuit(ir, obs, "numpy", "numpy").jacobian(
            weights, inputs, input_grad=True
        )
        tw, tx = Circuit(ir, obs, "tensorcircuit", "numpy").jacobian(
            weights, inputs, input_grad=True
        )
        assert np.allclose(rw, tw, atol=1e-8)
        assert np.allclose(rx, tx, atol=1e-8)

    def test_torch_backward_matches_numpy(self, ir, weights, inputs):
        torch = pytest.importorskip("torch")
        obs = [from_string("0.7 Z0 + 0.3 X1 Z2"), Z(0).to_observable()]
        rw = torch.tensor(weights, requires_grad=True)
        tw = torch.tensor(weights, requires_grad=True)
        x = torch.tensor(inputs)
        (Circuit(ir, obs, "numpy", "torch")(rw, x) ** 2).sum().backward()
        (Circuit(ir, obs, "tensorcircuit", "torch")(tw, x) ** 2).sum().backward()
        assert torch.allclose(rw.grad, tw.grad, atol=1e-8)


class TestProbabilities:
    def test_matches_numpy(self, ir, weights, inputs):
        ref = Circuit(ir, all_z(3), "numpy", "numpy").probabilities(weights, inputs)
        tcp = Circuit(ir, all_z(3), "tensorcircuit", "numpy").probabilities(
            weights, inputs
        )
        assert np.allclose(ref, tcp, atol=1e-9)


class TestCapabilities:
    def test_adjoint_gradient_method(self):
        assert TensorCircuitBackend().gradient_method == "adjoint"

    def test_supported_gates(self):
        caps = TensorCircuitBackend().capabilities
        for name in ("RX", "RY", "RZ", "H", "CNOT"):
            assert name in caps.supported_gates


class TestConfig:
    def test_polymorphic_provider_via_config(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), provider=TensorCircuitConfig(), interface="numpy")
        assert circuit(weights, inputs).shape == (3,)
