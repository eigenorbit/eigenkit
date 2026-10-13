"""User-facing top-level Circuit: blocks-list construction, polymorphic provider."""

from __future__ import annotations

import importlib.util
import itertools

import pytest
import torch

from eigenkit import Circuit as _Circuit
from eigenkit import (
    CircuitBuilder,
    Encoding,
    PennyLaneConfig,
    QiskitConfig,
    StrongVariational,
    Variational,
    Z,
    all_z,
    assemble,
)


# The top-level Circuit defaults to interface="numpy", provider="numpy".
# Every case in this module exercises the torch nn.Module dispatch path, so bind
# that as the local default here; `test_default_is_numpy` below pins the real
# out-of-the-box default.
def Circuit(*args, **kwargs):
    kwargs.setdefault("interface", "torch")
    return _Circuit(*args, **kwargs)


_HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
_HAS_BRAKET = importlib.util.find_spec("braket") is not None


def test_default_is_numpy():
    """The default: a bare Circuit(...) is a numpy circuit on the numpy sim."""
    import numpy as np

    from eigenkit.core.circuit import Circuit as NumpyCircuit

    circuit = _Circuit(blocks=[Encoding(in_features=2), Variational(depth=1)])
    assert isinstance(circuit, NumpyCircuit)
    assert circuit.provider.name == "numpy"
    out = circuit(np.zeros(circuit.spec.n_weights), np.array([0.1, 0.2]))
    assert out.shape == (2,)


def _default_providers():
    """Default-registered providers that are usable in this environment."""
    providers = ["numpy", "pennylane"]
    if _HAS_QISKIT:
        providers.append("qiskit")
    if _HAS_BRAKET:
        providers.append("braket")
    return providers


# Auto-registered on import (see ``_ir.backends``).
_PROVIDERS_DEFAULT = tuple(_default_providers())


class TestCircuitFromBlocks:
    def test_default_observables(self):
        layer = Circuit(blocks=[Encoding(in_features=3), Variational(depth=1)])
        out = layer(torch.randn(3, dtype=torch.float64))
        assert out.shape == (3,)

    def test_explicit_observable_scalar(self):
        layer = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1)],
            observables=Z(0).to_observable(),
        )
        out = layer(torch.randn(2, dtype=torch.float64))
        assert out.dim() == 0

    def test_explicit_observable_vector(self):
        layer = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1)],
            observables=all_z(2),
        )
        out = layer(torch.randn(2, dtype=torch.float64))
        assert out.shape == (2,)

    def test_strong_variational(self):
        layer = Circuit(blocks=[Encoding(in_features=2), StrongVariational(depth=2)])
        # 2 layers * 2 qubits * 3 rotations = 12 weights
        assert layer.weights.shape == (12,)

    def test_accepts_pre_built_ir(self):
        ir = assemble([Encoding(in_features=2), Variational(depth=1)])
        layer = Circuit(blocks=ir)
        assert layer.n_qubits == 2

    def test_accepts_circuit_builder_output(self):
        qc = CircuitBuilder(n_qubits=2)
        weights = qc.add_weights(2)
        inputs = qc.add_inputs(2)
        qc.ry(inputs[0], qubit=0).ry(inputs[1], qubit=1)
        qc.rx(weights[0], qubit=0).rx(weights[1], qubit=1)
        qc.cnot(0, 1)
        layer = Circuit(blocks=qc.build())
        assert layer.n_qubits == 2
        assert layer(torch.randn(2, dtype=torch.float64)).shape == (2,)

    def test_accepts_circuit_builder(self):
        qc = CircuitBuilder(n_qubits=2)
        weights = qc.add_weights(2)
        qc.ry(weights[0], qubit=0).ry(weights[1], qubit=1).cnot(0, 1)
        layer = Circuit(blocks=qc)
        assert layer.spec == qc.build()
        assert layer(torch.zeros(0, dtype=torch.float64)).shape == (2,)


class TestCircuitFromBuilderOnNumpy:
    """A `CircuitBuilder` is accepted wherever the numpy `Circuit` takes its
    blocks: as the whole circuit, or as one entry of a block list."""

    def test_builder_is_built_on_the_spot(self):
        import numpy as np

        qc = CircuitBuilder(n_qubits=2)
        qc.h(0).cnot(0, 1)
        circuit = _Circuit(qc, observables=Z(0) * Z(1))
        assert circuit.spec == qc.build()
        assert np.isclose(circuit([]), 1.0)

    def test_later_gates_do_not_reach_the_circuit(self):
        qc = CircuitBuilder(n_qubits=2)
        qc.h(0)
        circuit = _Circuit(qc)
        qc.cnot(0, 1)
        assert len(circuit.spec.gates) == 1

    def test_builder_in_a_block_list(self):
        import numpy as np

        qc = CircuitBuilder(n_qubits=2)
        inputs = qc.add_inputs(2)
        qc.ry(2 * inputs[0], qubit=0).ry(2 * inputs[1], qubit=1)
        circuit = _Circuit([qc, Variational(depth=1)])
        assert circuit.spec == assemble([qc.to_block(), Variational(depth=1)])
        weights = np.zeros(circuit.spec.n_weights)
        assert circuit(weights, np.array([0.1, 0.2])).shape == (2,)


@pytest.fixture
def with_provider_spec_init_x():
    """Shared IR, weights, and classical input for cross-provider checks."""
    spec = assemble([Encoding(in_features=2), Variational(depth=1)])
    torch.manual_seed(42)
    init = torch.randn(spec.n_weights, dtype=torch.float64) * 0.3
    x = torch.tensor([0.15, -0.25], dtype=torch.float64)
    return spec, init, x


class TestWithProvider:
    _ALL_PROVIDER_PAIRS = list(itertools.product(_PROVIDERS_DEFAULT, _PROVIDERS_DEFAULT))

    @pytest.mark.parametrize(
        "p_src,p_dst",
        _ALL_PROVIDER_PAIRS,
        ids=[f"{a}->{b}" for a, b in _ALL_PROVIDER_PAIRS],
    )
    def test_with_provider_all_pairs_forward_parity(
        self, with_provider_spec_init_x, p_src, p_dst
    ):
        """Native ``p_src`` output matches same spec/weights compiled on ``p_dst``.

        Covers every ordered pair, so ``pennylane->qiskit`` and ``qiskit->pennylane``
        (and Braket in both roles) are all exercised.
        """
        spec, init, x = with_provider_spec_init_x
        c_src = Circuit(
            blocks=spec,
            observables=all_z(2),
            provider=p_src,
            init=init.clone(),
        )
        y_src = c_src(x)
        c_dst = c_src.with_provider(p_dst)
        assert c_dst.spec is c_src.spec
        assert c_dst.weights.shape == c_src.weights.shape
        y_dst = c_dst(x)
        assert y_src.shape == y_dst.shape
        assert torch.allclose(y_src, y_dst, atol=2e-5, rtol=5e-4)

    _DISTINCT_PROVIDER_PAIRS = list(itertools.combinations(_PROVIDERS_DEFAULT, 2))

    @pytest.mark.parametrize(
        "p_a,p_b",
        _DISTINCT_PROVIDER_PAIRS,
        ids=[f"{a}<->{b}" for a, b in _DISTINCT_PROVIDER_PAIRS],
    )
    def test_with_provider_round_trip_preserves_forward(
        self, with_provider_spec_init_x, p_a, p_b
    ):
        """``A -> B -> A`` and ``B -> A -> B`` match staying on the original provider."""
        spec, init, x = with_provider_spec_init_x

        c_a = Circuit(blocks=spec, observables=all_z(2), provider=p_a, init=init.clone())
        y_a = c_a(x)
        c_a_b_a = c_a.with_provider(p_b).with_provider(p_a)
        assert torch.allclose(y_a, c_a_b_a(x), atol=2e-5, rtol=5e-4)

        c_b = Circuit(blocks=spec, observables=all_z(2), provider=p_b, init=init.clone())
        y_b = c_b(x)
        c_b_a_b = c_b.with_provider(p_a).with_provider(p_b)
        assert torch.allclose(y_b, c_b_a_b(x), atol=2e-5, rtol=5e-4)

    @pytest.mark.pennylane
    def test_with_provider_config(self):
        layer = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1)],
            provider="pennylane",
        )
        alt = layer.with_provider(PennyLaneConfig(device_name="default.qubit"))
        x = torch.randn(2, dtype=torch.float64)
        assert alt(x).shape == (2,)

    @pytest.mark.skipif(not _HAS_QISKIT, reason="requires qiskit extra")
    def test_copy_weights_false_gives_new_random_shape(self):
        layer = Circuit(blocks=[Encoding(in_features=2), Variational(depth=1)])
        w0 = layer.weights.data.clone()
        other = layer.with_provider("qiskit", copy_weights=False)
        assert other.weights.shape == w0.shape
        assert not torch.allclose(other.weights, w0)

    @pytest.mark.skipif(not _HAS_QISKIT, reason="requires qiskit extra")
    def test_explicit_init_overrides_copy_weights(self):
        layer = Circuit(blocks=[Encoding(in_features=2), Variational(depth=1)])
        custom = torch.zeros(layer.spec.n_weights, dtype=torch.float64)
        other = layer.with_provider("qiskit", copy_weights=True, init=custom)
        assert torch.allclose(other.weights, custom)


class TestPolymorphicProvider:
    @pytest.mark.pennylane
    def test_string(self):
        layer = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1)], provider="pennylane"
        )
        assert layer(torch.randn(2, dtype=torch.float64)).shape == (2,)

    @pytest.mark.skipif(not _HAS_QISKIT, reason="requires qiskit extra")
    def test_qiskit(self):
        layer = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1)], provider="qiskit"
        )
        assert layer(torch.randn(2, dtype=torch.float64)).shape == (2,)

    @pytest.mark.pennylane
    def test_pennylane_config(self):
        layer = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1)],
            provider=PennyLaneConfig(device_name="lightning.qubit"),
        )
        assert layer(torch.randn(2, dtype=torch.float64)).shape == (2,)

    @pytest.mark.skipif(not _HAS_QISKIT, reason="requires qiskit extra")
    def test_qiskit_config(self):
        layer = Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1)],
            provider=QiskitConfig(),
        )
        assert layer(torch.randn(2, dtype=torch.float64)).shape == (2,)


class TestTraining:
    def test_optimizer_step_decreases_loss(self):
        layer = Circuit(blocks=[Encoding(in_features=2), Variational(depth=2)])
        opt = torch.optim.SGD(layer.parameters(), lr=0.1)
        target = torch.tensor([1.0, -1.0], dtype=torch.float64)
        x = torch.tensor([0.3, -0.2], dtype=torch.float64)
        initial = ((layer(x) - target) ** 2).mean().item()
        for _ in range(5):
            opt.zero_grad()
            ((layer(x) - target) ** 2).mean().backward()
            opt.step()
        final = ((layer(x) - target) ** 2).mean().item()
        assert final < initial


class TestTorchBackwardOnNumpyProvider:
    """torch-HQNN training on the numpy provider (no PennyLane), powered by the
    batched adjoint jacobian in the backward pass."""

    @pytest.mark.pennylane
    def test_batched_grad_matches_pennylane(self):
        import numpy as np

        blocks = [Encoding(in_features=3, rotation="RX"), StrongVariational(depth=2)]
        x = torch.tensor(
            np.random.RandomState(0).uniform(0, np.pi, (8, 3)), dtype=torch.float64
        )
        numpy_layer = _Circuit(blocks, all_z(3), provider="numpy", interface="torch")
        init = numpy_layer.weights.detach().clone()
        pennylane_layer = _Circuit(
            blocks, all_z(3), provider="pennylane", interface="torch", init=init
        )

        xn = x.clone().requires_grad_(True)
        numpy_layer(xn).pow(2).sum().backward()
        xp = x.clone().requires_grad_(True)
        pennylane_layer(xp).pow(2).sum().backward()

        assert torch.allclose(
            numpy_layer.weights.grad, pennylane_layer.weights.grad, atol=1e-6
        )
        assert torch.allclose(xn.grad, xp.grad, atol=1e-6)

    def test_training_decreases_loss_on_numpy(self):
        import numpy as np

        layer = _Circuit(
            blocks=[Encoding(in_features=2), StrongVariational(depth=2)],
            observables=all_z(2),
            provider="numpy",
            interface="torch",
        )
        opt = torch.optim.Adam(layer.parameters(), lr=0.1)
        x = torch.tensor(
            np.random.RandomState(0).uniform(0, np.pi, (16, 2)), dtype=torch.float64
        )
        target = torch.zeros(16, 2, dtype=torch.float64)
        initial = (layer(x) - target).pow(2).mean().item()
        for _ in range(10):
            opt.zero_grad()
            (layer(x) - target).pow(2).mean().backward()
            opt.step()
        assert (layer(x) - target).pow(2).mean().item() < initial
