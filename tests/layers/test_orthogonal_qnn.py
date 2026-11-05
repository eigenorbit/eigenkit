"""Tests for the OrthogonalQNN layer (unary loader + RBS network + unary readout)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eigenkit import OrthogonalQNN, PennyLaneConfig
from eigenkit.core.backends.numpy_backend import NumpySimulator
from eigenkit.orthogonal import Topology, orthogonal_matrix, pyramid
from eigenkit.orthogonal.torch import unary_loader_angles


def _expected(layer, x):
    W = orthogonal_matrix(layer.topology, layer.circuit.weights.detach().numpy())
    x = x.detach().double().numpy()
    return (x @ W.T) ** 2 / np.sum(x**2, axis=-1, keepdims=True)


class TestConstruction:
    def test_minimal(self):
        layer = OrthogonalQNN(n_qubits=4)
        assert layer.n_qubits == layer.in_features == layer.out_features == 4
        assert layer.topology.name == "pyramid"
        assert layer.circuit.weights.shape == (6,)

    @pytest.mark.parametrize(
        ("topology", "n_angles"), [("butterfly", 4), ("x", 5), ("diagonal", 3)]
    )
    def test_the_weights_are_the_angles_of_the_topology(self, topology, n_angles):
        layer = OrthogonalQNN(n_qubits=4, topology=topology)
        assert layer.circuit.weights.shape == (n_angles,)

    def test_a_topology_object_must_match_the_width(self):
        layer = OrthogonalQNN(n_qubits=3, topology=pyramid(3, reflection=True))
        assert layer.topology.reflection
        with pytest.raises(ValueError, match="wires"):
            OrthogonalQNN(n_qubits=4, topology=pyramid(3))

    def test_the_x_layout_refuses_an_odd_width(self):
        with pytest.raises(ValueError, match="even width"):
            OrthogonalQNN(n_qubits=5, topology="x")

    def test_a_loader_that_does_not_fit_the_width_is_rejected(self):
        with pytest.raises(ValueError, match="power-of-two"):
            OrthogonalQNN(n_qubits=3, loader="parallel")

    def test_rejects_an_unknown_init(self):
        with pytest.raises(ValueError, match="init"):
            OrthogonalQNN(n_qubits=3, init="zeros")


class TestForward:
    def test_a_batch_gives_a_batch(self):
        out = OrthogonalQNN(n_qubits=4)(torch.randn(8, 4))
        assert out.shape == (8, 4)

    def test_one_sample_gives_one_row(self):
        out = OrthogonalQNN(n_qubits=4)(torch.randn(4))
        assert out.shape == (4,)

    @pytest.mark.parametrize("loader", ["diagonal", "parallel", "semi_diagonal"])
    @pytest.mark.parametrize("topology", ["pyramid", "butterfly", "x"])
    def test_the_output_is_the_squared_rotated_input(self, topology, loader):
        torch.manual_seed(0)
        layer = OrthogonalQNN(n_qubits=4, topology=topology, loader=loader)
        x = torch.randn(5, 4, dtype=torch.float64)
        np.testing.assert_allclose(
            layer(x).detach().numpy(), _expected(layer, x), atol=1e-12
        )

    def test_each_row_is_a_distribution(self):
        out = OrthogonalQNN(n_qubits=5)(torch.randn(6, 5, dtype=torch.float64))
        assert (out >= 0).all()
        torch.testing.assert_close(out.sum(dim=1), torch.ones(6, dtype=torch.float64))

    def test_identity_init_returns_the_squared_normalised_input(self):
        x = torch.tensor([[1.0, -2.0, 2.0]], dtype=torch.float64)
        out = OrthogonalQNN(n_qubits=3, init="identity")(x)
        torch.testing.assert_close(out, x**2 / 9.0)

    def test_a_zero_row_gives_the_uniform_distribution(self):
        # A ReLU in front of the layer can zero a whole row; that row has no
        # direction to load, so it maps to 1/n and the rest of the batch is kept.
        torch.manual_seed(0)
        layer = OrthogonalQNN(n_qubits=3)
        x = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]], dtype=torch.float64)
        out = layer(x)
        torch.testing.assert_close(out[0], torch.full((3,), 1 / 3, dtype=torch.float64))
        torch.testing.assert_close(out[1], layer(x[1:])[0])
        torch.testing.assert_close(layer(torch.zeros(3)), torch.full((3,), 1 / 3))

    def test_a_zero_row_sends_no_gradient_and_no_nan(self):
        torch.manual_seed(0)
        layer = OrthogonalQNN(n_qubits=3)
        x = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]], dtype=torch.float64, requires_grad=True
        )
        layer(x)[:, 0].sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(layer.circuit.weights.grad).all()
        assert torch.count_nonzero(x.grad[0]) == 0
        assert torch.count_nonzero(x.grad[1]) > 0

    def test_trains_after_a_relu_that_zeroes_rows(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.ReLU(), OrthogonalQNN(n_qubits=4)
        )
        x = torch.randn(64, 3)
        assert (torch.relu(model[0](x)) == 0).all(dim=1).any()
        loss = model(x)[:, 0].sum()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())

    def test_composes_in_sequential(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(6, 4),
            OrthogonalQNN(n_qubits=4),
            torch.nn.Linear(4, 2),
        )
        out = model(torch.randn(3, 6))
        assert out.shape == (3, 2)
        assert out.dtype == torch.float32


class TestGradients:
    def test_gradients_reach_the_weights_and_the_input(self):
        layer = OrthogonalQNN(n_qubits=4)
        x = torch.randn(3, 4, requires_grad=True)
        layer(x)[:, 0].sum().backward()
        assert layer.circuit.weights.grad.norm() > 0
        assert x.grad.norm() > 0

    def test_the_loader_angles_are_differentiable(self):
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda t: unary_loader_angles(t, "parallel"), (x,)
        )

    def test_training_moves_the_output_towards_a_target(self):
        torch.manual_seed(1)
        layer = OrthogonalQNN(n_qubits=3)
        x = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 1.0]])
        opt = torch.optim.Adam(layer.parameters(), lr=0.1)
        first = None
        for _ in range(60):
            opt.zero_grad()
            loss = (layer(x) - target).pow(2).sum()
            loss.backward()
            opt.step()
            first = loss.item() if first is None else first
        assert loss.item() < 0.1 * first


class TestProvider:
    def test_string(self):
        pytest.importorskip("qiskit")
        layer = OrthogonalQNN(n_qubits=3, provider="qiskit")
        reference = OrthogonalQNN(n_qubits=3)
        reference.load_state_dict(layer.state_dict())
        x = torch.randn(2, 3, dtype=torch.float64)
        torch.testing.assert_close(layer(x), reference(x))

    @pytest.mark.pennylane
    def test_config(self):
        layer = OrthogonalQNN(n_qubits=3, provider=PennyLaneConfig())
        assert layer(torch.randn(2, 3, dtype=torch.float64)).shape == (2, 3)

    def test_backend_instance(self):
        backend = NumpySimulator()
        layer = OrthogonalQNN(n_qubits=3, provider=backend)
        assert layer.provider is backend
        assert layer(torch.randn(2, 3)).shape == (2, 3)

    def test_a_custom_topology_runs(self):
        t = Topology(3, (((0, 2),), ((0, 1),)))
        layer = OrthogonalQNN(n_qubits=3, topology=t)
        x = torch.randn(4, 3, dtype=torch.float64)
        np.testing.assert_allclose(
            layer(x).detach().numpy(), _expected(layer, x), atol=1e-12
        )


def _closed_form(layer, x):
    from eigenkit.orthogonal.torch.modules import _topology_matrix

    W = _topology_matrix(layer.topology, layer.circuit.weights.detach().double())
    return (x @ W.t()) ** 2 / (x**2).sum(dim=-1, keepdim=True)


class TestInputGradientOnSparseInputs:
    @pytest.mark.parametrize("loader", ["diagonal", "semi_diagonal", "parallel"])
    def test_the_input_gradient_is_exact_where_x_has_zeros(self, loader):
        torch.manual_seed(2)
        layer = OrthogonalQNN(n_qubits=4, loader=loader)
        for v in (
            [1.0, 0, 0, 0],
            [0, 0, 0, 2.0],
            [0.5, -1.0, 0, 0],
            [0.3, 0.2, -1.0, 0.7],
        ):
            x = torch.tensor([v], dtype=torch.float64, requires_grad=True)
            layer(x).pow(2).sum().backward()
            reference = torch.tensor([v], dtype=torch.float64, requires_grad=True)
            _closed_form(layer, reference).pow(2).sum().backward()
            torch.testing.assert_close(x.grad, reference.grad, atol=1e-10, rtol=0)

    def test_the_values_and_the_weight_gradient_still_come_from_the_circuit(self):
        torch.manual_seed(3)
        layer = OrthogonalQNN(n_qubits=4)
        x = torch.randn(3, 4, dtype=torch.float64)
        circuit_only = (1.0 - layer.circuit(unary_loader_angles(x, "diagonal"))) / 2.0
        assert torch.equal(layer(x), circuit_only)
        layer(x)[:, 1].sum().backward()
        from_layer = layer.circuit.weights.grad.clone()
        layer.circuit.weights.grad = None
        circuit_only = (1.0 - layer.circuit(unary_loader_angles(x, "diagonal"))) / 2.0
        circuit_only[:, 1].sum().backward()
        assert torch.equal(from_layer, layer.circuit.weights.grad)


class TestReset:
    # The estimators call reset_parameters on every module after seeding.
    @staticmethod
    def _reset_all(model):
        model.apply(
            lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
        )

    def test_an_identity_start_survives_a_reset(self):
        layer = OrthogonalQNN(4, "pyramid", init="identity")
        torch.manual_seed(0)
        self._reset_all(layer)
        assert torch.count_nonzero(layer.circuit.weights) == 0

    def test_a_direct_reset_draws_new_random_angles(self):
        # As nn.Linear: m.reset_parameters() re-initialises the module itself.
        layer = OrthogonalQNN(4, "pyramid")
        drawn = []
        for seed in (1, 2, 1):
            torch.manual_seed(seed)
            layer.reset_parameters()
            drawn.append(layer.circuit.weights.detach().clone())
        assert torch.equal(drawn[0], drawn[2])
        assert not torch.equal(drawn[0], drawn[1])
        torch.manual_seed(1)
        layer.circuit.reset_parameters()
        assert torch.equal(layer.circuit.weights, drawn[0])

    def test_a_reset_through_apply_keeps_the_second_draw(self):
        # apply() resets the circuit, then the layer: two draws, the second kept.
        layer = OrthogonalQNN(4, "pyramid")
        torch.manual_seed(0)
        self._reset_all(layer)
        through_apply = layer.circuit.weights.detach().clone()
        torch.manual_seed(0)
        layer.circuit.reset_parameters()
        layer.circuit.reset_parameters()
        assert torch.equal(layer.circuit.weights, through_apply)

    def test_an_identity_start_survives_an_estimator_fit(self):
        from eigenkit.torch.estimators.generic import GenericClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(16, 4)).astype(np.float32)
        y = np.arange(16) % 4
        clf = GenericClassifier(
            torch.nn.Sequential(OrthogonalQNN(4, "pyramid", init="identity")),
            n_classes=4,
            lr=0.0,
            epochs=1,
            show_progress=False,
            random_state=0,
        ).fit(X, y)
        assert torch.count_nonzero(clf._model[0].circuit.weights) == 0


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_the_output_and_its_input_gradient_do_not_depend_on_the_scale(scale):
    # (W x)² / |x|² is scale-invariant; squaring these x would overflow or
    # underflow float64.
    torch.manual_seed(0)
    layer = OrthogonalQNN(4, "pyramid")
    v = [0.3, -1.2, 0.7, 2.0]
    x = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    xs = torch.tensor(v, dtype=torch.float64).mul(scale).requires_grad_()
    torch.testing.assert_close(layer(xs), layer(x), rtol=0, atol=1e-12)
    layer(xs)[1].backward()
    layer(x)[1].backward()
    assert torch.isfinite(xs.grad).all()
    torch.testing.assert_close(xs.grad * scale, x.grad, rtol=1e-12, atol=0)


def test_the_pair_indices_are_built_once_per_device(monkeypatch):
    import eigenkit.torch.layers.orthogonal_qnn as qnn_module
    from eigenkit.orthogonal.torch.modules import _topology_matrix

    calls = []
    original = qnn_module._pair_layers
    monkeypatch.setattr(
        qnn_module,
        "_pair_layers",
        lambda topology, device: calls.append(device) or original(topology, device),
    )
    torch.manual_seed(0)
    layer = OrthogonalQNN(4, "pyramid")
    x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    for _ in range(3):
        layer(x).sum().backward()
    assert len(calls) == 1
    # The cached indices give the matrix of the indices built afresh.
    theta = layer.circuit.weights.detach().to(torch.float64)
    cached = layer._layers_by_device[theta.device]
    assert torch.equal(
        _topology_matrix(layer.topology, theta, cached),
        _topology_matrix(layer.topology, theta),
    )
