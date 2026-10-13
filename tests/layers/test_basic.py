"""Tests for the Basic layer (one Encoding + one Variational + measurement)."""

from __future__ import annotations

import pytest
import torch

from eigenkit import Basic, PennyLaneConfig


class TestConstruction:
    def test_minimal(self):
        layer = Basic(n_qubits=4, depth=2)
        assert layer.n_qubits == 4
        assert layer.in_features == 4
        assert layer.out_features == 4

    def test_strong_variational(self):
        layer = Basic(n_qubits=4, depth=2, variational_type="strong")
        # 3 * depth * n_qubits = 3 * 2 * 4 = 24 weights.
        assert layer.circuit.weights.shape == (24,)

    def test_basic_variational_weight_count(self):
        layer = Basic(n_qubits=4, depth=2, variational_type="basic")
        # depth * n_qubits = 2 * 4 = 8 weights.
        assert layer.circuit.weights.shape == (8,)

    def test_rejects_invalid_basis(self):
        with pytest.raises(ValueError, match="basis"):
            Basic(n_qubits=4, depth=1, basis="W")  # type: ignore[arg-type]

    def test_rejects_invalid_variational_type(self):
        with pytest.raises(ValueError, match="variational_type"):
            Basic(n_qubits=4, depth=1, variational_type="weird")  # type: ignore[arg-type]


class TestForward:
    def test_shape(self):
        layer = Basic(n_qubits=4, depth=2)
        out = layer(torch.randn(8, 4, dtype=torch.float64))
        assert out.shape == (8, 4)

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_basis_runs(self, basis):
        layer = Basic(n_qubits=2, depth=1, basis=basis)
        out = layer(torch.randn(3, 2, dtype=torch.float64))
        assert out.shape == (3, 2)

    def test_preserves_float32_dtype(self):
        layer = Basic(n_qubits=2, depth=1)
        out = layer(torch.randn(2, 2, dtype=torch.float32))
        assert out.dtype == torch.float32

    def test_preserves_float64_dtype(self):
        layer = Basic(n_qubits=2, depth=1)
        out = layer(torch.randn(2, 2, dtype=torch.float64))
        assert out.dtype == torch.float64

    def test_composes_with_float32_linear(self):
        model = torch.nn.Sequential(
            Basic(n_qubits=2, depth=1),
            torch.nn.Linear(2, 1),
        )
        out = model(torch.randn(2, 2, dtype=torch.float32))
        assert out.shape == (2, 1)
        assert out.dtype == torch.float32


class TestGradients:
    def test_gradient_flows(self):
        layer = Basic(n_qubits=4, depth=1)
        out = layer(torch.randn(4, 4, dtype=torch.float64))
        out.pow(2).sum().backward()
        assert layer.circuit.weights.grad is not None
        assert layer.circuit.weights.grad.norm() > 0


class TestProvider:
    def test_string(self):
        pytest.importorskip("qiskit")
        layer = Basic(n_qubits=2, depth=1, provider="qiskit")
        assert layer(torch.randn(2, 2, dtype=torch.float64)).shape == (2, 2)

    @pytest.mark.pennylane
    def test_config(self):
        layer = Basic(
            n_qubits=2,
            depth=1,
            provider=PennyLaneConfig(device_name="default.qubit"),
        )
        assert layer(torch.randn(2, 2, dtype=torch.float64)).shape == (2, 2)
