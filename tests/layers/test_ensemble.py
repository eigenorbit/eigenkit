"""Tests for the Ensemble layer."""

from __future__ import annotations

import pytest
import torch

from eigenkit.torch.layers import Ensemble


class TestEnsembleConstruction:
    def test_basic(self):
        layer = Ensemble(in_features=8, n_qubits=4, depth=2)
        assert layer.in_features == 8
        assert layer.n_qubits == 4
        assert layer.q_layers == 2
        assert layer.out_features == 8  # q_layers * n_qubits = 2 * 4

    def test_rejects_non_divisible(self):
        with pytest.raises(ValueError, match="divisible"):
            Ensemble(in_features=7, n_qubits=4, depth=1)

    def test_strong_default(self):
        layer = Ensemble(in_features=4, n_qubits=2, depth=1, variational_type="strong")
        # 2 sub-circuits, each with 3*depth*n_qubits = 6 weights
        for circuit in layer.circuits:
            assert circuit.weights.shape == (6,)


class TestEnsembleForward:
    def test_batched(self):
        layer = Ensemble(in_features=8, n_qubits=4, depth=1)
        x = torch.randn(16, 8, dtype=torch.float64)
        y = layer(x)
        assert y.shape == (16, 8)

    def test_single_sample(self):
        layer = Ensemble(in_features=8, n_qubits=4, depth=1)
        x = torch.randn(8, dtype=torch.float64)
        y = layer(x)
        assert y.shape == (1, 8) or y.shape == (8,)  # tolerant on squeeze

    def test_gradient_flows(self):
        layer = Ensemble(in_features=4, n_qubits=2, depth=1)
        x = torch.randn(2, 4, dtype=torch.float64)
        loss = layer(x).pow(2).sum()
        loss.backward()
        # All sub-circuits have grad
        for circuit in layer.circuits:
            assert circuit.weights.grad is not None
            assert circuit.weights.grad.norm() > 0

    def test_preserves_float32_dtype(self):
        layer = Ensemble(in_features=4, n_qubits=2, depth=1)
        out = layer(torch.randn(2, 4, dtype=torch.float32))
        assert out.dtype == torch.float32

    def test_preserves_float64_dtype(self):
        layer = Ensemble(in_features=4, n_qubits=2, depth=1)
        out = layer(torch.randn(2, 4, dtype=torch.float64))
        assert out.dtype == torch.float64

    def test_composes_with_float32_linear(self):
        model = torch.nn.Sequential(
            Ensemble(in_features=4, n_qubits=2, depth=1),
            torch.nn.Linear(4, 1),
        )
        out = model(torch.randn(2, 4, dtype=torch.float32))
        assert out.shape == (2, 1)
        assert out.dtype == torch.float32
