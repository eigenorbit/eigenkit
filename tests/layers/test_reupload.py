"""Tests for the Reupload layer (standard data reuploading)."""

from __future__ import annotations

import pytest
import torch

from eigenkit import PennyLaneConfig, Reupload


class TestConstruction:
    def test_minimal(self):
        layer = Reupload(n_qubits=4, depth=2)
        assert layer.n_qubits == 4
        assert layer.in_features == 4
        assert layer.out_features == 4

    def test_basic_variational_weight_count(self):
        # depth * n_qubits variational weights (no trainable_scale).
        layer = Reupload(n_qubits=4, depth=3, variational_type="basic")
        assert layer.circuit.weights.shape == (3 * 4,)

    def test_strong_variational_weight_count(self):
        # 3 * depth * n_qubits variational weights.
        layer = Reupload(n_qubits=4, depth=2, variational_type="strong")
        assert layer.circuit.weights.shape == (3 * 2 * 4,)

    def test_trainable_scale_adds_weights(self):
        """Schuld pattern adds depth * n_qubits learnable scales."""
        no_scale = Reupload(n_qubits=4, depth=3, trainable_scale=False)
        scaled = Reupload(n_qubits=4, depth=3, trainable_scale=True)
        assert scaled.circuit.weights.numel() == no_scale.circuit.weights.numel() + 3 * 4

    def test_rejects_invalid_basis(self):
        with pytest.raises(ValueError, match="basis"):
            Reupload(n_qubits=4, depth=1, basis="W")  # type: ignore[arg-type]


class TestForward:
    def test_shape(self):
        layer = Reupload(n_qubits=4, depth=3)
        out = layer(torch.randn(8, 4, dtype=torch.float64))
        assert out.shape == (8, 4)

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_basis_runs(self, basis):
        # enc_rotation="RX" rather than the "RY" default: with RY encoding and
        # RY rotations every gate is real, so the statevector is real and <Y>
        # is identically zero, and the shape assertion below would hold
        # vacuously.
        layer = Reupload(n_qubits=2, depth=2, basis=basis, enc_rotation="RX")
        out = layer(torch.randn(3, 2, dtype=torch.float64))
        assert out.shape == (3, 2)
        assert out.abs().max() > 0

    def test_preserves_float32_dtype(self):
        layer = Reupload(n_qubits=2, depth=1)
        out = layer(torch.randn(2, 2, dtype=torch.float32))
        assert out.dtype == torch.float32

    def test_preserves_float64_dtype(self):
        layer = Reupload(n_qubits=2, depth=1)
        out = layer(torch.randn(2, 2, dtype=torch.float64))
        assert out.dtype == torch.float64

    def test_composes_with_float32_linear(self):
        model = torch.nn.Sequential(
            Reupload(n_qubits=2, depth=1),
            torch.nn.Linear(2, 1),
        )
        out = model(torch.randn(2, 2, dtype=torch.float32))
        assert out.shape == (2, 1)
        assert out.dtype == torch.float32


class TestGradients:
    def test_gradient_flows(self):
        layer = Reupload(n_qubits=4, depth=2)
        out = layer(torch.randn(4, 4, dtype=torch.float64))
        out.pow(2).sum().backward()
        assert layer.circuit.weights.grad is not None
        assert layer.circuit.weights.grad.norm() > 0

    @pytest.mark.parametrize("provider", ["pennylane", "qiskit"])
    def test_trainable_scale_gradient(self, provider):
        """Trainable-scale weights should receive non-zero gradients on
        both the native-autograd and parameter-shift backends."""
        if provider == "qiskit":
            pytest.importorskip("qiskit")
        layer = Reupload(
            n_qubits=2,
            depth=2,
            trainable_scale=True,
            provider=provider,
        )
        out = layer(torch.randn(3, 2, dtype=torch.float64))
        out.pow(2).sum().backward()
        assert layer.circuit.weights.grad is not None
        assert layer.circuit.weights.grad.norm() > 0


class TestProvider:
    def test_string(self):
        pytest.importorskip("qiskit")
        layer = Reupload(n_qubits=2, depth=1, provider="qiskit")
        assert layer(torch.randn(2, 2, dtype=torch.float64)).shape == (2, 2)

    @pytest.mark.pennylane
    def test_config(self):
        layer = Reupload(
            n_qubits=2,
            depth=1,
            provider=PennyLaneConfig(device_name="default.qubit"),
        )
        assert layer(torch.randn(2, 2, dtype=torch.float64)).shape == (2, 2)
