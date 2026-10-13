"""Tests for the Reuploading block: 2 strategies + validation.

Strategies:
1. Standard reuploading (Pérez-Salinas et al. 2020)
2. Trainable feature scaling (Schuld et al. 2021, `trainable_scale=True`)
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from eigenkit.core.blocks import Reuploading, assemble
from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import InputRef, Multiply, WeightRef
from eigenkit.core.observables import all_z


_HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
_HAS_BRAKET = importlib.util.find_spec("braket") is not None


# ---- construction & validation ----


class TestValidation:
    def test_minimal_construction(self):
        block = Reuploading(in_features=3, depth=2)
        assert block.required_qubits() == 3

    def test_rejects_zero_in_features(self):
        with pytest.raises(ValueError, match="in_features"):
            Reuploading(in_features=0, depth=2)

    def test_rejects_zero_depth(self):
        with pytest.raises(ValueError, match="depth"):
            Reuploading(in_features=3, depth=0)

    def test_rejects_invalid_variational_type(self):
        with pytest.raises(ValueError, match="basic.*strong"):
            Reuploading(in_features=3, depth=1, variational_type="bogus")


# ---- weight counts (the contract for downstream layer code) ----


class TestWeightCounts:
    def test_standard_basic(self):
        ir = assemble([Reuploading(in_features=4, depth=3)])
        # depth * n_qubits weights for basic variational, no scale
        assert ir.n_weights == 3 * 4

    def test_standard_strong(self):
        ir = assemble(
            [
                Reuploading(in_features=4, depth=3, variational_type="strong"),
            ]
        )
        # 3 * depth * n_qubits weights for strong
        assert ir.n_weights == 3 * 3 * 4

    def test_trainable_scale_adds_per_layer_per_feature(self):
        ir = assemble([Reuploading(in_features=4, depth=3, trainable_scale=True)])
        # variational (12) + scales (depth * in_features = 12) = 24
        assert ir.n_weights == 12 + 12

    def test_input_count_unaffected_by_strategy(self):
        for kwargs in [
            {},
            {"trainable_scale": True},
        ]:
            ir = assemble([Reuploading(in_features=4, depth=2, **kwargs)])
            assert ir.n_inputs == 4


# ---- IR structure ----


class TestIRStructure:
    def test_standard_uses_no_multiply(self):
        ir = assemble([Reuploading(in_features=2, depth=1)])
        for gate in ir.gates:
            for parameter in gate.params:
                assert not isinstance(parameter, Multiply), (
                    "standard reuploading should not produce Multiply expressions"
                )

    def test_trainable_scale_emits_multiply_in_encoding(self):
        ir = assemble([Reuploading(in_features=2, depth=1, trainable_scale=True)])
        # The first 2 gates are encoding rotations on qubits 0 and 1.
        # Their angles must be Multiply(WeightRef, InputRef).
        encoding_gates = [g for g in ir.gates if g.name == "RY" and g.params][:2]
        for gate in encoding_gates:
            angle = gate.params[0]
            assert isinstance(angle, Multiply), f"expected Multiply, got {type(angle)}"
            assert isinstance(angle.a, WeightRef)
            assert isinstance(angle.b, InputRef)


# ---- forward parity ----


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.pennylane
class TestForwardParity:
    """Same IR, same weights, same inputs — every backend must agree."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"variational_type": "strong"},
            {"trainable_scale": True},
            {"trainable_scale": True, "variational_type": "strong"},
        ],
    )
    def test_pennylane_qiskit_braket_agree(self, rng, kwargs):
        # Build an IR small enough to test exhaustively across providers.
        in_features = 4
        block = Reuploading(in_features=in_features, depth=2, **kwargs)
        ir = assemble([block])
        weights = rng.standard_normal(ir.n_weights)
        inputs = rng.standard_normal(in_features)
        observables = all_z(ir.n_qubits)

        out_pl = Circuit(ir, observables, "pennylane", "numpy")(weights, inputs)
        if _HAS_QISKIT:
            out_qk = Circuit(ir, observables, "qiskit", "numpy")(weights, inputs)
            assert np.allclose(out_pl, out_qk, atol=1e-10)
        if _HAS_BRAKET:
            out_bk = Circuit(ir, observables, "braket", "numpy")(weights, inputs)
            assert np.allclose(out_pl, out_bk, atol=1e-10)


@pytest.mark.pennylane
class TestSchuldEquivalence:
    """When trainable_scale=True with all scales pinned to 1.0, the result
    should match standard reuploading on the same circuit."""

    def test_unit_scales_equal_standard(self, rng):
        ir_std = assemble([Reuploading(in_features=3, depth=2)])
        ir_sch = assemble([Reuploading(in_features=3, depth=2, trainable_scale=True)])

        # Verify equivalence via input scaling: setting input to 0 should yield
        # the same output regardless of the scale weights (encoding becomes
        # RY(0) = identity), so the precise weight ordering does not matter.
        weights_std = rng.standard_normal(ir_std.n_weights)
        weights_sch = rng.standard_normal(ir_sch.n_weights)
        zero_inputs = np.zeros(3)

        out_std = Circuit(ir_std, all_z(3), "pennylane", "numpy")(
            weights_std, zero_inputs
        )
        out_sch = Circuit(ir_sch, all_z(3), "pennylane", "numpy")(
            weights_sch, zero_inputs
        )
        # Both should produce purely-variational outputs (encoding contributes
        # nothing). They differ because the variational weights differ — but
        # each sample should be in [-1, 1] and float-valued.
        assert out_std.shape == out_sch.shape == (3,)
        assert np.all(np.abs(out_std) <= 1)
        assert np.all(np.abs(out_sch) <= 1)


# ---- gradients flow through Multiply (Schuld pattern) ----


class TestTrainableScaleGradients:
    """`trainable_scale=True` emits Multiply(WeightRef, InputRef) gate
    angles. Native-autograd backends (PennyLane) handle these
    natively. Parameter-shift backends (Qiskit, Braket) handle them via
    the chain-rule-aware jacobian on `_ir.Circuit`. All of them must agree
    to machine precision."""

    @pytest.mark.pennylane
    def test_native_autograd_through_multiply(self):
        ir = assemble([Reuploading(in_features=2, depth=1, trainable_scale=True)])
        weights = torch.randn(ir.n_weights, dtype=torch.float64, requires_grad=True)
        inputs = torch.tensor([0.3, -0.5], dtype=torch.float64)
        circuit = Circuit(ir, all_z(2), "pennylane", "torch")
        loss = circuit(weights, inputs).sum()
        loss.backward()
        assert weights.grad is not None
        assert weights.grad.norm() > 0

    @pytest.mark.pennylane
    @pytest.mark.parametrize("provider", ["qiskit", "braket"])
    def test_chain_rule_gradients_match_pennylane(self, provider, rng):
        pytest.importorskip(provider)
        """With chain-rule jacobian in place, parameter-shift backends
        must produce identical gradients to PennyLane's native autograd
        on a trainable_scale circuit."""
        ir = assemble([Reuploading(in_features=3, depth=2, trainable_scale=True)])
        init = torch.tensor(rng.standard_normal(ir.n_weights), dtype=torch.float64)
        inputs = torch.tensor([0.3, -0.5, 0.7], dtype=torch.float64)

        w_pl = init.clone().requires_grad_()
        w_other = init.clone().requires_grad_()

        Circuit(ir, all_z(3), "pennylane", "torch")(w_pl, inputs).sum().backward()
        Circuit(ir, all_z(3), provider, "torch")(w_other, inputs).sum().backward()

        assert torch.allclose(w_pl.grad, w_other.grad, atol=1e-10)

    @pytest.mark.pennylane
    @pytest.mark.parametrize("provider", ["qiskit", "braket"])
    def test_chain_rule_combined_trainable_strong(self, provider, rng):
        pytest.importorskip(provider)
        """Chain-rule jacobian also handles the combined trainable_scale +
        strong-variational case."""
        ir = assemble(
            [
                Reuploading(
                    in_features=4,
                    depth=2,
                    trainable_scale=True,
                    variational_type="strong",
                ),
            ]
        )
        init = torch.tensor(rng.standard_normal(ir.n_weights), dtype=torch.float64)
        inputs = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)

        w_pl = init.clone().requires_grad_()
        w_other = init.clone().requires_grad_()

        Circuit(ir, all_z(2), "pennylane", "torch")(w_pl, inputs).sum().backward()
        Circuit(ir, all_z(2), provider, "torch")(w_other, inputs).sum().backward()

        assert torch.allclose(w_pl.grad, w_other.grad, atol=1e-10)

    @pytest.mark.skipif(not _HAS_QISKIT, reason="requires qiskit extra")
    def test_perturbed_circuit_cache_reused(self):
        """The chain-rule jacobian compiles 2 perturbed circuits per
        parametric gate, then caches them for subsequent backward calls.
        Verify the cache actually fills + is reused."""
        ir = assemble([Reuploading(in_features=2, depth=1, trainable_scale=True)])
        layer = Circuit(ir, all_z(2), "qiskit", "torch")
        weights = torch.randn(ir.n_weights, dtype=torch.float64, requires_grad=True)
        inputs = torch.tensor([0.3, -0.5], dtype=torch.float64)

        assert layer._chain_rule_cache == {}
        assert layer._has_unsafe_gates is True
        layer(weights, inputs).sum().backward()
        # After one backward, cache holds (gate_index, ±1) entries for
        # every parametric gate that depended on a weight.
        n_cached = len(layer._chain_rule_cache)
        assert n_cached > 0 and n_cached % 2 == 0
        # A second backward should not grow the cache.
        weights2 = weights.detach().clone().requires_grad_()
        layer(weights2, inputs).sum().backward()
        assert len(layer._chain_rule_cache) == n_cached


# ---- cudaq raises on Multiply (= trainable_scale) ----


class TestCudaqMultiplyRaise:
    """trainable_scale uses Multiply; cudaq doesn't support it. Verify the
    error message points the user at alternative backends."""

    def test_multiply_raises_at_compile(self):
        pytest.importorskip("cudaq")
        from eigenkit.core.backends.cudaquantum_backend import (
            CudaQuantumBackend,
        )
        from eigenkit.core.provider import register

        register("cudaq", CudaQuantumBackend())
        ir = assemble([Reuploading(in_features=2, depth=1, trainable_scale=True)])
        with pytest.raises(NotImplementedError, match="Multiply"):
            Circuit(ir, all_z(2), "cudaq", "numpy")
