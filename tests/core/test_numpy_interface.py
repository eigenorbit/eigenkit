"""The numpy interface (the default) across providers — forward *and*
gradients, with no torch involved.

`Circuit(..., interface="numpy")` returns the framework-free core circuit:
weights are passed per call, and `circuit.jacobian(weights, inputs)` exposes
gradients via the same provider-native → parameter-shift ladder the variational
spine uses (the numpy sim's analytic adjoint, PennyLane's autodiff, or
parameter-shift on Qiskit/Braket). These tests pin that the numpy interface is
a first-class, differentiable path on every backend — not just torch.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from eigenkit import Circuit, Encoding, StrongVariational, Variational, Z, all_z


def _is_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _providers():
    providers = ["numpy", "pennylane"]
    if _is_available("qiskit"):
        providers.append("qiskit")
    if _is_available("braket"):
        providers.append("braket")
    return providers


_PROVIDERS = _providers()
_BLOCKS = [Encoding(in_features=2, rotation="RX"), Variational(depth=2, rotation="RY")]
_X = np.array([0.3, -0.4], dtype=np.float64)


def _weights(n: int) -> np.ndarray:
    return np.linspace(0.1, 0.5, n)


def _finite_difference(circuit, weights, inputs, eps=1e-6):
    grad = np.zeros_like(weights)
    for i in range(len(weights)):
        wp, wm = weights.copy(), weights.copy()
        wp[i] += eps
        wm[i] -= eps
        grad[i] = (circuit(wp, inputs) - circuit(wm, inputs)) / (2 * eps)
    return grad


class TestForward:
    def test_returns_numpy_not_tensor(self):
        circuit = Circuit(
            _BLOCKS, observables=all_z(2), provider="numpy", interface="numpy"
        )
        out = circuit(_weights(circuit.spec.n_weights), _X)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)

    @pytest.mark.parametrize("provider", _PROVIDERS)
    def test_forward_matches_numpy_reference(self, provider):
        ref = Circuit(_BLOCKS, observables=all_z(2), provider="numpy", interface="numpy")
        w = _weights(ref.spec.n_weights)
        c = Circuit(_BLOCKS, observables=all_z(2), provider=provider, interface="numpy")
        assert np.allclose(c(w, _X), ref(w, _X), atol=1e-9)


class TestJacobian:
    @pytest.mark.parametrize("provider", _PROVIDERS)
    def test_scalar_jacobian_matches_finite_difference(self, provider):
        c = Circuit(
            _BLOCKS,
            observables=Z(0).to_observable(),
            provider=provider,
            interface="numpy",
        )
        w = _weights(c.spec.n_weights)
        jac = c.jacobian(w, _X)
        assert jac.shape == (c.spec.n_weights,)
        assert np.allclose(jac, _finite_difference(c, w, _X), atol=1e-5)

    @pytest.mark.parametrize("provider", _PROVIDERS)
    def test_jacobian_agrees_across_providers(self, provider):
        ref = Circuit(
            _BLOCKS, observables=Z(0).to_observable(), provider="numpy", interface="numpy"
        )
        w = _weights(ref.spec.n_weights)
        c = Circuit(
            _BLOCKS,
            observables=Z(0).to_observable(),
            provider=provider,
            interface="numpy",
        )
        assert np.allclose(c.jacobian(w, _X), ref.jacobian(w, _X), atol=1e-6)

    def test_vector_observable_jacobian_shape(self):
        c = Circuit(_BLOCKS, observables=all_z(2), provider="numpy", interface="numpy")
        jac = c.jacobian(_weights(c.spec.n_weights), _X)
        assert jac.shape == (2, c.spec.n_weights)  # (n_observables, n_weights)

    def test_input_grad_returns_both_jacobians(self):
        c = Circuit(_BLOCKS, observables=all_z(2), provider="numpy", interface="numpy")
        weight_jac, input_jac = c.jacobian(
            _weights(c.spec.n_weights), _X, input_grad=True
        )
        assert weight_jac.shape == (2, c.spec.n_weights)
        assert input_jac.shape == (2, 2)  # (n_observables, n_inputs)

    def test_input_jacobian_matches_finite_difference(self):
        c = Circuit(
            _BLOCKS, observables=Z(0).to_observable(), provider="numpy", interface="numpy"
        )
        w = _weights(c.spec.n_weights)
        _, input_jac = c.jacobian(w, _X, input_grad=True)

        eps = 1e-6
        fd = np.zeros_like(_X)
        for i in range(len(_X)):
            xp, xm = _X.copy(), _X.copy()
            xp[i] += eps
            xm[i] -= eps
            fd[i] = (c(w, xp) - c(w, xm)) / (2 * eps)
        assert np.allclose(input_jac, fd, atol=1e-5)

    def test_chain_rule_weights_match_finite_difference(self):
        # StrongVariational wraps weights so the chain-rule path (not the naive
        # parameter-shift) must engage; verify it still matches numerics.
        blocks = [Encoding(in_features=2), StrongVariational(depth=2)]
        c = Circuit(
            blocks, observables=Z(0).to_observable(), provider="numpy", interface="numpy"
        )
        w = _weights(c.spec.n_weights)
        assert np.allclose(c.jacobian(w, _X), _finite_difference(c, w, _X), atol=1e-5)

    def test_rejects_batched_inputs(self):
        c = Circuit(_BLOCKS, observables=all_z(2), provider="numpy", interface="numpy")
        batch = np.stack([_X, _X])
        with pytest.raises(ValueError, match="single 1-D inputs sample"):
            c.jacobian(_weights(c.spec.n_weights), batch)


def test_jacobian_drives_a_gradient_descent_step():
    """End-to-end: the numpy gradient is usable for optimization, no torch."""
    c = Circuit(
        [Variational(depth=2, rotation="RY")],
        observables=Z(0).to_observable(),
        provider="numpy",
        interface="numpy",
    )
    w = _weights(c.spec.n_weights)
    initial = c(w)
    for _ in range(50):
        w = w - 0.2 * c.jacobian(w)  # minimize <Z_0>
    assert c(w) < initial
