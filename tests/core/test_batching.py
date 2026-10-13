"""Batched-execution coverage across backends.

Goal: catch regressions in any backend's native broadcast path (PennyLane
qnode broadcasting, EstimatorV2 PUB parameter arrays, cudaq.observe
argument broadcasting) and
in the autograd Function's batched backward.

Each batched call is checked against the per-sample loop on the same
backend. Unavailable optional backends are skipped.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


# Skip rather than fail collection on a core install: the library's core is
# torch-free by design (see tests/core/test_torch_free_import.py), so a torch
# import at module scope would make this file uncollectable exactly there.
torch = pytest.importorskip("torch")

from eigenkit.core.blocks import (  # noqa: E402
    AngleEncoding,
    BasicVariational,
    assemble,
)
from eigenkit.core.circuit import Circuit  # noqa: E402
from eigenkit.core.observables import Z, all_z  # noqa: E402


_HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
_HAS_BRAKET = importlib.util.find_spec("braket") is not None
_HAS_CUDAQ = importlib.util.find_spec("cudaq") is not None
_HAS_QISKIT_IBM_RUNTIME = importlib.util.find_spec("qiskit_ibm_runtime") is not None
# FakeBrisbane runs on Aer; without it (Linux ARM with Python 3.14) the fallback
# BasicSimulator cannot hold the device's 127 qubits.
_HAS_QISKIT_AER = importlib.util.find_spec("qiskit_aer") is not None


def _provider_specs():
    """One spec per available backend, keyed by a short id used in test
    parametrize ids. Each value is a callable returning a fresh provider
    arg (registered name, Config, or backend instance) on each call —
    qiskit-runtime in particular wants a fresh FakeBackend per circuit
    to avoid state bleed between tests.
    """
    # numpy first: it is the default provider and the only one declaring
    # supports_batched_jacobian, so its batching is what every user gets by
    # default.
    specs: dict[str, callable] = {
        "numpy": lambda: "numpy",
        "pennylane": lambda: "pennylane",
    }
    if _HAS_QISKIT:
        specs["qiskit"] = lambda: "qiskit"
    if _HAS_BRAKET:
        specs["braket"] = lambda: "braket"
    if _HAS_CUDAQ:
        from eigenkit import CudaQuantumConfig

        specs["cudaq"] = lambda: CudaQuantumConfig(target="qpp-cpu")
    if _HAS_QISKIT and _HAS_QISKIT_IBM_RUNTIME and _HAS_QISKIT_AER:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane

        from eigenkit import QiskitRuntimeConfig

        # FakeBrisbane is a noisy simulator: shots-based estimates have
        # sampling error, so batched parity comparisons against it use a
        # generous tolerance below. We still verify shape + finite output.
        specs["qiskit-runtime"] = lambda: QiskitRuntimeConfig(
            backend_instance=FakeBrisbane(), shots=4096
        )
    return specs


_PROVIDER_SPECS = _provider_specs()
_PROVIDER_IDS = list(_PROVIDER_SPECS)


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
def inputs_batch():
    # Heterogeneous rows — uniform values would mask a wrong broadcasting
    # axis (transposed vs not), since every row would compute identically.
    return np.array(
        [
            [0.5, -0.2, 0.8],
            [-0.7, 0.1, 0.4],
            [0.3, 0.9, -0.6],
            [0.0, 0.4, 0.2],
        ],
        dtype=np.float64,
    )


def _tolerance(provider_id: str) -> float:
    """Loosen tolerance on shots-based backends; the others are analytic."""
    if provider_id == "qiskit-runtime":
        return 0.1  # FakeBrisbane is a noisy simulator with sampling error
    return 1e-9


# ---- forward parity ----


@pytest.mark.parametrize("provider_id", _PROVIDER_IDS)
class TestNumpyForwardParity:
    """Native batched output must match per-sample loop on every backend."""

    def test_shape(self, ir, weights, inputs_batch, provider_id):
        provider = _PROVIDER_SPECS[provider_id]()
        circuit = Circuit(ir, all_z(3), provider, "numpy")
        output = circuit(weights, inputs_batch)
        assert output.shape == (len(inputs_batch), 3)
        assert np.all(np.isfinite(output))

    def test_matches_per_sample(self, ir, weights, inputs_batch, provider_id):
        provider = _PROVIDER_SPECS[provider_id]()
        circuit = Circuit(ir, all_z(3), provider, "numpy")
        batched = circuit(weights, inputs_batch)
        per_sample = np.stack(
            [circuit(weights, inputs_batch[i]) for i in range(len(inputs_batch))]
        )
        assert np.allclose(batched, per_sample, atol=_tolerance(provider_id))

    def test_scalar_observable_batched_shape(
        self, ir, weights, inputs_batch, provider_id
    ):
        """Single observable returns `(B,)`, not `(B, 1)`."""
        provider = _PROVIDER_SPECS[provider_id]()
        circuit = Circuit(ir, Z(0).to_observable(), provider, "numpy")
        output = circuit(weights, inputs_batch)
        assert output.shape == (len(inputs_batch),)


@pytest.mark.parametrize("provider_id", _PROVIDER_IDS)
class TestTorchForwardParity:
    """Torch interface: batched parity + autograd graph survives."""

    def test_shape(self, ir, weights, inputs_batch, provider_id):
        provider = _PROVIDER_SPECS[provider_id]()
        circuit = Circuit(ir, all_z(3), provider, "torch")
        weights_t = torch.tensor(weights, dtype=torch.float64)
        inputs_t = torch.tensor(inputs_batch, dtype=torch.float64)
        output = circuit(weights_t, inputs_t)
        assert output.shape == (len(inputs_batch), 3)

    def test_matches_per_sample(self, ir, weights, inputs_batch, provider_id):
        provider = _PROVIDER_SPECS[provider_id]()
        circuit = Circuit(ir, all_z(3), provider, "torch")
        weights_t = torch.tensor(weights, dtype=torch.float64)
        inputs_t = torch.tensor(inputs_batch, dtype=torch.float64)
        batched = circuit(weights_t, inputs_t)
        per_sample = torch.stack(
            [circuit(weights_t, inputs_t[i]) for i in range(len(inputs_t))]
        )
        assert torch.allclose(batched, per_sample, atol=_tolerance(provider_id))


# ---- gradient correctness through batched inputs ----


@pytest.mark.parametrize("provider_id", _PROVIDER_IDS)
class TestBatchedGradient:
    """Backward through a 2-D `inputs` batch must produce per-sample input
    grads and a weight grad summed over the batch. Numerical correctness
    is checked against finite differences and against the per-sample loop.
    """

    def test_weight_grad_sums_across_batch(self, ir, weights, inputs_batch, provider_id):
        """`sum_b loss(x_b)`'s weight gradient must equal the sum of each
        sample's weight gradient. This is the contract every parameter-shift
        backward (and the native autograd path) has to honour."""
        if provider_id == "qiskit-runtime":
            pytest.skip(
                "qiskit-runtime declares gradient_method='none'; "
                ".backward() is not supported"
            )
        provider = _PROVIDER_SPECS[provider_id]()
        observable = Z(0).to_observable()
        inputs_t = torch.tensor(inputs_batch, dtype=torch.float64)

        circuit = Circuit(ir, observable, provider, "torch")

        # Batched: one .backward() over the summed batch output
        w_batched = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
        circuit(w_batched, inputs_t).sum().backward()

        # Per-sample: stack independent backward passes, sum the grads
        w_per_sample = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
        outputs = [circuit(w_per_sample, inputs_t[i]) for i in range(len(inputs_t))]
        torch.stack(outputs).sum().backward()

        atol = _tolerance(provider_id)
        assert torch.allclose(w_batched.grad, w_per_sample.grad, atol=atol)

    def test_input_grad_per_sample(self, ir, weights, inputs_batch, provider_id):
        """Each row of the input gradient must equal that sample's
        independently-computed input gradient. This catches a bug where
        backward might accidentally sum input grads across the batch."""
        if provider_id == "qiskit-runtime":
            pytest.skip("qiskit-runtime: gradient_method='none'")
        provider = _PROVIDER_SPECS[provider_id]()
        observable = Z(0).to_observable()
        circuit = Circuit(ir, observable, provider, "torch")
        weights_t = torch.tensor(weights, dtype=torch.float64)

        # Batched: input gradient should be (B, n_inputs)
        x_batched = torch.tensor(inputs_batch, dtype=torch.float64, requires_grad=True)
        circuit(weights_t, x_batched).sum().backward()
        assert x_batched.grad is not None
        assert x_batched.grad.shape == x_batched.shape

        # Per-sample: each row of the batched grad must match the
        # gradient from a fresh single-sample backward
        for i in range(len(inputs_batch)):
            x_i = torch.tensor(inputs_batch[i], dtype=torch.float64, requires_grad=True)
            circuit(weights_t, x_i).backward()
            atol = _tolerance(provider_id)
            assert torch.allclose(x_batched.grad[i], x_i.grad, atol=atol), (
                f"row {i} input grad mismatch on {provider_id}: "
                f"batched={x_batched.grad[i]} per-sample={x_i.grad}"
            )

    def test_input_grad_matches_finite_difference(
        self, ir, weights, inputs_batch, provider_id
    ):
        """End-to-end correctness: batched-autograd input grad must match
        a central finite difference on each sample. Catches sign / axis
        bugs that a parity check against per-sample autograd can't
        (because both sides would have the same bug)."""
        if provider_id == "qiskit-runtime":
            pytest.skip("qiskit-runtime: gradient_method='none'")
        provider = _PROVIDER_SPECS[provider_id]()
        observable = Z(0).to_observable()
        circuit = Circuit(ir, observable, provider, "torch")
        weights_t = torch.tensor(weights, dtype=torch.float64)

        x = torch.tensor(inputs_batch, dtype=torch.float64, requires_grad=True)
        circuit(weights_t, x).sum().backward()
        analytic = x.grad.clone()

        eps = 1e-5
        fd = torch.empty_like(x)
        x0 = x.detach().clone()
        for b in range(x0.shape[0]):
            for j in range(x0.shape[1]):
                xp = x0.clone()
                xm = x0.clone()
                xp[b, j] += eps
                xm[b, j] -= eps
                fp = circuit(weights_t, xp).sum().item()
                fm = circuit(weights_t, xm).sum().item()
                fd[b, j] = (fp - fm) / (2 * eps)
        # Finite difference loses ~5 digits; analytic should still match
        # within 1e-5 on noiseless backends.
        atol = max(1e-5, _tolerance(provider_id))
        assert torch.allclose(analytic, fd, atol=atol), (
            f"{provider_id}: analytic input grad disagrees with finite "
            f"difference. analytic={analytic} fd={fd}"
        )


# ---- edge cases ----


@pytest.mark.parametrize("provider_id", _PROVIDER_IDS)
class TestBatchedEdgeCases:
    def test_batch_of_one_still_returns_2d(self, ir, weights, provider_id):
        """`(1, n_inputs)` is a legitimate batch and must return `(1, n_obs)`,
        not get squeezed to a 1-D row. Some native broadcast paths can
        accidentally collapse the singleton."""
        provider = _PROVIDER_SPECS[provider_id]()
        circuit = Circuit(ir, all_z(3), provider, "numpy")
        single_row = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
        output = circuit(weights, single_row)
        assert output.shape == (1, 3)

    def test_rejects_3d_input(self, ir, weights, provider_id):
        provider = _PROVIDER_SPECS[provider_id]()
        circuit = Circuit(ir, all_z(3), provider, "numpy")
        with pytest.raises(ValueError, match="1D.*2D"):
            circuit(weights, np.zeros((2, 3, 3), dtype=np.float64))
