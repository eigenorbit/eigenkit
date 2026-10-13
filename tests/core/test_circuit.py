"""High-level Circuit dispatch: interface normalization, observable
handling, scalar vs vector output."""

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
from eigenkit.core.ir import Observable  # noqa: E402
from eigenkit.core.observables import X, Z, all_z  # noqa: E402


_HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
_needs_qiskit = pytest.mark.skipif(not _HAS_QISKIT, reason="requires qiskit extra")


@pytest.fixture
def ir():
    return assemble(
        [
            AngleEncoding(in_features=2),
            BasicVariational(depth=1, rotation="RY"),
        ]
    )


@pytest.fixture
def weights():
    return np.array([0.3, 0.7], dtype=np.float64)


@pytest.fixture
def inputs():
    return np.array([0.1, 0.2], dtype=np.float64)


class TestScalarVectorOutput:
    def test_single_observable_numpy_scalar(self, ir, weights, inputs):
        circuit = Circuit(ir, Z(0).to_observable(), "numpy", "numpy")
        result = circuit(weights, inputs)
        assert isinstance(result, float)

    def test_list_of_one_observable_vector(self, ir, weights, inputs):
        circuit = Circuit(ir, [Z(0).to_observable()], "numpy", "numpy")
        result = circuit(weights, inputs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_list_multiple_observables(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(2), "numpy", "numpy")
        result = circuit(weights, inputs)
        assert result.shape == (2,)

    def test_none_uses_all_z(self, ir, weights, inputs):
        circuit = Circuit(ir, None, "numpy", "numpy")
        result = circuit(weights, inputs)
        # ir.n_qubits == 2 → 2 Z observables
        assert result.shape == (2,)

    def test_pauli_expr_accepted(self, ir, weights, inputs):
        circuit = Circuit(ir, Z(0), "numpy", "numpy")
        result = circuit(weights, inputs)
        assert isinstance(result, float)

    def test_rejects_unknown_observable_type(self, ir):
        with pytest.raises(TypeError, match="not an observable"):
            Circuit(ir, ["not an observable"], "numpy", "numpy")


class TestTorchInterface:
    def test_returns_tensor(self, ir, weights, inputs):
        circuit = Circuit(ir, Z(0).to_observable(), "numpy", "torch")
        result = circuit(torch.tensor(weights), torch.tensor(inputs))
        assert isinstance(result, torch.Tensor)

    def test_gradient_flows_native(self, ir, inputs):
        circuit = Circuit(ir, Z(0).to_observable(), "numpy", "torch")
        weights = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        y = circuit(weights, torch.tensor(inputs))
        y.backward()
        assert weights.grad is not None
        assert weights.grad.norm() > 0

    @_needs_qiskit
    def test_gradient_flows_parameter_shift(self, ir, inputs):
        circuit = Circuit(ir, Z(0).to_observable(), "qiskit", "torch")
        weights = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        y = circuit(weights, torch.tensor(inputs))
        y.backward()
        assert weights.grad is not None


@_needs_qiskit
class TestInputGradient:
    """Parameter-shift backends must propagate gradients to inputs so that
    upstream trainable layers (encoders, RNN heads, input adapters) can be
    trained end-to-end. The native path (pennylane) provides this through
    torch interop; the parameter-shift path provides it through the
    chain-rule jacobian.
    """

    @pytest.mark.parametrize("provider", ["pennylane", "qiskit"])
    def test_input_grad_is_populated(self, ir, weights, provider):
        circuit = Circuit(ir, Z(0).to_observable(), provider, "torch")
        w = torch.tensor(weights, dtype=torch.float64)
        x = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        y = circuit(w, x)
        y.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad)).item()

    @pytest.mark.pennylane
    def test_parameter_shift_input_grad_matches_pennylane(self, ir):
        """The chain-rule input jacobian must match PennyLane's native
        autograd values within numerical tolerance."""
        ref = Circuit(ir, Z(0).to_observable(), "pennylane", "torch")
        sut = Circuit(ir, Z(0).to_observable(), "qiskit", "torch")
        for seed in range(3):
            g = torch.Generator().manual_seed(seed)
            w = torch.rand(2, generator=g, dtype=torch.float64)
            x_ref = torch.rand(2, generator=g, dtype=torch.float64).requires_grad_(True)
            x_sut = x_ref.detach().clone().requires_grad_(True)
            ref(w, x_ref).backward()
            sut(w, x_sut).backward()
            assert x_ref.grad is not None
            assert x_sut.grad is not None
            assert torch.allclose(x_ref.grad, x_sut.grad, atol=1e-7)

    def test_parameter_shift_input_grad_matches_finite_difference(self, ir):
        """Independent of any other backend: the parameter-shift input
        jacobian must agree with a central finite difference."""
        circuit = Circuit(ir, Z(0).to_observable(), "qiskit", "torch")
        w = torch.tensor([0.3, 0.7], dtype=torch.float64)
        x0 = torch.tensor([0.4, -0.2], dtype=torch.float64)
        x = x0.clone().requires_grad_(True)
        circuit(w, x).backward()
        eps = 1e-5
        fd = torch.empty_like(x0)
        for j in range(x0.numel()):
            xp = x0.clone()
            xm = x0.clone()
            xp[j] += eps
            xm[j] -= eps
            fd[j] = (circuit(w, xp) - circuit(w, xm)) / (2 * eps)
        assert x.grad is not None
        assert torch.allclose(x.grad, fd, atol=1e-6)

    def test_input_grad_supports_2d_batched_inputs(self, ir, weights):
        """Batched inputs must produce per-sample input gradients."""
        circuit = Circuit(ir, Z(0).to_observable(), "qiskit", "torch")
        w = torch.tensor(weights, dtype=torch.float64)
        x = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            dtype=torch.float64,
            requires_grad=True,
        )
        y = circuit(w, x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad)).item()

    @pytest.mark.pennylane
    def test_cudaq_input_grad_via_naive_path(self, ir):
        """cudaq cannot compile `Sum` parameter expressions, so the
        chain-rule perturbation path (which wraps in Sum) is unavailable.
        For simple encodings where each input-dependent gate is a direct
        `InputRef`, the naive input-shift fast path applies and cudaq
        gets correct input gradients.
        """
        cudaq = pytest.importorskip("cudaq")  # noqa: F841
        from eigenkit import CudaQuantumConfig

        circuit = Circuit(
            ir, Z(0).to_observable(), CudaQuantumConfig(target="qpp-cpu"), "torch"
        )
        ref = Circuit(ir, Z(0).to_observable(), "pennylane", "torch")
        w = torch.tensor([0.3, 0.7], dtype=torch.float64)
        x_cudaq = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        x_ref = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        circuit(w, x_cudaq).backward()
        ref(w, x_ref).backward()
        assert x_cudaq.grad is not None
        assert torch.allclose(x_cudaq.grad, x_ref.grad, atol=1e-7)

    @pytest.mark.pennylane
    def test_chain_rule_path_input_grad_with_scaled_encoding(self):
        """When `Encoding(scale != 1.0)` wraps inputs in `Scale`, naive
        input shift is not correct; the chain-rule path takes over.
        Verify it produces the right input gradient by matching pennylane.
        """
        ir = assemble(
            [
                AngleEncoding(in_features=2, scale=2.0),
                BasicVariational(depth=1, rotation="RY"),
            ]
        )
        ref = Circuit(ir, Z(0).to_observable(), "pennylane", "torch")
        sut = Circuit(ir, Z(0).to_observable(), "qiskit", "torch")
        w = torch.tensor([0.3, 0.7], dtype=torch.float64)
        x_ref = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        x_sut = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        ref(w, x_ref).backward()
        sut(w, x_sut).backward()
        assert torch.allclose(x_sut.grad, x_ref.grad, atol=1e-7)

    def test_input_grad_with_mixed_dtype_upstream(self, ir):
        """Real-world pipelines often have a float32 upstream (e.g. an
        RNN or Linear) feeding a float64 quantum block. The backward
        pass receives `grad_output` in the quantum block's float64 dtype
        but must return an input gradient in the upstream's float32. The
        backward must build its internal matmul tensors in a dtype that
        works regardless and cast the result to match the input.
        """
        circuit = Circuit(ir, Z(0).to_observable(), "qiskit", "torch")
        weights = torch.nn.Parameter(torch.tensor([0.3, 0.7], dtype=torch.float64))
        upstream = torch.nn.Linear(3, 2, dtype=torch.float32)
        raw_x = torch.randn(2, 3, dtype=torch.float32)
        x_f32 = upstream(raw_x)
        x_f64 = x_f32.to(torch.float64)
        y = torch.stack([circuit(weights, x_f64[i]) for i in range(2)])
        y.sum().backward()
        assert upstream.weight.grad is not None
        assert upstream.weight.grad.dtype == torch.float32
        assert torch.all(torch.isfinite(upstream.weight.grad)).item()

    def test_upstream_linear_trains_through_parameter_shift_backend(self, ir):
        """End-to-end smoke test: a trainable nn.Linear placed *before*
        the quantum block must receive a non-None gradient on the
        parameter-shift path.
        """
        circuit = Circuit(ir, Z(0).to_observable(), "qiskit", "torch")
        encoder = torch.nn.Linear(3, 2, dtype=torch.float64)
        weights = torch.nn.Parameter(torch.tensor([0.3, 0.7], dtype=torch.float64))
        raw_x = torch.randn(2, 3, dtype=torch.float64)
        encoded = encoder(raw_x)
        y = torch.stack([circuit(weights, encoded[i]) for i in range(2)])
        y.sum().backward()
        assert weights.grad is not None
        assert encoder.weight.grad is not None
        assert encoder.bias.grad is not None
        assert torch.all(torch.isfinite(encoder.weight.grad)).item()


class TestUnknownInterface:
    def test_raises(self, ir, weights, inputs):
        with pytest.raises(ValueError, match="unknown interface"):
            circuit = Circuit(ir, Z(0).to_observable(), "numpy", "numpy")
            circuit.interface = "jax"  # forcibly break
            circuit(weights, inputs)


class TestCompileValidation:
    def test_validates_at_construction(self, ir):
        from eigenkit.core.ir import CircuitSpec, Gate

        bad_ir = CircuitSpec(
            n_qubits=1, n_weights=0, n_inputs=0, gates=(Gate("U3", (0,), ()),)
        )
        with pytest.raises(ValueError, match="U3"):
            Circuit(bad_ir, Z(0).to_observable(), "numpy")


class TestObservableNormalization:
    def test_pauli_expr_inside_list(self, ir, weights, inputs):
        # List of PauliExpr -- exercises the per-item to_observable() branch.
        circuit = Circuit(ir, [Z(0), X(1)], "numpy", "numpy")
        result = circuit(weights, inputs)
        assert result.shape == (2,)


class TestInputDimensionality:
    def test_numpy_3d_input_rejected(self, ir, weights):
        circuit = Circuit(ir, all_z(2), "numpy", "numpy")
        with pytest.raises(ValueError, match="1D"):
            circuit(weights, np.zeros((4, 4, 2), dtype=np.float64))

    def test_torch_3d_input_rejected(self, ir, weights):
        circuit = Circuit(ir, all_z(2), "numpy", "torch")
        with pytest.raises(ValueError, match="1D"):
            circuit(torch.tensor(weights), torch.zeros(4, 4, 2, dtype=torch.float64))


@_needs_qiskit
class TestPerSampleFailureWrap:
    def test_sample_failure_includes_index(self, ir, weights, inputs):
        # Use qiskit (no native broadcast) so the loop in `batch_call` is
        # exercised. The per-sample error wrap lives there and reaches
        # `_run_one` once per sample.
        circuit = Circuit(ir, all_z(2), "qiskit", "numpy")

        original_run_one = circuit.compiled._run_one
        call_state = {"i": 0}

        def flaky_run_one(w, x):
            call_state["i"] += 1
            if call_state["i"] == 2:
                raise RuntimeError("synthetic backend failure")
            return original_run_one(w, x)

        circuit.compiled._run_one = flaky_run_one
        with pytest.raises(RuntimeError, match="batch sample index 1"):
            circuit(weights, np.stack([inputs, inputs]))


@_needs_qiskit
class TestChainRuleEarlyExits:
    """The chain-rule jacobian short-circuits gates whose angles either
    don't depend on any weight or whose chain factors all evaluate to
    zero. Both branches should leave the gradient unchanged."""

    def test_gate_with_no_weight_deps(self):
        # A circuit whose first gate's angle is a pure InputRef (no weight
        # dependency) AND whose other gates have a Multiply (so the
        # chain-rule path is selected). The first gate should hit the
        # "no weight deps" continue.
        from eigenkit.core.ir import (
            CircuitSpec,
            Gate,
            InputRef,
            Multiply,
            WeightRef,
        )

        spec = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=1,
            gates=(
                Gate("RX", (0,), (InputRef(0),)),
                Gate("RX", (0,), (Multiply(WeightRef(0), InputRef(0)),)),
            ),
        )
        circuit = Circuit(spec, Z(0).to_observable(), "qiskit", "torch")
        w = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([0.7], dtype=torch.float64)
        y = circuit(w, x)
        y.backward()
        assert w.grad is not None

    def test_gate_with_zero_chain_factor(self):
        # `Multiply(WeightRef, InputRef)` evaluated at inputs=0 has
        # chain factor 0 for the weight, so the chain-rule path's "no
        # chain factors" continue triggers.
        from eigenkit.core.ir import (
            CircuitSpec,
            Gate,
            InputRef,
            Multiply,
            WeightRef,
        )

        spec = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=1,
            gates=(Gate("RX", (0,), (Multiply(WeightRef(0), InputRef(0)),)),),
        )
        circuit = Circuit(spec, Z(0).to_observable(), "qiskit", "torch")
        w = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([0.0], dtype=torch.float64)  # zero chain factor
        y = circuit(w, x)
        y.backward()
        # w.grad is allowed to be 0 here -- the chain factor is 0.
        assert w.grad is not None
        assert torch.allclose(w.grad, torch.zeros_like(w.grad))


class TestZeroStateProbabilityDispatch:
    def test_torch_inputs_accepted(self, ir):
        circuit = Circuit(ir, all_z(2), "numpy", "torch")
        w = torch.tensor([0.3, 0.7], dtype=torch.float64)
        x = torch.tensor([0.1, 0.2], dtype=torch.float64)
        prob = circuit.zero_state_probability(w, x)
        assert 0.0 <= prob <= 1.0 + 1e-12

    def test_numpy_inputs_accepted(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(2), "numpy", "numpy")
        prob = circuit.zero_state_probability(weights, inputs)
        assert 0.0 <= prob <= 1.0 + 1e-12

    def test_no_inputs_default(self):
        # Circuit with n_inputs == 0: omitting inputs is allowed.
        from eigenkit.core.ir import CircuitSpec, Gate, WeightRef

        spec = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=0,
            gates=(Gate("RX", (0,), (WeightRef(0),)),),
        )
        circuit = Circuit(spec, Z(0).to_observable(), "numpy", "numpy")
        prob = circuit.zero_state_probability(np.array([0.3]))
        assert 0.0 <= prob <= 1.0 + 1e-12

    def test_unimplemented_raises(self, ir, weights, inputs):
        # Replace the compiled handle with a stub that doesn't expose
        # `zero_state_probability`.
        circuit = Circuit(ir, all_z(2), "numpy", "numpy")

        original_run = circuit.compiled.run

        class Stub:
            def run(self, w, x):
                return original_run(w, x)

        circuit.compiled = Stub()
        with pytest.raises(NotImplementedError, match="zero_state_probability"):
            circuit.zero_state_probability(weights, inputs)


class TestSubmitDispatch:
    def test_unsupported_backend_raises(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(2), "numpy", "numpy")
        with pytest.raises(NotImplementedError, match="submit"):
            circuit.submit(weights, inputs)

    def test_submit_2d_input_rejected(self, ir, weights):
        # The numpy simulator doesn't support submit; use a fake compiled object
        # with a `submit` method to drive the inputs-shape branch.
        circuit = Circuit(ir, all_z(2), "numpy", "numpy")

        class StubCompiled:
            def run(self, w, x):
                return circuit.compiled.run(w, x)

            def submit(self, w, x):
                return None

        circuit.compiled = StubCompiled()
        with pytest.raises(ValueError, match="1-D inputs"):
            circuit.submit(weights, np.zeros((3, 2), dtype=np.float64))

    def test_submit_default_inputs_when_none_consumed(self):
        from eigenkit.core.ir import CircuitSpec, Gate, WeightRef

        spec = CircuitSpec(
            n_qubits=1,
            n_weights=1,
            n_inputs=0,
            gates=(Gate("RX", (0,), (WeightRef(0),)),),
        )
        circuit = Circuit(spec, Z(0).to_observable(), "numpy", "numpy")

        class StubCompiled:
            received: tuple | None = None

            def run(self, w, x):
                return circuit.compiled.run(w, x)

            def submit(self, w, x):
                StubCompiled.received = (w.shape, x.shape)
                return None

        circuit.compiled = StubCompiled()
        circuit.submit(np.array([0.3]))  # inputs default
        assert StubCompiled.received == ((1,), (0,))


class TestEmptyObservableIsRejectedOnce:
    """An observable with no Pauli terms cannot be measured.

    It is a legitimate value -- `jordan_wigner` returns one with an offset when
    the operator reduces to a multiple of the identity -- but left to the
    backends, measuring it would not give one answer: qsim, qulacs and
    tensorcircuit would return 0.0 while numpy, pennylane, qiskit, qiskit-aer
    and braket would raise three different exception types. `Circuit` rejects
    it before any backend sees it, so every provider gives the same error.
    """

    @staticmethod
    def _spec():
        return assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])

    def test_rejected_on_every_provider(self):
        for provider in ("numpy", "pennylane"):
            with pytest.raises(ValueError, match="no Pauli terms"):
                Circuit(self._spec(), Observable(terms=()), provider, "numpy")

    def test_the_message_points_at_the_offset_convention(self):
        with pytest.raises(ValueError, match="offset"):
            Circuit(self._spec(), Observable(terms=()), "numpy", "numpy")

    def test_an_empty_one_among_several_names_its_position(self):
        observables = [Observable.z(0), Observable(terms=())]
        with pytest.raises(ValueError, match="at index 1"):
            Circuit(self._spec(), observables, "numpy", "numpy")


class TestSingleSampleMethodsRejectABatch:
    """`probabilities` and `zero_state_probability` answer for one sample.

    Both refuse a 2-D `inputs`. Passed through to the backend, a batch would
    produce silent nonsense rather than an error: a 2-qubit circuit with a
    3-row batch would come back with shape (16,) summing to 4.0 on numpy and
    (2, 4) summing to 2.0 on pennylane. Neither is a probability distribution.
    """

    @staticmethod
    def _circuit(provider="numpy"):
        spec = assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])
        return Circuit(spec, all_z(2), provider, "numpy")

    def test_probabilities_rejects_a_batch(self):
        circuit = self._circuit()
        weights = np.zeros(circuit.spec.n_weights)
        with pytest.raises(ValueError, match="takes one sample"):
            circuit.probabilities(weights, np.zeros((3, 2)))

    def test_zero_state_probability_rejects_a_batch(self):
        circuit = self._circuit()
        weights = np.zeros(circuit.spec.n_weights)
        with pytest.raises(ValueError, match="takes one sample"):
            circuit.zero_state_probability(weights, np.zeros((3, 2)))

    def test_the_single_sample_case_still_works(self):
        circuit = self._circuit()
        weights = np.zeros(circuit.spec.n_weights)
        probabilities = circuit.probabilities(weights, np.zeros(2))
        assert probabilities.shape == (4,)
        np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)

    def test_the_message_points_at_the_batched_sibling(self):
        circuit = self._circuit()
        weights = np.zeros(circuit.spec.n_weights)
        with pytest.raises(ValueError, match="zero_state_probabilities"):
            circuit.zero_state_probability(weights, np.zeros((3, 2)))

    def test_the_torch_wrapper_documents_what_it_actually_does(self):
        """`TorchCircuit.zero_state_probability` takes one sample, not an
        input batch, and returns a plain float, not a tensor on the tape."""
        torch = pytest.importorskip("torch")

        from eigenkit.torch import TorchCircuit

        spec = assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])
        circuit = TorchCircuit(spec)

        probability = circuit.zero_state_probability(torch.zeros(2, dtype=torch.float64))
        assert isinstance(probability, float)
        with pytest.raises(ValueError, match="takes one sample"):
            circuit.zero_state_probability(torch.zeros((3, 2), dtype=torch.float64))
