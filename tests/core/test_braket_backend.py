"""Braket backend: cross-backend parity, capabilities, configs."""

from __future__ import annotations

import itertools

import pytest


pytest.importorskip("braket")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eigenkit.core.backends.braket_backend import (  # noqa: E402
    BraketBackend,
    BraketConfig,
)
from eigenkit.core.blocks import (  # noqa: E402
    AngleEncoding,
    BasicVariational,
    StrongVariational,
    assemble,
)
from eigenkit.core.circuit import Circuit  # noqa: E402
from eigenkit.core.observables import Z, all_z, from_string  # noqa: E402


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
def inputs():
    return np.array([0.5, -0.2, 0.8], dtype=np.float64)


class TestForwardParity:
    """Braket should match the numpy simulator on every forward."""

    def test_single_observable(self, ir, weights, inputs):
        reference = Circuit(ir, Z(1).to_observable(), "numpy", "numpy")
        braket = Circuit(ir, Z(1).to_observable(), "braket", "numpy")
        assert abs(reference(weights, inputs) - braket(weights, inputs)) < 1e-10

    def test_all_z(self, ir, weights, inputs):
        reference = Circuit(ir, all_z(3), "numpy", "numpy")
        braket = Circuit(ir, all_z(3), "braket", "numpy")
        assert np.allclose(
            reference(weights, inputs), braket(weights, inputs), atol=1e-10
        )

    def test_hamiltonian(self, ir, weights, inputs):
        observable = from_string("0.5 Z0 X1 - 0.3 Y2 + 0.7 Z0 Z2")
        reference = Circuit(ir, observable, "numpy", "numpy")
        braket = Circuit(ir, observable, "braket", "numpy")
        assert abs(reference(weights, inputs) - braket(weights, inputs)) < 1e-10


class TestGradientParity:
    def test_param_shift_matches_native(self, ir, weights, inputs):
        observables = [
            from_string("0.7 Z0 + 0.3 X1 Z2"),
            Z(0).to_observable(),
        ]

        ref_w = torch.tensor(weights, requires_grad=True)
        bk_w = torch.tensor(weights, requires_grad=True)
        x_t = torch.tensor(inputs)

        reference = Circuit(ir, observables, "numpy", "torch")
        braket = Circuit(ir, observables, "braket", "torch")

        (reference(ref_w, x_t) ** 2).sum().backward()
        (braket(bk_w, x_t) ** 2).sum().backward()

        assert torch.allclose(ref_w.grad, bk_w.grad, atol=1e-8)


class TestCapabilities:
    def test_declaration(self):
        caps = BraketBackend().capabilities
        assert "RX" in caps.supported_gates
        assert "CNOT" in caps.supported_gates
        assert caps.supports_shots is True
        assert caps.max_qubits is None


class TestCompile:
    def test_returns_compiled_circuit(self, ir):
        compiled = BraketBackend().compile(ir, all_z(3))
        assert hasattr(compiled, "run")

    def test_no_run_torch(self, ir):
        """Braket has no native autograd; the IR layer wraps it."""
        compiled = BraketBackend().compile(ir, all_z(3))
        assert not hasattr(compiled, "run_torch")

    def test_compile_reusable(self, ir, weights, inputs):
        compiled = BraketBackend().compile(ir, all_z(3))
        first = compiled.run(weights, inputs)
        second = compiled.run(weights, inputs)
        assert np.allclose(first, second)


class TestConfig:
    def test_default(self):
        backend = BraketBackend()
        assert backend.config.device_arn is None
        assert backend.config.local_backend == "braket_sv"
        assert backend.config.shots == 0

    def test_via_make_backend(self):
        config = BraketConfig(local_backend="braket_sv")
        backend = config.make_backend()
        assert isinstance(backend, BraketBackend)
        assert backend.config.local_backend == "braket_sv"

    def test_polymorphic_provider_arg(self, ir, weights, inputs):
        """Pass BraketConfig directly as provider — no register call."""
        circuit = Circuit(
            ir,
            all_z(3),
            provider=BraketConfig(),
            interface="numpy",
        )
        default_circuit = Circuit(ir, all_z(3), "braket", "numpy")
        assert np.allclose(
            circuit(weights, inputs),
            default_circuit(weights, inputs),
            atol=1e-10,
        )


class TestDryRunInfo:
    """`Circuit.dry_run_info()` proxies to `BraketCompiled.dry_run_info()`."""

    def test_returns_dict_with_expected_keys(self, ir):
        circuit = Circuit(ir, all_z(3), "braket", "numpy")
        info = circuit.dry_run_info()
        assert isinstance(info, dict)
        assert {
            "device_arn",
            "shots",
            "n_qubits",
            "n_gates",
            "n_observables",
            "is_cloud",
            "is_qpu",
        }.issubset(info.keys())

    def test_numpy_raises(self, ir):
        circuit = Circuit(ir, all_z(3), "numpy", "numpy")
        with pytest.raises(NotImplementedError, match="dry_run_info"):
            circuit.dry_run_info()


class TestExpressions:
    """Verify Scale/Negate via FreeParameterExpression arithmetic."""

    def test_scaled_encoding_matches_numpy(self):
        """Encoding(scale=2.0) uses Scale expression internally."""
        ir = assemble(
            [
                AngleEncoding(in_features=2, scale=2.0),
                BasicVariational(depth=1, rotation="RY"),
            ]
        )
        weights = np.array([0.3, 0.7], dtype=np.float64)
        inputs = np.array([0.4, -0.6], dtype=np.float64)

        ref_out = Circuit(ir, all_z(2), "numpy", "numpy")(weights, inputs)
        bk_out = Circuit(ir, all_z(2), "braket", "numpy")(weights, inputs)
        assert np.allclose(ref_out, bk_out, atol=1e-10)


class TestStrongVariationalParity:
    def test_strong_variational_matches(self):
        ir = assemble(
            [
                AngleEncoding(in_features=2),
                StrongVariational(depth=2),
            ]
        )
        weights = np.random.default_rng(0).standard_normal(ir.n_weights)
        inputs = np.array([0.3, -0.5], dtype=np.float64)

        ref_out = Circuit(ir, all_z(2), "numpy", "numpy")(weights, inputs)
        bk_out = Circuit(ir, all_z(2), "braket", "numpy")(weights, inputs)
        assert np.allclose(ref_out, bk_out, atol=1e-10)


@pytest.mark.parametrize(
    "gate, qubits",
    [
        (gate, qubits)
        for gate in ("mcz", "mcx", "cry")
        for qubits in itertools.permutations(range(4), 2)
    ]
    + [("mcz", (4, 1, 3)), ("mcx", (2, 0, 4)), ("mcx", (3, 1, 0, 4))],
)
def test_controlled_gates_match_numpy_for_every_qubit_ordering(gate, qubits):
    """Braket's local simulator silently skips `ctrl @` gates for some control
    and target positions on four or more qubits, so the backend lowers them to
    native gates instead; checked here over every two-qubit ordering."""
    from eigenkit import CircuitBuilder, X

    n = max(max(qubits) + 1, 4)
    qc = CircuitBuilder(n_qubits=n)
    for q in range(n):
        qc.h(q)
        qc.ry(0.3 + 0.2 * q, q)
    if gate == "cry":
        qc.cry(0.9, *qubits)
    else:
        getattr(qc, gate)(*qubits)
    spec = qc.build()
    observables = [Z(q) for q in range(n)] + [X(q) for q in range(n)]
    expected = Circuit(spec, observables=observables)([])
    actual = Circuit(spec, observables=observables, provider="braket")([])
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_shots_measure_non_commuting_observables_in_separate_tasks():
    """At shots > 0 Braket measures only qubit-wise-commuting terms in one task,
    so the backend splits the rest into one task per group."""
    from eigenkit import CircuitBuilder, X, Y

    qc = CircuitBuilder(n_qubits=2)
    qc.h(0).cnot(0, 1).ry(0.4, 1)
    spec = qc.build()
    observables = [Z(0), X(0), Z(0) * Z(1) + 0.5 * X(0) * X(1) - 0.3 * Y(0) * Y(1)]
    expected = Circuit(spec, observables=observables)([])
    circuit = Circuit(spec, observables=observables, provider=BraketConfig(shots=20000))
    assert circuit.compiled.dry_run_info()["n_circuit_runs_per_call"] == 3
    np.testing.assert_allclose(circuit([]), expected, atol=0.05)
    with pytest.raises(ValueError, match="one per group"):
        circuit.compiled.submit(np.empty(0), np.empty(0))
