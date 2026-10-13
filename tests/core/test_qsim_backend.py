"""qsim (Cirq) backend: parity tests, skipped when qsimcirq is not installed.

qsim is **single precision (fp32)**, so parity against the numpy fp64 reference is checked at a
relaxed ~1e-4 tolerance — tight enough to catch a real translation/endianness bug, loose enough to
pass fp32 accumulation error. The tolerance is the point: qsim trades precision for speed.

Unlike the Aer backend (which reuses the Qiskit backend's proven IR translation), the Cirq lowering
here is hand-written, so the gate table and the parameter-expression translation each get their own
parity class against the numpy oracle."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest


qsimcirq = pytest.importorskip("qsimcirq")  # skips entire module when missing

from eigenkit.core.backends.qsim_backend import (  # noqa: E402
    QsimBackend,
    QsimConfig,
)
from eigenkit.core.blocks import (  # noqa: E402
    AngleEncoding,
    Reuploading,
    StrongVariational,
    assemble,
)
from eigenkit.core.builder import CircuitBuilder  # noqa: E402
from eigenkit.core.circuit import Circuit  # noqa: E402
from eigenkit.core.ir import (  # noqa: E402
    DATA_GATES,
    CircuitSpec,
    Constant,
    Gate,
    InputRef,
    Multiply,
    Negate,
    Scale,
    Sum,
    WeightRef,
)
from eigenkit.core.observables import Z, all_z, from_string  # noqa: E402
from eigenkit.core.provider import get, register  # noqa: E402


# fp32 accumulation: tight enough to catch a real bug (endianness/gate/ordering would be O(1)),
# loose enough for single-precision rounding.
FP32_TOL = 1e-4
# Parameter-shift differences two fp32 evaluations, so gradients carry roughly twice the forward
# residual. Still four orders of magnitude below a real translation bug.
GRAD_TOL = 5e-4


@pytest.fixture(autouse=True)
def _register_qsim():
    register("qsim", QsimBackend())


@pytest.fixture
def ir():
    return assemble([AngleEncoding(in_features=3), StrongVariational(depth=2)])


@pytest.fixture
def weights(ir):
    rng = np.random.RandomState(0)
    return rng.uniform(-np.pi, np.pi, ir.n_weights).astype(np.float64)


@pytest.fixture
def inputs():
    return np.array([0.5, -0.2, 0.8], dtype=np.float64)


class TestForwardParity:
    def test_single_observable(self, ir, weights, inputs):
        got = Circuit(ir, Z(1).to_observable(), "qsim", "numpy")(weights, inputs)
        ref = Circuit(ir, Z(1).to_observable(), "numpy", "numpy")(weights, inputs)
        assert abs(got - ref) < FP32_TOL

    def test_multi_observable(self, ir, weights, inputs):
        obs = all_z(3)
        got = np.asarray(Circuit(ir, obs, "qsim", "numpy")(weights, inputs))
        ref = np.asarray(Circuit(ir, obs, "numpy", "numpy")(weights, inputs))
        assert np.max(np.abs(got - ref)) < FP32_TOL

    def test_nondiagonal_observable(self, ir, weights, inputs):
        obs = from_string("0.6 Z0 + 0.3 X0 X1 + 0.2 Y2")
        got = Circuit(ir, obs, "qsim", "numpy")(weights, inputs)
        ref = Circuit(ir, obs, "numpy", "numpy")(weights, inputs)
        assert abs(got - ref) < FP32_TOL

    def test_batched_forward(self, ir, weights):
        rng = np.random.RandomState(1)
        x = rng.uniform(-1, 1, (5, 3)).astype(np.float64)
        obs = all_z(3)
        got = np.asarray(Circuit(ir, obs, "qsim", "numpy")(weights, x))
        ref = np.asarray(Circuit(ir, obs, "numpy", "numpy")(weights, x))
        assert got.shape == (5, 3)
        assert np.max(np.abs(got - ref)) < FP32_TOL


def _spec_parity(build_fn, n_qubits, obs_qubit=0):
    """Build a constant-angle circuit through the fluent builder and compare qsim to numpy."""
    builder = CircuitBuilder(n_qubits=n_qubits)
    build_fn(builder)
    spec = builder.build()
    obs = Z(obs_qubit).to_observable()
    w = np.zeros(spec.n_weights)
    x = np.zeros(spec.n_inputs)
    got = Circuit(spec, obs, "qsim", "numpy")(w, x)
    ref = Circuit(spec, obs, "numpy", "numpy")(w, x)
    return got, ref


class TestGateCoverage:
    """Every gate qsim advertises must match numpy. The Cirq lowering is hand-written — the
    controlled gates in particular encode a control/target convention (``qubits[:-1]`` controls,
    ``qubits[-1]`` target) that a silent swap would break."""

    @pytest.mark.parametrize(
        "name, build, n, obs_q",
        [
            ("RX", lambda qc: qc.rx(math.pi / 3, 0), 2, 0),
            ("RY", lambda qc: qc.ry(math.pi / 3, 0), 2, 0),
            ("RZ", lambda qc: qc.h(0).rz(math.pi / 3, 0).h(0), 2, 0),
            ("H", lambda qc: qc.h(0), 2, 0),
            ("X", lambda qc: qc.x(0), 2, 0),
            ("Z", lambda qc: qc.h(0).z(0).h(0), 2, 0),
            ("S/Sdg", lambda qc: qc.h(0).s(0).sdg(0).h(0), 2, 0),
            ("S alone", lambda qc: qc.h(0).s(0).h(0), 2, 0),
            ("Sdg alone", lambda qc: qc.h(0).sdg(0).h(0), 2, 0),
            ("CNOT", lambda qc: qc.x(0).cnot(0, 1), 2, 1),
            ("SWAP", lambda qc: qc.x(0).swap(0, 1), 2, 1),
            ("PhaseShift", lambda qc: qc.h(0).phase_shift(math.pi / 3, 0).h(0), 2, 0),
            ("CRY on", lambda qc: qc.x(0).cry(math.pi / 3, 0, 1), 2, 1),
            # control left in |0>: the target must NOT rotate. Catches a control/target swap.
            ("CRY off", lambda qc: qc.cry(math.pi / 3, 0, 1), 2, 1),
            ("CPHASE", lambda qc: qc.h(0).h(1).cphase(math.pi / 3, 0, 1), 2, 0),
            ("MCX", lambda qc: qc.x(0).x(1).mcx(0, 1, 2), 3, 2),
            ("MCX partial", lambda qc: qc.x(0).mcx(0, 1, 2), 3, 2),
            ("MCZ", lambda qc: qc.x(0).x(1).h(2).mcz(0, 1, 2).h(2), 3, 2),
        ],
    )
    def test_gate_matches_numpy(self, name, build, n, obs_q):
        got, ref = _spec_parity(build, n, obs_q)
        assert abs(got - ref) < FP32_TOL, name

    @pytest.mark.parametrize(
        "name, build",
        [
            ("PhaseShift", lambda qc, w: qc.h(0).phase_shift(w, 0).h(0)),
            ("CPHASE", lambda qc, w: qc.h(0).h(1).cphase(w, 0, 1)),
        ],
    )
    def test_phase_gates_with_symbolic_angle(self, name, build):
        """The phase-gate exponents are the one place the lowering does arithmetic on the angle.
        A constant angle and a weight-driven angle take different paths through cirq's resolver,
        so both need pinning — a sympy.pi form can pass this test while crashing on the
        constant-angle case above."""
        builder = CircuitBuilder(n_qubits=2)
        (w,) = builder.add_weights(1)
        build(builder, w)
        spec = builder.build()
        angles = np.array([[0.0], [math.pi / 3], [-1.2], [math.pi]])
        for weights in angles:
            got = Circuit(spec, Z(0).to_observable(), "qsim", "numpy")(
                weights, np.empty(0)
            )
            ref = Circuit(spec, Z(0).to_observable(), "numpy", "numpy")(
                weights, np.empty(0)
            )
            assert abs(got - ref) < FP32_TOL, f"{name} @ {weights[0]}"

    def test_cry_target_is_second_qubit(self):
        """Pin the convention explicitly: CRY(control=0, target=1) with control set must leave
        qubit 0 alone and rotate qubit 1."""
        builder = CircuitBuilder(n_qubits=2)
        builder.x(0).cry(math.pi, 0, 1)
        spec = builder.build()
        got = np.asarray(
            Circuit(spec, all_z(2), "qsim", "numpy")(np.empty(0), np.empty(0))
        )
        # qubit 0 flipped to |1> -> <Z0> = -1; qubit 1 fully rotated to |1> -> <Z1> = -1
        assert np.allclose(got, [-1.0, -1.0], atol=FP32_TOL)


class TestParameterExpressions:
    """``_resolve_symbolic`` translates the IR parameter algebra into sympy. Every node type needs
    a parity check — this is the path where cudaq has a documented limitation."""

    @staticmethod
    def _check(angle, weights, inputs, n_weights, n_inputs):
        builder = CircuitBuilder(n_qubits=2)
        builder.add_weights(n_weights)
        builder.add_inputs(n_inputs)
        builder.ry(angle, 0)
        spec = builder.build()
        obs = Z(0).to_observable()
        w = np.asarray(weights, dtype=np.float64)
        x = np.asarray(inputs, dtype=np.float64)
        got = Circuit(spec, obs, "qsim", "numpy")(w, x)
        ref = Circuit(spec, obs, "numpy", "numpy")(w, x)
        assert abs(got - ref) < FP32_TOL

    def test_constant(self):
        self._check(Constant(0.7), [], [], 0, 0)

    def test_weight_ref(self):
        self._check(WeightRef(0), [0.9], [], 1, 0)

    def test_input_ref(self):
        self._check(InputRef(0), [], [0.4], 0, 1)

    def test_scale(self):
        self._check(Scale(WeightRef(0), 2.5), [0.6], [], 1, 0)

    def test_negate(self):
        self._check(Negate(WeightRef(0)), [0.6], [], 1, 0)

    def test_sum(self):
        self._check(Sum((WeightRef(0), InputRef(0), Constant(0.2))), [0.6], [0.3], 1, 1)

    def test_multiply_weight_by_input(self):
        """Trainable feature scaling: RY(lambda * x)."""
        self._check(Multiply(WeightRef(0), InputRef(0)), [1.3], [0.7], 1, 1)

    def test_nested_expression(self):
        angle = Sum(
            (Scale(Negate(WeightRef(0)), 0.5), Multiply(WeightRef(1), InputRef(0)))
        )
        self._check(angle, [0.8, 1.1], [0.35], 2, 1)

    def test_reuploading_trainable_scale_end_to_end(self):
        spec = assemble([Reuploading(in_features=2, depth=2, trainable_scale=True)])
        w = np.random.default_rng(3).standard_normal(spec.n_weights)
        x = np.array([0.4, -0.6], dtype=np.float64)
        got = np.asarray(Circuit(spec, all_z(2), "qsim", "numpy")(w, x))
        ref = np.asarray(Circuit(spec, all_z(2), "numpy", "numpy")(w, x))
        assert np.max(np.abs(got - ref)) < FP32_TOL


class TestGradientParity:
    """qsim declares ``gradient_method = "parameter_shift"``; the dispatch layer must produce the
    same jacobian as the numpy adjoint reference."""

    def test_weight_jacobian(self, ir, weights, inputs):
        obs = from_string("0.7 Z0 + 0.3 X1 Z2 - 0.5 Y0 X2")
        got = Circuit(ir, obs, "qsim", "numpy").jacobian(weights, inputs)
        ref = Circuit(ir, obs, "numpy", "numpy").jacobian(weights, inputs)
        assert np.max(np.abs(np.asarray(got) - np.asarray(ref))) < GRAD_TOL

    def test_input_jacobian(self, ir, weights, inputs):
        _, got = Circuit(ir, all_z(3), "qsim", "numpy").jacobian(
            weights, inputs, input_grad=True
        )
        _, ref = Circuit(ir, all_z(3), "numpy", "numpy").jacobian(
            weights, inputs, input_grad=True
        )
        assert np.max(np.abs(np.asarray(got) - np.asarray(ref))) < GRAD_TOL


class TestCapabilities:
    def test_parameter_shift_gradient_method(self):
        assert QsimBackend().gradient_method == "parameter_shift"

    def test_supported_gates_are_all_but_data_gates(self):
        """qsim lowers the full circuit-algebra gate set; only the data-carrying primitives
        (AmplitudeEncoding / QubitUnitary / ControlledQubitUnitary) are out of scope."""
        caps = QsimBackend().capabilities
        assert caps.supported_gates.isdisjoint(DATA_GATES)
        for name in ("RX", "RY", "RZ", "H", "X", "Z", "S", "Sdg", "CNOT", "SWAP"):
            assert name in caps.supported_gates
        for name in ("PhaseShift", "CRY", "CPHASE", "MCZ", "MCX"):
            assert name in caps.supported_gates

    def test_gate_table_covers_every_advertised_gate(self):
        """Guard against the capability set and the lowering drifting apart: anything advertised
        in ``capabilities`` must actually compile."""
        from eigenkit.core.backends.qsim_backend import SUPPORTED

        assert QsimBackend().capabilities.supported_gates == SUPPORTED

    def test_no_batching_or_shots(self):
        caps = QsimBackend().capabilities
        assert caps.supports_batching is False
        assert caps.supports_shots is False
        assert caps.max_qubits is None

    def test_data_gate_rejected_with_clear_error(self):
        """Data gates are out of scope for this adapter; ``validate_circuit`` must reject them up
        front rather than letting the lowering raise a bare NotImplementedError mid-compile."""
        spec = CircuitSpec(
            n_qubits=2,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("QubitUnitary", (0,), (), data=np.eye(2, dtype=np.complex128)),),
        )
        with pytest.raises(ValueError, match="not supported by provider"):
            Circuit(spec, all_z(2), "qsim", "numpy")

    def test_no_native_torch(self, ir):
        """qsim is not a differentiable-tensor library: no `run_torch` fast path."""
        compiled = QsimBackend().compile(ir, all_z(3))
        assert not hasattr(compiled, "run_torch")


class TestStateVector:
    def test_readout_matches_numpy(self, ir, weights, inputs):
        got = np.asarray(get("qsim").compile(ir, all_z(3)).statevector(weights, inputs))
        ref = np.asarray(get("numpy").compile(ir, all_z(3)).statevector(weights, inputs))
        assert got.shape == (8,)
        # phase-invariant comparison: probabilities
        assert np.max(np.abs(np.abs(got) ** 2 - np.abs(ref) ** 2)) < FP32_TOL

    def test_endianness_qubit0_is_msb(self):
        """X on qubit 0 must land at flat index 4 (|100>), qubit-0-MSB order. A Cirq
        qubit_order mistake would put it at index 1."""
        builder = CircuitBuilder(n_qubits=3)
        builder.x(0)
        spec = builder.build()
        sv = np.asarray(
            get("qsim").compile(spec, all_z(3)).statevector(np.empty(0), np.empty(0))
        )
        assert int(np.argmax(np.abs(sv))) == 4

    def test_batched_statevector_shape(self, ir, weights):
        x = np.random.default_rng(2).uniform(-1, 1, (4, 3))
        sv = np.asarray(get("qsim").compile(ir, all_z(3)).statevector(weights, x))
        assert sv.shape == (4, 8)

    def test_initial_state_raises(self, ir, weights, inputs):
        """Injection is deliberately not wired; it must fail loudly rather than silently
        ignoring the requested initial state."""
        compiled = get("qsim").compile(ir, all_z(3))
        with pytest.raises(NotImplementedError, match="initial_state"):
            compiled.statevector(weights, inputs, initial_state=np.eye(8)[0])


class TestZeroStateProbability:
    """The fidelity-kernel primitive behind QuantumKernel / QSVC / QSVR."""

    def test_matches_numpy(self, ir, weights, inputs):
        got = get("qsim").compile(ir, all_z(3)).zero_state_probability(weights, inputs)
        ref = get("numpy").compile(ir, all_z(3)).zero_state_probability(weights, inputs)
        assert abs(got - ref) < FP32_TOL

    def test_hadamard_gives_half(self):
        builder = CircuitBuilder(n_qubits=2)
        builder.h(0)
        spec = builder.build()
        compiled = get("qsim").compile(spec, all_z(2))
        assert (
            abs(compiled.zero_state_probability(np.empty(0), np.empty(0)) - 0.5)
            < FP32_TOL
        )


class TestConfig:
    def test_polymorphic_provider_via_config(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), provider=QsimConfig(), interface="numpy")
        assert np.asarray(circuit(weights, inputs)).shape == (3,)

    def test_explicit_threads_respected(self, ir, weights, inputs):
        """An explicit thread count must not change the numbers, only the schedule."""
        one = Circuit(ir, all_z(3), QsimConfig(threads=1), "numpy")(weights, inputs)
        two = Circuit(ir, all_z(3), QsimConfig(threads=2), "numpy")(weights, inputs)
        assert np.allclose(np.asarray(one), np.asarray(two), atol=FP32_TOL)

    def test_default_backend_name(self):
        assert QsimBackend().name == "qsim"


@pytest.mark.skipif(sys.platform != "darwin", reason="the OpenMP clash is macOS-only")
def test_trains_with_torch_in_a_fresh_process():
    """qsimcirq and torch each ship libomp on macOS; unguarded, the second to start aborts.

    The root conftest sets `KMP_DUPLICATE_LIB_OK` for the whole session, so the
    check runs in a child process without it, as a user's script would.
    """
    pytest.importorskip("torch")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("KMP_DUPLICATE_LIB_OK", "OMP_NUM_THREADS")
    }
    code = """
import torch
from eigenkit import Circuit, Encoding, Variational, all_z

c = Circuit(
    blocks=[Encoding(in_features=4), Variational(depth=2)],
    observables=all_z(4),
    provider="qsim",
    interface="torch",
)
c(torch.randn(8, 4)).pow(2).mean().backward()
torch.ones(500, 500) @ torch.ones(500, 500)
"""
    run = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr[-2000:]
