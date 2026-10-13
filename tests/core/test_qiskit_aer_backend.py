"""qiskit-aer backend: parity tests, skipped when qiskit-aer is not installed.

The numpy reference simulator is the ground truth (adjoint-exact, always available). These compare
both Aer methods — ``statevector`` (exact) and ``matrix_product_state`` (exact at unbounded bond) —
against ``provider="numpy"`` for forward expectation values, gradients, statevector I/O, the
fidelity-kernel primitive, and a batched forward. Aer statevector matches to ~1e-12; MPS to ~1e-9
(SVD truncation).

The IR translation is shared verbatim with the plain Qiskit backend, so the gate lowering itself is
covered by the Qiskit parity tests in ``test_backends.py``. What this file adds is the *execution*
path — Aer's ``save_expectation_value`` / ``save_statevector`` / ``set_statevector`` instructions —
which the MPS method reaches through a different simulator core. Gate coverage is therefore
parametrized over both methods rather than skipped."""

from __future__ import annotations

import math

import numpy as np
import pytest


qiskit_aer = pytest.importorskip("qiskit_aer")  # skips entire module when missing

from eigenkit.core.backends.qiskit_aer_backend import (  # noqa: E402
    QiskitAerBackend,
    QiskitAerConfig,
)
from eigenkit.core.blocks import (  # noqa: E402
    AngleEncoding,
    StrongVariational,
    assemble,
)
from eigenkit.core.builder import CircuitBuilder  # noqa: E402
from eigenkit.core.circuit import Circuit  # noqa: E402
from eigenkit.core.ir import (  # noqa: E402
    ALL_GATES,
    CircuitSpec,
    Constant,
    Gate,
    InputRef,
    WeightRef,
)
from eigenkit.core.observables import Z, all_z, from_string  # noqa: E402
from eigenkit.core.provider import get, register  # noqa: E402


SV_TOL = 1e-10
MPS_TOL = 1e-8
# Parameter-shift differences two forward evaluations, so it inherits roughly twice the
# method's forward residual.
SV_GRAD_TOL = 1e-9
MPS_GRAD_TOL = 1e-7

METHODS = [("qiskit-aer", SV_TOL), ("qiskit-aer-mps", MPS_TOL)]


@pytest.fixture(autouse=True)
def _register_aer():
    register("qiskit-aer", QiskitAerBackend())
    register(
        "qiskit-aer-mps", QiskitAerConfig(method="matrix_product_state").make_backend()
    )


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


@pytest.mark.parametrize("pid,tol", METHODS)
class TestParity:
    def test_single_observable(self, ir, weights, inputs, pid, tol):
        got = Circuit(ir, Z(1).to_observable(), pid, "numpy")(weights, inputs)
        ref = Circuit(ir, Z(1).to_observable(), "numpy", "numpy")(weights, inputs)
        assert abs(got - ref) < tol

    def test_multi_observable(self, ir, weights, inputs, pid, tol):
        obs = all_z(3)
        got = np.asarray(Circuit(ir, obs, pid, "numpy")(weights, inputs))
        ref = np.asarray(Circuit(ir, obs, "numpy", "numpy")(weights, inputs))
        assert np.max(np.abs(got - ref)) < tol

    def test_nondiagonal_observable(self, ir, weights, inputs, pid, tol):
        obs = from_string("0.6 Z0 + 0.3 X0 X1 + 0.2 Y2")
        got = Circuit(ir, obs, pid, "numpy")(weights, inputs)
        ref = Circuit(ir, obs, "numpy", "numpy")(weights, inputs)
        assert abs(got - ref) < tol

    def test_statevector(self, ir, weights, inputs, pid, tol):
        got = _sv(pid, ir, weights, inputs)
        ref = _sv("numpy", ir, weights, inputs)
        # phase-invariant: compare probabilities
        assert np.max(np.abs(np.abs(got) ** 2 - np.abs(ref) ** 2)) < tol

    def test_batched_forward(self, ir, weights, pid, tol):
        rng = np.random.RandomState(1)
        x = rng.uniform(-1, 1, (5, 3)).astype(np.float64)
        obs = all_z(3)
        got = np.asarray(Circuit(ir, obs, pid, "numpy")(weights, x))
        ref = np.asarray(Circuit(ir, obs, "numpy", "numpy")(weights, x))
        assert got.shape == (5, 3)
        assert np.max(np.abs(got - ref)) < tol


def _sv(pid, ir, weights, inputs, initial_state=None):
    compiled = get(pid).compile(ir, all_z(ir.n_qubits))
    return np.asarray(
        compiled.statevector(weights, inputs, initial_state=initial_state),
        dtype=np.complex128,
    )


@pytest.mark.parametrize(
    "pid,tol", [("qiskit-aer", SV_GRAD_TOL), ("qiskit-aer-mps", MPS_GRAD_TOL)]
)
class TestGradientParity:
    """Both methods declare ``gradient_method = "parameter_shift"``; the dispatch layer must
    reproduce the numpy adjoint jacobian."""

    def test_weight_jacobian(self, ir, weights, inputs, pid, tol):
        obs = from_string("0.7 Z0 + 0.3 X1 Z2 - 0.5 Y0 X2")
        got = Circuit(ir, obs, pid, "numpy").jacobian(weights, inputs)
        ref = Circuit(ir, obs, "numpy", "numpy").jacobian(weights, inputs)
        assert np.max(np.abs(np.asarray(got) - np.asarray(ref))) < tol

    def test_input_jacobian(self, ir, weights, inputs, pid, tol):
        _, got = Circuit(ir, all_z(3), pid, "numpy").jacobian(
            weights, inputs, input_grad=True
        )
        _, ref = Circuit(ir, all_z(3), "numpy", "numpy").jacobian(
            weights, inputs, input_grad=True
        )
        assert np.max(np.abs(np.asarray(got) - np.asarray(ref))) < tol


def _spec_parity(pid, build_fn, n_qubits, obs_qubit=0):
    builder = CircuitBuilder(n_qubits=n_qubits)
    build_fn(builder)
    spec = builder.build()
    obs = Z(obs_qubit).to_observable()
    w = np.zeros(spec.n_weights)
    x = np.zeros(spec.n_inputs)
    got = Circuit(spec, obs, pid, "numpy")(w, x)
    ref = Circuit(spec, obs, "numpy", "numpy")(w, x)
    return got, ref


@pytest.mark.parametrize("pid,tol", METHODS)
class TestGateCoverage:
    """The lowering is shared with the Qiskit backend, but each gate still has to survive
    execution under both simulator cores. ``CRY`` is the one the MPS core refuses to run
    directly — it reaches this test only via the basis translation in ``_to_native_basis``."""

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
            ("CNOT", lambda qc: qc.x(0).cnot(0, 1), 2, 1),
            ("SWAP", lambda qc: qc.x(0).swap(0, 1), 2, 1),
            ("PhaseShift", lambda qc: qc.h(0).phase_shift(math.pi / 3, 0).h(0), 2, 0),
            ("CRY on", lambda qc: qc.x(0).cry(math.pi / 3, 0, 1), 2, 1),
            ("CRY off", lambda qc: qc.cry(math.pi / 3, 0, 1), 2, 1),
            ("CPHASE", lambda qc: qc.h(0).h(1).cphase(math.pi / 3, 0, 1), 2, 0),
            ("MCX", lambda qc: qc.x(0).x(1).mcx(0, 1, 2), 3, 2),
            ("MCZ", lambda qc: qc.x(0).x(1).h(2).mcz(0, 1, 2).h(2), 3, 2),
        ],
    )
    def test_gate_matches_numpy(self, name, build, n, obs_q, pid, tol):
        got, ref = _spec_parity(pid, build, n, obs_q)
        assert abs(got - ref) < tol, name


@pytest.mark.parametrize("pid,tol", METHODS)
class TestStateVector:
    def test_endianness_qubit0_is_msb(self, pid, tol):
        """X on qubit 0 must land at flat index 4 (|100>). Aer is qubit-0-LSB internally, so this
        pins the reordering in ``_swap_endianness``."""
        spec = CircuitSpec(3, 0, 0, (Gate("X", (0,)),))
        sv = _sv(pid, spec, np.empty(0), np.empty(0))
        assert int(np.argmax(np.abs(sv))) == 4

    def test_injection_matches_readout(self, pid, tol):
        """Split a circuit in two: running the tail from the head's statevector must equal running
        the whole thing. Exercises ``set_statevector`` — which the MPS core reaches through a
        dense-to-MPS conversion, a genuinely different path from the statevector core."""
        prep = CircuitSpec(
            3, 0, 0, (Gate("RY", (0,), (Constant(0.7),)), Gate("CNOT", (0, 1)))
        )
        rest = CircuitSpec(
            3, 0, 0, (Gate("RY", (2,), (Constant(0.3),)), Gate("CNOT", (1, 2)))
        )
        full = CircuitSpec(3, 0, 0, prep.gates + rest.gates)
        psi = _sv(pid, prep, np.empty(0), np.empty(0))
        direct = _sv(pid, full, np.empty(0), np.empty(0))
        injected = _sv(pid, rest, np.empty(0), np.empty(0), initial_state=psi)
        assert np.max(np.abs(np.abs(direct) ** 2 - np.abs(injected) ** 2)) < tol

    def test_unnormalized_initial_state_raises(self, ir, weights, inputs, pid, tol):
        compiled = get(pid).compile(ir, all_z(3))
        with pytest.raises(ValueError, match="normalized"):
            compiled.statevector(weights, inputs, initial_state=np.ones(8))

    def test_batched_statevector_shape(self, ir, weights, pid, tol):
        x = np.random.default_rng(2).uniform(-1, 1, (4, 3))
        sv = np.asarray(get(pid).compile(ir, all_z(3)).statevector(weights, x))
        assert sv.shape == (4, 8)


@pytest.mark.parametrize("pid,tol", METHODS)
class TestZeroStateProbability:
    """The fidelity-kernel primitive behind QuantumKernel / QSVC / QSVR."""

    def test_matches_numpy(self, ir, weights, inputs, pid, tol):
        got = get(pid).compile(ir, all_z(3)).zero_state_probability(weights, inputs)
        ref = get("numpy").compile(ir, all_z(3)).zero_state_probability(weights, inputs)
        assert abs(got - ref) < tol

    def test_hadamard_gives_half(self, pid, tol):
        spec = CircuitSpec(2, 0, 0, (Gate("H", (0,)),))
        compiled = get(pid).compile(spec, all_z(2))
        prob = compiled.zero_state_probability(np.empty(0), np.empty(0))
        assert abs(prob - 0.5) < tol


class TestCapabilities:
    def test_parameter_shift_gradient_method(self):
        assert QiskitAerBackend().gradient_method == "parameter_shift"

    def test_supports_full_gate_set(self):
        assert QiskitAerBackend().capabilities.supported_gates == ALL_GATES

    def test_amplitude_encoding_adjoint_unsupported(self):
        """Matches the plain Qiskit backend: prepare_state() is not invertible here."""
        caps = QiskitAerBackend().capabilities
        assert "AmplitudeEncoding" not in caps.supported_adjoint_gates
        assert "RX" in caps.supported_adjoint_gates

    def test_no_batching_or_shots(self):
        caps = QiskitAerBackend().capabilities
        assert caps.supports_batching is False
        assert caps.supports_shots is False
        assert caps.max_qubits is None

    def test_no_native_torch(self, ir):
        compiled = QiskitAerBackend().compile(ir, all_z(3))
        assert not hasattr(compiled, "run_torch")


class TestConfig:
    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown Aer method"):
            QiskitAerConfig(method="tensor_network")

    def test_both_methods_accepted(self):
        assert QiskitAerConfig(method="statevector").method == "statevector"
        assert QiskitAerConfig("matrix_product_state").method == "matrix_product_state"

    def test_name_tracks_method(self):
        """The two provider ids must be self-describing, since benchmark output is keyed on
        ``backend.name`` and the two methods are the whole point of this backend."""
        assert QiskitAerBackend().name == "qiskit-aer"
        assert (
            QiskitAerConfig(method="matrix_product_state").make_backend().name
            == "qiskit-aer-mps"
        )
        assert QiskitAerBackend(name="custom").name == "custom"

    def test_shots_not_wired_yet(self, ir):
        """``shots`` is accepted by the config but must fail loudly at compile rather than
        silently returning analytic values."""
        backend = QiskitAerConfig(shots=1024).make_backend()
        with pytest.raises(NotImplementedError, match="finite-shot"):
            backend.compile(ir, all_z(3))

    def test_polymorphic_provider_via_config(self, ir, weights, inputs):
        circuit = Circuit(ir, all_z(3), provider=QiskitAerConfig(), interface="numpy")
        assert np.asarray(circuit(weights, inputs)).shape == (3,)

    def test_mps_config_as_provider(self, ir, weights, inputs):
        mps = QiskitAerConfig(method="matrix_product_state")
        got = np.asarray(Circuit(ir, all_z(3), mps, "numpy")(weights, inputs))
        ref = np.asarray(Circuit(ir, all_z(3), "numpy", "numpy")(weights, inputs))
        assert np.max(np.abs(got - ref)) < MPS_TOL


@pytest.mark.parametrize("pid,tol", METHODS)
class TestASpecMayDeclareAParameterItNeverUses:
    """`assign_parameters` is strict about extras, so binding the full declared
    list would raise `CircuitError: Cannot bind parameters (w1) not present in
    the circuit` here, for a spec every other backend runs.

    Every execution path binds, and each binds a different circuit object:
    `run` the expectation-value copy, `statevector` the save-statevector copy,
    and an injected initial state a third, composed one. The parameter set is
    recorded once from the circuit all three derive from, so each path binds
    only the parameters its circuit carries.
    """

    @staticmethod
    def _spec():
        # w1 is declared and never referenced by a gate.
        return CircuitSpec(
            n_qubits=2,
            n_weights=2,
            n_inputs=1,
            gates=(
                Gate("RY", (0,), (WeightRef(0),)),
                Gate("RX", (1,), (InputRef(0),)),
            ),
        )

    def test_the_expectation_value_path_binds(self, pid, tol):
        spec = self._spec()
        weights = np.array([0.4, 99.0])
        inputs = np.array([0.2])
        got = get(pid).compile(spec, all_z(2)).run(weights, inputs)
        ref = get("numpy").compile(spec, all_z(2)).run(weights, inputs)
        np.testing.assert_allclose(got, ref, atol=tol)

    def test_the_statevector_path_binds(self, pid, tol):
        spec = self._spec()
        weights = np.array([0.4, 99.0])
        inputs = np.array([0.2])
        got = np.asarray(get(pid).compile(spec, all_z(2)).statevector(weights, inputs))
        ref = np.asarray(
            get("numpy").compile(spec, all_z(2)).statevector(weights, inputs)
        )
        np.testing.assert_allclose(np.abs(got) ** 2, np.abs(ref) ** 2, atol=tol)

    def test_the_injected_initial_state_path_binds(self, pid, tol):
        spec = self._spec()
        weights = np.array([0.4, 99.0])
        inputs = np.array([0.2])
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0
        got = np.asarray(
            get(pid)
            .compile(spec, all_z(2))
            .statevector(weights, inputs, initial_state=psi)
        )
        ref = np.asarray(
            get("numpy").compile(spec, all_z(2)).statevector(weights, inputs)
        )
        np.testing.assert_allclose(np.abs(got) ** 2, np.abs(ref) ** 2, atol=tol)

    def test_the_fidelity_primitive_binds(self, pid, tol):
        spec = self._spec()
        weights = np.array([0.4, 99.0])
        inputs = np.array([0.2])
        got = get(pid).compile(spec, all_z(2)).zero_state_probability(weights, inputs)
        ref = get("numpy").compile(spec, all_z(2)).zero_state_probability(weights, inputs)
        assert abs(got - ref) < tol
