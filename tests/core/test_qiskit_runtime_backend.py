"""Qiskit Runtime backend: tested against `FakeBrisbane` from
`qiskit_ibm_runtime.fake_provider`. Real-hardware tests would require IBM
credentials and consume compute time, so they're gated behind
`@pytest.mark.qpu` and skipped by default."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch


# qiskit / qiskit-ibm-runtime are optional extras; skip cleanly if absent.
pytest.importorskip("qiskit")
qiskit_ibm_runtime = pytest.importorskip("qiskit_ibm_runtime")
# FakeBrisbane runs on Aer. Without it (the qiskit extra leaves Aer out on
# Linux ARM with Python 3.14), qiskit-ibm-runtime falls back to BasicSimulator,
# which cannot hold the device's 127 qubits.
pytest.importorskip("qiskit_aer")
from qiskit_ibm_runtime.fake_provider import FakeBrisbane  # noqa: E402

from eigenkit import Circuit as _Circuit  # noqa: E402
from eigenkit import (  # noqa: E402
    CostWarning,
    Encoding,
    Observable,
    QiskitRuntimeBackend,
    QiskitRuntimeConfig,
    StrongVariational,
    Variational,
    Z,
    all_z,
)


# These tests build torch circuits on the Qiskit Runtime backend, while the
# top-level Circuit defaults to the numpy interface. Provider is passed
# explicitly per test.
def Circuit(*args, **kwargs):
    kwargs.setdefault("interface", "torch")
    return _Circuit(*args, **kwargs)


@pytest.fixture
def fake_backend():
    """A noisy IBM Eagle simulation. ~127 qubits available; we use 2-3."""
    return FakeBrisbane()


@pytest.fixture
def small_circuit():
    return [Encoding(in_features=2), Variational(depth=1, rotation="RY")]


class TestCompile:
    def test_compiles_with_fake_backend(self, fake_backend, small_circuit):
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=1024),
        )
        info = layer._inner.compiled.dry_run_info()
        assert info["backend_name"] == "fake_brisbane"
        assert info["is_fake"] is True
        assert info["n_logical_qubits"] == 2
        assert info["n_observables"] == 1

    def test_transpilation_uses_native_gates(self, fake_backend, small_circuit):
        """After transpile, gate counts should reflect IBM Eagle's native
        basis: rz, sx, ecr (and possibly id/x/measure)."""
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_instance=fake_backend, shots=1024, optimization_level=1
            ),
        )
        info = layer._inner.compiled.dry_run_info()
        gates = info["transpiled_gate_counts"]
        # All gate names should be in the IBM Eagle native set.
        eagle_gates = {"rz", "sx", "ecr", "x", "id", "measure", "delay", "barrier"}
        assert set(gates).issubset(eagle_gates), gates

    def test_optimization_level(self, fake_backend, small_circuit):
        """opt_level=3 should produce ≤ depth of opt_level=0 (in expectation)."""
        l0 = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_instance=fake_backend,
                shots=1024,
                optimization_level=0,
            ),
        )
        l3 = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_instance=fake_backend,
                shots=1024,
                optimization_level=3,
            ),
        )
        d0 = l0._inner.compiled.dry_run_info()["transpiled_depth"]
        d3 = l3._inner.compiled.dry_run_info()["transpiled_depth"]
        assert d3 <= d0


class TestForward:
    def test_returns_tensor_in_unit_interval(self, fake_backend, small_circuit):
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=4096),
        )
        x = torch.tensor([0.3, -0.5], dtype=torch.float64)
        y = layer(x)
        assert isinstance(y, torch.Tensor)
        assert -1.0 <= y.item() <= 1.0

    def test_multi_observable_vector(self, fake_backend):
        layer = Circuit(
            blocks=[Encoding(in_features=2), StrongVariational(depth=1)],
            observables=all_z(2),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=4096),
        )
        x = torch.tensor([0.3, -0.5], dtype=torch.float64)
        y = layer(x)
        assert y.shape == (2,)


def _paid_looking_backend():
    """A noiseless Aer simulator defined outside qiskit_aer, so the backend
    treats it as a paid device: it warns like one but runs locally."""
    from qiskit_aer import AerSimulator

    class PaidLookingBackend(AerSimulator):
        pass

    return PaidLookingBackend()


class TestGradientPolicy:
    def test_gradient_method_is_none(self):
        backend = QiskitRuntimeBackend()
        assert backend.gradient_method == "none"

    def test_gradient_method_follows_enable_gradients(self):
        backend = QiskitRuntimeBackend(QiskitRuntimeConfig(enable_gradients=True))
        assert backend.gradient_method == "parameter_shift"

    def test_backward_raises(self, fake_backend, small_circuit):
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=1024),
        )
        x = torch.tensor([0.3, -0.5], dtype=torch.float64)
        y = layer(x)
        with pytest.raises(
            RuntimeError, match="do not support automatic differentiation"
        ):
            y.backward()

    def test_numpy_jacobian_is_refused(self, fake_backend, small_circuit):
        """`jacobian()` obeys the same policy as `.backward()`: unasked, it
        would run parameter-shift on hardware — 2 * n_weights paid jobs with
        no warning."""
        circuit = _Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=1024),
        )
        with pytest.raises(RuntimeError, match="enable_gradients=True"):
            circuit.jacobian(np.zeros(circuit.spec.n_weights), np.zeros(2))

    def test_fake_backend_opt_in_does_not_warn(self, fake_backend, small_circuit):
        """A local fake device costs nothing, with gradients or without."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", CostWarning)
            Circuit(
                blocks=small_circuit,
                observables=Z(0),
                provider=QiskitRuntimeConfig(
                    backend_instance=fake_backend, shots=1024, enable_gradients=True
                ),
            )

    def test_opt_in_warns_with_the_backward_job_count(self, small_circuit):
        """Two RY weights -> 4 shift evaluations, x 2 estimator jobs (one per
        observable) = 8 jobs per backward pass."""
        with pytest.warns(CostWarning, match="gradients are enabled") as record:
            Circuit(
                blocks=small_circuit,
                observables=all_z(2),
                provider=QiskitRuntimeConfig(
                    backend_instance=_paid_looking_backend(),
                    shots=1024,
                    enable_gradients=True,
                ),
            )
        messages = [str(w.message) for w in record if w.category is CostWarning]
        assert any("at least 8 PUBs per sample" in m for m in messages)

    def test_no_gradient_warning_without_opt_in(self, small_circuit):
        with pytest.warns(CostWarning) as record:
            Circuit(
                blocks=small_circuit,
                observables=all_z(2),
                provider=QiskitRuntimeConfig(
                    backend_instance=_paid_looking_backend(), shots=1024
                ),
            )
        assert not any("gradients are enabled" in str(w.message) for w in record)

    def test_opt_in_backward_matches_the_exact_gradient(self, small_circuit):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            layer = Circuit(
                blocks=small_circuit,
                observables=Z(0),
                provider=QiskitRuntimeConfig(
                    backend_instance=_paid_looking_backend(),
                    shots=20000,
                    enable_gradients=True,
                ),
            )
        x = torch.tensor([0.3, -0.5], dtype=torch.float64)
        layer(x).backward()
        weights = next(layer.parameters())

        exact = _Circuit(blocks=small_circuit, observables=Z(0), provider="numpy")
        expected = exact.jacobian(weights.detach().numpy(), x.numpy())
        # Shot noise only: each entry is half a difference of two estimates at
        # 20000 shots, sigma ~ 0.007.
        np.testing.assert_allclose(weights.grad.numpy(), expected, atol=0.05)

    def test_opt_in_chain_rule_recompiles_quietly(self):
        """Trainable feature scales take the chain-rule path, which compiles
        one shifted circuit per gate. The first compile warned; the shifted
        copies must not warn again."""
        from eigenkit import Reuploading

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            circuit = _Circuit(
                blocks=[Reuploading(in_features=2, depth=1, trainable_scale=True)],
                observables=Z(0),
                provider=QiskitRuntimeConfig(
                    backend_instance=_paid_looking_backend(),
                    shots=256,
                    enable_gradients=True,
                ),
            )
        assert circuit._has_unsafe_gates
        with warnings.catch_warnings():
            warnings.simplefilter("error", CostWarning)
            grad = circuit.jacobian(np.full(circuit.spec.n_weights, 0.1), np.zeros(2))
        assert grad.shape == (circuit.spec.n_weights,)

    def test_shifted_copies_stay_on_the_resolved_device(self, monkeypatch):
        """`"least_busy"` is asked once: the chain-rule copies run where the
        circuit itself was compiled, transpiled onto the same qubits."""
        from eigenkit import Reuploading
        from eigenkit.core.backends import qiskit_runtime_backend as runtime

        class Service:
            def least_busy(self, **_):
                return FakeBrisbane()  # a different device object every time

        monkeypatch.setattr(runtime, "_resolve_service", lambda config: Service())
        circuit = _Circuit(
            blocks=[Reuploading(in_features=2, depth=1, trainable_scale=True)],
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_name="least_busy", shots=64, enable_gradients=True
            ),
        )
        circuit.jacobian(np.full(circuit.spec.n_weights, 0.1), np.zeros(2))

        base = circuit.compiled
        layout = base._transpiled_circuit.layout.initial_index_layout(
            filter_ancillas=True
        )
        assert circuit._chain_rule_cache
        for shifted in circuit._chain_rule_cache.values():
            assert shifted._backend is base._backend
            assert (
                shifted._transpiled_circuit.layout.initial_index_layout(
                    filter_ancillas=True
                )
                == layout
            )

    def test_frozen_weights_skip_their_shifts(
        self, monkeypatch, fake_backend, small_circuit
    ):
        """With the quantum weights frozen, backward shifts only the inputs;
        the weight shifts would be paid jobs whose result is thrown away."""
        from eigenkit.core import circuit as circuit_module

        def weight_shifts(*args, **kwargs):
            raise AssertionError("weight shifts ran for frozen weights")

        monkeypatch.setattr(circuit_module, "parameter_shift_jacobian", weight_shifts)
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_instance=fake_backend, shots=64, enable_gradients=True
            ),
        )
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        x = torch.tensor([0.3, -0.5], dtype=torch.float64, requires_grad=True)
        layer(x).backward()
        assert x.grad is not None

    def test_no_gradient_warning_without_weights(self):
        """A kernel-style circuit has no weight pass to put a price on."""
        with pytest.warns(CostWarning) as record:
            Circuit(
                blocks=[Encoding(in_features=2)],
                observables=Z(0),
                provider=QiskitRuntimeConfig(
                    backend_instance=_paid_looking_backend(),
                    shots=64,
                    enable_gradients=True,
                ),
            )
        assert not any("gradients are enabled" in str(w.message) for w in record)


class TestSubmit:
    def test_submit_returns_handle(self, fake_backend, small_circuit):
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=1024),
        )
        x = torch.tensor([0.3, -0.5], dtype=torch.float64)
        handle = layer.submit(x)
        assert hasattr(handle, "result")
        assert hasattr(handle, "status")
        assert hasattr(handle, "job_id")
        result = handle.result()
        assert result.shape == (1,)

    def test_submit_matches_run(self, fake_backend, small_circuit):
        """Both paths should give comparable values (same circuit, same
        shots — within Monte-Carlo noise on the FakeBackend)."""
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=8192),
        )
        x = torch.tensor([0.3, -0.5], dtype=torch.float64)
        a = layer(x).item()
        b = float(layer.submit(x).result()[0])
        assert abs(a - b) < 0.15  # noise bound; 8192 shots gives ~0.01 stderr


class TestValidation:
    def test_zero_shots_rejected(self, fake_backend, small_circuit):
        with pytest.raises(ValueError, match="shots > 0"):
            Circuit(
                blocks=small_circuit,
                observables=Z(0),
                provider=QiskitRuntimeConfig(
                    backend_instance=fake_backend,
                    shots=0,
                ),
            )


class TestNoCostWarningOnFakeBackend:
    def test_silent_for_fakes(self, fake_backend, small_circuit):
        """Compiling against a FakeBackend should not emit CostWarning —
        only real IBM hardware does."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", CostWarning)
            Circuit(
                blocks=small_circuit,
                observables=Z(0),
                provider=QiskitRuntimeConfig(
                    backend_instance=fake_backend,
                    shots=1024,
                ),
            )


class TestServiceAuthChain:
    """`_resolve_backend` builds a `QiskitRuntimeService` when
    `backend_instance` is None. Verify the token / channel / instance
    arguments fall through the documented priority order without
    actually contacting IBM."""

    def test_explicit_token_forwarded(self, fake_backend, small_circuit, monkeypatch):
        captured = self._patch_service(monkeypatch, fake_backend)

        Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_name="ibm_brisbane",
                token="explicit-token",
                channel="ibm_quantum_platform",
                shots=1024,
            ),
        )
        assert captured["kwargs"]["token"] == "explicit-token"
        assert captured["kwargs"]["channel"] == "ibm_quantum_platform"
        assert captured["backend_name"] == "ibm_brisbane"

    def test_env_var_token(
        self,
        fake_backend,
        small_circuit,
        monkeypatch,
    ):
        captured = self._patch_service(monkeypatch, fake_backend)
        monkeypatch.setenv("QISKIT_IBM_TOKEN", "env-token")

        Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_name="ibm_brisbane",
                shots=1024,
            ),
        )
        assert captured["kwargs"]["token"] == "env-token"

    def test_no_token_when_the_variable_is_unset(
        self,
        fake_backend,
        small_circuit,
        monkeypatch,
    ):
        captured = self._patch_service(monkeypatch, fake_backend)
        monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)

        Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_name="ibm_brisbane",
                shots=1024,
            ),
        )
        # No token forwarded -- the SDK falls back to its saved-account file.
        assert "token" not in captured["kwargs"]

    def test_instance_forwarded(self, fake_backend, small_circuit, monkeypatch):
        captured = self._patch_service(monkeypatch, fake_backend)

        Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_name="ibm_brisbane",
                token="t",
                instance="my-hub/my-group/my-project",
                shots=1024,
            ),
        )
        assert captured["kwargs"]["instance"] == "my-hub/my-group/my-project"

    @staticmethod
    def _patch_service(monkeypatch, fake_backend):
        """Stub out `QiskitRuntimeService` so resolution is offline.

        Returns a dict that captures the kwargs the service was built
        with and the backend name requested.
        """
        captured: dict = {"kwargs": None, "backend_name": None}

        class FakeService:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def backend(self, name):
                captured["backend_name"] = name
                return fake_backend

        # Patch at the module the runtime backend imports from.
        import qiskit_ibm_runtime

        monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", FakeService)
        return captured


class TestSamplerPrimitive:
    """`primitive="sampler"` evaluates the same expectation values through
    `SamplerV2` and raw bitstring counts. Run against a noiseless
    `AerSimulator` so the only deviation from the analytic value is shot
    noise, which is what the tolerances below are sized for."""

    @staticmethod
    def _noiseless():
        from qiskit_aer import AerSimulator

        return AerSimulator()

    @staticmethod
    def _sampler_circuit(observables, backend, **kwargs):
        # AerSimulator is local and does not warn; the filter keeps these
        # tests independent of that.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            return _Circuit(
                blocks=[Encoding(in_features=2), Variational(depth=1, rotation="RY")],
                observables=observables,
                provider=QiskitRuntimeConfig(
                    backend_instance=backend, shots=20000, primitive="sampler", **kwargs
                ),
            )

    def test_unknown_primitive_rejected(self):
        with pytest.raises(ValueError, match="estimator.*sampler"):
            QiskitRuntimeConfig(primitive="bogus")

    def test_dry_run_reports_the_primitive_and_its_job_count(self, fake_backend):
        sampler = self._sampler_circuit(all_z(2), fake_backend)
        info = sampler.compiled.dry_run_info()
        assert info["primitive"] == "sampler"
        # all_z(2) is two Z terms on distinct qubits: one measurement circuit.
        assert info["n_observables"] == 2
        assert info["n_measurement_groups"] == 1
        assert info["n_circuit_runs_per_call"] == 1

    def test_estimator_stays_one_job_per_observable(self, fake_backend):
        estimator = _Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1, rotation="RY")],
            observables=all_z(2),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=1024),
        )
        info = estimator.compiled.dry_run_info()
        assert info["primitive"] == "estimator"
        assert info["n_circuit_runs_per_call"] == info["n_observables"] == 2

    def test_matches_the_exact_simulator(self):
        """Non-commuting observables across all three Paulis, so the basis
        rotation and the grouping both have to be right."""
        observables = [
            Observable.from_string("1.0 Z0 Z1"),
            Observable.from_string("0.5 X0 - 2.0 Z1"),
            Observable.from_string("1.0 Y0 Y1"),
        ]
        blocks = [Encoding(in_features=2), Variational(depth=1, rotation="RY")]
        exact = _Circuit(blocks=blocks, observables=observables, provider="numpy")
        rng = np.random.default_rng(0)
        weights = rng.normal(size=exact.spec.n_weights)
        inputs = rng.normal(size=2)

        sampler = self._sampler_circuit(observables, self._noiseless())
        np.testing.assert_allclose(
            sampler(weights, inputs),
            exact(weights, inputs),
            rtol=0,
            atol=0.1,  # 20k shots; coefficients up to 2.0
        )

    def test_grouping_off_costs_more_jobs_and_agrees(self):
        observables = [Observable.from_string("1.0 Z0 + 1.0 Z1")]
        blocks = [Encoding(in_features=2), Variational(depth=1, rotation="RY")]
        exact = _Circuit(blocks=blocks, observables=observables, provider="numpy")
        rng = np.random.default_rng(1)
        weights = rng.normal(size=exact.spec.n_weights)
        inputs = rng.normal(size=2)

        backend = self._noiseless()
        grouped = self._sampler_circuit(observables, backend)
        ungrouped = self._sampler_circuit(observables, backend, group_commuting=False)
        assert grouped.compiled.dry_run_info()["n_measurement_groups"] == 1
        assert ungrouped.compiled.dry_run_info()["n_measurement_groups"] == 2
        for circuit in (grouped, ungrouped):
            np.testing.assert_allclose(
                circuit(weights, inputs), exact(weights, inputs), rtol=0, atol=0.1
            )

    def test_batch_broadcasts_without_extra_jobs(self):
        blocks = [Encoding(in_features=2), Variational(depth=1, rotation="RY")]
        exact = _Circuit(blocks=blocks, observables=all_z(2), provider="numpy")
        rng = np.random.default_rng(2)
        weights = rng.normal(size=exact.spec.n_weights)
        inputs = rng.normal(size=(4, 2))

        sampler = self._sampler_circuit(all_z(2), self._noiseless())
        values = sampler(weights, inputs)
        assert values.shape == (4, 2)
        np.testing.assert_allclose(values, exact(weights, inputs), rtol=0, atol=0.1)

    def test_submit_returns_expectation_values(self):
        sampler = self._sampler_circuit(all_z(2), self._noiseless())
        rng = np.random.default_rng(3)
        weights = rng.normal(size=sampler.spec.n_weights)
        handle = sampler.submit(weights, rng.normal(size=2))
        assert hasattr(handle, "status") and hasattr(handle, "job_id")
        assert handle.result().shape == (2,)

    def test_sample_returns_a_histogram(self):
        sampler = self._sampler_circuit(all_z(2), self._noiseless())
        rng = np.random.default_rng(4)
        counts = sampler.sample(
            rng.normal(size=sampler.spec.n_weights), rng.normal(size=2)
        )
        assert sum(counts.values()) == 20000
        assert all(len(key) == 2 and set(key) <= {"0", "1"} for key in counts)

    def test_sample_works_under_the_estimator_too(self, fake_backend):
        """`sample()` is a sampler operation whichever primitive computes the
        expectation values; EstimatorV2 has no bitstrings to give."""
        estimator = _Circuit(
            blocks=[Encoding(in_features=2), Variational(depth=1, rotation="RY")],
            observables=all_z(2),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=512),
        )
        counts = estimator.sample(np.zeros(estimator.spec.n_weights), np.zeros(2))
        assert sum(counts.values()) == 512

    def test_probabilities_track_the_exact_distribution(self):
        blocks = [Encoding(in_features=2), Variational(depth=1, rotation="RY")]
        exact = _Circuit(blocks=blocks, observables=all_z(2), provider="numpy")
        rng = np.random.default_rng(5)
        weights = rng.normal(size=exact.spec.n_weights)
        inputs = rng.normal(size=2)

        sampler = self._sampler_circuit(all_z(2), self._noiseless())
        shot_probabilities = sampler.probabilities(weights, inputs)
        assert shot_probabilities.shape == (4,)
        assert shot_probabilities.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(
            shot_probabilities, exact.probabilities(weights, inputs), rtol=0, atol=0.05
        )

    def test_cost_warning_names_the_primitive_and_its_job_count(self):
        """The warning has to quote the sampler's job count, not the
        estimator's — they differ, and it is the number being paid for."""
        from qiskit_aer import AerSimulator

        class PaidLookingBackend(AerSimulator):
            """Defined outside qiskit_aer, so it is not recognised as local."""

        with pytest.warns(CostWarning, match="sampler.*one job of 1 PUBs per call"):
            _Circuit(
                blocks=[Encoding(in_features=2), Variational(depth=1, rotation="RY")],
                observables=all_z(2),
                provider=QiskitRuntimeConfig(
                    backend_instance=PaidLookingBackend(),
                    shots=1024,
                    primitive="sampler",
                ),
            )

    def test_aer_is_local_and_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", CostWarning)
            circuit = _Circuit(
                blocks=[Encoding(in_features=2), Variational(depth=1, rotation="RY")],
                observables=all_z(2),
                provider=QiskitRuntimeConfig(
                    backend_instance=self._noiseless(), shots=1024, primitive="sampler"
                ),
            )
        assert circuit.dry_run_info()["is_fake"] is True


class TestSeedTranspiler:
    """Without a seed, SABRE layout and routing pick a different answer on
    every run, so two compilations of the same circuit are not comparable —
    which matters most on the circuits that need routing in the first place."""

    @staticmethod
    def _gates(backend, seed):
        layer = Circuit(
            blocks=[Encoding(in_features=4), StrongVariational(depth=2)],
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_instance=backend,
                shots=1024,
                optimization_level=3,
                seed_transpiler=seed,
            ),
        )
        info = layer._inner.compiled.dry_run_info()
        return info["transpiled_gate_counts"].get("ecr", 0), info["transpiled_depth"]

    def test_same_seed_is_reproducible(self, fake_backend):
        assert self._gates(fake_backend, 11) == self._gates(fake_backend, 11)

    def test_seed_is_reported(self, fake_backend, small_circuit):
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_instance=fake_backend, shots=1024, seed_transpiler=7
            ),
        )
        assert layer._inner.compiled.dry_run_info()["seed_transpiler"] == 7

    def test_default_is_unseeded(self, fake_backend, small_circuit):
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(backend_instance=fake_backend, shots=1024),
        )
        assert layer._inner.compiled.dry_run_info()["seed_transpiler"] is None


class TestJobHandleCounts:
    """One job should yield both the raw counts and the job id: keeping them
    together is what lets a run be re-scored offline instead of re-paid for."""

    @staticmethod
    def _circuit(backend, primitive):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            return _Circuit(
                blocks=[Encoding(in_features=2), Variational(depth=1, rotation="RY")],
                observables=all_z(2),
                provider=QiskitRuntimeConfig(
                    backend_instance=backend, shots=4096, primitive=primitive
                ),
            )

    def test_sampler_handle_exposes_the_raw_counts(self, fake_backend):
        circuit = self._circuit(fake_backend, "sampler")
        handle = circuit.submit(np.zeros(circuit.spec.n_weights), np.zeros(2))
        histograms = handle.counts()
        # all_z(2) is one measurement group, so one histogram.
        assert len(histograms) == 1
        assert sum(histograms[0].values()) == 4096
        assert all(len(key) == 2 for key in histograms[0])

    def test_counts_and_result_come_from_one_job(self, fake_backend):
        """`result()` must be `counts()` reduced, not a second submission."""
        circuit = self._circuit(fake_backend, "sampler")
        handle = circuit.submit(np.zeros(circuit.spec.n_weights), np.zeros(2))
        histograms = handle.counts()
        expected = np.array(
            [
                sum(shots * (1 - 2 * int(key[q])) for key, shots in histograms[0].items())
                / sum(histograms[0].values())
                for q in range(2)
            ]
        )
        np.testing.assert_allclose(handle.result(), expected, atol=1e-12)

    def test_estimator_handle_says_it_has_no_counts(self, fake_backend):
        circuit = self._circuit(fake_backend, "estimator")
        handle = circuit.submit(np.zeros(circuit.spec.n_weights), np.zeros(2))
        with pytest.raises(NotImplementedError, match="primitive='sampler'"):
            handle.counts()


# ---- markers for opt-in real-hardware integration tests ----


@pytest.mark.qpu
class TestRealQpuIntegration:
    """These run only when explicitly enabled (e.g. `pytest -m qpu`).
    They consume IBM Quantum compute time and require credentials in
    `QISKIT_IBM_TOKEN` or a saved account."""

    def test_brisbane_smoke(self, small_circuit):
        layer = Circuit(
            blocks=small_circuit,
            observables=Z(0),
            provider=QiskitRuntimeConfig(
                backend_name="ibm_brisbane",
                shots=1024,
                optimization_level=1,
            ),
        )
        x = torch.tensor([0.3, -0.5], dtype=torch.float64)
        y = layer(x)
        assert -1.0 <= y.item() <= 1.0


class TestQuantumKernelOnRuntime:
    """Quantum kernels run on Qiskit Runtime through the inversion circuit:
    one SamplerV2 job per kernel matrix, all-zeros share read from counts."""

    X = np.random.default_rng(0).uniform(0, np.pi, size=(6, 2))

    @staticmethod
    def _kernel(backend, **kwargs):
        from eigenkit import QuantumKernel

        return QuantumKernel(
            n_features=2,
            feature_map="AngleCNOT",
            reps=2,
            provider=QiskitRuntimeConfig(
                backend_instance=backend, primitive="sampler", seed_transpiler=0, **kwargs
            ),
        )

    def test_matches_the_exact_kernel_within_shot_noise(self):
        from qiskit_aer import AerSimulator

        from eigenkit import QuantumKernel

        exact = QuantumKernel(n_features=2, feature_map="AngleCNOT", reps=2)
        sampled = self._kernel(AerSimulator(), shots=20000)
        np.testing.assert_allclose(
            sampled.compute_kernel_matrix(self.X),
            exact.compute_kernel_matrix(self.X),
            atol=0.03,
        )
        np.testing.assert_allclose(
            sampled.compute_kernel_matrix(self.X, self.X[:3]),
            exact.compute_kernel_matrix(self.X, self.X[:3]),
            atol=0.03,
        )

    def test_a_kernel_matrix_is_one_job_and_one_transpile(self, monkeypatch):
        from qiskit_aer import AerSimulator

        kernel = self._kernel(AerSimulator(), shots=1000)
        compiled = kernel.circuit.compiled
        runs = []
        real = compiled._sampler

        def counting():
            sampler = real()
            original = sampler.run

            def run(pubs, *args, **kwargs):
                runs.append([len(np.atleast_2d(p[1])) for p in pubs])
                return original(pubs, *args, **kwargs)

            sampler.run = run
            return sampler

        monkeypatch.setattr(compiled, "_sampler", counting)
        kernel.compute_kernel_matrix(self.X)
        assert runs == [[15]], "one PUB with the 6 * 5 / 2 pairs above the diagonal"
        assert kernel._forward is None, "the forward map is never compiled on hardware"

    def test_qsvc_fits_on_a_noisy_fake_device(self, fake_backend):
        from eigenkit import QSVC

        y = (self.X[:, 0] > np.pi / 2).astype(int)
        model = QSVC(kernel=self._kernel(fake_backend, shots=2000)).fit(self.X, y)
        assert model.predict(self.X).shape == (6,)

    def test_submit_and_collect(self):
        from qiskit_aer import AerSimulator

        kernel = self._kernel(AerSimulator(), shots=2000)
        job = kernel.submit_kernel_matrix(self.X[:4])
        assert job.n_circuits == 6
        K = job.result()
        assert K.shape == (4, 4)
        np.testing.assert_allclose(np.diag(K), 1.0)

    def test_submit_rejects_a_matrix_with_nothing_to_measure(self):
        from qiskit_aer import AerSimulator

        kernel = self._kernel(AerSimulator(), shots=100)
        with pytest.raises(ValueError, match="no kernel entries"):
            kernel.submit_kernel_matrix(self.X[:1])

    def test_retrieve_needs_a_service(self):
        from qiskit_aer import AerSimulator

        kernel = self._kernel(AerSimulator(), shots=100)
        with pytest.raises(NotImplementedError, match="job history"):
            kernel.retrieve_kernel_matrix("some-job-id", n_x=4)

    def test_retrieve_checks_the_job_id_first(self):
        from qiskit_aer import AerSimulator

        kernel = self._kernel(AerSimulator(), shots=100)
        with pytest.raises(ValueError, match="job_id"):
            kernel.circuit.compiled.retrieve_zero_state_probabilities("not a job id")

    def test_retrieve_through_the_service(self, tmp_path):
        from qiskit_aer import AerSimulator

        kernel = self._kernel(AerSimulator(), shots=2000)
        submitted = kernel.submit_kernel_matrix(self.X[:4])
        path = submitted.save(tmp_path / "job")
        jobs = {submitted.job_id(): submitted._handle._job}

        class Service:
            def job(self, job_id):
                return jobs[job_id]

        # A backend looked up through QiskitRuntimeService carries it as
        # `backend.service`; a local simulator has none, so give it one.
        kernel.circuit.compiled._backend.service = Service()
        restored = kernel.retrieve_kernel_matrix(path)
        np.testing.assert_allclose(restored.result(), submitted.result())

    def test_resources_report(self):
        from qiskit_ibm_runtime.fake_provider import FakeFez

        kernel = self._kernel(FakeFez(), shots=100, optimization_level=3)
        report = kernel.circuit.resources()
        assert report["two_qubit_gate_counts"].keys() <= {"cz"}
        assert report["n_2q_gates"] >= report["n_2q_gates_unrouted"] >= 1
        assert report["routing_overhead_2q"] == (
            report["n_2q_gates"] - report["n_2q_gates_unrouted"]
        )
        assert len(report["physical_qubits"]) == 2
        assert 0.0 < report["estimated_success_probability"] < 1.0
        assert kernel.circuit.transpiled_circuit().num_qubits == FakeFez().num_qubits

    def test_estimate_cost(self):
        from qiskit_aer import AerSimulator

        info = self._kernel(AerSimulator(), shots=1000).estimate_cost(self.X, self.X[:2])
        assert info["method"] == "inversion"
        assert info["n_circuits"] == 12 and info["n_jobs"] == 1
        assert info["total_shots"] == 12000


class TestBackendName:
    def test_a_device_is_required(self, small_circuit):
        with pytest.raises(ValueError, match="needs a device"):
            Circuit(
                blocks=small_circuit,
                observables=Z(0),
                provider=QiskitRuntimeConfig(shots=100),
            )

    def test_least_busy(self, fake_backend, small_circuit, monkeypatch):
        import qiskit_ibm_runtime

        captured = {}

        class FakeService:
            def __init__(self, **kwargs):
                pass

            def least_busy(self, **kwargs):
                captured.update(kwargs)
                return fake_backend

        monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", FakeService)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            Circuit(
                blocks=small_circuit,
                observables=Z(0),
                provider=QiskitRuntimeConfig(backend_name="least_busy", shots=100),
            )
        assert captured == {"operational": True, "simulator": False}
