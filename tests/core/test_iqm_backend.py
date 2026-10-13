"""IQM backend, tested against IQM's own fake backends.

`IQMFakeAdonis` and friends run locally on Aer with a modelled IQM noise
profile, so the whole path — transpilation to the native `r` / `cz` basis,
MOVE-gate routing on the Star architectures, submission, counts, reduction —
is exercised with no token and no network. Real-hardware tests would consume
IQM quota, so they are gated behind `@pytest.mark.qpu` and skipped by default.

Because the fake backends are *noisy* by construction, the numeric assertions
here check structure, sign and ordering rather than closeness to the analytic
value; the exactness of the counts arithmetic is pinned in `test_sampling.py`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


# qiskit and iqm-client are optional extras; skip cleanly if absent.
pytest.importorskip("qiskit")
pytest.importorskip("iqm.qiskit_iqm")
from iqm.qiskit_iqm import IQMFakeAdonis, IQMFakeDeneb  # noqa: E402

from eigenkit import (  # noqa: E402
    Circuit,
    CostWarning,
    Encoding,
    IQMBackend,
    IQMConfig,
    Observable,
    Variational,
    Z,
    all_z,
)
from eigenkit.core.builder import CircuitBuilder  # noqa: E402
from eigenkit.core.provider import registered_names  # noqa: E402


@pytest.fixture
def fake_backend():
    """A noise-modelled 5-qubit IQM Crystal device."""
    return IQMFakeAdonis()


@pytest.fixture
def blocks():
    return [Encoding(in_features=2), Variational(depth=1, rotation="RY")]


def _circuit(blocks, observables, backend, **kwargs):
    return Circuit(
        blocks=blocks,
        observables=observables,
        provider=IQMConfig(backend_instance=backend, shots=4000, **kwargs),
    )


class TestRegistration:
    def test_registered_when_iqm_is_installed(self):
        assert "iqm" in registered_names()

    def test_gradient_method_is_none(self):
        assert IQMBackend().gradient_method == "none"

    def test_capabilities_declare_shots(self):
        capabilities = IQMBackend().capabilities
        assert capabilities.supports_shots
        assert not capabilities.supports_batching
        assert "AmplitudeEncoding" not in capabilities.supported_gates


class TestConfigValidation:
    def test_default_config_is_constructible(self):
        """`provider="iqm"` builds `IQMBackend()`, which builds `IQMConfig()`.
        Refusing a target-less config here would make the registered name
        unusable rather than merely unresolvable."""
        assert IQMConfig().url is None

    def test_missing_target_is_caught_at_compile(self, monkeypatch, blocks):
        monkeypatch.delenv("IQM_SERVER_URL", raising=False)
        with pytest.raises(ValueError, match="no IQM target"):
            Circuit(blocks=blocks, observables=all_z(2), provider="iqm")

    def test_env_token_is_not_forwarded(self, monkeypatch, fake_backend, blocks):
        """iqm-client reads IQM_TOKEN itself and refuses a token supplied both
        ways, so an env-sourced token must be left for it to find."""
        captured = {}

        class FakeProvider:
            def __init__(self, url=None, **kwargs):
                captured.update(kwargs)

            def get_backend(self, **kwargs):
                return fake_backend

        monkeypatch.setenv("IQM_TOKEN", "from-the-environment")
        monkeypatch.setattr("iqm.qiskit_iqm.IQMProvider", FakeProvider)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            Circuit(
                blocks=blocks,
                observables=all_z(2),
                provider=IQMConfig(url="https://example.invalid/", shots=100),
            )
        assert "token" not in captured

    def test_explicit_token_is_forwarded(self, monkeypatch, fake_backend, blocks):
        captured = {}

        class FakeProvider:
            def __init__(self, url=None, **kwargs):
                captured.update(kwargs)

            def get_backend(self, **kwargs):
                return fake_backend

        monkeypatch.delenv("IQM_TOKEN", raising=False)
        monkeypatch.setattr("iqm.qiskit_iqm.IQMProvider", FakeProvider)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            Circuit(
                blocks=blocks,
                observables=all_z(2),
                provider=IQMConfig(
                    url="https://example.invalid/", shots=100, token="explicit"
                ),
            )
        assert captured["token"] == "explicit"

    def test_url_falls_back_to_the_environment(self, monkeypatch, fake_backend, blocks):
        """`url=None` has to pick up IQM_SERVER_URL. iqm-client takes the url
        positionally, so passing `None` through overrides its own env lookup and
        fails deep inside with an AttributeError instead of connecting."""
        captured = {}

        class FakeProvider:
            def __init__(self, url=None, **kwargs):
                captured["url"] = url

            def get_backend(self, **kwargs):
                return fake_backend

        monkeypatch.setenv("IQM_SERVER_URL", "https://example.invalid/")
        monkeypatch.setattr("iqm.qiskit_iqm.IQMProvider", FakeProvider)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            Circuit(
                blocks=blocks,
                observables=all_z(2),
                provider=IQMConfig(quantum_computer="garnet", shots=100),
            )
        assert captured["url"] == "https://example.invalid/"

    def test_zero_shots_rejected(self, fake_backend):
        with pytest.raises(ValueError, match="shots must be > 0"):
            IQMConfig(backend_instance=fake_backend, shots=0)

    def test_unknown_transpile_mode_rejected(self, fake_backend):
        with pytest.raises(ValueError, match="parametric.*per_call"):
            IQMConfig(backend_instance=fake_backend, transpile_mode="fast")

    def test_use_metrics_defaults_to_off(self):
        """iqm-client's own default, so callers who do not set the knob get
        iqm-client's behaviour."""
        assert IQMConfig().use_metrics is False


class TestSeedTranspiler:
    """Without a seed, SABRE layout and routing pick a different answer on
    every run, so two compilations of the same circuit are not comparable.
    Adonis is a 5-qubit star, so a CNOT ladder on all 5 has to be routed and
    the seed actually bites."""

    @pytest.fixture
    def routed(self):
        return [Encoding(in_features=5), Variational(depth=2, rotation="RY")]

    def _gates(self, backend, blocks, seed):
        circuit = _circuit(blocks, all_z(5), backend, seed_transpiler=seed)
        counts = circuit.compiled.dry_run_info()["transpiled_gate_counts"]
        # IQM keys the counts by measurement basis; all_z is one group.
        return sum(v.get("cz", 0) for v in counts.values())

    def test_same_seed_is_reproducible(self, fake_backend, routed):
        assert self._gates(fake_backend, routed, 11) == self._gates(
            fake_backend, routed, 11
        )

    def test_seed_is_reported(self, fake_backend, blocks):
        circuit = _circuit(blocks, all_z(2), fake_backend, seed_transpiler=7)
        assert circuit.compiled.dry_run_info()["seed_transpiler"] == 7

    def test_default_is_unseeded(self, fake_backend, blocks):
        circuit = _circuit(blocks, all_z(2), fake_backend)
        assert circuit.compiled.dry_run_info()["seed_transpiler"] is None


class TestUseMetrics:
    """`use_metrics` has to reach `IQMProvider.get_backend`: without it the
    target carries no error rates and Qiskit's layout pass is noise-blind."""

    def _patch_provider(self, monkeypatch, backend):
        captured = {}

        class FakeProvider:
            def __init__(self, url=None, **kwargs):
                captured["provider_kwargs"] = kwargs

            def get_backend(self, **kwargs):
                captured["get_backend_kwargs"] = kwargs
                return backend

        monkeypatch.setattr("iqm.qiskit_iqm.IQMProvider", FakeProvider)
        return captured

    @pytest.mark.parametrize("use_metrics", [True, False])
    def test_forwarded_to_get_backend(
        self, monkeypatch, fake_backend, blocks, use_metrics
    ):
        captured = self._patch_provider(monkeypatch, fake_backend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            Circuit(
                blocks=blocks,
                observables=all_z(2),
                provider=IQMConfig(
                    url="https://example.invalid/",
                    shots=100,
                    use_metrics=use_metrics,
                ),
            )
        assert captured["get_backend_kwargs"]["use_metrics"] is use_metrics


class TestCompile:
    def test_transpiles_to_the_native_basis(self, fake_backend, blocks):
        circuit = _circuit(blocks, all_z(2), fake_backend)
        info = circuit.compiled.dry_run_info()
        assert info["is_local"] is True
        assert info["n_logical_qubits"] == 2
        assert info["n_physical_qubits"] == 5
        # IQM Crystal devices are r + cz natively; nothing else may survive.
        native = {"r", "cz", "measure", "barrier", "id", "delay"}
        for counts in info["transpiled_gate_counts"].values():
            assert set(counts) <= native, counts

    def test_one_job_per_measurement_group(self, fake_backend, blocks):
        # all_z(2) is two commuting Z terms: one circuit resolves both.
        grouped = _circuit(blocks, all_z(2), fake_backend)
        assert grouped.compiled.dry_run_info()["n_circuit_runs_per_call"] == 1
        # X and Z on the same qubit cannot share a measurement.
        split = _circuit(
            blocks,
            [Observable.from_string("1.0 X0"), Observable.from_string("1.0 Z0")],
            fake_backend,
        )
        assert split.compiled.dry_run_info()["n_measurement_groups"] == 2

    def test_grouping_can_be_switched_off(self, fake_backend, blocks):
        ungrouped = _circuit(blocks, all_z(2), fake_backend, group_commuting=False)
        assert ungrouped.compiled.dry_run_info()["n_measurement_groups"] == 2

    def test_per_call_mode_caches_nothing(self, fake_backend, blocks):
        per_call = _circuit(blocks, all_z(2), fake_backend, transpile_mode="per_call")
        info = per_call.compiled.dry_run_info()
        assert info["transpile_mode"] == "per_call"
        assert "transpiled_depth" not in info

    def test_sample_does_not_cost_a_compile_time_transpile(self, fake_backend, blocks):
        """The all-Z circuit that `sample()` needs is built on first use, so a
        caller who only reads expectation values never transpiles it."""
        circuit = _circuit(
            blocks,
            [Observable.from_string("1.0 X0 X1")],
            fake_backend,
        )
        assert list(circuit.compiled.dry_run_info()["transpiled_depth"]) == ["XX"]
        circuit.sample(np.zeros(circuit.spec.n_weights), np.zeros(2))
        assert sorted(circuit.compiled.dry_run_info()["transpiled_depth"]) == [
            "XX",
            "ZZ",
        ]

    def test_no_cost_warning_for_a_local_backend(self, fake_backend, blocks):
        with warnings.catch_warnings():
            warnings.simplefilter("error", CostWarning)
            _circuit(blocks, all_z(2), fake_backend)


class TestForward:
    def test_qubit_order_and_sign(self, fake_backend):
        """X on qubit 0 alone: Z0 must come back negative and the others
        positive. This pins the layout mapping and the bit order together —
        a reversed histogram key would flip which qubit reads as excited."""
        builder = CircuitBuilder(n_qubits=3)
        builder.x(0)
        circuit = _circuit(builder.build(), all_z(3), fake_backend)
        values = circuit(np.zeros(0), np.zeros(0))
        assert values.shape == (3,)
        assert values[0] < -0.8
        assert values[1] > 0.8
        assert values[2] > 0.8

    def test_values_stay_in_range(self, fake_backend, blocks):
        circuit = _circuit(blocks, Z(0), fake_backend)
        rng = np.random.default_rng(0)
        # A single observable collapses to a scalar, as everywhere else.
        value = circuit(rng.normal(size=circuit.spec.n_weights), rng.normal(size=2))
        assert -1.0 <= float(value) <= 1.0

    def test_batch_returns_one_row_per_sample(self, fake_backend, blocks):
        circuit = _circuit(blocks, all_z(2), fake_backend)
        rng = np.random.default_rng(1)
        weights = rng.normal(size=circuit.spec.n_weights)
        values = circuit(weights, rng.normal(size=(3, 2)))
        assert values.shape == (3, 2)

    @pytest.mark.parametrize("transpile_mode", ["parametric", "per_call"])
    def test_both_transpile_modes_agree(self, fake_backend, blocks, transpile_mode):
        """Same circuit, same device: the two modes differ in gate count, not
        in what they compute."""
        circuit = _circuit(blocks, all_z(2), fake_backend, transpile_mode=transpile_mode)
        values = circuit(np.zeros(circuit.spec.n_weights), np.zeros(2))
        # Zero angles leave |00>, so both Z values sit near +1 up to readout error.
        assert values[0] > 0.8
        assert values[1] > 0.8

    def test_star_architecture_routes_through_the_resonator(self, blocks):
        """Deneb is a Star device: two-qubit gates go through a resonator with
        MOVE gates, which the transpiler has to insert."""
        circuit = _circuit(blocks, all_z(2), IQMFakeDeneb())
        counts = next(
            iter(circuit.compiled.dry_run_info()["transpiled_gate_counts"].values())
        )
        assert counts.get("move", 0) > 0
        values = circuit(np.zeros(circuit.spec.n_weights), np.zeros(2))
        assert values.shape == (2,)


class TestSampling:
    def test_sample_returns_a_histogram(self, fake_backend, blocks):
        circuit = _circuit(blocks, all_z(2), fake_backend)
        counts = circuit.sample(np.zeros(circuit.spec.n_weights), np.zeros(2))
        assert sum(counts.values()) == 4000
        assert all(len(key) == 2 and set(key) <= {"0", "1"} for key in counts)

    def test_sample_keys_are_qubit_0_msb(self, fake_backend):
        builder = CircuitBuilder(n_qubits=3)
        builder.x(0)
        circuit = _circuit(builder.build(), all_z(3), fake_backend)
        counts = circuit.sample(np.zeros(0), np.zeros(0))
        assert max(counts, key=counts.get) == "100"

    def test_probabilities_are_a_distribution(self, fake_backend, blocks):
        circuit = _circuit(blocks, all_z(2), fake_backend)
        probabilities = circuit.probabilities(
            np.zeros(circuit.spec.n_weights), np.zeros(2)
        )
        assert probabilities.shape == (4,)
        assert probabilities.sum() == pytest.approx(1.0)
        # Zero angles leave |00>, which noise dents but does not dethrone.
        assert probabilities.argmax() == 0

    def test_probabilities_reject_a_batch(self, fake_backend, blocks):
        circuit = _circuit(blocks, all_z(2), fake_backend)
        with pytest.raises(ValueError):
            circuit.probabilities(np.zeros(circuit.spec.n_weights), np.zeros((2, 2)))


class TestSubmit:
    def test_submit_returns_a_handle(self, fake_backend, blocks):
        circuit = _circuit(blocks, all_z(2), fake_backend)
        handle = circuit.submit(np.zeros(circuit.spec.n_weights), np.zeros(2))
        assert hasattr(handle, "status") and hasattr(handle, "job_id")
        assert handle.result().shape == (2,)

    def test_handle_exposes_the_raw_counts(self, fake_backend, blocks):
        circuit = _circuit(blocks, all_z(2), fake_backend)
        handle = circuit.submit(np.zeros(circuit.spec.n_weights), np.zeros(2))
        histograms = handle.counts()
        # all_z(2) is one measurement group, so one histogram.
        assert len(histograms) == 1
        assert sum(histograms[0].values()) == 4000


class TestGradientPolicy:
    def test_torch_backward_is_refused(self, fake_backend, blocks):
        """The deploy workflow is train-on-a-simulator, then run here: a torch
        layer on IQM has to say so rather than quietly spending 2 * n_weights
        jobs on a parameter-shift pass."""
        torch = pytest.importorskip("torch")
        layer = Circuit(
            blocks=blocks,
            observables=Z(0),
            provider=IQMConfig(backend_instance=fake_backend, shots=1000),
            interface="torch",
        )
        y = layer(torch.tensor([0.3, -0.5], dtype=torch.float64))
        with pytest.raises(
            RuntimeError, match="do not support automatic differentiation"
        ):
            y.backward()

    def test_numpy_jacobian_is_refused(self, fake_backend, blocks):
        circuit = _circuit(blocks, Z(0), fake_backend)
        with pytest.raises(RuntimeError, match="enable_gradients=True"):
            circuit.jacobian(np.zeros(circuit.spec.n_weights), np.zeros(2))

    def test_gradient_method_follows_enable_gradients(self):
        assert IQMBackend().gradient_method == "none"
        opted_in = IQMBackend(IQMConfig(enable_gradients=True))
        assert opted_in.gradient_method == "parameter_shift"

    def test_opt_in_differentiates(self, fake_backend, blocks):
        """Opted in, both gradient paths run parameter-shift on the device.
        The fake backend is noisy, so this pins shape and finiteness, not
        closeness to the analytic value."""
        torch = pytest.importorskip("torch")
        circuit = _circuit(blocks, Z(0), fake_backend, enable_gradients=True)
        grad = circuit.jacobian(np.full(circuit.spec.n_weights, 0.4), np.zeros(2))
        assert grad.shape == (circuit.spec.n_weights,)
        assert np.all(np.isfinite(grad))

        layer = Circuit(
            blocks=blocks,
            observables=Z(0),
            provider=IQMConfig(
                backend_instance=fake_backend, shots=1000, enable_gradients=True
            ),
            interface="torch",
        )
        layer(torch.tensor([0.3, -0.5], dtype=torch.float64)).backward()
        weights = next(layer.parameters())
        assert weights.grad is not None
        assert torch.all(torch.isfinite(weights.grad))

    def test_opt_in_warns_on_a_real_device(self, monkeypatch, fake_backend, blocks):
        """all_z(2) is one measurement group, so 2 RY weights cost 4 shift
        evaluations x 1 circuit = 4 circuits per backward pass."""
        monkeypatch.setattr(
            "eigenkit.core.backends.iqm_backend._is_local_backend", lambda _: False
        )
        with pytest.warns(CostWarning, match="gradients are enabled") as record:
            _circuit(blocks, all_z(2), fake_backend, enable_gradients=True)
        assert any("at least 4 circuits per sample" in str(w.message) for w in record)

    def test_fake_backend_opt_in_does_not_warn(self, fake_backend, blocks):
        with warnings.catch_warnings():
            warnings.simplefilter("error", CostWarning)
            _circuit(blocks, all_z(2), fake_backend, enable_gradients=True)

    def test_shifted_copies_share_the_device_and_seed(self, fake_backend):
        """The chain-rule copies reuse the resolved backend and the handle's
        transpiler seed, so every gradient term lands on the same qubits."""
        from eigenkit import Reuploading

        circuit = _circuit(
            [Reuploading(in_features=2, depth=1, trainable_scale=True)],
            Z(0),
            fake_backend,
            enable_gradients=True,
        )
        circuit.jacobian(np.full(circuit.spec.n_weights, 0.1), np.zeros(2))

        base = circuit.compiled
        assert circuit._chain_rule_cache
        for shifted in circuit._chain_rule_cache.values():
            assert shifted._backend is base._backend
            assert shifted._seed == base._seed


@pytest.mark.qpu
class TestRealHardwareIntegration:
    """Runs only when explicitly enabled (`pytest -m qpu`). Consumes IQM
    quota and needs `IQM_TOKEN` plus a reachable server."""

    def test_resonance_smoke(self, blocks):
        circuit = Circuit(
            blocks=blocks,
            observables=all_z(2),
            provider=IQMConfig(
                url="https://resonance.iqm.tech/",
                quantum_computer="garnet",
                shots=1000,
            ),
        )
        values = circuit(np.zeros(circuit.spec.n_weights), np.zeros(2))
        assert values.shape == (2,)
        assert np.all(np.abs(values) <= 1.0)


class TestQuantumKernelOnIQM:
    """Quantum kernels run on IQM through the inversion circuit: every kernel
    entry is one circuit of a single job, read as an all-zeros share."""

    X = np.random.default_rng(1).uniform(0, np.pi, size=(5, 2))

    @staticmethod
    def _kernel(backend, **kwargs):
        from eigenkit import QuantumKernel

        return QuantumKernel(
            n_features=2,
            feature_map="AngleCNOT",
            reps=2,
            provider=IQMConfig(backend_instance=backend, seed_transpiler=0, **kwargs),
        )

    def test_close_to_the_exact_kernel(self, fake_backend):
        from eigenkit import QuantumKernel

        exact = QuantumKernel(n_features=2, feature_map="AngleCNOT", reps=2)
        sampled = self._kernel(fake_backend, shots=8000)
        K = sampled.compute_kernel_matrix(self.X)
        # Noisy device: structure, not exactness.
        assert K.shape == (5, 5)
        np.testing.assert_allclose(np.diag(K), 1.0)
        assert np.abs(K - exact.compute_kernel_matrix(self.X)).max() < 0.15

    def test_one_job_per_kernel_matrix(self, fake_backend, monkeypatch):
        kernel = self._kernel(fake_backend, shots=500)
        compiled = kernel.circuit.compiled
        sizes = []
        real = compiled._backend.run

        def recording(circuits, **kwargs):
            sizes.append(len(circuits))
            return real(circuits, **kwargs)

        monkeypatch.setattr(compiled._backend, "run", recording)
        kernel.compute_kernel_matrix(self.X, self.X[:2])
        assert sizes == [10]

    def test_resources_on_crystal_and_star(self, fake_backend):
        crystal = self._kernel(fake_backend, shots=100).circuit.resources()
        assert crystal["two_qubit_gate_counts"].keys() <= {"cz"}
        assert crystal["n_2q_gates"] >= 1
        assert 0.0 < crystal["estimated_success_probability"] < 1.0

        star = self._kernel(IQMFakeDeneb(), shots=100).circuit.resources()
        assert star["two_qubit_gate_counts"].get("move", 0) > 0
        assert star["routing_overhead_2q"] > 0

    def test_retrieve_from_a_fake_backend_is_refused(self, fake_backend):
        kernel = self._kernel(fake_backend, shots=100)
        with pytest.raises(NotImplementedError, match="job history"):
            kernel.retrieve_kernel_matrix("job", n_x=3)
