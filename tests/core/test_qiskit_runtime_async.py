"""Qiskit Runtime async hooks: method-aware submit, job tags, attach(job_id).

These are the public seams a job service uses to submit a
job, tag it for crash recovery, and re-attach to it by job id from another
process. Nothing here contacts IBM: a fake backend in local testing mode covers
the result paths, and stubbed primitives or a mocked `service` cover the rest.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")
# FakeBrisbane runs on Aer. Without it (the qiskit extra leaves Aer out on
# Linux ARM with Python 3.14), qiskit-ibm-runtime falls back to BasicSimulator,
# which cannot hold the device's 127 qubits.
pytest.importorskip("qiskit_aer")

import numpy as np  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeBrisbane  # noqa: E402

from eigenkit.core.backends import qiskit_runtime_backend  # noqa: E402
from eigenkit.core.backends.qiskit_runtime_backend import (  # noqa: E402
    QiskitRuntimeBackend,
    QiskitRuntimeCompiled,
    QiskitRuntimeConfig,
    QiskitRuntimeJobHandle,
)
from eigenkit.core.blocks import (  # noqa: E402
    AngleEncoding,
    BasicVariational,
    assemble,
)
from eigenkit.core.observables import all_z  # noqa: E402
from eigenkit.core.provider import CostWarning  # noqa: E402


_SHOTS = 512
_JOB_ID = "0b7c6f8e-1d2a-4c3b-9e8f-7a6b5c4d3e2f"


@pytest.fixture(scope="module")
def fake_backend():
    return FakeBrisbane()


@pytest.fixture
def ir():
    return assemble(
        [
            AngleEncoding(in_features=2),
            BasicVariational(depth=2, rotation="RY"),
        ]
    )


@pytest.fixture
def weights():
    return np.array([0.4, 1.1, -0.3, 0.9], dtype=np.float64)


@pytest.fixture
def inputs():
    return np.array([0.5, -0.2], dtype=np.float64)


def _compile(ir, backend, primitive="estimator") -> QiskitRuntimeCompiled:
    config = QiskitRuntimeConfig(
        backend_instance=backend, shots=_SHOTS, primitive=primitive
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostWarning)
        return QiskitRuntimeBackend(config=config).compile(ir, all_z(2))


class _Recorder:
    """Seeds a local-mode primitive and records the jobs it returns.

    Seeding makes the local simulator deterministic, so a value submitted
    asynchronously can be compared exactly with the blocking method's.
    """

    def __init__(self, primitive, jobs: list, seed: int):
        primitive.options.simulator.seed_simulator = seed
        self._primitive = primitive
        self._jobs = jobs

    @property
    def options(self):
        return self._primitive.options

    def run(self, pubs):
        job = self._primitive.run(pubs)
        self._jobs.append(job)
        return job


def _seed(compiled: QiskitRuntimeCompiled, seed: int = 1234) -> list:
    """Seed every primitive `compiled` builds; return the recorded jobs."""
    jobs: list = []
    for name in ("_sampler", "_estimator"):
        build = getattr(compiled, name)
        setattr(
            compiled,
            name,
            lambda build=build: _Recorder(build(), jobs, seed),
        )
    return jobs


def _stub_primitive():
    return SimpleNamespace(
        options=SimpleNamespace(environment=SimpleNamespace(job_tags=None)),
        run=MagicMock(name="run", return_value=MagicMock(name="job")),
    )


@pytest.fixture
def stubbed(ir, fake_backend):
    """A compiled circuit whose primitives are stubs: nothing is simulated."""
    compiled = _compile(ir, fake_backend)
    sampler, estimator = _stub_primitive(), _stub_primitive()
    compiled._sampler = lambda: sampler
    compiled._estimator = lambda: estimator
    return SimpleNamespace(compiled=compiled, sampler=sampler, estimator=estimator)


class TestSubmitMethods:
    def test_submit_methods_lists_async_methods(self):
        assert QiskitRuntimeCompiled.submit_methods == frozenset(
            {"run", "sample", "probabilities"}
        )

    def test_zero_state_probability_is_not_offered(self):
        assert "zero_state_probability" not in QiskitRuntimeCompiled.submit_methods

    def test_unknown_method_raises_naming_submit_methods(self, stubbed, weights, inputs):
        with pytest.raises(ValueError, match="submit_methods"):
            stubbed.compiled.submit(weights, inputs, method="zero_state_probability")
        stubbed.sampler.run.assert_not_called()
        stubbed.estimator.run.assert_not_called()

    def test_two_d_weights_rejected(self, stubbed, weights, inputs):
        with pytest.raises(ValueError, match="1-D"):
            stubbed.compiled.submit(np.stack([weights, weights]), inputs)
        stubbed.estimator.run.assert_not_called()

    def test_two_d_inputs_rejected(self, stubbed, weights, inputs):
        with pytest.raises(ValueError, match="1-D"):
            stubbed.compiled.submit(weights, np.stack([inputs, inputs]), method="sample")
        stubbed.sampler.run.assert_not_called()


class TestSubmitDecodesLikeTheBlockingMethods:
    """Seeded, so the asynchronous path must give the blocking path's values."""

    def test_sample(self, ir, fake_backend, weights, inputs):
        compiled = _compile(ir, fake_backend)
        _seed(compiled)
        submitted = compiled.submit(weights, inputs, method="sample").result()
        blocking = compiled.sample(weights, inputs)
        assert isinstance(submitted, dict)
        assert submitted == blocking
        assert sum(submitted.values()) == _SHOTS
        assert all(len(key) == 2 and set(key) <= {"0", "1"} for key in submitted)

    def test_sample_is_qubit_zero_msb(self, ir, fake_backend):
        """RX(pi) on qubit 0 only: nearly every shot reads '1' leftmost."""
        compiled = _compile(ir, fake_backend)
        _seed(compiled)
        counts = compiled.submit(
            np.zeros(4), np.array([np.pi, 0.0]), method="sample"
        ).result()
        leading_one = sum(n for key, n in counts.items() if key[0] == "1")
        assert leading_one / _SHOTS > 0.9

    def test_probabilities(self, ir, fake_backend, weights, inputs):
        compiled = _compile(ir, fake_backend)
        _seed(compiled)
        submitted = compiled.submit(weights, inputs, method="probabilities").result()
        blocking = compiled.probabilities(weights, inputs)
        assert submitted.dtype == np.float64
        assert submitted.shape == (4,)
        assert submitted.sum() == pytest.approx(1.0)
        np.testing.assert_array_equal(submitted, blocking)

    @pytest.mark.parametrize("primitive", ["estimator", "sampler"])
    def test_run(self, ir, fake_backend, weights, inputs, primitive):
        compiled = _compile(ir, fake_backend, primitive)
        _seed(compiled)
        submitted = compiled.submit(weights, inputs, method="run").result()
        blocking = compiled.run(weights, inputs)
        assert submitted.dtype == np.float64
        assert submitted.shape == (2,)
        np.testing.assert_array_equal(submitted, blocking)

    def test_run_is_the_default(self, ir, fake_backend, weights, inputs):
        compiled = _compile(ir, fake_backend)
        _seed(compiled)
        handle = compiled.submit(weights, inputs)
        assert handle.method == "run"
        assert handle.result().shape == (2,)

    def test_handle_exposes_method_and_job_id(self, ir, fake_backend, weights, inputs):
        compiled = _compile(ir, fake_backend)
        handle = compiled.submit(weights, inputs, method="sample")
        assert handle.method == "sample"
        assert isinstance(handle.job_id(), str) and handle.job_id()

    def test_sample_handle_counts_is_the_one_histogram(
        self, ir, fake_backend, weights, inputs
    ):
        compiled = _compile(ir, fake_backend)
        handle = compiled.submit(weights, inputs, method="sample")
        assert handle.counts() == [handle.result()]

    def test_sample_and_probabilities_run_under_the_estimator(
        self, stubbed, weights, inputs
    ):
        """Sampling methods are always one SamplerV2 job, whatever the config's
        primitive; the estimator is never touched."""
        for method in ("sample", "probabilities"):
            stubbed.compiled.submit(weights, inputs, method=method)
        assert stubbed.sampler.run.call_count == 2
        stubbed.estimator.run.assert_not_called()
        for call in stubbed.sampler.run.call_args_list:
            assert len(call.args[0]) == 1  # one Z-basis PUB


class TestJobTags:
    def test_tags_become_sorted_key_value_job_tags(self, stubbed, weights, inputs):
        stubbed.compiled.submit(
            weights,
            inputs,
            method="sample",
            tags={"job_id": _JOB_ID, "env": "dev"},
        )
        assert stubbed.sampler.options.environment.job_tags == [
            "env=dev",
            f"job_id={_JOB_ID}",
        ]
        stubbed.sampler.run.assert_called_once()

    def test_estimator_run_is_tagged_too(self, stubbed, weights, inputs):
        stubbed.compiled.submit(weights, inputs, tags={"env": "prod"})
        assert stubbed.estimator.options.environment.job_tags == ["env=prod"]
        stubbed.estimator.run.assert_called_once()

    def test_sampler_run_is_tagged_too(self, ir, fake_backend, weights, inputs):
        compiled = _compile(ir, fake_backend, "sampler")
        sampler = _stub_primitive()
        compiled._sampler = lambda: sampler
        compiled.submit(weights, inputs, tags={"env": "prod"})
        assert sampler.options.environment.job_tags == ["env=prod"]

    @pytest.mark.parametrize("tags", [None, {}])
    def test_no_tags_leaves_job_tags_unset(self, stubbed, weights, inputs, tags):
        stubbed.compiled.submit(weights, inputs, method="sample", tags=tags)
        assert stubbed.sampler.options.environment.job_tags is None
        stubbed.sampler.run.assert_called_once()

    def test_limits_are_inclusive(self, stubbed, weights, inputs):
        tags = {f"k{i}": "v" for i in range(7)}
        tags["long"] = "x" * (86 - len("long="))
        stubbed.compiled.submit(weights, inputs, method="sample", tags=tags)
        assert len(stubbed.sampler.options.environment.job_tags) == 8

    def test_tags_reach_a_local_mode_job(self, ir, fake_backend, weights, inputs):
        """A real primitive accepts the option; the job still runs."""
        compiled = _compile(ir, fake_backend)
        handle = compiled.submit(weights, inputs, method="sample", tags={"env": "dev"})
        assert sum(handle.result().values()) == _SHOTS

    @pytest.mark.parametrize(
        "tags",
        [
            pytest.param({f"k{i}": "v" for i in range(9)}, id="nine-tags"),
            pytest.param({"long": "x" * (87 - len("long="))}, id="87-characters"),
            pytest.param({"env": ""}, id="empty-value"),
            pytest.param({"": "dev"}, id="empty-key"),
            pytest.param({"a=b": "c"}, id="equals-in-key"),
            pytest.param({"env": 3}, id="non-str-value"),
        ],
    )
    def test_invalid_tags_raise_before_any_job(self, stubbed, weights, inputs, tags):
        with pytest.raises(ValueError, match="tag"):
            stubbed.compiled.submit(weights, inputs, method="sample", tags=tags)
        stubbed.sampler.run.assert_not_called()


class TestAttach:
    @pytest.mark.parametrize(
        ("primitive", "method"),
        [
            ("estimator", "run"),
            ("sampler", "run"),
            ("estimator", "sample"),
            ("estimator", "probabilities"),
            ("sampler", "sample"),
        ],
    )
    def test_attach_decodes_like_submit(
        self, ir, fake_backend, weights, inputs, primitive, method
    ):
        compiled = _compile(ir, fake_backend, primitive)
        jobs = _seed(compiled)
        submitted = compiled.submit(weights, inputs, method=method)
        expected = submitted.result()

        compiled._backend = MagicMock(name="IBMBackend")
        compiled._backend.service.job.return_value = jobs[-1]
        attached = compiled.attach(submitted.job_id(), method=method)

        compiled._backend.service.job.assert_called_once_with(submitted.job_id())
        assert attached.method == method
        assert attached.job_id() == submitted.job_id()
        if isinstance(expected, dict):
            assert attached.result() == expected
        else:
            assert attached.result().dtype == expected.dtype
            np.testing.assert_array_equal(attached.result(), expected)

    def test_attach_uses_the_backend_service(self, ir, fake_backend):
        compiled = _compile(ir, fake_backend)
        compiled._backend = MagicMock(name="IBMBackend")
        compiled.attach("d1abc2xyz", method="sample")
        compiled._backend.service.job.assert_called_once_with("d1abc2xyz")

    @pytest.mark.parametrize(
        "job_id", ["", "bad id!", "x" * 65, "../etc/passwd", None, 12345]
    )
    def test_malformed_job_id_raises_without_touching_the_service(
        self, ir, fake_backend, job_id
    ):
        compiled = _compile(ir, fake_backend)
        compiled._backend = MagicMock(name="IBMBackend")
        with pytest.raises(ValueError, match="job_id"):
            compiled.attach(job_id, method="sample")
        compiled._backend.service.job.assert_not_called()

    def test_unknown_method_raises_without_touching_the_service(self, ir, fake_backend):
        compiled = _compile(ir, fake_backend)
        compiled._backend = MagicMock(name="IBMBackend")
        with pytest.raises(ValueError, match="submit_methods"):
            compiled.attach("d1abc2xyz", method="zero_state_probability")
        compiled._backend.service.job.assert_not_called()

    def test_backend_without_a_service_cannot_attach(self, ir, fake_backend):
        compiled = _compile(ir, fake_backend)
        with pytest.raises(ValueError, match="IBM Runtime backend"):
            compiled.attach("d1abc2xyz", method="sample")

    def test_wrong_method_is_reported_as_a_mismatch(
        self, ir, fake_backend, weights, inputs
    ):
        """A sample job has one PUB; the estimator `run` decode expects one
        per observable."""
        compiled = _compile(ir, fake_backend)
        jobs = _seed(compiled)
        submitted = compiled.submit(weights, inputs, method="sample")
        compiled._backend = MagicMock(name="IBMBackend")
        compiled._backend.service.job.return_value = jobs[-1]
        attached = compiled.attach(submitted.job_id(), method="run")
        with pytest.raises(ValueError, match="method='run'"):
            attached.result()


class _FakeJob:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def job_id(self):
        return "d1abc2xyz"


class TestHandle:
    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="submit_methods"):
            QiskitRuntimeJobHandle(_FakeJob(None), 1, 60, method="zero_state_probability")

    def test_probabilities_wider_than_the_dense_cap_raise(self):
        width = qiskit_runtime_backend._sampling.MAX_DENSE_QUBITS + 1
        handle = QiskitRuntimeJobHandle(
            _FakeJob(object()),
            1,
            60,
            counts=lambda result: [{"0" * width: _SHOTS}],
            method="probabilities",
            n_qubits=width,
        )
        with pytest.raises(ValueError, match="sample"):
            handle.result()


class TestBackendResolution:
    def test_compile_resolves_the_backend_once(self, ir, fake_backend, monkeypatch):
        calls = []
        original = qiskit_runtime_backend._resolve_backend

        def counting(config):
            calls.append(config)
            return original(config)

        monkeypatch.setattr(qiskit_runtime_backend, "_resolve_backend", counting)
        _compile(ir, fake_backend)
        assert len(calls) == 1

    def test_no_service_or_token_lookup_with_a_backend_instance(
        self, ir, fake_backend, weights, inputs, monkeypatch
    ):
        """A job service passes a ready backend; nothing may build a service from
        an environment token or the saved account."""
        import qiskit_ibm_runtime

        def refuse(**kwargs):
            raise AssertionError("QiskitRuntimeService must not be constructed")

        monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", refuse)
        monkeypatch.setenv("QISKIT_IBM_TOKEN", "must-not-be-read")
        compiled = _compile(ir, fake_backend)
        for method in sorted(QiskitRuntimeCompiled.submit_methods):
            compiled.submit(weights, inputs, method=method).result()
        compiled._backend = MagicMock(name="IBMBackend")
        compiled.attach("d1abc2xyz", method="sample")
