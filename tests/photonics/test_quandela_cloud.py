"""Quandela Cloud backend tests.

The offline tests need no network or credentials — they cover token resolution, the
offline ``dry_run_info``, the counts->distribution conversion, protocol conformance,
gradient rejection, and (against a fake processor/sampler/job, with the SDK installed)
the submission path: the one-time ``CostWarning``, processor reuse, job timeouts and
failures. The live test (marked ``qpu``, gated on a resolvable token) actually submits to
Quandela Cloud; it is skipped by default (the conftest skips ``qpu`` tests unless run
with ``-m qpu``) and also skips without a token.
"""

import math
import os
import warnings

import numpy as np
import pytest

from eigenkit.core.ir import Constant
from eigenkit.core.provider import CostWarning
from eigenkit.photonics import (
    BeamSplitter,
    Compiled,
    Executor,
    FockProbability,
    PhotonicCircuitSpec,
    PhotonNumber,
    QuandelaCloudBackend,
)
from eigenkit.photonics.backends import perceval_remote
from eigenkit.photonics.backends.perceval_remote import (
    RemoteCompiled,
    RemoteJobHandle,
    _counts_to_distribution,
    _resolve_token,
)


BALANCED = Constant(math.pi / 2)


def _hom_spec():
    return PhotonicCircuitSpec(
        n_modes=2,
        n_weights=0,
        n_inputs=0,
        components=(BeamSplitter(modes=(0, 1), theta=BALANCED),),
        input_state=(1, 1),
    )


def test_protocol_conformance():
    backend = QuandelaCloudBackend("sim:slos", token="dummy")
    assert isinstance(backend, Executor)
    assert isinstance(backend.compile(_hom_spec()), Compiled)


def test_dry_run_info_is_offline():
    # No token, no network, no SDK.
    info = (
        QuandelaCloudBackend("qpu:ascella", shots=5000)
        .compile(_hom_spec())
        .dry_run_info()
    )
    assert info["platform"] == "qpu:ascella"
    assert info["n_modes"] == 2
    assert info["n_photons"] == 2
    assert info["shots"] == 5000
    assert info["n_components"] == 1


@pytest.fixture
def no_env_token(monkeypatch):
    for var in ("QUANDELA_TOKEN", "PERCEVAL_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def test_missing_token_raises(monkeypatch, no_env_token):
    monkeypatch.setattr(perceval_remote, "_saved_token", lambda: None)
    with pytest.raises(RuntimeError, match="token"):
        _resolve_token(None)


def test_token_resolution(monkeypatch):
    monkeypatch.delenv("PERCEVAL_TOKEN", raising=False)
    monkeypatch.setenv("QUANDELA_TOKEN", "envtoken")
    monkeypatch.setattr(perceval_remote, "_saved_token", lambda: "saved")
    assert _resolve_token(None) == "envtoken"  # env wins over Perceval's saved token
    assert _resolve_token("explicit") == "explicit"  # explicit wins over env


def test_falls_back_to_perceval_saved_token(monkeypatch, no_env_token):
    # A token stored the standard Perceval way (pcvl.RemoteConfig) must still work.
    monkeypatch.setattr(perceval_remote, "_saved_token", lambda: "saved")
    assert _resolve_token(None) == "saved"


def test_counts_to_distribution():
    dist = _counts_to_distribution({(2, 0): 993, (0, 2): 1007}, 2)
    assert dist[(2, 0)] == pytest.approx(0.4965, abs=1e-3)
    assert dist[(0, 2)] == pytest.approx(0.5035, abs=1e-3)
    assert sum(dist.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("counts", [{}, {(2, 0): 0, (0, 2): 0}])
def test_empty_postselection_raises(counts):
    # Every shot discarded must not read as a valid (all-zero) measurement.
    with pytest.raises(RuntimeError, match="post-selection"):
        _counts_to_distribution(counts, 2)


# ---- submission path, against fakes (no network) ----


class _FakeStatus:
    def __init__(self, name, stop_message=None):
        self.name = name
        self.stop_message = stop_message

    def __call__(self):
        return self.name


class _FakeJob:
    def __init__(self, counts=None, *, complete=True, success=True):
        self._counts = counts if counts is not None else {(2, 0): 1, (0, 2): 1}
        self.is_complete = complete
        self.is_success = success
        name = "RUNNING" if not complete else ("SUCCESS" if success else "ERROR")
        self.status = _FakeStatus(name, None if success else "platform error")
        self.id = "job-123"
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def get_results(self):
        return {"results": self._counts}


class _FakeSampler:
    def __init__(self, processor, max_shots_per_call=None):
        self.sample_count = self

    def execute_async(self, shots):
        return _FakeJob()


class _FakeProcessor:
    def set_circuit(self, circuit):
        pass

    def min_detected_photons_filter(self, n):
        pass

    def with_input(self, state):
        pass


@pytest.fixture
def fake_cloud(monkeypatch):
    """Patch the network edges: the RemoteProcessor factory and perceval's Sampler.
    Returns the list of processors built, so tests can count connections."""
    pytest.importorskip("perceval")
    built = []

    def connect(self):
        built.append(_FakeProcessor())
        return built[-1]

    monkeypatch.setattr(RemoteCompiled, "_remote_processor", connect)
    monkeypatch.setattr("perceval.algorithm.Sampler", _FakeSampler)
    return built


def _no_params():
    return np.array([]), np.array([])


def test_processor_is_built_once_per_handle(fake_cloud):
    compiled = QuandelaCloudBackend("sim:slos", token="dummy").compile(_hom_spec())
    with pytest.warns(CostWarning):
        compiled.run(*_no_params())
    compiled.run(*_no_params())
    compiled.submit(*_no_params()).result()
    assert len(fake_cloud) == 1


def test_failed_attempt_keeps_the_cost_warning(monkeypatch, fake_cloud):
    compiled = QuandelaCloudBackend("qpu:ascella", token="dummy").compile(_hom_spec())

    def no_connection(self):
        raise RuntimeError("no Quandela Cloud token found")

    with monkeypatch.context() as patch:
        patch.setattr(RemoteCompiled, "_remote_processor", no_connection)
        with pytest.raises(RuntimeError, match="token"):
            compiled.run(*_no_params())

    # The retry is the first submission that goes through, so it must still warn...
    with pytest.warns(CostWarning):
        compiled.run(*_no_params())
    # ...and only once.
    with warnings.catch_warnings():
        warnings.simplefilter("error", CostWarning)
        compiled.run(*_no_params())


def test_expectation_rejects_bad_target_before_submitting(fake_cloud):
    compiled = QuandelaCloudBackend("sim:slos", token="dummy").compile(_hom_spec())
    with pytest.raises(ValueError, match="FockProbability"):
        compiled.expectation(*_no_params(), FockProbability(target=(1, 0)))
    assert not fake_cloud  # nothing connected, nothing paid for


def test_expectation_allows_lossy_target_under_post_selection(fake_cloud):
    # With min_detected_photons=1, one-photon outputs survive post-selection.
    compiled = QuandelaCloudBackend(
        "sim:slos", token="dummy", min_detected_photons=1
    ).compile(_hom_spec())
    with pytest.warns(CostWarning):
        value = compiled.expectation(*_no_params(), FockProbability(target=(1, 0)))
    assert value == 0.0  # the fake job only returns two-photon outputs


def test_result_times_out():
    job = _FakeJob(complete=False)
    handle = RemoteJobHandle(job, 2, poll_interval=0.001, poll_timeout_seconds=0.01)
    # The id lets a submit() user find the job in the Quandela console.
    with pytest.raises(TimeoutError, match="job-123 still RUNNING"):
        handle.result()
    assert not job.cancelled  # a submit() caller keeps the handle and decides


def _stalled_cloud(monkeypatch, job):
    monkeypatch.setattr(_FakeSampler, "execute_async", lambda self, shots: job)
    return QuandelaCloudBackend(
        "qpu:ascella", token="dummy", poll_timeout_seconds=0
    ).compile(_hom_spec())


def test_run_cancels_the_job_when_it_times_out(monkeypatch, fake_cloud):
    # run() drops the handle, so a job left running could never be cancelled and a
    # retry would pay for a second one.
    job = _FakeJob(complete=False)
    compiled = _stalled_cloud(monkeypatch, job)
    with pytest.warns(CostWarning), pytest.raises(TimeoutError, match="job-123"):
        compiled.run(*_no_params())
    assert job.cancelled


class _InterruptedJob(_FakeJob):
    @property
    def is_complete(self):
        raise KeyboardInterrupt

    @is_complete.setter
    def is_complete(self, value):
        pass


def test_run_cancels_the_job_on_ctrl_c(monkeypatch, fake_cloud):
    job = _InterruptedJob(complete=False)
    compiled = _stalled_cloud(monkeypatch, job)
    with pytest.warns(CostWarning), pytest.raises(KeyboardInterrupt):
        compiled.run(*_no_params())
    assert job.cancelled


def test_failed_cancel_keeps_the_original_error(monkeypatch, fake_cloud):
    job = _FakeJob(complete=False)

    def broken_cancel():
        raise ConnectionError("cloud unreachable")

    job.cancel = broken_cancel
    compiled = _stalled_cloud(monkeypatch, job)
    with pytest.warns(CostWarning), pytest.raises(TimeoutError):
        compiled.run(*_no_params())


def test_failed_job_raises():
    handle = RemoteJobHandle(_FakeJob(success=False), 2)
    with pytest.raises(RuntimeError, match="platform error"):
        handle.result()


def test_gradients_unsupported_on_hardware():
    compiled = QuandelaCloudBackend("sim:slos", token="dummy").compile(_hom_spec())
    with pytest.raises(NotImplementedError, match="gradient"):
        compiled.jacobian(np.array([]), np.array([]), PhotonNumber(0))
    # The torch bridge passes the gradient flags; they must reach the same error,
    # not a TypeError about unexpected keywords.
    with pytest.raises(NotImplementedError, match="gradient"):
        compiled.jacobian(
            np.array([]),
            np.array([]),
            PhotonNumber(0),
            need_weight_grad=True,
            need_input_grad=False,
        )


@pytest.mark.qpu
def test_quandela_cloud_live():
    # Skipped unless run with `-m qpu` (conftest) AND a token is present. Run:
    #   QUANDELA_TOKEN=... [QUANDELA_PLATFORM=qpu:ascella] \
    #     pytest tests/photonics/test_quandela_cloud.py -m qpu -s
    pytest.importorskip("perceval")
    try:
        token = _resolve_token(None)
    except RuntimeError:
        pytest.skip("set QUANDELA_TOKEN to run the live Quandela Cloud test")
    from eigenkit.photonics import PercevalBackend

    platform = os.environ.get("QUANDELA_PLATFORM", "sim:slos")
    spec = _hom_spec()
    no_params = np.array([])

    backend = QuandelaCloudBackend(platform, token=token, shots=20000)
    with pytest.warns(CostWarning):
        sampled = backend.compile(spec).run(no_params, no_params)
    exact = PercevalBackend().compile(spec).run(no_params, no_params)

    print(
        f"\n[quandela:{platform}] sampled: "
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(sampled.items()))
    )
    print(
        "[local exact]        "
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(exact.items()))
    )

    assert abs(sum(sampled.values()) - 1.0) < 1e-6
    assert all(sum(state) == spec.n_photons for state in sampled)  # photon conservation
