"""Braket async hooks: method-aware submit, task tags, attach(task_id).

These are the public seams a job service uses to submit a task, tag it for crash recovery, and re-attach to it by ARN later. Nothing here
contacts AWS: the local simulator covers the result paths, and a mocked device
or a monkeypatched `AwsQuantumTask` covers the cloud paths.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytest.importorskip("braket")

import numpy as np  # noqa: E402

from eigenkit.core.backends.braket_backend import (  # noqa: E402
    BraketBackend,
    BraketCompiled,
    BraketConfig,
    BraketJobHandle,
    _aggregate_values,
)
from eigenkit.core.blocks import (  # noqa: E402
    AngleEncoding,
    BasicVariational,
    assemble,
)
from eigenkit.core.observables import all_z  # noqa: E402
from eigenkit.core.provider import CostWarning  # noqa: E402


_SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
_TASK_ARN = "arn:aws:braket:us-east-1:123:quantum-task/x"


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


def _compile(ir, config: BraketConfig) -> BraketCompiled:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostWarning)
        return BraketBackend(config=config).compile(ir, all_z(2))


@pytest.fixture
def local_compiled(ir):
    return _compile(ir, BraketConfig())


@pytest.fixture
def cloud_compiled(ir):
    compiled = _compile(ir, BraketConfig(device_arn=_SV1_ARN, shots=100))
    compiled._device = MagicMock()
    compiled._device.run.return_value = MagicMock(name="task")
    return compiled


class _FakeResult:
    def __init__(self, values):
        self.values = values


class _FakeAwsQuantumTask:
    """Stands in for `braket.aws.AwsQuantumTask` re-attached from an ARN."""

    values: list | None = []
    # A whole result object, for the tasks whose decode reads more than
    # `values` (measurement counts, the carried result type).
    full_result: object | None = None
    task_state: str = "COMPLETED"
    created: list = []

    def __init__(self, arn, aws_session=None, poll_timeout_seconds=None, **kwargs):
        self.id = arn
        self.poll_timeout_seconds = poll_timeout_seconds
        _FakeAwsQuantumTask.created.append(self)

    def state(self):
        return _FakeAwsQuantumTask.task_state

    def result(self):
        # The real task returns None, not a result, once it ends FAILED or
        # CANCELLED, or when polling times out.
        if _FakeAwsQuantumTask.full_result is not None:
            return _FakeAwsQuantumTask.full_result
        if _FakeAwsQuantumTask.values is None:
            return None
        return _FakeResult(_FakeAwsQuantumTask.values)


@pytest.fixture
def fake_aws_task(monkeypatch):
    import braket.aws

    _FakeAwsQuantumTask.created = []
    _FakeAwsQuantumTask.values = []
    _FakeAwsQuantumTask.full_result = None
    _FakeAwsQuantumTask.task_state = "COMPLETED"
    monkeypatch.setattr(braket.aws, "AwsQuantumTask", _FakeAwsQuantumTask)
    return _FakeAwsQuantumTask


class TestSubmitMethods:
    def test_submit_methods_lists_async_methods(self):
        assert BraketCompiled.submit_methods == frozenset(
            {"run", "sample", "probabilities", "zero_state_probability"}
        )

    def test_unknown_method_raises_naming_submit_methods(
        self, cloud_compiled, weights, inputs
    ):
        with pytest.raises(ValueError, match="submit_methods"):
            cloud_compiled.submit(weights, inputs, method="statevector")
        cloud_compiled._device.run.assert_not_called()


class TestLocalSubmit:
    def test_default_submit_matches_run(self, local_compiled, weights, inputs):
        handle = local_compiled.submit(weights, inputs)
        assert handle.method == "run"
        assert np.allclose(handle.result(), local_compiled.run(weights, inputs))

    def test_zsp_submit_matches_sync_zsp(self, local_compiled, weights, inputs):
        handle = local_compiled.submit(weights, inputs, method="zero_state_probability")
        value = handle.result()
        assert isinstance(value, np.ndarray)
        assert value.ndim == 0
        assert value.dtype == np.float64
        expected = local_compiled.zero_state_probability(weights, inputs)
        assert float(value) == pytest.approx(expected, abs=1e-12)

    def test_local_submit_does_not_forward_tags(self, local_compiled, weights, inputs):
        local_compiled._device = MagicMock()
        local_compiled.submit(weights, inputs, tags={"job_id": "abc"})
        _, kwargs = local_compiled._device.run.call_args
        assert "tags" not in kwargs


class TestCloudSubmitTags:
    def test_tags_forwarded_to_device_run(self, cloud_compiled, weights, inputs):
        cloud_compiled.submit(weights, inputs, tags={"job_id": "abc"})
        _, kwargs = cloud_compiled._device.run.call_args
        assert kwargs["tags"] == {"job_id": "abc"}

    def test_no_tags_kwarg_when_tags_omitted(self, cloud_compiled, weights, inputs):
        cloud_compiled.submit(weights, inputs)
        _, kwargs = cloud_compiled._device.run.call_args
        assert "tags" not in kwargs

    def test_zsp_submit_uses_probability_circuit(self, cloud_compiled, weights, inputs):
        handle = cloud_compiled.submit(
            weights, inputs, method="zero_state_probability", tags={"k": "v"}
        )
        args, kwargs = cloud_compiled._device.run.call_args
        assert args[0] is cloud_compiled._zsp_circuit
        assert args[0] is not cloud_compiled._parametric_circuit
        assert kwargs["tags"] == {"k": "v"}
        assert handle.method == "zero_state_probability"

    def test_tags_at_aws_limits_accepted(self, cloud_compiled, weights, inputs):
        tags = {f"k{i}": "" for i in range(49)}
        tags["k" * 128] = "v" * 256
        cloud_compiled.submit(weights, inputs, tags=tags)
        _, kwargs = cloud_compiled._device.run.call_args
        assert kwargs["tags"] == tags

    @pytest.mark.parametrize(
        "tags",
        [
            {1: "v"},
            {"k": 1},
            {"": "v"},
            {"k" * 129: "v"},
            {"k": "v" * 257},
            {f"k{i}": "v" for i in range(51)},
        ],
        ids=[
            "non-str-key",
            "non-str-value",
            "empty-key",
            "key-too-long",
            "value-too-long",
            "too-many-tags",
        ],
    )
    def test_invalid_tags_rejected_before_device_call(
        self, cloud_compiled, weights, inputs, tags
    ):
        with pytest.raises(ValueError):
            cloud_compiled.submit(weights, inputs, tags=tags)
        cloud_compiled._device.run.assert_not_called()


class TestAttach:
    def test_attach_run_rebuilds_handle(self, ir, fake_aws_task):
        compiled = _compile(ir, BraketConfig(device_arn=_SV1_ARN, shots=100))
        fake_aws_task.values = [0.5, -0.25]
        handle = compiled.attach(_TASK_ARN, method="run")
        assert isinstance(handle, BraketJobHandle)
        assert handle.task_id() == _TASK_ARN
        assert handle.method == "run"
        expected = _aggregate_values([0.5, -0.25], compiled._aggregation_plan)
        assert np.array_equal(handle.result(), expected)
        assert fake_aws_task.created[-1].poll_timeout_seconds == 3600

    def test_attach_default_method_is_run(self, ir, fake_aws_task):
        compiled = _compile(ir, BraketConfig(device_arn=_SV1_ARN, shots=100))
        assert compiled.attach(_TASK_ARN).method == "run"

    def test_attach_zsp_decodes_like_sync_zsp(self, ir, fake_aws_task):
        compiled = _compile(ir, BraketConfig(device_arn=_SV1_ARN, shots=100))
        fake_aws_task.values = [[0.75, 0.05, 0.15, 0.05]]
        value = compiled.attach(_TASK_ARN, method="zero_state_probability").result()
        assert value.ndim == 0
        assert value.dtype == np.float64
        assert float(value) == 0.75

    def test_attach_matches_submit_handle_for_same_values(
        self, cloud_compiled, weights, inputs, fake_aws_task
    ):
        task = MagicMock()
        task.result.return_value = _FakeResult([0.5, -0.25])
        cloud_compiled._device.run.return_value = task
        submitted = cloud_compiled.submit(weights, inputs)
        fake_aws_task.values = [0.5, -0.25]
        attached = cloud_compiled.attach(_TASK_ARN)
        assert np.array_equal(submitted.result(), attached.result())

    def test_attach_on_local_config_raises(self, local_compiled):
        with pytest.raises(ValueError):
            local_compiled.attach(_TASK_ARN)

    def test_attach_unknown_method_raises(self, ir, fake_aws_task):
        compiled = _compile(ir, BraketConfig(device_arn=_SV1_ARN, shots=100))
        with pytest.raises(ValueError, match="submit_methods"):
            compiled.attach(_TASK_ARN, method="statevector")
        assert fake_aws_task.created == []


class TestJobHandleMethod:
    def test_handle_method_defaults_to_run(self):
        handle = BraketJobHandle(MagicMock(), [[(1.0, 0)]], 10)
        assert handle.method == "run"

    def test_handle_method_is_read_only(self):
        handle = BraketJobHandle(
            MagicMock(), [[(1.0, 0)]], 10, method="zero_state_probability"
        )
        assert handle.method == "zero_state_probability"
        with pytest.raises(AttributeError):
            handle.method = "run"


def _cloud(ir) -> BraketCompiled:
    return _compile(ir, BraketConfig(device_arn=_SV1_ARN, shots=100))


def _no_result_task(state: str) -> MagicMock:
    task = MagicMock()
    task.id = _TASK_ARN
    task.result.return_value = None
    task.state.return_value = state
    return task


class TestTaskWithoutResult:
    @pytest.mark.parametrize("state", ["FAILED", "CANCELLED"])
    @pytest.mark.parametrize("method", ["run", "zero_state_probability"])
    def test_terminal_task_without_result_raises_clearly(
        self, cloud_compiled, weights, inputs, state, method
    ):
        cloud_compiled._device.run.return_value = _no_result_task(state)
        handle = cloud_compiled.submit(weights, inputs, method=method)
        with pytest.raises(RuntimeError) as excinfo:
            handle.result()
        message = str(excinfo.value)
        assert _TASK_ARN in message
        assert state in message

    @pytest.mark.parametrize("method", ["run", "zero_state_probability"])
    def test_attached_failed_task_raises_clearly(self, ir, fake_aws_task, method):
        fake_aws_task.values = None
        fake_aws_task.task_state = "FAILED"
        handle = _cloud(ir).attach(_TASK_ARN, method=method)
        with pytest.raises(RuntimeError, match="FAILED"):
            handle.result()

    def test_poll_timeout_names_the_live_state(self, cloud_compiled, weights, inputs):
        cloud_compiled._device.run.return_value = _no_result_task("RUNNING")
        handle = cloud_compiled.submit(weights, inputs)
        with pytest.raises(RuntimeError, match="RUNNING") as excinfo:
            handle.result()
        assert "poll_timeout_seconds" in str(excinfo.value)


class TestTagCharacters:
    @pytest.mark.parametrize(
        "tags",
        [
            {"aws:created-by": "me"},
            {"AWS:Created-By": "me"},
            {"Aws:x": "v"},
            {"k#": "v"},
            {"k": "v;"},
            {"k\n": "v"},
            {"k": "v\tw"},
            {"k": "a*b"},
        ],
        ids=[
            "aws-prefix",
            "aws-prefix-upper",
            "aws-prefix-mixed",
            "hash-in-key",
            "semicolon-in-value",
            "newline-in-key",
            "tab-in-value",
            "star-in-value",
        ],
    )
    def test_reserved_or_disallowed_tags_rejected_before_device_call(
        self, cloud_compiled, weights, inputs, tags
    ):
        with pytest.raises(ValueError):
            cloud_compiled.submit(weights, inputs, tags=tags)
        cloud_compiled._device.run.assert_not_called()

    @pytest.mark.parametrize(
        "tags",
        [
            {"job_id": "0f8c2b3e-1d2a-4c5b-9e8f-7a6b5c4d3e2f"},
            {"a b_.:/=+-@": "x y_.:/=+-@ 09"},
            {"clé": "été"},
            {"k": "aws:value-is-fine"},
            {"app:aws": "v"},
        ],
        ids=[
            "uuid",
            "every-allowed-symbol",
            "unicode-letters",
            "aws-in-value",
            "aws-not-prefix",
        ],
    )
    def test_allowed_tags_accepted(self, cloud_compiled, weights, inputs, tags):
        cloud_compiled.submit(weights, inputs, tags=tags)
        _, kwargs = cloud_compiled._device.run.call_args
        assert kwargs["tags"] == tags


class TestAttachTaskIdShape:
    @pytest.mark.parametrize(
        "task_id",
        [
            "",
            "x",
            "0f8c2b3e-1d2a-4c5b-9e8f-7a6b5c4d3e2f",
            "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            "arn:aws:s3:us-east-1:123:quantum-task/x",
            None,
        ],
        ids=["empty", "word", "bare-uuid", "device-arn", "other-service", "none"],
    )
    def test_non_task_arn_rejected_before_sdk(self, ir, fake_aws_task, task_id):
        with pytest.raises(ValueError, match="quantum-task"):
            _cloud(ir).attach(task_id)
        assert fake_aws_task.created == []


class TestSubmitShapes:
    @pytest.mark.parametrize(
        ("w_shape", "x_shape"),
        [((1, 4), (2,)), ((4,), (1, 2)), ((2, 4), (2, 2)), ((), (2,))],
        ids=["2d-weights", "2d-inputs", "2d-both", "0d-weights"],
    )
    def test_non_1d_rejected_before_device_call(self, cloud_compiled, w_shape, x_shape):
        weights = np.full(w_shape, 0.1)
        inputs = np.full(x_shape, 0.2)
        with pytest.raises(ValueError, match="1-D"):
            cloud_compiled.submit(weights, inputs)
        cloud_compiled._device.run.assert_not_called()

    def test_single_sample_batch_of_one_value_is_rejected(self):
        # With one weight and one input, a (1, 1) batch could bind silently:
        # float() of a one-element array only warns. It must be refused.
        spec = assemble(
            [AngleEncoding(in_features=1), BasicVariational(depth=1, rotation="RY")]
        )
        assert (spec.n_weights, spec.n_inputs) == (1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostWarning)
            compiled = BraketBackend(
                config=BraketConfig(device_arn=_SV1_ARN, shots=100)
            ).compile(spec, all_z(1))
        compiled._device = MagicMock()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(ValueError, match="1-D"):
                compiled.submit(np.array([[0.3]]), np.array([[0.1]]))
        compiled._device.run.assert_not_called()


class TestResultShapeGuard:
    def test_run_handle_given_probability_vector_raises(self, ir, fake_aws_task):
        fake_aws_task.values = [[0.75, 0.05, 0.15, 0.05]]
        handle = _cloud(ir).attach(_TASK_ARN, method="run")
        with pytest.raises(ValueError, match="method"):
            handle.result()

    def test_run_handle_given_wrong_value_count_raises(self, ir, fake_aws_task):
        fake_aws_task.values = [0.5, -0.25, 0.1]
        handle = _cloud(ir).attach(_TASK_ARN, method="run")
        with pytest.raises(ValueError, match="3"):
            handle.result()

    def test_run_handle_single_term_given_vector_raises(self):
        task = MagicMock()
        task.result.return_value = _FakeResult([np.array([0.5, 0.5])])
        handle = BraketJobHandle(task, [[(1.0, 0)]], 10)
        with pytest.raises(ValueError, match="method"):
            handle.result()

    def test_zsp_handle_given_expectation_values_raises(self, ir, fake_aws_task):
        fake_aws_task.values = [0.5, -0.25]
        handle = _cloud(ir).attach(_TASK_ARN, method="zero_state_probability")
        with pytest.raises(ValueError, match="method"):
            handle.result()

    def test_zsp_handle_given_one_expectation_value_raises(self, ir, fake_aws_task):
        fake_aws_task.values = [0.5]
        handle = _cloud(ir).attach(_TASK_ARN, method="zero_state_probability")
        with pytest.raises(ValueError, match="method"):
            handle.result()

    def test_zsp_handle_given_wrong_width_vector_raises(self, ir, fake_aws_task):
        fake_aws_task.values = [[0.75, 0.25]]
        handle = _cloud(ir).attach(_TASK_ARN, method="zero_state_probability")
        with pytest.raises(ValueError, match="4"):
            handle.result()


def _typed_result(result_type, value):
    """A task result carrying one result type, as the SDK builds it."""
    return SimpleNamespace(
        values=[value],
        result_types=[SimpleNamespace(type=result_type, value=value)],
    )


class TestSampleAndProbabilitiesHooks:
    """submit/attach for the probabilities and zero_state_probability methods."""

    def test_local_submit_sample_counts_sum_to_shots(self, ir, weights, inputs):
        compiled = _compile(ir, BraketConfig(shots=300))
        counts = compiled.submit(weights, inputs, method="sample").result()
        assert sum(counts.values()) == 300
        assert all(len(key) == 2 and set(key) <= {"0", "1"} for key in counts)

    def test_local_submit_probabilities_matches_blocking(
        self, local_compiled, weights, inputs
    ):
        handle = local_compiled.submit(weights, inputs, method="probabilities")
        assert handle.method == "probabilities"
        assert np.allclose(
            handle.result(), local_compiled.probabilities(weights, inputs), atol=1e-12
        )

    def test_cloud_submit_sample_uses_a_circuit_without_result_types(
        self, cloud_compiled, weights, inputs
    ):
        handle = cloud_compiled.submit(
            weights, inputs, method="sample", tags={"job_id": "abc"}
        )
        args, kwargs = cloud_compiled._device.run.call_args
        assert args[0].result_types == []
        assert kwargs["tags"] == {"job_id": "abc"}
        assert kwargs["shots"] == 100
        assert handle.method == "sample"

    def test_cloud_submit_probabilities_with_shots_uses_probability_circuit(
        self, cloud_compiled, weights, inputs
    ):
        cloud_compiled.submit(weights, inputs, method="probabilities")
        args, _ = cloud_compiled._device.run.call_args
        assert args[0] is cloud_compiled._zsp_circuit

    def test_attach_sample_decodes_measurement_counts(self, ir, fake_aws_task):
        fake_aws_task.full_result = SimpleNamespace(
            values=[],
            result_types=[],
            measurement_counts={"10": 60, "01": 40},
            measured_qubits=[0, 1],
        )
        counts = _cloud(ir).attach(_TASK_ARN, method="sample").result()
        assert counts == {"10": 60, "01": 40}
        assert all(type(count) is int for count in counts.values())

    @pytest.mark.parametrize("kind", ["Probability", "StateVector", "DensityMatrix"])
    def test_attach_probabilities_decodes_each_result_type(self, ir, fake_aws_task, kind):
        from braket.ir import jaqcd

        probs = np.array([0.1, 0.2, 0.3, 0.4])
        amplitudes = np.sqrt(probs) * np.exp(1j * np.arange(4))
        value, result_type = {
            "Probability": (probs.copy(), jaqcd.Probability(targets=[0, 1])),
            "StateVector": (amplitudes, jaqcd.StateVector()),
            "DensityMatrix": (
                np.outer(amplitudes, amplitudes.conj()),
                jaqcd.DensityMatrix(targets=[0, 1]),
            ),
        }[kind]
        fake_aws_task.full_result = _typed_result(result_type, value)
        decoded = _cloud(ir).attach(_TASK_ARN, method="probabilities").result()
        assert decoded.dtype == np.float64
        assert decoded.shape == (4,)
        assert np.allclose(decoded, probs)

    def test_attach_sample_given_expectation_task_raises(self, ir, fake_aws_task):
        fake_aws_task.values = [0.5, -0.25]
        handle = _cloud(ir).attach(_TASK_ARN, method="sample")
        with pytest.raises(ValueError, match="method"):
            handle.result()

    @pytest.mark.parametrize("method", ["sample", "probabilities"])
    def test_attached_cancelled_task_raises_clearly(self, ir, fake_aws_task, method):
        fake_aws_task.values = None
        fake_aws_task.task_state = "CANCELLED"
        with pytest.raises(RuntimeError, match="CANCELLED"):
            _cloud(ir).attach(_TASK_ARN, method=method).result()
