"""Live Braket tests: real SV1, DM1 and QPU tasks, billed to the AWS account.

Everything else in the suite fakes the device. These create real quantum tasks,
so they are gated twice, and neither gate alone runs anything:

1. every test is marked `qpu`, which the root `conftest.py` skips unless `-m`
   names `qpu`;
2. the whole module skips unless the environment sets `EIGENKIT_BRAKET_LIVE=1`.

Cost, at Amazon Braket's published prices:

- SV1 and DM1 bill $0.075 per minute with a 3-second minimum per task, so each
  of these tiny tasks costs about $0.004. `TestSimulatorExact` creates four
  tasks, about $0.015.
- A QPU task costs $0.30 plus shots times the per-shot price: $0.000425 on
  Rigetti Cepheus-1-108Q, $0.00145 on IQM Garnet. `TestHardware` creates two
  10-shot tasks: $0.30425 each on Cepheus ($0.61 for both), $0.3145 each on
  Garnet ($0.63). Only these two devices are accepted; IonQ, AQT and every other
  QPU are refused before any task is created.

The whole module on Cepheus is about $0.62.

Environment:

- `EIGENKIT_BRAKET_LIVE=1`: required, together with `-m qpu`.
- `EIGENKIT_BRAKET_LIVE_S3_BUCKET`: an existing results bucket in the simulator
  region. `TestSimulatorExact` skips without it.
- `EIGENKIT_BRAKET_LIVE_SIM_REGION`: where SV1 and DM1 run, default `us-east-1`.
  The ARNs of the managed simulators carry no region, so the SDK uses the
  session's; the tests set `AWS_DEFAULT_REGION` to this for the duration.
- `EIGENKIT_BRAKET_LIVE_QPU_ARN`: the QPU, default Rigetti Cepheus-1-108Q in
  us-west-1. IQM Garnet (eu-north-1) is the only alternative accepted.
- `EIGENKIT_BRAKET_LIVE_QPU_S3_BUCKET`: an existing results bucket in the QPU's
  region. `TestHardware` skips without it.
- `EIGENKIT_BRAKET_LIVE_S3_PREFIX`: key prefix inside both buckets, default
  `kit-live-tests`.
- `EIGENKIT_BRAKET_LIVE_QPU_TIMEOUT`: seconds to wait for a QPU task, default 1800.
  A task still queued or running then is cancelled, so it is not billed later;
  Braket cancels a queued QPU task on a best-effort basis only.
- AWS credentials come from the standard boto3 chain.

Results bucket: `BraketConfig(s3_bucket=..., s3_prefix=...)` hands the SDK an
explicit `s3_destination_folder`. Without one, `AwsDevice.run()` falls back to
`AMZN_BRAKET_TASK_RESULTS_S3_URI` and then to `AwsSession.default_bucket()`,
which calls CreateBucket for `amazon-braket-<region>-<account>` when that bucket
does not exist. A missing bucket variable therefore skips the tests here rather
than letting the SDK choose, and create, one. Braket writes the results under
`<bucket>/<prefix>/<task id>/`.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest


pytest.importorskip("braket")

from eigenkit.core.backends.braket_backend import (  # noqa: E402
    BraketBackend,
    BraketCompiled,
    BraketConfig,
)
from eigenkit.core.backends.numpy_backend import NumpySimulator  # noqa: E402
from eigenkit.core.ir import (  # noqa: E402
    CircuitSpec,
    Gate,
    InputRef,
    WeightRef,
)
from eigenkit.core.observables import all_z  # noqa: E402
from eigenkit.core.provider import CostWarning  # noqa: E402


_LIVE_VARIABLE = "EIGENKIT_BRAKET_LIVE"

pytestmark = pytest.mark.skipif(
    os.environ.get(_LIVE_VARIABLE) != "1",
    reason=(
        f"live Braket test: creates billed AWS tasks (SV1/DM1 about $0.004 per "
        f"task, a QPU task $0.30 plus shots x the per-shot price, about $0.62 for "
        f"the module on Cepheus). Set {_LIVE_VARIABLE}=1 and run with -m qpu."
    ),
)

_SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
_DM1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"
_CEPHEUS_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
_GARNET_ARN = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
# The only QPUs this module will submit to, with their per-shot price in USD.
_CHEAP_QPUS = {_CEPHEUS_ARN: 0.000425, _GARNET_ARN: 0.00145}

_QPU_SHOTS = 10
_SIM_POLL_TIMEOUT_SECONDS = 600
_EXACT_ATOL = 1e-9

# Three qubits with qubit 1 idle, so every decode has to report a qubit no gate
# touches. The simulator circuit is parametric, so the bound inputs travel too.
_SIM_SPEC = CircuitSpec(
    n_qubits=3,
    n_weights=2,
    n_inputs=1,
    gates=(
        Gate("RY", (0,), (WeightRef(0),)),
        Gate("CNOT", (0, 2)),
        Gate("RX", (2,), (WeightRef(1),)),
        Gate("RY", (2,), (InputRef(0),)),
    ),
)
_SIM_WEIGHTS = np.array([0.7, 1.3])
_SIM_INPUTS = np.array([0.4])

# A Bell pair on qubits 0 and 2 around the idle qubit 1.
_QPU_SPEC = CircuitSpec(
    n_qubits=3,
    n_weights=0,
    n_inputs=0,
    gates=(Gate("H", (0,)), Gate("CNOT", (0, 2))),
)
_NO_ARGS = (np.zeros(0), np.zeros(0))
_IDLE_QUBIT = 1


def _compile(spec: CircuitSpec, config: BraketConfig) -> BraketCompiled:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostWarning)
        return BraketBackend(config=config).compile(spec, all_z(spec.n_qubits))


def _reference():
    compiled = NumpySimulator().compile(_SIM_SPEC, all_z(_SIM_SPEC.n_qubits))
    return (
        compiled.probabilities(_SIM_WEIGHTS, _SIM_INPUTS),
        compiled.zero_state_probability(_SIM_WEIGHTS, _SIM_INPUTS),
    )


def _prefix() -> str:
    return os.environ.get("EIGENKIT_BRAKET_LIVE_S3_PREFIX", "kit-live-tests")


def _region(arn: str) -> str:
    return arn.split(":")[3]


@pytest.fixture
def simulator_config(monkeypatch):
    """Build a shots=0 config for a managed simulator, pinned to its region."""
    bucket = os.environ.get("EIGENKIT_BRAKET_LIVE_S3_BUCKET")
    if not bucket:
        pytest.skip(
            "set EIGENKIT_BRAKET_LIVE_S3_BUCKET to an existing results bucket in the "
            "simulator region; without one the SDK would create its default bucket"
        )
    monkeypatch.setenv(
        "AWS_DEFAULT_REGION",
        os.environ.get("EIGENKIT_BRAKET_LIVE_SIM_REGION", "us-east-1"),
    )

    def build(arn: str) -> BraketConfig:
        return BraketConfig(
            device_arn=arn,
            shots=0,
            s3_bucket=bucket,
            s3_prefix=_prefix(),
            poll_timeout_seconds=_SIM_POLL_TIMEOUT_SECONDS,
        )

    return build


@pytest.fixture
def qpu_config(monkeypatch) -> BraketConfig:
    """A 10-shot config for the one QPU this run may use."""
    arn = os.environ.get("EIGENKIT_BRAKET_LIVE_QPU_ARN", _CEPHEUS_ARN)
    if arn not in _CHEAP_QPUS:
        pytest.fail(
            f"EIGENKIT_BRAKET_LIVE_QPU_ARN={arn!r} is not one of the QPUs this module "
            f"may bill: {sorted(_CHEAP_QPUS)}."
        )
    bucket = os.environ.get("EIGENKIT_BRAKET_LIVE_QPU_S3_BUCKET")
    if not bucket:
        pytest.skip(
            f"set EIGENKIT_BRAKET_LIVE_QPU_S3_BUCKET to an existing results bucket in "
            f"{_region(arn)}; without one the SDK would create its default bucket"
        )
    monkeypatch.setenv("AWS_DEFAULT_REGION", _region(arn))
    return BraketConfig(
        device_arn=arn,
        shots=_QPU_SHOTS,
        s3_bucket=bucket,
        s3_prefix=_prefix(),
        poll_timeout_seconds=int(
            os.environ.get("EIGENKIT_BRAKET_LIVE_QPU_TIMEOUT", "1800")
        ),
    )


_TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})


def _run_on_qpu(compiled: BraketCompiled, method: str):
    """Submit one task and wait for it, cancelling it if it never finishes."""
    handle = compiled.submit(*_NO_ARGS, method=method)
    print(f"{method}: Braket task {handle.task_id()}")
    try:
        return handle.result()
    except BaseException:
        # Still queued or running after the poll timeout, or interrupted while
        # waiting: cancel, so the task is not billed after the test gave up.
        if handle.status() not in _TERMINAL_STATES:
            handle.cancel()
        raise


_SIMULATORS = pytest.mark.parametrize(
    ("arn", "result_type"),
    [(_SV1_ARN, "Amplitude"), (_DM1_ARN, "DensityMatrix")],
    ids=["sv1", "dm1"],
)


class TestSimulatorExact:
    """shots=0 on SV1 and DM1 is exact, so it matches the numpy statevector
    reference to double precision. One task per test."""

    @pytest.mark.qpu
    @_SIMULATORS
    def test_probabilities_match_the_statevector(
        self, simulator_config, arn, result_type
    ):
        compiled = _compile(_SIM_SPEC, simulator_config(arn))
        expected, _ = _reference()
        probs = compiled.probabilities(_SIM_WEIGHTS, _SIM_INPUTS)
        assert probs.dtype == np.float64
        assert probs.shape == (2**_SIM_SPEC.n_qubits,)
        np.testing.assert_allclose(probs, expected, atol=_EXACT_ATOL, rtol=0)
        # The capability read from GetDevice picked the exact result type.
        assert ("probabilities", result_type) in compiled._measurement_circuits

    @pytest.mark.qpu
    @_SIMULATORS
    def test_zero_state_probability_matches_the_statevector(
        self, simulator_config, arn, result_type
    ):
        compiled = _compile(_SIM_SPEC, simulator_config(arn))
        _, expected = _reference()
        value = compiled.zero_state_probability(_SIM_WEIGHTS, _SIM_INPUTS)
        assert type(value) is float
        assert value == pytest.approx(expected, abs=_EXACT_ATOL)
        assert ("zero_state_probability", result_type) in compiled._measurement_circuits


class TestHardware:
    """10 shots on the configured QPU. Hardware is noisy, so these check shape,
    normalisation and bit order, never the physics. One task per test."""

    @pytest.mark.qpu
    def test_probabilities_cover_the_idle_qubit(self, qpu_config):
        compiled = _compile(_QPU_SPEC, qpu_config)
        probs = _run_on_qpu(compiled, "probabilities")
        # The idle qubit went to the device as an identity gate.
        assert any(
            type(instruction.operator).__name__ == "I"
            and list(instruction.target) == [_IDLE_QUBIT]
            for instruction in compiled._zsp_circuit.instructions
        )
        assert probs.dtype == np.float64
        assert probs.shape == (2**_QPU_SPEC.n_qubits,)
        assert np.all(probs >= 0)
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)
        # A 10-shot frequency: every entry is a whole number of shots.
        np.testing.assert_allclose(
            probs * _QPU_SHOTS, np.round(probs * _QPU_SHOTS), atol=1e-9, rtol=0
        )

    @pytest.mark.qpu
    def test_sample_reports_the_idle_qubit_as_zero(self, qpu_config):
        compiled = _compile(_QPU_SPEC, qpu_config)
        counts = _run_on_qpu(compiled, "sample")
        assert sum(counts.values()) == _QPU_SHOTS
        assert all(type(count) is int for count in counts.values())
        assert all(len(key) == _QPU_SPEC.n_qubits for key in counts)
        assert all(set(key) <= {"0", "1"} for key in counts)
        # Braket measures only the qubits the circuit uses; the idle one is 0.
        assert all(key[_IDLE_QUBIT] == "0" for key in counts)
