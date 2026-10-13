"""Braket `sample`, `probabilities` and the exact shots=0 paths.

`sample` reads the task's measurement counts. `probabilities` and
`zero_state_probability` read the result type the device publishes for the
requested shots: `Probability` whenever shots > 0, and at shots=0 on the
managed simulators an exact one instead (`Amplitude` on SV1, `DensityMatrix`
on DM1), because neither publishes a shots=0 `Probability`.

Nothing here contacts AWS. The local simulator covers the values; a fake
`AwsDevice` that wraps the local simulator and publishes SV1-, DM1- or
QPU-like capabilities covers the result-type selection end to end.
"""

from __future__ import annotations

import re
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytest.importorskip("braket")

import numpy as np  # noqa: E402
from braket.devices import LocalSimulator  # noqa: E402
from braket.ir import jaqcd  # noqa: E402

from eigenkit.core.backends import braket_backend as bb  # noqa: E402
from eigenkit.core.backends.braket_backend import (  # noqa: E402
    BraketBackend,
    BraketCompiled,
    BraketConfig,
    BraketJobHandle,
)
from eigenkit.core.blocks import (  # noqa: E402
    AngleEncoding,
    BasicVariational,
    assemble,
)
from eigenkit.core.ir import CircuitSpec, Gate  # noqa: E402
from eigenkit.core.observables import all_z  # noqa: E402
from eigenkit.core.provider import CostWarning  # noqa: E402


_SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
_DM1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"
_QPU_ARN = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
_TASK_ARN = "arn:aws:braket:us-east-1:123:quantum-task/x"

# supportedResultTypes of the braket.ir.openqasm.program action, as GetDevice
# reported them for SV1 and DM1 in us-east-1 on 2026-09-28, and the shape a
# gate-based QPU publishes (one shot range for all four result types).
SV1_LIMITS = {
    "Sample": (1, 100000),
    "Expectation": (0, 100000),
    "Variance": (0, 100000),
    "Probability": (1, 100000),
    "Amplitude": (0, 0),
    "AdjointGradient": (0, 0),
}
DM1_LIMITS = {
    "Sample": (1, 100000),
    "Expectation": (0, 100000),
    "Variance": (0, 100000),
    "DensityMatrix": (0, 0),
    "Probability": (1, 100000),
}
QPU_LIMITS = {
    "Sample": (1, 20000),
    "Expectation": (1, 20000),
    "Variance": (1, 20000),
    "Probability": (1, 20000),
}


def _compile(spec, config: BraketConfig) -> BraketCompiled:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostWarning)
        return BraketBackend(config=config).compile(spec, all_z(spec.n_qubits))


def _x_on_qubit_0() -> CircuitSpec:
    # Qubit 1 carries no gate on purpose: every decode must still report it.
    return CircuitSpec(n_qubits=2, n_weights=0, n_inputs=0, gates=(Gate("X", (0,)),))


def _h_on_qubit_0() -> CircuitSpec:
    return CircuitSpec(n_qubits=2, n_weights=0, n_inputs=0, gates=(Gate("H", (0,)),))


def _variational() -> CircuitSpec:
    return assemble(
        [AngleEncoding(in_features=3), BasicVariational(depth=2, rotation="RY")]
    )


_NO_ARGS = (np.zeros(0), np.zeros(0))
_W = np.array([0.4, 1.1, -0.3, 0.9, 0.2, -0.7], dtype=np.float64)
_X = np.array([0.5, -0.2, 0.8], dtype=np.float64)


class _FakeAwsDevice:
    """An `AwsDevice` stand-in: runs on the local simulator, drops the
    cloud-only keyword arguments, and publishes the given capabilities."""

    def __init__(self, local_backend: str, limits: dict[str, tuple[int, int]]):
        self._simulator = LocalSimulator(local_backend)
        entries = [
            SimpleNamespace(name=name, minShots=low, maxShots=high)
            for name, (low, high) in limits.items()
        ]
        self.properties = SimpleNamespace(
            action={
                "braket.ir.openqasm.program": SimpleNamespace(
                    supportedResultTypes=entries
                )
            }
        )
        self.circuits: list = []

    def run(self, circuit, *, shots=0, inputs=None, **cloud_kwargs):
        self.circuits.append(circuit)
        with warnings.catch_warnings():
            # The local DM simulator warns on a noise-free circuit; DM1 does not.
            warnings.filterwarnings("ignore", "You are running a noise-free circuit")
            return self._simulator.run(circuit, shots=shots, inputs=inputs or {})


def _cloud(spec, arn, shots, local_backend, limits) -> BraketCompiled:
    compiled = _compile(spec, BraketConfig(device_arn=arn, shots=shots))
    compiled._device = _FakeAwsDevice(local_backend, limits)
    return compiled


def _result_type_name(circuit) -> str:
    (result_type,) = circuit.result_types
    return type(result_type).__name__


class TestSubmitMethods:
    def test_submit_methods_include_sample_and_probabilities(self):
        assert BraketCompiled.submit_methods == frozenset(
            {"run", "sample", "probabilities", "zero_state_probability"}
        )

    @pytest.mark.parametrize(
        "method", ["run", "sample", "probabilities", "zero_state_probability"]
    )
    def test_handle_accepts_every_submit_method(self, method):
        handle = BraketJobHandle(MagicMock(), [[(1.0, 0)]], 10, method=method)
        assert handle.method == method


class TestLocalSample:
    def test_x_on_qubit_0_is_qubit0_msb(self):
        compiled = _compile(_x_on_qubit_0(), BraketConfig(shots=1000))
        assert compiled.sample(*_NO_ARGS) == {"10": 1000}

    def test_hadamard_counts_sum_to_shots(self):
        compiled = _compile(_h_on_qubit_0(), BraketConfig(shots=500))
        counts = compiled.sample(*_NO_ARGS)
        assert set(counts) <= {"00", "10"}
        assert sum(counts.values()) == 500
        assert all(type(count) is int for count in counts.values())

    def test_sample_with_zero_shots_raises_before_device_access(self):
        compiled = _compile(_x_on_qubit_0(), BraketConfig(shots=0))
        compiled._device = MagicMock()
        with pytest.raises(ValueError, match="shots > 0"):
            compiled.sample(*_NO_ARGS)
        compiled._device.run.assert_not_called()

    def test_submit_sample_with_zero_shots_raises_before_device_access(self):
        compiled = _compile(_x_on_qubit_0(), BraketConfig(shots=0))
        compiled._device = MagicMock()
        with pytest.raises(ValueError, match="shots > 0"):
            compiled.submit(*_NO_ARGS, method="sample")
        compiled._device.run.assert_not_called()

    def test_submit_sample_matches_blocking_shape(self):
        compiled = _compile(_x_on_qubit_0(), BraketConfig(shots=200))
        assert compiled.submit(*_NO_ARGS, method="sample").result() == {"10": 200}


class TestLocalProbabilities:
    def test_x_on_qubit_0_exact(self):
        compiled = _compile(_x_on_qubit_0(), BraketConfig(shots=0))
        probs = compiled.probabilities(*_NO_ARGS)
        assert probs.dtype == np.float64
        assert probs.shape == (4,)
        assert np.array_equal(probs, [0.0, 0.0, 1.0, 0.0])

    def test_x_on_qubit_0_with_shots(self):
        compiled = _compile(_x_on_qubit_0(), BraketConfig(shots=1000))
        assert np.array_equal(compiled.probabilities(*_NO_ARGS), [0.0, 0.0, 1.0, 0.0])

    def test_hadamard_exact(self):
        compiled = _compile(_h_on_qubit_0(), BraketConfig(shots=0))
        assert np.allclose(
            compiled.probabilities(*_NO_ARGS), [0.5, 0.0, 0.5, 0.0], atol=1e-12, rtol=0
        )

    def test_zsp_on_a_circuit_with_an_idle_qubit(self):
        compiled = _compile(_h_on_qubit_0(), BraketConfig(shots=0))
        assert compiled.zero_state_probability(*_NO_ARGS) == pytest.approx(0.5, abs=1e-12)

    def test_zsp_is_the_first_entry_of_probabilities(self):
        compiled = _compile(_variational(), BraketConfig(shots=0))
        probs = compiled.probabilities(_W, _X)
        assert probs.sum() == pytest.approx(1.0, abs=1e-12)
        assert compiled.zero_state_probability(_W, _X) == pytest.approx(
            probs[0], abs=1e-12
        )

    def test_submit_probabilities_matches_blocking(self):
        compiled = _compile(_variational(), BraketConfig(shots=0))
        handle = compiled.submit(_W, _X, method="probabilities")
        assert np.allclose(handle.result(), compiled.probabilities(_W, _X), atol=1e-12)


class TestResultTypeSelection:
    @pytest.mark.parametrize(
        ("limits", "method", "expected"),
        [
            (SV1_LIMITS, "zero_state_probability", "Amplitude"),
            (DM1_LIMITS, "zero_state_probability", "DensityMatrix"),
            (SV1_LIMITS, "probabilities", "Amplitude"),
            (DM1_LIMITS, "probabilities", "DensityMatrix"),
        ],
        ids=["sv1-zsp", "dm1-zsp", "sv1-probabilities", "dm1-probabilities"],
    )
    def test_shots_zero_picks_the_exact_result_type(self, limits, method, expected):
        assert bb._select_result_type(method, 0, limits, "dev") == expected

    def test_statevector_is_preferred_when_published_at_shots_zero(self):
        limits = {**SV1_LIMITS, "StateVector": (0, 0)}
        assert bb._select_result_type("probabilities", 0, limits, "dev") == "StateVector"

    @pytest.mark.parametrize(
        "limits", [SV1_LIMITS, DM1_LIMITS, QPU_LIMITS], ids=["sv1", "dm1", "hardware"]
    )
    @pytest.mark.parametrize("method", ["zero_state_probability", "probabilities"])
    def test_shots_positive_picks_probability(self, limits, method):
        assert bb._select_result_type(method, 100, limits, "dev") == "Probability"

    @pytest.mark.parametrize(
        "limits", [SV1_LIMITS, DM1_LIMITS, QPU_LIMITS], ids=["sv1", "dm1", "hardware"]
    )
    def test_sample_reads_measurement_counts(self, limits):
        assert bb._select_result_type("sample", 100, limits, "dev") is None

    def test_sample_with_zero_shots_raises(self):
        with pytest.raises(ValueError, match="shots > 0"):
            bb._select_result_type("sample", 0, SV1_LIMITS, "dev")

    @pytest.mark.parametrize("method", ["zero_state_probability", "probabilities"])
    def test_no_candidate_raises_naming_device_and_method(self, method):
        with pytest.raises(ValueError) as excinfo:
            bb._select_result_type(method, 0, QPU_LIMITS, _QPU_ARN)
        assert _QPU_ARN in str(excinfo.value)
        assert method in str(excinfo.value)

    def test_shots_above_the_device_range_raise(self):
        with pytest.raises(ValueError, match="probabilities"):
            bb._select_result_type("probabilities", 30000, QPU_LIMITS, _QPU_ARN)

    def test_unknown_capabilities_keep_the_probability_path(self):
        assert (
            bb._select_result_type("zero_state_probability", 0, None, "dev")
            == "Probability"
        )
        assert bb._select_result_type("probabilities", 100, None, "dev") == "Probability"


def _local_exact(spec, weights, inputs):
    local = _compile(spec, BraketConfig(shots=0))
    return local.probabilities(weights, inputs), local.zero_state_probability(
        weights, inputs
    )


_EXACT_DEVICES = [
    (_SV1_ARN, "braket_sv", SV1_LIMITS, "Amplitude"),
    (_DM1_ARN, "braket_dm", DM1_LIMITS, "DensityMatrix"),
]


class TestExactCloudPaths:
    @pytest.mark.parametrize(
        ("arn", "local_backend", "limits", "result_type"),
        _EXACT_DEVICES,
        ids=["sv1", "dm1"],
    )
    def test_zsp_shots_zero_matches_local_exact(
        self, arn, local_backend, limits, result_type
    ):
        compiled = _cloud(_variational(), arn, 0, local_backend, limits)
        _, expected = _local_exact(_variational(), _W, _X)
        value = compiled.zero_state_probability(_W, _X)
        assert type(value) is float
        assert value == pytest.approx(expected, abs=1e-12)
        assert _result_type_name(compiled._device.circuits[-1]) == result_type

    @pytest.mark.parametrize(
        ("arn", "local_backend", "limits", "result_type"),
        _EXACT_DEVICES,
        ids=["sv1", "dm1"],
    )
    def test_probabilities_shots_zero_matches_local_exact(
        self, arn, local_backend, limits, result_type
    ):
        compiled = _cloud(_variational(), arn, 0, local_backend, limits)
        expected, _ = _local_exact(_variational(), _W, _X)
        probs = compiled.probabilities(_W, _X)
        assert probs.dtype == np.float64
        assert probs.shape == (8,)
        assert np.allclose(probs, expected, atol=1e-12, rtol=0)
        assert _result_type_name(compiled._device.circuits[-1]) == result_type

    @pytest.mark.parametrize(
        ("arn", "local_backend", "limits", "result_type"),
        _EXACT_DEVICES,
        ids=["sv1", "dm1"],
    )
    def test_x_on_qubit_0_bit_order_on_exact_paths(
        self, arn, local_backend, limits, result_type
    ):
        compiled = _cloud(_x_on_qubit_0(), arn, 0, local_backend, limits)
        assert np.allclose(compiled.probabilities(*_NO_ARGS), [0, 0, 1, 0], atol=1e-12)
        assert compiled.zero_state_probability(*_NO_ARGS) == pytest.approx(0.0, abs=1e-12)

    def test_zsp_with_shots_keeps_the_probability_circuit(self):
        compiled = _cloud(_variational(), _SV1_ARN, 100, "braket_sv", SV1_LIMITS)
        compiled.zero_state_probability(_W, _X)
        circuit = compiled._device.circuits[-1]
        assert circuit is compiled._zsp_circuit
        assert _result_type_name(circuit) == "Probability"

    def test_probabilities_with_shots_uses_the_probability_result_type(self):
        compiled = _cloud(_x_on_qubit_0(), _SV1_ARN, 300, "braket_sv", QPU_LIMITS)
        assert np.array_equal(compiled.probabilities(*_NO_ARGS), [0.0, 0.0, 1.0, 0.0])
        assert _result_type_name(compiled._device.circuits[-1]) == "Probability"

    def test_sample_circuit_has_no_result_types(self):
        compiled = _cloud(_x_on_qubit_0(), _SV1_ARN, 50, "braket_sv", SV1_LIMITS)
        assert compiled.sample(*_NO_ARGS) == {"10": 50}
        assert compiled._device.circuits[-1].result_types == []

    def test_device_without_an_exact_result_type_raises_before_running(self):
        # A QPU config refuses shots=0 at compile time already; the selection
        # is the second line of defence for any device without an exact type.
        compiled = _cloud(_x_on_qubit_0(), _SV1_ARN, 0, "braket_sv", QPU_LIMITS)
        with pytest.raises(ValueError, match=re.escape(_SV1_ARN)):
            compiled.probabilities(*_NO_ARGS)
        assert compiled._device.circuits == []


def _h_on_qubit_0_of(n_qubits: int) -> CircuitSpec:
    # Every qubit but 0 is idle, which the exact paths still have to cover.
    return CircuitSpec(
        n_qubits=n_qubits, n_weights=0, n_inputs=0, gates=(Gate("H", (0,)),)
    )


_AMPLITUDE_CAP = bb._MAX_AMPLITUDE_LISTING_QUBITS
_DENSITY_CAP = bb._MAX_DENSITY_MATRIX_QUBITS


class TestExactWidthCaps:
    """The exact shots=0 result types that grow with the circuit are capped at
    the width Braket accepts, and the refusal comes before any task."""

    def test_sv1_exact_probabilities_above_the_cap_raise_before_running(self):
        compiled = _cloud(
            _h_on_qubit_0_of(_AMPLITUDE_CAP + 1), _SV1_ARN, 0, "braket_sv", SV1_LIMITS
        )
        with pytest.raises(ValueError) as excinfo:
            compiled.probabilities(*_NO_ARGS)
        message = str(excinfo.value)
        assert _SV1_ARN in message
        assert f"2**{_AMPLITUDE_CAP + 1}" in message
        assert "shots > 0" in message
        assert "sample()" in message
        assert compiled._device.circuits == []
        # The 2**n state list is never even built.
        assert compiled._measurement_circuits == {}

    def test_sv1_submit_probabilities_above_the_cap_raises_before_running(self):
        compiled = _cloud(
            _h_on_qubit_0_of(_AMPLITUDE_CAP + 1), _SV1_ARN, 0, "braket_sv", SV1_LIMITS
        )
        with pytest.raises(ValueError, match="shots > 0"):
            compiled.submit(*_NO_ARGS, method="probabilities")
        assert compiled._device.circuits == []

    def test_sv1_exact_zsp_above_the_cap_reads_one_amplitude(self):
        compiled = _cloud(
            _h_on_qubit_0_of(_AMPLITUDE_CAP + 1), _SV1_ARN, 0, "braket_sv", SV1_LIMITS
        )
        assert compiled.zero_state_probability(*_NO_ARGS) == pytest.approx(0.5, abs=1e-12)
        (result_type,) = compiled._device.circuits[-1].result_types
        assert type(result_type).__name__ == "Amplitude"
        assert result_type.state == ["0" * (_AMPLITUDE_CAP + 1)]

    @pytest.mark.parametrize("method", ["probabilities", "zero_state_probability"])
    def test_dm1_exact_paths_above_the_cap_raise_before_running(self, method):
        compiled = _cloud(
            _h_on_qubit_0_of(_DENSITY_CAP + 1), _DM1_ARN, 0, "braket_dm", DM1_LIMITS
        )
        with pytest.raises(ValueError) as excinfo:
            getattr(compiled, method)(*_NO_ARGS)
        message = str(excinfo.value)
        assert _DM1_ARN in message
        assert f"{method}()" in message
        assert f"at most {_DENSITY_CAP}" in message
        assert "shots > 0" in message
        assert compiled._device.circuits == []
        assert compiled._measurement_circuits == {}

    def test_dm1_exact_paths_at_the_cap_run(self):
        compiled = _cloud(
            _h_on_qubit_0_of(_DENSITY_CAP), _DM1_ARN, 0, "braket_dm", DM1_LIMITS
        )
        expected = np.zeros(2**_DENSITY_CAP)
        # H on qubit 0, the most significant bit: |00..0> and |10..0>.
        expected[0] = expected[2 ** (_DENSITY_CAP - 1)] = 0.5
        assert np.allclose(compiled.probabilities(*_NO_ARGS), expected, atol=1e-12)
        assert compiled.zero_state_probability(*_NO_ARGS) == pytest.approx(0.5, abs=1e-12)
        assert _result_type_name(compiled._device.circuits[-1]) == "DensityMatrix"

    @pytest.mark.parametrize(
        ("method", "result_type", "cap"),
        [
            ("probabilities", "Amplitude", _AMPLITUDE_CAP),
            ("probabilities", "DensityMatrix", _DENSITY_CAP),
            ("zero_state_probability", "DensityMatrix", _DENSITY_CAP),
        ],
        ids=["sv1-probabilities", "dm1-probabilities", "dm1-zsp"],
    )
    def test_the_cap_itself_is_allowed(self, method, result_type, cap):
        bb._check_result_width(method, result_type, cap, "dev", cloud=True)
        with pytest.raises(ValueError, match="dev"):
            bb._check_result_width(method, result_type, cap + 1, "dev", cloud=True)

    @pytest.mark.parametrize("cloud", [True, False], ids=["cloud", "local"])
    @pytest.mark.parametrize(
        ("method", "result_type"),
        [
            ("zero_state_probability", "Amplitude"),
            ("probabilities", "StateVector"),
            ("sample", None),
        ],
    )
    def test_other_paths_are_not_capped_here(self, method, result_type, cloud):
        bb._check_result_width(method, result_type, 30, "dev", cloud=cloud)


def _x_on_qubit_0_of(n_qubits: int) -> CircuitSpec:
    # Deterministic at any shot count: every shot reads 1 on qubit 0, 0 elsewhere.
    return CircuitSpec(
        n_qubits=n_qubits, n_weights=0, n_inputs=0, gates=(Gate("X", (0,)),)
    )


def _one_hot_on_qubit_0(n_qubits: int) -> np.ndarray:
    # Qubit 0 is the most significant bit, so |10..0> is index 2**(n - 1).
    expected = np.zeros(2**n_qubits)
    expected[2 ** (n_qubits - 1)] = 1.0
    return expected


# "Managed simulators and QPUs are limited to 15 qubits maximum" for the
# Probability result type (Amazon Braket Developer Guide, "List of result
# types"). Pinned here as the number AWS publishes, not read from the module.
_PROBABILITY_CAP = 15

# The fakes all wrap the local state-vector simulator: `Probability` with shots
# reads the same there, and the local density-matrix simulator would need
# 4**16 entries for a circuit one qubit over the cap.
_CLOUD_PROBABILITY_DEVICES = [
    (_SV1_ARN, SV1_LIMITS),
    (_DM1_ARN, DM1_LIMITS),
    (_QPU_ARN, QPU_LIMITS),
]
_CLOUD_PROBABILITY_IDS = ["sv1", "dm1", "qpu"]
_PROBABILITY_METHODS = ["probabilities", "zero_state_probability"]


def _run_method(compiled: BraketCompiled, method: str, how: str):
    if how == "submit":
        return compiled.submit(*_NO_ARGS, method=method).result()
    return getattr(compiled, method)(*_NO_ARGS)


class TestProbabilityWidthCap:
    """Braket returns the `Probability` result type over at most 15 qubits on
    its managed simulators and QPUs. Both methods read it over every qubit
    whenever the shots are > 0 on a cloud device, so a wider circuit is refused
    before any task or circuit; the local simulator has no such limit."""

    def test_the_cap_is_the_documented_aws_limit(self):
        assert bb._MAX_PROBABILITY_QUBITS == _PROBABILITY_CAP

    @pytest.mark.parametrize("how", ["blocking", "submit"])
    @pytest.mark.parametrize("method", _PROBABILITY_METHODS)
    @pytest.mark.parametrize(
        ("arn", "limits"), _CLOUD_PROBABILITY_DEVICES, ids=_CLOUD_PROBABILITY_IDS
    )
    def test_cloud_above_the_cap_raises_before_any_task(self, arn, limits, method, how):
        width = _PROBABILITY_CAP + 1
        compiled = _cloud(_x_on_qubit_0_of(width), arn, 100, "braket_sv", limits)
        with pytest.raises(ValueError) as excinfo:
            _run_method(compiled, method, how)
        message = str(excinfo.value)
        assert f"{method}()" in message
        assert arn in message
        assert f"all {width} qubits" in message
        assert f"at most {_PROBABILITY_CAP} qubits" in message
        assert "sample()" in message
        if method == "probabilities":
            assert f"{bb._sampling.MAX_DENSE_QUBITS}-qubit dense cap" in message
        assert compiled._device.circuits == []
        assert compiled._zsp_circuit is None

    @pytest.mark.parametrize("how", ["blocking", "submit"])
    @pytest.mark.parametrize(
        ("arn", "limits"), _CLOUD_PROBABILITY_DEVICES, ids=_CLOUD_PROBABILITY_IDS
    )
    def test_cloud_device_without_readable_capabilities_is_capped_too(
        self, arn, limits, how
    ):
        # `Probability` is the fallback when GetDevice lists nothing readable,
        # and it is the same result type at AWS.
        compiled = _cloud(
            _x_on_qubit_0_of(_PROBABILITY_CAP + 1), arn, 100, "braket_sv", limits
        )
        compiled._device.properties = SimpleNamespace()
        with pytest.raises(ValueError, match=f"at most {_PROBABILITY_CAP} qubits"):
            _run_method(compiled, "zero_state_probability", how)
        assert compiled._device.circuits == []

    @pytest.mark.parametrize("how", ["blocking", "submit"])
    @pytest.mark.parametrize(
        ("arn", "limits"), _CLOUD_PROBABILITY_DEVICES, ids=_CLOUD_PROBABILITY_IDS
    )
    def test_cloud_at_the_cap_runs(self, arn, limits, how):
        compiled = _cloud(
            _x_on_qubit_0_of(_PROBABILITY_CAP), arn, 100, "braket_sv", limits
        )
        probs = _run_method(compiled, "probabilities", how)
        assert np.array_equal(probs, _one_hot_on_qubit_0(_PROBABILITY_CAP))
        assert float(_run_method(compiled, "zero_state_probability", how)) == 0.0
        assert _result_type_name(compiled._device.circuits[-1]) == "Probability"
        assert len(compiled._device.circuits) == 2

    @pytest.mark.parametrize("how", ["blocking", "submit"])
    @pytest.mark.parametrize("shots", [0, 100])
    def test_local_simulator_above_the_cap_is_not_capped(self, shots, how):
        width = _PROBABILITY_CAP + 1
        compiled = _compile(_x_on_qubit_0_of(width), BraketConfig(shots=shots))
        probs = _run_method(compiled, "probabilities", how)
        assert np.array_equal(probs, _one_hot_on_qubit_0(width))
        assert float(_run_method(compiled, "zero_state_probability", how)) == 0.0

    @pytest.mark.parametrize("how", ["blocking", "submit"])
    @pytest.mark.parametrize(
        ("arn", "limits"), _CLOUD_PROBABILITY_DEVICES, ids=_CLOUD_PROBABILITY_IDS
    )
    def test_cloud_sample_above_the_cap_is_not_capped(self, arn, limits, how):
        width = _PROBABILITY_CAP + 1
        compiled = _cloud(_x_on_qubit_0_of(width), arn, 50, "braket_sv", limits)
        if how == "submit":
            counts = compiled.submit(*_NO_ARGS, method="sample").result()
        else:
            counts = compiled.sample(*_NO_ARGS)
        assert counts == {"1" + "0" * (width - 1): 50}
        assert compiled._device.circuits[-1].result_types == []

    @pytest.mark.parametrize("method", _PROBABILITY_METHODS)
    def test_the_check_caps_cloud_probability_only(self, method):
        bb._check_result_width(method, "Probability", _PROBABILITY_CAP, "dev", cloud=True)
        with pytest.raises(ValueError, match="dev"):
            bb._check_result_width(
                method, "Probability", _PROBABILITY_CAP + 1, "dev", cloud=True
            )
        bb._check_result_width(method, "Probability", 30, "dev", cloud=False)


def _typed_result(result_type, value):
    return SimpleNamespace(
        values=[value],
        result_types=[SimpleNamespace(type=result_type, value=value)],
    )


def _handle(result, method, n_qubits=2) -> BraketJobHandle:
    task = MagicMock()
    task.id = _TASK_ARN
    task.result.return_value = result
    return BraketJobHandle(task, [[(1.0, 0)]], 10, method=method, n_qubits=n_qubits)


_EXPECTED = np.array([0.1, 0.2, 0.3, 0.4])
_AMPLITUDES = np.sqrt(_EXPECTED) * np.exp(1j * np.arange(4))


def _wire(value) -> list:
    """`value` as a managed simulator's result carries it: nested Python lists
    with every complex number a `[real, imag]` pair."""
    array = np.asarray(value, dtype=np.complex128)
    return np.stack([array.real, array.imag], axis=-1).tolist()


# Row 0 of the DensityMatrix value a live DM1 task returned for a 3-qubit
# circuit (quantum-task/0824acc9-9003-4fec-be6c-22cbb88239f1, us-east-1), as
# `AwsQuantumTask.result()` hands it over: `[real, imag]` pairs.
_DM1_ROW_0 = [
    [0.5499172089346925, 0.0],
    [0.04596043957327822, 0.42513203388188786],
    [0.0, 0.0],
    [0.0, 0.0],
    [-0.06271754596250358, 0.14293513259699475],
    [0.19142281501917768, 0.06043200478144935],
    [0.0, 0.0],
    [0.0, 0.0],
]


def _dm1_density_matrix() -> np.ndarray:
    # A noise-free DM1 result is a pure state |psi><psi|, so its row 0,
    # psi_0 * conj(psi), fixes psi up to a global phase (psi_0 taken real).
    row = np.array([re + 1j * im for re, im in _DM1_ROW_0])
    psi = row.conj() / np.sqrt(row[0].real)
    rho = np.outer(psi, psi.conj())
    assert np.allclose(rho[0], row, atol=1e-15)
    # Keep the live row bit for bit (and the matrix Hermitian).
    rho[0, :] = row
    rho[:, 0] = row.conj()
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
    return rho


class TestDecode:
    def test_probability_result(self):
        result = _typed_result(jaqcd.Probability(targets=[0, 1]), _EXPECTED.copy())
        probs = _handle(result, "probabilities").result()
        assert probs.dtype == np.float64
        assert np.allclose(probs, _EXPECTED)

    def test_statevector_result(self):
        result = _typed_result(jaqcd.StateVector(), _AMPLITUDES)
        assert np.allclose(_handle(result, "probabilities").result(), _EXPECTED)

    def test_density_matrix_result(self):
        rho = np.outer(_AMPLITUDES, _AMPLITUDES.conj())
        result = _typed_result(jaqcd.DensityMatrix(targets=[0, 1]), rho)
        probs = _handle(result, "probabilities").result()
        assert probs.dtype == np.float64
        assert np.allclose(probs, _EXPECTED)
        zsp = _handle(result, "zero_state_probability").result()
        assert zsp.ndim == 0
        assert float(zsp) == pytest.approx(0.1)

    def test_amplitude_result(self):
        states = ["00", "01", "10", "11"]
        result = _typed_result(
            jaqcd.Amplitude(states=states), dict(zip(states, _AMPLITUDES))
        )
        assert np.allclose(_handle(result, "probabilities").result(), _EXPECTED)
        zsp_result = _typed_result(jaqcd.Amplitude(states=["00"]), {"00": _AMPLITUDES[0]})
        zsp = _handle(zsp_result, "zero_state_probability").result()
        assert zsp.ndim == 0
        assert float(zsp) == pytest.approx(0.1)

    def test_measurement_counts_result_pads_unmeasured_qubits(self):
        result = SimpleNamespace(
            values=[], result_types=[], measurement_counts={"1": 7}, measured_qubits=[0]
        )
        assert _handle(result, "sample").result() == {"10": 7}

    def test_sample_handle_given_a_result_type_raises(self):
        result = _typed_result(jaqcd.Probability(targets=[0, 1]), _EXPECTED.copy())
        result.measurement_counts = {"10": 5}
        with pytest.raises(ValueError, match="method"):
            _handle(result, "sample").result()

    def test_probabilities_handle_given_the_wrong_width_raises(self):
        result = _typed_result(jaqcd.Probability(targets=[0]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="4"):
            _handle(result, "probabilities").result()

    def test_partial_amplitude_result_for_probabilities_raises(self):
        result = _typed_result(jaqcd.Amplitude(states=["00"]), {"00": _AMPLITUDES[0]})
        with pytest.raises(ValueError, match="method"):
            _handle(result, "probabilities").result()


class TestDecodeCloudWireForm:
    """The managed simulators serialize a complex number as a `[real, imag]`
    pair, and the SDK does not turn every result type back into complex
    numbers (a DM1 `DensityMatrix` arrives as `(d, d, 2)` nested lists). The
    local simulator the fake devices wrap hands over complex arrays, so these
    decode the cloud form directly."""

    def test_density_matrix_wire_form_probabilities(self):
        rho = _dm1_density_matrix()
        wire = _wire(rho)
        assert np.asarray(wire).shape == (8, 8, 2)
        assert wire[0] == _DM1_ROW_0
        result = _typed_result(jaqcd.DensityMatrix(targets=[0, 1, 2]), wire)
        probs = _handle(result, "probabilities", n_qubits=3).result()
        assert probs.dtype == np.float64
        assert probs.shape == (8,)
        assert np.allclose(probs, np.real(np.diagonal(rho)), atol=1e-15, rtol=0)

    def test_density_matrix_wire_form_zero_state_probability(self):
        wire = _wire(_dm1_density_matrix())
        result = _typed_result(jaqcd.DensityMatrix(targets=[0, 1, 2]), wire)
        zsp = _handle(result, "zero_state_probability", n_qubits=3).result()
        assert zsp.dtype == np.float64
        assert zsp.ndim == 0
        assert float(zsp) == 0.5499172089346925

    def test_statevector_wire_form(self):
        result = _typed_result(jaqcd.StateVector(), _wire(_AMPLITUDES))
        probs = _handle(result, "probabilities").result()
        assert probs.dtype == np.float64
        assert np.allclose(probs, _EXPECTED)
        zsp = _handle(result, "zero_state_probability").result()
        assert zsp.ndim == 0
        assert float(zsp) == pytest.approx(0.1)

    def test_amplitude_wire_form(self):
        states = ["00", "01", "10", "11"]
        pairs = dict(zip(states, _wire(_AMPLITUDES)))
        result = _typed_result(jaqcd.Amplitude(states=states), pairs)
        assert np.allclose(_handle(result, "probabilities").result(), _EXPECTED)
        zsp_result = _typed_result(jaqcd.Amplitude(states=["00"]), {"00": pairs["00"]})
        zsp = _handle(zsp_result, "zero_state_probability").result()
        assert zsp.ndim == 0
        assert float(zsp) == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "shape",
        [(8, 8, 3), (8, 4, 2), (4, 8, 2), (8, 8, 2, 2), (8,), (8, 2)],
        ids=["three-parts", "wide", "tall", "four-d", "vector", "pairs-vector"],
    )
    @pytest.mark.parametrize("method", ["probabilities", "zero_state_probability"])
    def test_malformed_density_matrix_raises(self, shape, method):
        result = _typed_result(jaqcd.DensityMatrix(targets=[0, 1, 2]), np.zeros(shape))
        with pytest.raises(ValueError, match="not square"):
            _handle(result, method, n_qubits=3).result()

    @pytest.mark.parametrize(
        "shape", [(8, 3), (8, 2, 2), (8, 8, 2)], ids=["three-parts", "three-d", "matrix"]
    )
    def test_malformed_statevector_raises(self, shape):
        result = _typed_result(jaqcd.StateVector(), np.zeros(shape))
        with pytest.raises(ValueError, match="not 1-D"):
            _handle(result, "probabilities", n_qubits=3).result()

    def test_amplitude_that_is_not_a_complex_number_raises(self):
        result = _typed_result(jaqcd.Amplitude(states=["00"]), {"00": [0.1, 0.2, 0.3]})
        with pytest.raises(ValueError, match="method"):
            _handle(result, "zero_state_probability").result()
