"""Building and reading a Quandela Cloud job without a connection."""

import json
from pathlib import Path

import numpy as np
import pytest


pcvl = pytest.importorskip("perceval")

from eigenkit.core.ir import Constant, resolve  # noqa: E402
from eigenkit.photonics.backends.perceval_remote import (  # noqa: E402
    cloud_job_body,
    cloud_job_counts,
)
from eigenkit.photonics.ir import (  # noqa: E402
    BeamSplitter,
    PhaseShifter,
    PhotonicCircuitSpec,
)
from eigenkit.photonics.reservoir import random_reservoir_spec  # noqa: E402


FIXTURES = Path(__file__).parent / "fixtures"
DETAILS = json.loads((FIXTURES / "quandela_platform_sim_slos.json").read_text())
SPEC, WEIGHTS = random_reservoir_spec(
    4, 2, depth=2, input_state=(1, 1, 0, 0), rng=np.random.default_rng(0)
)


def _body(inputs, **overrides):
    kwargs = {
        "platform": "sim:slos",
        "platform_details": DETAILS,
        "shots": 200,
        "job_name": "eo-test",
        **overrides,
    }
    return cloud_job_body(SPEC, WEIGHTS, inputs, **kwargs)


def _angles(spec):
    for comp in spec.components:
        if isinstance(comp, PhaseShifter):
            yield comp.phi
        elif isinstance(comp, BeamSplitter):
            yield comp.theta
            yield comp.phi


def _result(name):
    return json.loads((FIXTURES / name).read_text())


def test_body_names_the_platform_and_the_job():
    body = _body(np.zeros(2))
    assert body["platform_name"] == "sim:slos"
    assert body["job_name"] == "eo-test"
    assert body["payload"]["command"] == "sample_count"


def test_one_iteration_per_row_with_the_resolved_angles():
    rows = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    iterator = _body(rows)["payload"]["iterator"]
    assert len(iterator) == 3
    for row, iteration in zip(rows, iterator, strict=True):
        values = sorted(iteration["circuit_params"].values())
        expected = sorted(float(resolve(e, WEIGHTS, row)) for e in _angles(SPEC))
        assert values == pytest.approx(expected)


def test_one_d_inputs_are_a_batch_of_one():
    assert len(_body(np.array([0.1, 0.2]))["payload"]["iterator"]) == 1


def test_shots_set_the_samples_and_the_shot_limit_of_every_row():
    for iteration in _body(np.zeros((2, 2)), shots=321)["payload"]["iterator"]:
        assert iteration["max_samples"] == 321
        assert iteration["max_shots"] == 321


def test_the_experiment_carries_the_modes_and_the_input_state():
    body = _body(np.zeros(2))
    experiment = pcvl.serialization.deserialize(body["payload"]["experiment"])
    assert experiment.m == 4
    assert list(experiment.input_state) == [1, 1, 0, 0]


def test_the_photon_filter_defaults_to_the_input_photons():
    default = pcvl.serialization.deserialize(_body(np.zeros(2))["payload"]["experiment"])
    lowered = pcvl.serialization.deserialize(
        _body(np.zeros(2), min_detected_photons=1)["payload"]["experiment"]
    )
    assert default.min_photons_filter == 2
    assert lowered.min_photons_filter == 1


def test_body_is_json_and_survives_numpy_rows():
    body = _body(np.zeros((2, 2), dtype=np.float32))
    assert json.loads(json.dumps(body)) == body


def test_no_connection_is_opened(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr("requests.get", refuse)
    monkeypatch.setattr("requests.post", refuse)
    _body(np.zeros((2, 2)))


@pytest.mark.parametrize(
    "inputs", [np.zeros((0, 2)), np.zeros((2, 3)), np.zeros((2, 2, 2)), np.zeros(3)]
)
def test_a_wrong_batch_shape_is_refused(inputs):
    with pytest.raises(ValueError, match="inputs"):
        _body(inputs)


def test_two_d_weights_are_refused():
    with pytest.raises(ValueError, match="weights"):
        cloud_job_body(
            SPEC,
            np.zeros((2, SPEC.n_weights)),
            np.zeros(2),
            platform="sim:slos",
            platform_details=DETAILS,
            shots=1,
            job_name="x",
        )


def test_a_circuit_without_inputs_keeps_its_row_count():
    spec = PhotonicCircuitSpec(
        n_modes=2,
        n_weights=0,
        n_inputs=0,
        components=(BeamSplitter((0, 1), Constant(np.pi / 2)),),
        input_state=(1, 1),
    )
    body = cloud_job_body(
        spec,
        np.array([]),
        np.zeros((3, 0)),
        platform="sim:slos",
        platform_details=DETAILS,
        shots=10,
        job_name="x",
    )
    assert len(body["payload"]["iterator"]) == 3


def test_counts_of_a_single_run():
    (counts,) = cloud_job_counts(_result("quandela_result_single.json"), 4)
    assert all(isinstance(key, tuple) and len(key) == 4 for key in counts)
    assert all(type(n) is int for key in counts for n in key)
    assert sum(counts.values()) == 1000


def test_counts_of_a_batch_keep_the_row_order():
    result = _result("quandela_result_batch.json")
    rows = cloud_job_counts(result, 4)
    assert len(rows) == 3
    assert all(sum(row.values()) == 200 for row in rows)
    entries = json.loads(result["results"])["results_list"]
    for row, entry in zip(rows, entries, strict=True):
        direct = pcvl.serialization.deserialize(entry["results"])
        assert sum(row.values()) == sum(direct.values())
        assert row == {
            tuple(int(n) for n in state): count for state, count in direct.items()
        }


def test_a_row_with_no_surviving_shot_is_empty():
    result = _result("quandela_result_batch.json")
    inner = json.loads(result["results"])
    inner["results_list"][1]["results"] = pcvl.serialization.serialize(pcvl.BSCount())
    result["results"] = json.dumps(inner)
    rows = cloud_job_counts(result, 4)
    assert rows[1] == {}
    assert rows[0] and rows[2]


@pytest.mark.parametrize(
    "results", [None, "null", json.dumps("Error: Incorrect padding"), json.dumps({})]
)
def test_a_job_without_counts_is_refused(results):
    with pytest.raises(ValueError, match="no counts"):
        cloud_job_counts({"results": results, "job_id": "x"}, 4)


def test_a_result_without_the_results_key_is_refused():
    with pytest.raises(ValueError, match="no counts"):
        cloud_job_counts({"job_id": "x"}, 4)
