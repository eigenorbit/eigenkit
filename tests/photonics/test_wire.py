"""The JSON codec for photonic circuit specs."""

import json
import subprocess
import sys

import numpy as np
import pytest

from eigenkit.core.ir import Constant, InputRef, WeightRef
from eigenkit.photonics.ir import (
    BeamSplitter,
    Permutation,
    PhaseShifter,
    PhotonicCircuitSpec,
)
from eigenkit.photonics.reservoir import random_reservoir_spec
from eigenkit.photonics.wire import photonic_spec_from_json, photonic_spec_to_json


SPEC = PhotonicCircuitSpec(
    n_modes=4,
    n_weights=1,
    n_inputs=1,
    components=(
        PhaseShifter(1, InputRef(0)),
        BeamSplitter((0, 1), WeightRef(0)),
        Permutation((1, 0, 2, 3)),
    ),
    input_state=(1, 0, 1, 0),
)
GOLDEN = {
    "n_modes": 4,
    "n_weights": 1,
    "n_inputs": 1,
    "input_state": [1, 0, 1, 0],
    "components": [
        {"c": "PS", "mode": 1, "phi": {"t": "InputRef", "i": 0}},
        {
            "c": "BS",
            "modes": [0, 1],
            "theta": {"t": "WeightRef", "i": 0},
            "phi": {"t": "Constant", "v": 0.0},
        },
        {"c": "PERM", "perm": [1, 0, 2, 3]},
    ],
}


def test_golden_encoding():
    assert photonic_spec_to_json(SPEC) == GOLDEN


def test_golden_decoding():
    assert photonic_spec_from_json(GOLDEN) == SPEC


def test_round_trip_of_a_reservoir():
    spec, _ = random_reservoir_spec(
        4, 2, depth=2, input_state=(1, 1, 0, 0), rng=np.random.default_rng(0)
    )
    wire = json.loads(json.dumps(photonic_spec_to_json(spec)))
    assert photonic_spec_from_json(wire) == spec


def test_numpy_scalars_encode_as_plain_json():
    spec = PhotonicCircuitSpec(
        n_modes=np.int64(2),
        n_weights=0,
        n_inputs=0,
        components=(PhaseShifter(np.int64(0), Constant(np.float64(0.5))),),
        input_state=(np.int64(1), np.int64(0)),
    )
    data = photonic_spec_to_json(spec)
    assert json.loads(json.dumps(data)) == data
    assert type(data["n_modes"]) is int
    assert type(data["components"][0]["mode"]) is int
    assert type(data["input_state"][0]) is int


@pytest.mark.parametrize(
    "bad, error",
    [
        ({**GOLDEN, "components": [{"c": "XX"}]}, ValueError),
        ({k: v for k, v in GOLDEN.items() if k != "input_state"}, KeyError),
        (
            {
                **GOLDEN,
                "components": [
                    {
                        "c": "BS",
                        "modes": [0, 2],
                        "theta": {"t": "Constant", "v": 0.0},
                        "phi": {"t": "Constant", "v": 0.0},
                    }
                ],
            },
            ValueError,
        ),
        ({**GOLDEN, "input_state": [1, 0, 1]}, ValueError),
        ("not a dict", TypeError),
    ],
)
def test_malformed_input_is_refused(bad, error):
    with pytest.raises(error):
        photonic_spec_from_json(bad)


def test_importing_the_codec_loads_no_perceval():
    code = "import sys, eigenkit.photonics.wire; assert 'perceval' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
