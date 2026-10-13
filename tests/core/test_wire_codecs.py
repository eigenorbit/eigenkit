"""Public wire codecs in `eigenkit.core.wire`.

Saved model bundles store circuits and observables in this JSON form, so it
must stay byte-identical to the encodings the `.eigenkit` bundles already use.
"""

from __future__ import annotations

import json
from decimal import Decimal

import numpy as np
import pytest

from eigenkit.core import wire
from eigenkit.core.blocks import (
    AngleEncoding,
    BasicVariational,
    _circuit_spec_from_json,
    _circuit_spec_to_json,
    assemble,
)
from eigenkit.core.ir import Observable, PauliTerm
from eigenkit.core.observables import Z, all_z, from_string
from eigenkit.qml._serialization import (
    _observables_from_dict,
    _observables_to_dict,
)


def _dumps(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@pytest.fixture
def spec():
    return assemble(
        [
            AngleEncoding(in_features=3),
            BasicVariational(depth=2, rotation="RY"),
        ]
    )


class TestCircuitSpecCodec:
    def test_to_json_matches_private_encoding(self, spec):
        assert _dumps(wire.circuit_spec_to_json(spec)) == _dumps(
            _circuit_spec_to_json(spec)
        )

    def test_round_trip_equals_spec(self, spec):
        data = wire.circuit_spec_to_json(spec)
        assert wire.circuit_spec_from_json(data) == spec

    def test_from_json_matches_private_decoder(self, spec):
        data = _circuit_spec_to_json(spec)
        assert wire.circuit_spec_from_json(data) == _circuit_spec_from_json(data)

    def test_survives_json_text_round_trip(self, spec):
        text = json.dumps(wire.circuit_spec_to_json(spec))
        assert wire.circuit_spec_from_json(json.loads(text)) == spec

    def test_top_level_shape(self, spec):
        data = wire.circuit_spec_to_json(spec)
        assert set(data) == {"n_qubits", "n_weights", "n_inputs", "gates"}
        assert data["n_qubits"] == 3
        assert data["n_inputs"] == 3


class TestObservablesCodec:
    def test_list_matches_golden_encoding(self):
        observables = [Z(0).to_observable(), *all_z(2)]
        assert wire.observables_to_json(observables) == [
            {"terms": [{"coeff": 1.0, "paulis": [["Z", 0]]}]},
            {"terms": [{"coeff": 1.0, "paulis": [["Z", 0]]}]},
            {"terms": [{"coeff": 1.0, "paulis": [["Z", 1]]}]},
        ]

    def test_single_matches_golden_encoding(self):
        observable = Observable(
            terms=(
                PauliTerm(coeff=0.5, paulis=(("Z", 0), ("X", 1))),
                PauliTerm(coeff=-0.3, paulis=(("Y", 2),)),
            )
        )
        assert wire.observables_to_json(observable) == {
            "terms": [
                {"coeff": 0.5, "paulis": [["Z", 0], ["X", 1]]},
                {"coeff": -0.3, "paulis": [["Y", 2]]},
            ]
        }

    @pytest.mark.parametrize(
        "observables",
        [
            [Z(0).to_observable(), *all_z(2)],
            all_z(3),
            from_string("0.5 Z0 X1 - 0.3 Y2 + 0.7 Z0 Z2"),
        ],
        ids=["list", "all_z", "single"],
    )
    def test_matches_qml_private_encoding(self, observables):
        assert _dumps(wire.observables_to_json(observables)) == _dumps(
            _observables_to_dict(observables)
        )

    @pytest.mark.parametrize(
        "observables",
        [
            [Z(0).to_observable(), *all_z(2)],
            from_string("0.5 Z0 X1 - 0.3 Y2 + 0.7 Z0 Z2"),
        ],
        ids=["list", "single"],
    )
    def test_round_trip_equals_input(self, observables):
        text = json.dumps(wire.observables_to_json(observables))
        assert wire.observables_from_json(json.loads(text)) == observables

    def test_qml_private_decoder_agrees(self):
        data = wire.observables_to_json(all_z(2))
        assert _observables_from_dict(data) == wire.observables_from_json(data)

    def test_output_is_pure_json(self):
        observables = [from_string("0.5 Z0 X1 - 0.3 Y2"), *all_z(2)]
        assert json.loads(json.dumps(wire.observables_to_json(observables))) == (
            wire.observables_to_json(observables)
        )


def _coeffs(encoded) -> list:
    items = encoded if isinstance(encoded, list) else [encoded]
    return [term["coeff"] for item in items for term in item["terms"]]


def _term(coeff) -> Observable:
    return Observable(terms=(PauliTerm(coeff=coeff, paulis=(("Z", 0),)),))


class TestObservableCoefficients:
    """Coefficients must leave the codec as plain `int` or `float`.

    numpy scalars are the usual source: `np.float64` subclasses `float`, so
    it would pass straight through and orjson would then refuse it; and
    `np.int64` must stay an int, as a Python int does, not become a float.
    """

    @pytest.mark.parametrize(
        ("coeff", "expected_type"),
        [
            (np.float64(0.25), float),
            (np.float32(0.5), float),
            (np.int64(3), int),
            (np.int32(-2), int),
        ],
        ids=["float64", "float32", "int64", "int32"],
    )
    def test_numpy_scalar_becomes_plain_number(self, coeff, expected_type):
        (value,) = _coeffs(wire.observables_to_json(_term(coeff)))
        assert type(value) is expected_type
        assert value == coeff

    def test_numpy_coefficients_in_a_list_are_all_plain(self):
        observables = [
            _term(np.float64(0.25)),
            _term(np.int64(2)),
            _term(np.float32(0.5)),
        ]
        encoded = wire.observables_to_json(observables)
        assert [type(c) for c in _coeffs(encoded)] == [float, int, float]

    def test_numpy_coefficients_serialize_with_orjson(self):
        orjson = pytest.importorskip("orjson")
        encoded = wire.observables_to_json(
            [_term(np.float64(0.25)), _term(np.int64(2)), _term(np.float32(0.5))]
        )
        assert orjson.loads(orjson.dumps(encoded)) == encoded

    @pytest.mark.parametrize(
        "coeff", [2, -0.3, 1e-12], ids=["int", "float", "tiny-float"]
    )
    def test_plain_python_number_is_kept_as_is(self, coeff):
        (value,) = _coeffs(wire.observables_to_json(_term(coeff)))
        assert type(value) is type(coeff)
        assert value == coeff

    @pytest.mark.parametrize(
        "coeff",
        [True, np.bool_(False), "1.0", Decimal("0.5"), None],
        ids=["bool", "numpy-bool", "str", "decimal", "none"],
    )
    def test_non_real_coefficient_rejected(self, coeff):
        # A bool is not a JSON number, and turning it into 1.0 would silently
        # change what a `.eigenkit` bundle stores, so it is refused like a string.
        with pytest.raises(TypeError, match="coeff"):
            wire.observables_to_json(_term(coeff))
