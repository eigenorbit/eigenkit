"""The JSON encoding of photonic circuits.

This module is the photonic counterpart of `core.wire`. It turns a
`PhotonicCircuitSpec` into plain JSON data and back, so that a circuit can be sent
to a service or stored. It holds only data, and uses neither Perceval nor NumPy.

The angles of the components use the same encoding as the parameters of the qubit
circuits, so a `WeightRef` or an `InputRef` means the same thing in both.
"""

from __future__ import annotations

from ..core.blocks import _param_from_json, _param_to_json
from .ir import BeamSplitter, Permutation, PhaseShifter, PhotonicCircuitSpec


def _component_to_json(comp) -> dict:
    if isinstance(comp, PhaseShifter):
        return {"c": "PS", "mode": int(comp.mode), "phi": _param_to_json(comp.phi)}
    if isinstance(comp, BeamSplitter):
        return {
            "c": "BS",
            "modes": [int(m) for m in comp.modes],
            "theta": _param_to_json(comp.theta),
            "phi": _param_to_json(comp.phi),
        }
    if isinstance(comp, Permutation):
        return {"c": "PERM", "perm": [int(m) for m in comp.perm]}
    raise TypeError(f"unknown photonic component: {type(comp).__name__}")


def _component_from_json(data: dict):
    tag = data["c"]
    if tag == "PS":
        return PhaseShifter(data["mode"], _param_from_json(data["phi"]))
    if tag == "BS":
        return BeamSplitter(
            tuple(data["modes"]),
            _param_from_json(data["theta"]),
            _param_from_json(data["phi"]),
        )
    if tag == "PERM":
        return Permutation(tuple(data["perm"]))
    raise ValueError(f"unknown photonic component in JSON: {tag!r}")


def photonic_spec_to_json(spec: PhotonicCircuitSpec) -> dict:
    """Encode a photonic circuit as JSON data.

    Args:
        spec (PhotonicCircuitSpec): The circuit to encode.

    Returns:
        A dictionary with the keys `"n_modes"`, `"n_weights"`, `"n_inputs"`,
            `"input_state"` and `"components"`. Every integer in it is a plain
            Python `int`, so `json.dumps` accepts it.
    """
    return {
        "n_modes": int(spec.n_modes),
        "n_weights": int(spec.n_weights),
        "n_inputs": int(spec.n_inputs),
        "input_state": [int(n) for n in spec.input_state],
        "components": [_component_to_json(comp) for comp in spec.components],
    }


def photonic_spec_from_json(data: dict) -> PhotonicCircuitSpec:
    """Build a photonic circuit from the data that `photonic_spec_to_json` produced.

    The circuit goes through the same checks as one built by hand.

    Args:
        data (dict): The encoded circuit.

    Returns:
        The decoded `PhotonicCircuitSpec`.

    Raises:
        TypeError: If `data` is not a dictionary.
        KeyError: If a key is missing.
        ValueError: If a component is unknown, or the circuit is not valid.
    """
    if not isinstance(data, dict):
        raise TypeError(
            f"a photonic circuit is encoded as a dict, got {type(data).__name__}"
        )
    return PhotonicCircuitSpec(
        n_modes=data["n_modes"],
        n_weights=data["n_weights"],
        n_inputs=data["n_inputs"],
        components=tuple(_component_from_json(comp) for comp in data["components"]),
        input_state=tuple(data["input_state"]),
    )
