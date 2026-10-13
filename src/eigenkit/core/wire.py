"""A stable JSON form for circuits and observables.

The functions of this module convert a `CircuitSpec` and its observables to plain
JSON types (`dict`, `list`, `str`, `int`, `float`, `bool` and `None`), and back.
`.eigenkit` bundles use the same encoding, so a change to it would break the saved
models.

Example:
    ```python
    import json

    from eigenkit.core import wire
    from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
    from eigenkit.core.observables import all_z

    spec = assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])
    payload = json.dumps(
        {
            "circuit": wire.circuit_spec_to_json(spec),
            "observables": wire.observables_to_json(all_z(2)),
        }
    )

    data = json.loads(payload)
    assert wire.circuit_spec_from_json(data["circuit"]) == spec
    assert wire.observables_from_json(data["observables"]) == all_z(2)
    ```
"""

from __future__ import annotations

import numbers
from typing import Sequence

from .blocks import _circuit_spec_from_json, _circuit_spec_to_json
from .ir import CircuitSpec, Observable, PauliTerm


__all__ = [
    "circuit_spec_from_json",
    "circuit_spec_to_json",
    "observables_from_json",
    "observables_to_json",
]


def circuit_spec_to_json(spec: CircuitSpec) -> dict:
    """Encode a circuit as a dictionary that can be saved as JSON.

    Args:
        spec (CircuitSpec): The assembled circuit.

    Returns:
        A dictionary with the keys `n_qubits`, `n_weights`, `n_inputs` and `gates`,
            made of plain JSON types only.
    """
    return _circuit_spec_to_json(spec)


def circuit_spec_from_json(data: dict) -> CircuitSpec:
    """Decode a circuit encoded by `circuit_spec_to_json`.

    Args:
        data (dict): The encoded circuit, for example as returned by `json.loads`.

    Returns:
        The circuit, equal to the one that was encoded.
    """
    return _circuit_spec_from_json(data)


def _coeff_to_json(coeff) -> int | float:
    # An exact Python int or float is kept as-is, so the bytes match what
    # `.eigenkit` bundles already store. Subclasses are not: `np.float64` is a
    # `float` subclass that orjson refuses to serialize.
    if type(coeff) in (int, float):
        return coeff
    # A bool is an `Integral`, but JSON `true` is not a number and turning it
    # into 1 would silently change what a bundle stores. `np.bool_` is not a
    # `numbers.Real`, so it falls through to the TypeError below as well.
    if isinstance(coeff, bool):
        raise TypeError(f"PauliTerm.coeff must be a real number, got bool {coeff!r}.")
    if isinstance(coeff, numbers.Integral):
        return int(coeff)
    if isinstance(coeff, numbers.Real):
        return float(coeff)
    raise TypeError(f"PauliTerm.coeff must be a real number, got {coeff!r}.")


def _observable_to_json(observable: Observable) -> dict:
    return {
        "terms": [
            {
                "coeff": _coeff_to_json(term.coeff),
                "paulis": [[str(op), int(qubit)] for op, qubit in term.paulis],
            }
            for term in observable.terms
        ]
    }


def _observable_from_json(data: dict) -> Observable:
    return Observable(
        terms=tuple(
            PauliTerm(
                coeff=term["coeff"],
                paulis=tuple(tuple(pauli) for pauli in term["paulis"]),
            )
            for term in data["terms"]
        )
    )


def observables_to_json(
    observables: Observable | Sequence[Observable],
) -> list | dict:
    """Encode one observable, or a list of them, as data that can be saved as JSON.

    Each observable becomes `{"terms": [{"coeff": c, "paulis": [[op, q], ...]}]}`.
    A coefficient that is a Python `int` or `float` is written unchanged. A NumPy
    integer becomes an `int`, and any other real number (a NumPy float or a
    `Fraction`, for example) becomes a `float`, so the output is plain JSON for any
    encoder.

    Args:
        observables (Observable | Sequence[Observable]): A single observable, or a
            list or tuple of observables.

    Returns:
        A dictionary for a single observable, or, for a list or tuple, a list of
            dictionaries in the order of the input.

    Raises:
        TypeError: If a coefficient is a `bool` or is not a real number.
    """
    if isinstance(observables, (list, tuple)):
        return [_observable_to_json(o) for o in observables]
    return _observable_to_json(observables)


def observables_from_json(data: list | dict) -> Observable | list[Observable]:
    """Decode observables encoded by `observables_to_json`.

    Args:
        data (list | dict): The encoded observables, for example as returned by
            `json.loads`.

    Returns:
        An `Observable` when `data` is a dictionary, or a list of observables when
            `data` is a list.
    """
    if isinstance(data, list):
        return [_observable_from_json(d) for d in data]
    return _observable_from_json(data)
