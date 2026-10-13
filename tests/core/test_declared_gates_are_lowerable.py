"""Every gate a backend declares must be a gate it can actually execute.

`Capabilities.supported_gates` is a promise: `validate_circuit` rejects
anything outside it *before* compilation, "with a message naming the offending
gate". A backend that declares a gate it cannot lower breaks that promise in
the worst direction — the circuit passes validation and then crashes inside
`compile()`, which the user reads as a library bug rather than an unsupported
feature.

Two ways this can happen. A backend that keeps a private copy of the lowering
falls behind the shared one and declares gates its copy does not implement. A
backend that declares a name which is not an IR gate name (say "Y" or "CZ") has
a branch that can never run.

This test checks every declared name against the IR vocabulary and executes
each declared gate on each installed backend, so either form fails here.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import (
    ALL_GATES,
    CircuitSpec,
    Constant,
    Gate,
    Observable,
    PauliTerm,
)
from eigenkit.core.provider import get


# Backend name -> the SDK module that must import for it to be testable.
BACKENDS = {
    "numpy": None,
    "pennylane": "pennylane",
    "qiskit": "qiskit",
    "qiskit-aer": "qiskit_aer",
    "qsim": "qsimcirq",
    "braket": "braket",
    "qulacs": "qulacs",
    "tensorcircuit": "tensorcircuit",
}

# One runnable instance of every gate in the vocabulary, on 3 qubits.
_UNITARY_1Q = ((0.0, 1.0), (1.0, 0.0))  # Pauli X
_AMPLITUDES = (0.5, 0.5, 0.5, 0.5)
GATE_FIXTURES: dict[str, Gate] = {
    "RX": Gate("RX", (0,), (Constant(0.3),)),
    "RY": Gate("RY", (0,), (Constant(0.3),)),
    "RZ": Gate("RZ", (0,), (Constant(0.3),)),
    "PhaseShift": Gate("PhaseShift", (0,), (Constant(0.3),)),
    "CRY": Gate("CRY", (0, 1), (Constant(0.3),)),
    "CPHASE": Gate("CPHASE", (0, 1), (Constant(0.3),)),
    "H": Gate("H", (0,)),
    "X": Gate("X", (0,)),
    "Z": Gate("Z", (0,)),
    "S": Gate("S", (0,)),
    "Sdg": Gate("Sdg", (0,)),
    "CNOT": Gate("CNOT", (0, 1)),
    "SWAP": Gate("SWAP", (0, 1)),
    "MCZ": Gate("MCZ", (0, 1)),
    "MCX": Gate("MCX", (0, 1)),
    "QubitUnitary": Gate("QubitUnitary", (0,), data=(_UNITARY_1Q,)),
    "AmplitudeEncoding": Gate("AmplitudeEncoding", (0, 1), data=(_AMPLITUDES,)),
    "ControlledQubitUnitary": Gate(
        "ControlledQubitUnitary", (0, 1), data=(_UNITARY_1Q, None)
    ),
}


def test_the_fixture_table_covers_the_whole_vocabulary():
    """Otherwise a new gate would be silently exempt from this file."""
    assert set(GATE_FIXTURES) == set(ALL_GATES)


def _provider(name):
    module = BACKENDS[name]
    if module is not None:
        pytest.importorskip(module)
    return name


@pytest.mark.parametrize("backend_name", list(BACKENDS))
def test_every_declared_gate_lowers_and_runs(backend_name):
    provider = _provider(backend_name)
    declared = get(provider).capabilities.supported_gates

    unknown = sorted(set(declared) - set(ALL_GATES))
    assert not unknown, (
        f"{backend_name} declares {unknown}, which are not IR gate names, so "
        f"they can never match a real gate"
    )

    observable = Observable(terms=(PauliTerm(1.0, (("Z", 0),)),))
    failures = []
    for name in sorted(declared):
        spec = CircuitSpec(
            n_qubits=3, n_weights=0, n_inputs=0, gates=(GATE_FIXTURES[name],)
        )
        try:
            value = Circuit(spec, observable, provider, "numpy")(np.empty(0))
        except Exception as exc:  # noqa: BLE001 — the point is to catch any
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not np.isfinite(np.asarray(value)).all():
            failures.append(f"{name}: produced a non-finite value {value!r}")

    assert not failures, (
        f"{backend_name} declares gates it cannot execute:\n  " + "\n  ".join(failures)
    )
