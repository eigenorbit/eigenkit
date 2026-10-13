"""A weight or input that feeds several gates needs the per-gate chain rule.

Shifting a shared vector entry by +/- pi/2 moves every gate it feeds at once,
so the expectation carries sums of the gates' frequencies and the naive two-
and four-term rules are wrong. `Reuploading` encodes each feature once per
layer, so its input gradients hit this on every backend that differentiates by
shifting. numpy differentiates natively (adjoint), so it is the oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.blocks import Reuploading, assemble
from eigenkit.core.circuit import Circuit
from eigenkit.core.ir import (
    CircuitSpec,
    Gate,
    InputRef,
    Observable,
    PauliTerm,
    WeightRef,
    find_input_shift_unsafe_gates,
    find_parameter_shift_unsafe_gates,
)


Z0 = Observable(terms=(PauliTerm(1.0, (("Z", 0),)),))
X1 = Observable(terms=(PauliTerm(1.0, (("X", 1),)),))
SHIFT_BACKENDS = {
    "qiskit": "qiskit",
    "qiskit-aer": "qiskit_aer",
    "qsim": "qsimcirq",
    "braket": "braket",
}


def _backend(name):
    pytest.importorskip(SHIFT_BACKENDS[name])
    return name


def _tied_weight_spec() -> CircuitSpec:
    """One weight on an RY and on a CRY, one input on two RX gates."""
    return CircuitSpec(
        n_qubits=2,
        n_weights=1,
        n_inputs=1,
        gates=(
            Gate("H", (0,)),
            Gate("RY", (1,), (WeightRef(0),)),
            Gate("RX", (1,), (InputRef(0),)),
            Gate("CRY", (0, 1), (WeightRef(0),)),
            Gate("RX", (0,), (InputRef(0),)),
        ),
    )


def test_shared_references_are_flagged():
    spec = _tied_weight_spec()
    assert find_parameter_shift_unsafe_gates(spec)
    assert find_input_shift_unsafe_gates(spec)
    reuploading = assemble([Reuploading(in_features=2, depth=2)])
    assert find_input_shift_unsafe_gates(reuploading)


def test_single_use_references_stay_on_the_fast_path():
    spec = CircuitSpec(
        n_qubits=1,
        n_weights=1,
        n_inputs=1,
        gates=(Gate("RX", (0,), (InputRef(0),)), Gate("RY", (0,), (WeightRef(0),))),
    )
    assert not find_parameter_shift_unsafe_gates(spec)
    assert not find_input_shift_unsafe_gates(spec)


@pytest.mark.parametrize("provider", SHIFT_BACKENDS)
def test_tied_weight_and_input_match_the_adjoint(provider):
    spec = _tied_weight_spec()
    weights, inputs = np.array([0.7]), np.array([-0.4])
    expected = Circuit(spec, [Z0, X1], "numpy", "numpy").jacobian(
        weights, inputs, input_grad=True
    )
    actual = Circuit(spec, [Z0, X1], _backend(provider), "numpy").jacobian(
        weights, inputs, input_grad=True
    )
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("provider", SHIFT_BACKENDS)
def test_reuploading_input_gradient_matches_the_adjoint(provider):
    spec = assemble([Reuploading(in_features=2, depth=2)])
    weights = np.linspace(-1.0, 1.0, spec.n_weights)
    inputs = np.array([0.3, -0.8])
    expected = Circuit(spec, Z0, "numpy", "numpy").jacobian(
        weights, inputs, input_grad=True
    )
    actual = Circuit(spec, Z0, _backend(provider), "numpy").jacobian(
        weights, inputs, input_grad=True
    )
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=1e-6)
