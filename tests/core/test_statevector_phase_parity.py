"""A statevector must agree on phase, not just on expectation values.

qulacs has no parametric U1 gate, so a *trainable* `PhaseShift` or `CPHASE` is
lowered to RZ slots, which realise the gate up to `e^{-i·c·theta}`. That factor
is unobservable in an expectation value -- but `statevector()` is public, and
two consumers read the amplitudes themselves:

  * `amplitude_features` concatenates the real and imaginary parts into ML
    features, so the whole feature vector would shift;
  * the `leaky` reservoir policy mixes an evolved state with a reset one,
    which turns a global phase into a *relative* one and moves even `<Z>`.

Left uncorrected, the factor is far from negligible: reservoir predictions
differ between numpy and qulacs by 0.33 (`amplitudes`) and 0.28 (the default
`z` readout).
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.builder import CircuitBuilder
from eigenkit.core.circuit import Circuit
from eigenkit.core.observables import all_z


# provider -> (import name, tolerance). qsim is a single-precision simulator
# and documents ~1e-6 accumulation error, so it gets the tolerance its own
# module docstring states rather than a double-precision one. The point of
# this file is phase, and a phase error shows up far above either bound.
BACKENDS = {
    "pennylane": ("pennylane", 1e-9),
    "qiskit": ("qiskit", 1e-9),
    "qulacs": ("qulacs", 1e-9),
    "qsim": ("qsimcirq", 1e-5),
}


def _phase_spec(gate: str):
    builder = CircuitBuilder(n_qubits=2)
    weights = builder.add_weights(1)
    builder.h(0)
    builder.h(1)
    if gate == "PhaseShift":
        builder.phase_shift(weights[0], qubit=0)
    else:
        builder.cphase(weights[0], control=0, target=1)
    return builder.build()


@pytest.mark.parametrize("gate", ["PhaseShift", "CPHASE"])
@pytest.mark.parametrize("theta", [0.0, 0.7, 2.9, -1.3])
@pytest.mark.parametrize(
    "provider,module,tolerance",
    [(name, module, tol) for name, (module, tol) in sorted(BACKENDS.items())],
)
def test_trainable_phase_gate_statevector_matches_numpy(
    provider, module, tolerance, gate, theta
):
    pytest.importorskip(module)
    spec = _phase_spec(gate)
    weights = np.array([theta])
    try:
        actual = Circuit(spec, all_z(2), provider, "numpy").statevector(weights)
    except (NotImplementedError, ValueError) as exc:
        pytest.skip(f"{provider} cannot run this circuit: {exc}")
    expected = Circuit(spec, all_z(2), "numpy", "numpy").statevector(weights)
    np.testing.assert_allclose(actual, expected, atol=tolerance)


@pytest.mark.parametrize("gate", ["PhaseShift", "CPHASE"])
def test_a_constant_phase_gate_is_exact(gate):
    """The constant-angle path uses qulacs' exact U1 and carries no such factor;
    it is checked so the two paths cannot diverge from each other."""
    pytest.importorskip("qulacs")
    builder = CircuitBuilder(n_qubits=2)
    builder.h(0)
    builder.h(1)
    if gate == "PhaseShift":
        builder.phase_shift(0.8, qubit=0)
    else:
        builder.cphase(0.8, control=0, target=1)
    spec = builder.build()
    empty = np.empty(0)
    np.testing.assert_allclose(
        Circuit(spec, all_z(2), "qulacs", "numpy").statevector(empty),
        Circuit(spec, all_z(2), "numpy", "numpy").statevector(empty),
        atol=1e-9,
    )
