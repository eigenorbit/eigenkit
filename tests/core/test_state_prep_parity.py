"""Every backend prepares the same state, from any starting state.

`AmplitudeEncoding` asks for a unitary whose first column is the requested
state. The other columns are unconstrained by that request, so backends left
to choose their own completion (a QR one, a Gram-Schmidt one, an SDK's
state-preparation synthesis) agree when the target register is |0...0> and
disagree everywhere else — the same circuit can return -0.74, -1.0, 0.0 and
+0.98, with nothing raising.

The completion lives in one place. These tests pin that, and they place
`AmplitudeEncoding` after something else in the block list, which is where
the completion columns actually matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.backends._state_prep import state_prep_unitary
from eigenkit.core.blocks import AmplitudeEncoding, AngleEncoding, assemble
from eigenkit.core.circuit import Circuit
from eigenkit.core.observables import Z


# Backends declaring AmplitudeEncoding, with the module that must be importable.
PROVIDERS = {
    "numpy": None,
    "pennylane": "pennylane",
    "qiskit": "qiskit",
    "qiskit-aer": "qiskit_aer",
    "qulacs": "qulacs",
}


def _provider(name):
    module = PROVIDERS[name]
    if module is not None:
        pytest.importorskip(module)
    return name


class TestTheUnitaryItself:
    @pytest.mark.parametrize(
        "amps", [[1, 0], [0.5, 0.5, 0.5, 0.5], [0.6, 0.8], [1, 2, 3, 4]]
    )
    def test_first_column_is_the_normalized_state(self, amps):
        psi = np.asarray(amps, dtype=np.complex128)
        psi = psi / np.linalg.norm(psi)
        np.testing.assert_allclose(state_prep_unitary(amps)[:, 0], psi, atol=1e-12)

    @pytest.mark.parametrize("amps", [[0.5, 0.5, 0.5, 0.5], [1, 2, 3, 4]])
    def test_is_unitary(self, amps):
        u = state_prep_unitary(amps)
        np.testing.assert_allclose(u.conj().T @ u, np.eye(len(u)), atol=1e-12)

    def test_is_deterministic(self):
        a = state_prep_unitary([0.5, 0.5, 0.5, 0.5])
        b = state_prep_unitary([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(a, b)


class TestCrossBackendParity:
    """numpy is the reference; every other backend must match it."""

    @staticmethod
    def _spec_after_another_block():
        # AmplitudeEncoding second, so the target register is NOT |0...0> and
        # the completion columns are actually exercised.
        return assemble(
            [AngleEncoding(in_features=2), AmplitudeEncoding([0.5, 0.5, 0.5, 0.5])]
        )

    @staticmethod
    def _spec_alone():
        return assemble([AmplitudeEncoding([0.1, 0.2, 0.3, 0.9])])

    @pytest.mark.parametrize("provider", list(PROVIDERS))
    @pytest.mark.parametrize("angle", [np.pi / 2, 0.3, -1.2])
    def test_after_another_block(self, provider, angle):
        spec = self._spec_after_another_block()
        obs = Z(0).to_observable()
        inputs = np.array([angle, 0.0])
        expected = Circuit(spec, obs, "numpy", "numpy")(np.empty(0), inputs)
        actual = Circuit(spec, obs, _provider(provider), "numpy")(np.empty(0), inputs)
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    @pytest.mark.parametrize("provider", list(PROVIDERS))
    def test_from_the_zero_state(self, provider):
        spec = self._spec_alone()
        obs = Z(0).to_observable()
        expected = Circuit(spec, obs, "numpy", "numpy")(np.empty(0))
        actual = Circuit(spec, obs, _provider(provider), "numpy")(np.empty(0))
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    @pytest.mark.parametrize("provider", list(PROVIDERS))
    def test_two_encodings_in_a_row(self, provider):
        # The second one lands on a fully populated register.
        spec = assemble(
            [
                AmplitudeEncoding([0.5, 0.5, 0.5, 0.5]),
                AmplitudeEncoding([0.1, 0.2, 0.3, 0.9]),
            ]
        )
        obs = Z(0).to_observable()
        expected = Circuit(spec, obs, "numpy", "numpy")(np.empty(0))
        actual = Circuit(spec, obs, _provider(provider), "numpy")(np.empty(0))
        np.testing.assert_allclose(actual, expected, atol=1e-9)
