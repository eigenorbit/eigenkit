"""A reservoir must give the same answer on every backend that can run it.

The reservoir reads the statevector itself rather than an expectation value,
so it is the consumer most exposed to a backend that is only correct up to a
phase. Both readouts are covered, and both state policies: `leaky` mixes the
evolved state with a reset one, which converts a global phase into a relative
one and moves even the default `<Z>` features.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit.core.blocks import AngleEncoding, BasicVariational
from eigenkit.qml.qrc import QuantumReservoirRegressor


BACKENDS = {"pennylane": "pennylane", "qiskit": "qiskit", "qulacs": "qulacs"}


def _data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(24, 4, 2)), rng.normal(size=(24, 4))


def _blocks():
    # RZ rotations make the variational block a trainable phase gate, the case
    # where a lowering that is correct only up to a phase would diverge.
    return [AngleEncoding(in_features=2), BasicVariational(depth=1, rotation="RZ")]


@pytest.mark.parametrize("provider,module", sorted(BACKENDS.items()))
@pytest.mark.parametrize("features", ["z", "amplitudes"])
@pytest.mark.parametrize("policy", ["carry", "leaky"])
def test_predictions_match_the_numpy_reference(provider, module, features, policy):
    pytest.importorskip(module)
    X, y = _data()

    def fit(backend):
        return QuantumReservoirRegressor(
            blocks=_blocks(),
            state_policy=policy,
            features=features,
            provider=backend,
            random_state=0,
            washout=1,
            leak=0.5,
        ).fit(X, y)

    try:
        actual = fit(provider).predict(X)
    except NotImplementedError as exc:
        pytest.skip(f"{provider} cannot drive a reservoir: {exc}")
    np.testing.assert_allclose(actual, fit("numpy").predict(X), atol=1e-8)


@pytest.mark.parametrize("features", ["z", "amplitudes"])
def test_the_amplitude_readout_actually_reads_phases(features):
    """Guard the guard: `amplitudes` must depend on the imaginary part, or the
    parity test above would pass for the wrong reason."""
    from eigenkit.reservoir.features import amplitude_features

    real_only = np.array([[0.6 + 0j, 0.8 + 0j]])
    with_phase = np.array([[0.6 + 0j, 0.8j]])
    assert not np.allclose(amplitude_features(real_only), amplitude_features(with_phase))
