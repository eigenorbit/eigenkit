"""End-to-end proof that the photonic gradient actually *trains* a circuit.

Gradient-descends the internal phase of a single-photon Mach-Zehnder
interferometer — a tunable beam splitter — to route the photon deterministically
to a chosen output, and to hit an arbitrary target split ratio, using only
``compiled.expectation`` + ``compiled.jacobian``. If the gradient were wrong the
loss would not fall.
"""

import numpy as np
import pytest

from eigenkit.core.ir import Constant, WeightRef
from eigenkit.photonics import (
    BeamSplitter,
    FockProbability,
    PercevalBackend,
    PhaseShifter,
    PhotonicCircuitSpec,
    PhotonNumber,
)


pytest.importorskip("perceval")

_EMPTY = np.array([])
_BALANCED = Constant(np.pi / 2)


def _mzi_spec():
    """MZI: balanced BS, trainable phase on mode 0, balanced BS. One photon in mode 0."""
    return PhotonicCircuitSpec(
        n_modes=2,
        n_weights=1,
        n_inputs=0,
        components=(
            BeamSplitter(modes=(0, 1), theta=_BALANCED),
            PhaseShifter(mode=0, phi=WeightRef(0)),
            BeamSplitter(modes=(0, 1), theta=_BALANCED),
        ),
        input_state=(1, 0),
    )


def train(spec, observable, target, *, init, lr, steps):
    """Minimize ``(E[observable] - target)**2`` by gradient descent through the
    photonic parameter-shift gradient. Returns ``(weights, loss_history, value)``."""
    compiled = PercevalBackend().compile(spec)
    weights = np.array(init, dtype=float)
    history = []
    for _ in range(steps):
        value = compiled.expectation(weights, _EMPTY, observable)
        history.append((value - target) ** 2)
        grad_w, _ = compiled.jacobian(
            weights, _EMPTY, observable, method="parameter_shift"
        )
        weights = weights - lr * 2.0 * (value - target) * grad_w
    value = compiled.expectation(weights, _EMPTY, observable)
    return weights, history, value


def test_training_routes_photon_to_target_output():
    _, history, value = train(
        _mzi_spec(),
        FockProbability(target=(0, 1)),
        target=1.0,
        init=[0.3],
        lr=1.5,
        steps=300,
    )
    assert history[-1] < history[0] * 0.01  # loss collapsed
    assert value > 0.99  # photon (almost) fully routed to mode 1


def test_training_hits_arbitrary_split_ratio():
    _, history, value = train(
        _mzi_spec(),
        PhotonNumber(mode=1),
        target=0.3,
        init=[0.2],
        lr=1.5,
        steps=300,
    )
    assert abs(value - 0.3) < 0.02
    assert history[-1] < history[0] * 0.05
