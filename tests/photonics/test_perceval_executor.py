"""End-to-end tests of the Perceval executor.

Validates two things the photonics ring rests on: (1) the Perceval executor computes
genuine boson statistics (the Hong-Ou-Mandel dip), and (2) the qubit IR's Parameter
expression tree is reused verbatim for photonic components (trainable + data-
encoding). The lazy import boundary is pinned in ``test_import_surface.py``, which
runs without perceval.
"""

import math

import numpy as np
import pytest

from eigenkit.core.ir import Constant, InputRef, Scale, Sum, WeightRef
from eigenkit.photonics import (
    BeamSplitter,
    PercevalBackend,
    PhaseShifter,
    PhotonicCircuitSpec,
)


# These tests exercise the real SDK; skip cleanly if the [perceval] extra is absent.
pytest.importorskip("perceval")

BALANCED = Constant(math.pi / 2)  # theta = pi/2 -> 50:50 beam splitter


def _run(spec, weights=(), inputs=()):
    compiled = PercevalBackend().compile(spec)
    return compiled.run(np.asarray(weights, dtype=float), np.asarray(inputs, dtype=float))


def test_single_photon_5050_split():
    spec = PhotonicCircuitSpec(
        n_modes=2,
        n_weights=0,
        n_inputs=0,
        components=(BeamSplitter(modes=(0, 1), theta=BALANCED),),
        input_state=(1, 0),
    )
    dist = _run(spec)
    assert dist[(1, 0)] == pytest.approx(0.5, abs=1e-9)
    assert dist[(0, 1)] == pytest.approx(0.5, abs=1e-9)


def test_hong_ou_mandel_dip():
    # Two indistinguishable photons, one per mode, into a balanced beam splitter
    # bunch: P(|1,1>) collapses to 0. This is a genuinely non-classical,
    # permanent-driven result, so passing it proves the executor computes boson
    # statistics, not classical light routing.
    spec = PhotonicCircuitSpec(
        n_modes=2,
        n_weights=0,
        n_inputs=0,
        components=(BeamSplitter(modes=(0, 1), theta=BALANCED),),
        input_state=(1, 1),
    )
    dist = _run(spec)
    assert dist.get((1, 1), 0.0) == pytest.approx(0.0, abs=1e-9)
    assert dist[(2, 0)] == pytest.approx(0.5, abs=1e-9)
    assert dist[(0, 2)] == pytest.approx(0.5, abs=1e-9)


def test_distribution_normalized():
    spec = PhotonicCircuitSpec(
        n_modes=3,
        n_weights=0,
        n_inputs=0,
        components=(
            BeamSplitter(modes=(0, 1), theta=Constant(1.0)),
            BeamSplitter(modes=(1, 2), theta=Constant(2.0)),
        ),
        input_state=(1, 1, 0),
    )
    assert sum(_run(spec).values()) == pytest.approx(1.0, abs=1e-9)


def test_parameter_tree_reuse_weight_and_input():
    # The photonic IR reuses the qubit IR's Parameter tree verbatim. A phase of
    # Sum(WeightRef(0), Scale(InputRef(0), 2.0)) must resolve and thread through to
    # perceval exactly like a hand-built PS(w0 + 2*x0) — proving trainable AND
    # data-encoding components fall out of the shared substrate.
    w, x = 0.3, 0.4
    phi = Sum((WeightRef(0), Scale(InputRef(0), 2.0)))
    mzi = (
        BeamSplitter(modes=(0, 1), theta=BALANCED),
        PhaseShifter(mode=0, phi=phi),
        BeamSplitter(modes=(0, 1), theta=BALANCED),
    )
    symbolic = _run(
        PhotonicCircuitSpec(
            n_modes=2, n_weights=1, n_inputs=1, components=mzi, input_state=(1, 0)
        ),
        weights=[w],
        inputs=[x],
    )
    reference = _run(
        PhotonicCircuitSpec(
            n_modes=2,
            n_weights=0,
            n_inputs=0,
            components=(
                BeamSplitter(modes=(0, 1), theta=BALANCED),
                PhaseShifter(mode=0, phi=Constant(w + 2 * x)),
                BeamSplitter(modes=(0, 1), theta=BALANCED),
            ),
            input_state=(1, 0),
        )
    )
    for state in set(symbolic) | set(reference):
        assert symbolic.get(state, 0.0) == pytest.approx(
            reference.get(state, 0.0), abs=1e-9
        )

    # ...and the parameters actually matter (non-trivial dependence).
    other = _run(
        PhotonicCircuitSpec(
            n_modes=2, n_weights=1, n_inputs=1, components=mzi, input_state=(1, 0)
        ),
        weights=[1.9],
        inputs=[0.0],
    )
    assert abs(other.get((1, 0), 0.0) - symbolic.get((1, 0), 0.0)) > 1e-3


def test_input_state_validation():
    with pytest.raises(ValueError, match="input_state"):
        PhotonicCircuitSpec(
            n_modes=2, n_weights=0, n_inputs=0, components=(), input_state=(1, 0, 0)
        )
