"""Executor-protocol conformance and registry tests for the photonic ring."""

import math

import numpy as np
import pytest

from eigenkit.core.ir import Constant, InputRef, Sum, WeightRef
from eigenkit.photonics import (
    BeamSplitter,
    Compiled,
    Executor,
    PercevalBackend,
    PhaseShifter,
    PhotonicCircuitSpec,
)
from eigenkit.photonics import executor as executor_mod


pytest.importorskip("perceval")


def _spec():
    return PhotonicCircuitSpec(
        n_modes=2,
        n_weights=0,
        n_inputs=0,
        components=(BeamSplitter(modes=(0, 1), theta=Constant(math.pi / 2)),),
        input_state=(1, 0),
    )


def test_perceval_backend_satisfies_protocols():
    backend = PercevalBackend()
    assert isinstance(backend, Executor)
    assert isinstance(backend.compile(_spec()), Compiled)


def test_registry_resolves_perceval():
    assert "perceval" in executor_mod.registered_names()
    assert isinstance(executor_mod.get("perceval"), Executor)


@pytest.mark.parametrize("name", ["CliffordClifford2017", "SamplingStepper"])
def test_sampling_backend_rejected(name):
    # run() returns the probability distribution, which sampling backends don't have.
    with pytest.raises(ValueError, match="only samples"):
        PercevalBackend(name)


def test_unknown_backend_rejected():
    # perceval itself would silently fall back to SLOS.
    with pytest.raises(ValueError, match="unknown perceval backend"):
        PercevalBackend("NoSuchBackend")


def test_exact_alternative_backend_matches_slos():
    w, x = np.array([]), np.array([])
    slos = PercevalBackend("SLOS").compile(_spec()).run(w, x)
    naive = PercevalBackend("Naive").compile(_spec()).run(w, x)
    for state in set(slos) | set(naive):
        assert naive.get(state, 0.0) == pytest.approx(slos.get(state, 0.0), abs=1e-12)


@pytest.mark.parametrize(
    "weights, inputs",
    [
        ([0.1, 0.2], [0.3]),  # one weight too many
        ([0.1], [0.3, 0.4]),  # two samples passed as one flat vector
        ([[0.1]], [0.3]),  # 2-D weights
    ],
)
def test_run_rejects_mismatched_lengths(weights, inputs):
    spec = PhotonicCircuitSpec(
        n_modes=2,
        n_weights=1,
        n_inputs=1,
        components=(
            BeamSplitter(modes=(0, 1), theta=Constant(math.pi / 2)),
            PhaseShifter(mode=0, phi=Sum((WeightRef(0), InputRef(0)))),
            BeamSplitter(modes=(0, 1), theta=Constant(math.pi / 2)),
        ),
        input_state=(1, 0),
    )
    with pytest.raises(ValueError, match="shape"):
        PercevalBackend().compile(spec).run(np.asarray(weights), np.asarray(inputs))
