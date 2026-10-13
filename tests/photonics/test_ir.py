"""Photonic IR construction: angle coercion, tuple freezing and validation.

Not gated on perceval: building a spec is pure data. The one test that runs a
circuit skips on its own when the SDK is absent.
"""

import math

import numpy as np
import pytest

from eigenkit.core.ir import Constant, InputRef, Scale, Sum, WeightRef
from eigenkit.photonics import (
    BeamSplitter,
    Permutation,
    PhaseShifter,
    PhotonicCircuitSpec,
)


@pytest.mark.parametrize("angle", [0.5, 2, np.float64(0.5)])
def test_plain_number_angles_become_constants(angle):
    assert PhaseShifter(0, angle).phi == Constant(float(angle))
    splitter = BeamSplitter((0, 1), angle, angle)
    assert splitter.theta == Constant(float(angle))
    assert splitter.phi == Constant(float(angle))


def test_expression_angles_are_kept_as_given():
    phase = Sum((WeightRef(0), Scale(InputRef(0), 2.0)))
    assert PhaseShifter(0, phase).phi is phase
    assert BeamSplitter((0, 1), WeightRef(1)).theta == WeightRef(1)


def _mzi(balanced, phase):
    return PhotonicCircuitSpec(
        n_modes=2,
        n_weights=0,
        n_inputs=0,
        components=(
            BeamSplitter((0, 1), balanced),
            PhaseShifter(0, phase),
            BeamSplitter((0, 1), balanced),
        ),
        input_state=(1, 0),
    )


def test_number_built_spec_equals_constant_built_spec():
    by_number = _mzi(math.pi / 2, 0.9)
    by_constant = _mzi(Constant(math.pi / 2), Constant(0.9))
    assert by_number == by_constant
    assert hash(by_number) == hash(by_constant)


def test_number_built_spec_runs():
    pytest.importorskip("perceval")
    from eigenkit.photonics import PercevalBackend

    empty = np.array([])
    by_number = PercevalBackend().compile(_mzi(math.pi / 2, 0.9)).run(empty, empty)
    by_constant = (
        PercevalBackend()
        .compile(_mzi(Constant(math.pi / 2), Constant(0.9)))
        .run(empty, empty)
    )
    assert by_number == pytest.approx(by_constant, abs=1e-12)


# ---- validation: a bad circuit fails when it is built, not inside Perceval ----
# Perceval's own checks are asserts, which `python -O` strips: there a reversed
# beam splitter silently acts as (0, 1) and an out-of-range phase is ignored.


@pytest.mark.parametrize("modes", [(0, 2), (1, 0), (0,), (0, 1, 2)])
def test_beam_splitter_needs_adjacent_modes(modes):
    with pytest.raises(ValueError, match="adjacent"):
        BeamSplitter(modes, 0.5)


@pytest.mark.parametrize("perm", [(0, 0), (1, 2)])
def test_permutation_must_reorder_leading_modes(perm):
    with pytest.raises(ValueError, match="perm"):
        Permutation(perm)


def _spec(component, n_modes=2):
    return PhotonicCircuitSpec(
        n_modes=n_modes,
        n_weights=0,
        n_inputs=0,
        components=(component,),
        input_state=(1,) + (0,) * (n_modes - 1),
    )


@pytest.mark.parametrize(
    "component",
    [
        PhaseShifter(5, 0.1),
        PhaseShifter(-1, 0.1),
        BeamSplitter((1, 2), 0.5),
        Permutation((2, 0, 1)),
    ],
)
def test_component_outside_the_circuit_rejected(component):
    with pytest.raises(ValueError, match="outside modes"):
        _spec(component)


def test_unknown_component_rejected():
    with pytest.raises(TypeError, match="unknown photonic component"):
        _spec("not a component")


@pytest.mark.parametrize("input_state", [(1.5, 0), (-1, 1)])
def test_photon_counts_must_be_non_negative_integers(input_state):
    with pytest.raises(ValueError, match="non-negative integers"):
        PhotonicCircuitSpec(2, 0, 0, (), input_state)
