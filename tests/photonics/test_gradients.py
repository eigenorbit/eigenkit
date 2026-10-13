"""Gradient tests for the photonic substrate.

The correctness oracle is central finite differences. The headline result is
``test_two_photon_requires_multiterm_rule``: with two photons the naive ±π/2
(order=1) rule is *wrong*, and only the multi-term photonic parameter-shift rule
matches finite differences — the whole reason the photonic rule exists.
"""

import math

import numpy as np
import pytest

from eigenkit.core.ir import (
    Constant,
    InputRef,
    Multiply,
    Scale,
    Sum,
    WeightRef,
)
from eigenkit.photonics import (
    BeamSplitter,
    FockProbability,
    PercevalBackend,
    PhaseShifter,
    PhotonicCircuitSpec,
    PhotonNumber,
)
from eigenkit.photonics import gradients as grad


pytest.importorskip("perceval")

BALANCED = Constant(math.pi / 2)


def _mzi(phase, n_weights, n_inputs, input_state=(1, 0)):
    """A Mach-Zehnder: balanced BS, a trainable/data phase on mode 0, balanced BS."""
    return PhotonicCircuitSpec(
        n_modes=2,
        n_weights=n_weights,
        n_inputs=n_inputs,
        components=(
            BeamSplitter(modes=(0, 1), theta=BALANCED),
            PhaseShifter(mode=0, phi=phase),
            BeamSplitter(modes=(0, 1), theta=BALANCED),
        ),
        input_state=input_state,
    )


def _jac(spec, observable, weights, inputs, method):
    compiled = PercevalBackend().compile(spec)
    return compiled.jacobian(
        np.asarray(weights, dtype=float),
        np.asarray(inputs, dtype=float),
        observable,
        method=method,
    )


def test_single_photon_shift_matches_finite_diff():
    spec = _mzi(WeightRef(0), n_weights=1, n_inputs=0)
    obs = PhotonNumber(mode=0)
    ps, _ = _jac(spec, obs, [0.7], [], "parameter_shift")
    fd, _ = _jac(spec, obs, [0.7], [], "finite_difference")
    assert np.allclose(ps, fd, atol=1e-6)
    assert abs(fd[0]) > 1e-2  # a real, non-trivial gradient


@pytest.mark.parametrize(
    "phase",
    [
        Scale(WeightRef(0), 2.0),  # chain factor 2
        Sum((Scale(WeightRef(0), 2.0), InputRef(0))),  # weight + input
        Multiply(WeightRef(0), InputRef(0)),  # bilinear feature scaling
    ],
)
def test_chain_rule_angles_match_finite_diff(phase):
    spec = _mzi(phase, n_weights=1, n_inputs=1)
    obs = FockProbability(target=(1, 0))
    ps_w, ps_x = _jac(spec, obs, [0.4], [0.9], "parameter_shift")
    fd_w, fd_x = _jac(spec, obs, [0.4], [0.9], "finite_difference")
    assert np.allclose(ps_w, fd_w, atol=1e-6)
    assert np.allclose(ps_x, fd_x, atol=1e-6)


def test_trainable_beamsplitter_matches_finite_diff():
    # The rule also covers a beam-splitter angle (SU(2) generator), not just phases.
    spec = PhotonicCircuitSpec(
        n_modes=2,
        n_weights=1,
        n_inputs=0,
        components=(BeamSplitter(modes=(0, 1), theta=WeightRef(0)),),
        input_state=(1, 0),
    )
    obs = PhotonNumber(mode=0)
    ps, _ = _jac(spec, obs, [0.6], [], "parameter_shift")
    fd, _ = _jac(spec, obs, [0.6], [], "finite_difference")
    assert np.allclose(ps, fd, atol=1e-6)


@pytest.mark.parametrize("obs", [FockProbability(target=(0, 1, 0)), PhotonNumber(mode=0)])
def test_trainable_beamsplitter_with_spectator_mode_matches_finite_diff(obs):
    # With a third mode, the components around the trainable splitter superpose
    # sectors with an odd and an even photon count in its two modes, so its theta
    # carries half-integer frequencies. The integer-frequency rule gets these wrong
    # (even the sign); the half-angle rule must match finite differences.
    spec = PhotonicCircuitSpec(
        n_modes=3,
        n_weights=1,
        n_inputs=0,
        components=(
            BeamSplitter(modes=(0, 1), theta=BALANCED),
            BeamSplitter(modes=(1, 2), theta=WeightRef(0)),
            BeamSplitter(modes=(0, 1), theta=BALANCED),
        ),
        input_state=(1, 0, 0),
    )
    ps, _ = _jac(spec, obs, [0.8], [], "parameter_shift")
    fd, _ = _jac(spec, obs, [0.8], [], "finite_difference")
    assert np.allclose(ps, fd, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_random_multimode_mesh_matches_finite_diff(seed):
    # Every slot trainable (splitter theta and phi, phases), three photons in four
    # modes, so each rule — integer for phases, half-angle for splitter angles — is
    # exercised away from the special two-mode case.
    components, n_weights = [], 0
    for _ in range(3):
        for m in range(3):
            components.append(
                BeamSplitter(
                    modes=(m, m + 1),
                    theta=WeightRef(n_weights),
                    phi=WeightRef(n_weights + 1),
                )
            )
            n_weights += 2
        for m in range(4):
            components.append(PhaseShifter(mode=m, phi=WeightRef(n_weights)))
            n_weights += 1
    spec = PhotonicCircuitSpec(
        n_modes=4,
        n_weights=n_weights,
        n_inputs=0,
        components=tuple(components),
        input_state=(1, 1, 1, 0),
    )
    weights = np.random.default_rng(seed).uniform(0.0, 2 * math.pi, n_weights)
    for obs in (FockProbability(target=(1, 0, 1, 1)), PhotonNumber(mode=2)):
        ps, _ = _jac(spec, obs, weights, [], "parameter_shift")
        fd, _ = _jac(spec, obs, weights, [], "finite_difference")
        assert np.allclose(ps, fd, atol=1e-6)


@pytest.mark.parametrize("method", ["parameter_shift", "finite_difference"])
def test_unwanted_side_is_skipped(method):
    spec = _mzi(Sum((WeightRef(0), InputRef(0))), n_weights=1, n_inputs=1)
    obs = PhotonNumber(mode=0)
    compiled = PercevalBackend().compile(spec)
    w, x = np.array([0.4]), np.array([0.9])
    full_w, full_x = compiled.jacobian(w, x, obs, method=method)

    only_w, no_x = compiled.jacobian(w, x, obs, method=method, need_input_grad=False)
    assert no_x is None
    assert np.allclose(only_w, full_w)

    no_w, only_x = compiled.jacobian(w, x, obs, method=method, need_weight_grad=False)
    assert no_w is None
    assert np.allclose(only_x, full_x)


def test_spec_built_from_lists_is_hashable_and_differentiable():
    spec = PhotonicCircuitSpec(
        2,
        1,
        0,
        [
            BeamSplitter([0, 1], BALANCED),
            PhaseShifter(0, WeightRef(0)),
            BeamSplitter([0, 1], BALANCED),
        ],
        [1, 0],
    )
    assert hash(spec) == hash(_mzi(WeightRef(0), n_weights=1, n_inputs=0))
    ps, _ = _jac(spec, PhotonNumber(mode=0), [0.7], [], "parameter_shift")
    fd, _ = _jac(spec, PhotonNumber(mode=0), [0.7], [], "finite_difference")
    assert np.allclose(ps, fd, atol=1e-6)


def test_two_photon_requires_multiterm_rule():
    # Two photons (one per input mode) => the phase-dependence of any Fock-diagonal
    # observable carries frequency 2. The naive order-1 (±π/2) rule cannot see it
    # and returns the wrong gradient; the default multi-term rule (order = n_photons
    # = 2) is exact.
    spec = _mzi(WeightRef(0), n_weights=1, n_inputs=0, input_state=(1, 1))
    obs = FockProbability(target=(2, 0))
    weights = np.array([0.7])
    empty = np.array([])

    fd, _ = grad.finite_difference_jacobian(PercevalBackend(), spec, weights, empty, obs)
    exact, _ = grad.photonic_parameter_shift_jacobian(
        PercevalBackend(), spec, weights, empty, obs
    )  # order defaults to n_photons = 2
    naive, _ = grad.photonic_parameter_shift_jacobian(
        PercevalBackend(), spec, weights, empty, obs, order=1
    )

    assert abs(fd[0]) > 1e-2  # there is a real gradient to get right
    assert np.allclose(exact, fd, atol=1e-6)  # multi-term rule nails it
    assert not np.allclose(naive, fd, atol=1e-3)  # ±π/2 rule is wrong here


def test_expectation_helper_agrees_with_distribution():
    spec = _mzi(Constant(0.9), n_weights=0, n_inputs=0)
    compiled = PercevalBackend().compile(spec)
    dist = compiled.run(np.array([]), np.array([]))
    expected_n0 = sum(prob * state[0] for state, prob in dist.items())
    assert compiled.expectation(
        np.array([]), np.array([]), PhotonNumber(mode=0)
    ) == pytest.approx(expected_n0, abs=1e-12)


@pytest.mark.parametrize(
    "obs",
    [
        FockProbability(target=(1, 0, 0)),  # three modes on a two-mode circuit
        FockProbability(target=(2, 0)),  # two photons; the circuit carries one
        PhotonNumber(mode=5),  # no such mode
    ],
)
def test_observable_that_cannot_match_is_rejected(obs):
    # A target that is never in the distribution would read 0.0 and give a zero
    # gradient, so training would silently never move.
    spec = _mzi(WeightRef(0), n_weights=1, n_inputs=0)
    compiled = PercevalBackend().compile(spec)
    w, x = np.array([0.7]), np.array([])
    for method in ("parameter_shift", "finite_difference"):
        with pytest.raises(ValueError, match="PhotonNumber|FockProbability"):
            compiled.jacobian(w, x, obs, method=method)
    with pytest.raises(ValueError, match="PhotonNumber|FockProbability"):
        compiled.expectation(w, x, obs)
