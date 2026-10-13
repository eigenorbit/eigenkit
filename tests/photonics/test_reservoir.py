"""Photonic reservoir computing tests.

Headline: a fixed random interferometer + a trained linear readout learns the
bilinear memory task ``y_t = x_t * x_{t-1}`` — which a linear readout on the raw
windowed inputs cannot (the reservoir supplies the nonlinearity), confirming the
frozen optics act as a useful feature map.
"""

import numpy as np
import pytest

from eigenkit.photonics.reservoir import (
    fock_basis,
    photon_number_features,
    random_reservoir_spec,
    run_photonic_reservoir,
)
from eigenkit.reservoir import Carry, LinearReadout, Stateless


pytest.importorskip("perceval")


def test_fock_basis():
    assert fock_basis(2, 1) == [(0, 1), (1, 0)]
    basis = fock_basis(3, 2)
    assert len(basis) == 6  # C(2 + 3 - 1, 3 - 1)
    assert all(sum(state) == 2 for state in basis)
    assert basis == sorted(basis)


def test_feature_trajectories_are_well_formed():
    rng = np.random.default_rng(0)
    spec, weights = random_reservoir_spec(
        4, n_inputs=1, depth=2, input_state=(1, 1, 0, 0), rng=rng
    )
    seqs = rng.uniform(-1.0, 1.0, size=(3, 5, 1))

    fock = run_photonic_reservoir(spec, weights, seqs, features="fock")
    assert fock.shape == (3, 5, len(fock_basis(4, 2)))
    assert np.allclose(fock.sum(axis=-1), 1.0, atol=1e-9)  # probabilities

    pn = run_photonic_reservoir(spec, weights, seqs, features="photon_number")
    assert pn.shape == (3, 5, 4)
    assert np.allclose(pn.sum(axis=-1), 2.0, atol=1e-9)  # 2 photons conserved


def test_photon_number_features_helper():
    # one photon certainly in mode 1 -> <n> = [0, 1]
    assert np.allclose(photon_number_features({(0, 1): 1.0}, 2), [0.0, 1.0])


def test_depth_zero_rejected():
    # depth=0 encodes the input as trailing phase shifters with no subsequent mixing,
    # which is inert for Fock-diagonal readouts -> the reservoir would ignore its input.
    with pytest.raises(ValueError, match="depth"):
        random_reservoir_spec(3, n_inputs=1, depth=0)


def test_single_mode_rejected():
    # One mode has no beam splitter to mix with, the same inert case as depth=0.
    with pytest.raises(ValueError, match="n_modes"):
        random_reservoir_spec(1, n_inputs=1)


@pytest.mark.parametrize("n_inputs", [0, 3])
def test_n_inputs_outside_bounds_rejected(n_inputs):
    # 0 ignores the input. With every mode phase-encoded, adding a constant to all
    # inputs is a global phase (photon number is conserved), so the features would
    # see only differences between inputs.
    with pytest.raises(ValueError, match="n_inputs"):
        random_reservoir_spec(3, n_inputs=n_inputs)


def test_vacuum_input_rejected():
    with pytest.raises(ValueError, match="photon"):
        random_reservoir_spec(3, n_inputs=1, input_state=(0, 0, 0))


class _CountingExecutor:
    """Stands in for a billable executor: records every run."""

    def __init__(self):
        self.runs = 0

    def compile(self, spec):
        return self

    def run(self, weights, inputs):
        self.runs += 1
        return {(1, 0, 0): 1.0}


def test_unknown_features_rejected_before_any_run():
    spec, weights = random_reservoir_spec(3, n_inputs=1, rng=np.random.default_rng(0))
    executor = _CountingExecutor()
    with pytest.raises(ValueError, match="features"):
        run_photonic_reservoir(
            spec,
            weights,
            np.zeros((2, 3, 1)),
            executor=executor,
            features="photon-number",
        )
    assert executor.runs == 0


def test_photon_number_features_skip_the_fock_basis(monkeypatch):
    # The basis grows as C(n_modes + N - 1, N); photon-number features never read it.
    from eigenkit.photonics import reservoir

    def no_basis(*args):
        raise AssertionError("fock_basis built for photon_number features")

    monkeypatch.setattr(reservoir, "fock_basis", no_basis)
    spec, weights = random_reservoir_spec(3, n_inputs=1, rng=np.random.default_rng(0))
    out = run_photonic_reservoir(
        spec, weights, np.zeros((2, 3, 1)), features="photon_number"
    )
    assert out.shape == (2, 3, 3)


@pytest.mark.parametrize("shape", [(0, 4, 1), (2, 0, 1)])
@pytest.mark.parametrize("features, n_features", [("fock", 3), ("photon_number", 3)])
def test_empty_batch_returns_empty_trajectory(shape, features, n_features):
    spec, weights = random_reservoir_spec(3, n_inputs=1, rng=np.random.default_rng(0))
    out = run_photonic_reservoir(spec, weights, np.zeros(shape), features=features)
    assert out.shape == shape[:2] + (n_features,)


def test_sequential_policy_rejected():
    rng = np.random.default_rng(0)
    spec, weights = random_reservoir_spec(3, n_inputs=1, rng=rng)
    seqs = rng.uniform(-1.0, 1.0, size=(2, 4, 1))
    with pytest.raises(NotImplementedError, match="stateless"):
        run_photonic_reservoir(spec, weights, seqs, policy=Carry())


def test_reservoir_learns_bilinear_memory_task():
    rng = np.random.default_rng(1)
    batch, timesteps = 40, 12
    seqs = rng.uniform(-1.0, 1.0, size=(batch, timesteps, 1))
    prev = np.concatenate([np.zeros((batch, 1, 1)), seqs[:, :-1, :]], axis=1)
    targets = (seqs * prev)[..., 0]  # y_t = x_t * x_{t-1}: needs memory + nonlinearity

    window = 2
    spec, weights = random_reservoir_spec(
        n_modes=4,
        n_inputs=window,
        depth=2,
        input_state=(1, 1, 0, 0),
        input_scale=0.5,  # keep sin(scale*x) quasi-linear so sin*sin ~ x*x'
        rng=np.random.default_rng(3),
    )
    feats = run_photonic_reservoir(spec, weights, seqs, policy=Stateless(window=window))

    reservoir_x = feats.reshape(batch * timesteps, -1)
    raw_x = Stateless(window=window).expand_inputs(seqs).reshape(batch * timesteps, -1)
    y = targets.reshape(batch * timesteps)

    perm = np.random.default_rng(2).permutation(batch * timesteps)
    n_train = int(0.7 * len(perm))
    train, test = perm[:n_train], perm[n_train:]

    def test_mse(x_matrix):
        head = LinearReadout(ridge=1e-6).fit(x_matrix[train], y[train])
        return float(np.mean((head.predict(x_matrix[test]) - y[test]) ** 2))

    reservoir_mse = test_mse(reservoir_x)
    raw_mse = test_mse(raw_x)
    target_var = float(np.var(y[test]))

    # The linear-on-raw baseline can't fit a product (best linear fit ~ 0 -> MSE ~ var).
    assert reservoir_mse < 0.4 * raw_mse
    assert reservoir_mse < 0.3 * target_var
