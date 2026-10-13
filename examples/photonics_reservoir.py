"""Photonic reservoir computing: learn a bilinear memory task with frozen optics.

A fixed (random, untrained) linear-optical interferometer maps each timestep's input
window into photon-number statistics; only a classical ridge readout is trained. The
task ``y_t = x_t * x_{t-1}`` needs both memory (a 2-step window) and nonlinearity (a
product), so a linear readout on the raw inputs fails (its MSE ~ the target variance)
while the reservoir's Fock features succeed.

Run: ``python examples/photonics_reservoir.py``
"""

import numpy as np

from eigenkit.photonics.reservoir import (
    random_reservoir_spec,
    run_photonic_reservoir,
)
from eigenkit.reservoir import LinearReadout, Stateless


def main() -> None:
    rng = np.random.default_rng(1)
    batch, timesteps, window = 40, 12, 2
    seqs = rng.uniform(-1.0, 1.0, size=(batch, timesteps, 1))
    prev = np.concatenate([np.zeros((batch, 1, 1)), seqs[:, :-1, :]], axis=1)
    y = (seqs * prev)[..., 0].reshape(-1)  # y_t = x_t * x_{t-1}

    # A frozen random photonic reservoir: 4 modes, 2 photons, encoding the 2-step window.
    spec, weights = random_reservoir_spec(
        n_modes=4,
        n_inputs=window,
        depth=2,
        input_state=(1, 1, 0, 0),
        input_scale=0.5,
        rng=np.random.default_rng(3),
    )
    feats = run_photonic_reservoir(spec, weights, seqs, policy=Stateless(window=window))
    reservoir_x = feats.reshape(batch * timesteps, -1)
    raw_x = Stateless(window=window).expand_inputs(seqs).reshape(batch * timesteps, -1)

    perm = np.random.default_rng(2).permutation(batch * timesteps)
    n_train = int(0.7 * len(perm))
    train, test = perm[:n_train], perm[n_train:]

    def test_mse(features: np.ndarray) -> float:
        head = LinearReadout(ridge=1e-6).fit(features[train], y[train])
        return float(np.mean((head.predict(features[test]) - y[test]) ** 2))

    print(f"target variance      : {np.var(y[test]):.4f}")
    print(f"linear on raw inputs : MSE {test_mse(raw_x):.4f}   (cannot fit a product)")
    print(
        f"photonic reservoir   : MSE {test_mse(reservoir_x):.4f}   "
        f"({reservoir_x.shape[1]} Fock features, optics frozen)"
    )


if __name__ == "__main__":
    main()
