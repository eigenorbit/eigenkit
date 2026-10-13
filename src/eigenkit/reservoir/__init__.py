"""Quantum reservoir computing (QRC).

A circuit with fixed weights, usually random, is driven over input sequences.
The quantum state is carried from one timestep to the next, and classical
features are read out from it.

The reservoir circuit is never trained; only a classical readout is. The
driver, the policies, the feature extractors and the ridge-regression readout
use only NumPy. Like `optimization` and `chemistry`, this package belongs to the
core of the library, so it needs neither PyTorch nor a backend package.

Three policies cover the common variants of QRC:

- [`Stateless`][eigenkit.reservoir.Stateless] starts every step from `|0…0⟩`
  and encodes a sliding window of inputs. All timesteps run in parallel, in one
  batched evaluation.
- [`Carry`][eigenkit.reservoir.Carry] carries the state from one step to the
  next with the circuit's unitary: `ψ_t = U(x_t) ψ_{t-1}`.
- [`LeakyCarry`][eigenkit.reservoir.LeakyCarry] mixes the carried state with a
  reset state, and renormalises the result.

Example:
    ```python
    from eigenkit import Circuit
    from eigenkit.reservoir import (
        run_reservoir, Carry, expectation_features, LinearReadout,
    )

    states = run_reservoir(circuit, Carry(), sequences, weights)  # (B, T, 2**n)
    feats = expectation_features(states, "Z0 + Z1 + 0.5 X0 X1")   # (B, T, 1)
    readout = LinearReadout(ridge=1e-4).fit(feats.reshape(B * T, -1), targets)
    ```
"""

from .driver import run_reservoir
from .features import (
    amplitude_features,
    expectation_features,
    pauli_z_features,
    probability_features,
)
from .policies import Carry, LeakyCarry, ReservoirPolicy, ReservoirState, Stateless
from .readout import LinearReadout


__all__ = [
    "run_reservoir",
    "ReservoirPolicy",
    "ReservoirState",
    "Stateless",
    "Carry",
    "LeakyCarry",
    "expectation_features",
    "probability_features",
    "pauli_z_features",
    "amplitude_features",
    "LinearReadout",
]
