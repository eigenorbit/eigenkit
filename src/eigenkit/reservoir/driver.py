r"""The reservoir driver: run a frozen circuit over input sequences, carrying
quantum state across timesteps per a policy, and return the statevector
trajectory.

The reservoir (circuit + fixed ``weights``) is never trained here — this is the
forward pass that produces the features a classical readout learns on. Two
dispatch paths, chosen by the policy:

- **Stateless / parallel** (policy exposes ``expand_inputs``): all \(B \cdot T\)
  timesteps are independent, so they evolve in a single batched ``statevector``
  sweep. No Python timestep loop.
- **Sequential** (policy implements ``initial`` / ``inject`` / ``advance``): the
  quantum state is carried step to step, so timesteps run in order. The ``B``
  sequences still evolve in parallel within each step via per-sample injection.
"""

from __future__ import annotations

import numpy as np


def _require_n_inputs(got: int, circuit, *, timestep: int | None = None) -> None:
    expected = circuit.spec.n_inputs
    if got != expected:
        where = "" if timestep is None else f" at timestep {timestep}"
        raise ValueError(
            f"policy produced inputs with {got} feature(s){where} but the "
            f"circuit expects n_inputs={expected}. For a windowed Stateless "
            f"policy the circuit must encode window*F inputs."
        )


def run_reservoir(circuit, policy, sequences, weights) -> np.ndarray:
    """Evolve `circuit` over `sequences` under `policy`, with frozen `weights`.

    The result is the trajectory of statevectors, a complex array of shape
    `(B, T, 2**n)`.

    `sequences` has shape `(B, T, F)`: `B` independent series, `T` timesteps and
    `F` features per step. `weights` is the fixed 1-D parameter vector of the
    reservoir, shared by the whole batch. Features are extracted separately from
    the returned trajectory (see the feature extractors), so the same run can
    produce expectation, probability or amplitude features without evolving the
    circuit again.

    Args:
        circuit (Circuit): The reservoir circuit, run with frozen weights.
        policy (Policy): How the state passes from one timestep to the next.
        sequences (np.ndarray): The inputs, of shape `(B, T, F)`.
        weights (np.ndarray): The frozen weights of the reservoir.

    Returns:
        The trajectory of statevectors, of shape `(B, T, 2**n)`.
    """
    sequences = np.asarray(sequences, dtype=np.float64)
    if sequences.ndim != 3:
        raise ValueError(f"sequences must be 3-D (B, T, F), got ndim={sequences.ndim}")
    weights = np.asarray(weights, dtype=np.float64)
    B, T, _ = sequences.shape

    expand = getattr(policy, "expand_inputs", None)
    if expand is not None:
        # Stateless / parallel: one batched sweep over all B*T samples.
        inputs_bt = np.asarray(expand(sequences), dtype=np.float64)
        _require_n_inputs(inputs_bt.shape[-1], circuit)
        flat = inputs_bt.reshape(B * T, -1)
        states = circuit.statevector(weights, flat)  # (B*T, 2**n)
        return states.reshape(B, T, -1)

    # Sequential: fold over time, carrying quantum state.
    state = policy.initial(batch=B)
    traj = []
    for t in range(T):
        x_t = sequences[:, t, :]
        init, inputs = policy.inject(x_t, state)
        inputs = np.asarray(inputs, dtype=np.float64)
        _require_n_inputs(inputs.shape[-1], circuit, timestep=t)
        evolved = circuit.statevector(weights, inputs, initial_state=init)  # (B, 2**n)
        traj.append(evolved)
        state = policy.advance(x_t, evolved, state)
    return np.stack(traj, axis=1)  # (B, T, 2**n)
