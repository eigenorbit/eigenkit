r"""Reservoir policies — how each timestep maps the carried state and the
current input to the circuit's ``(initial_state, gate_inputs)``.

Two execution shapes, because they are genuinely different computations:

- **Sequential (a fold over time).** The quantum state *is* the memory: \(\psi_t\)
  depends on \(\psi_{t-1}\), so timesteps must run in order. These policies implement
  ``initial`` / ``inject`` / ``advance``
  ([`Carry`][eigenkit.reservoir.policies.Carry],
  [`LeakyCarry`][eigenkit.reservoir.policies.LeakyCarry]).
- **Parallel (a map over time).** The quantum state carries nothing between steps — each
  step's input is a pure function of the raw input sequence (e.g. a sliding window).
  These policies expose ``expand_inputs(sequences)`` and the driver runs all ``B·T``
  evaluations in **one batched sweep** instead of a Python timestep loop
  ([`Stateless`][eigenkit.reservoir.policies.Stateless]).

The driver prefers ``expand_inputs`` when a policy has it (idiomatic optional
method, like the backends' ``statevector`` / ``probabilities``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class ReservoirState:
    """The state that sequential policies carry from one timestep to the next.

    `vector` is the `(B, 2**n)` statevector to inject at the next step, and `None`
    means that the step starts from `|0…0⟩`. `history` holds the recent raw inputs,
    of shape `(B, k, F)`, for a sequential policy that encodes a window of inputs.

    Attributes:
        vector: The statevector to inject at the next step, of shape `(B, 2**n)`.
            `None` means that the step starts from `|0…0⟩`.
        history: The last `k` raw inputs, of shape `(B, k, F)`, for a policy that
            encodes a window of them. It is `None` when the policy keeps no history.
    """

    vector: np.ndarray | None = None
    history: np.ndarray | None = None


@runtime_checkable
class ReservoirPolicy(Protocol):
    """A reservoir policy that carries state from one timestep to the next.

    Parallel policies, which carry no state, do not implement this protocol. They
    provide `expand_inputs(sequences)`, which returns the inputs of every timestep
    with shape `(B, T, n_inputs)`, and the driver evaluates all timesteps at once.
    See [`Stateless`][eigenkit.reservoir.Stateless].
    """

    def initial(self, batch: int) -> ReservoirState:
        """Return the state carried into the first timestep.

        Args:
            batch (int): The number of sequences that run in parallel.

        Returns:
            The initial `ReservoirState`.
        """
        ...

    def inject(
        self, x_t: np.ndarray, state: ReservoirState
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Turn the input of this timestep into the arguments of the circuit call.

        Args:
            x_t (np.ndarray): The input of this timestep, of shape `(B, F)`.
            state (ReservoirState): The state carried over from the previous step.

        Returns:
            initial_state: The state to evaluate the circuit from, or `None` to start
                from `|0…0⟩`.
            gate_inputs: The values to pass to the gates of the circuit.
        """
        ...

    def advance(
        self, x_t: np.ndarray, evolved: np.ndarray, state: ReservoirState
    ) -> ReservoirState:
        """Build the state for the next timestep from the result of this step.

        Args:
            x_t (np.ndarray): The input of this timestep, of shape `(B, F)`.
            evolved (np.ndarray): The statevector that the circuit produced.
            state (ReservoirState): The state that went into this step.

        Returns:
            The state for the next timestep.
        """
        ...


@dataclass
class Stateless:
    """Parallel policy that carries no state.

    Every step starts from `|0…0⟩`, and the last `window` inputs are encoded as
    rotation angles of the circuit's gates. The memory is purely classical (the
    window), and the quantum state carries nothing from one step to the next.
    Every timestep is therefore independent, and the driver evaluates all of them
    in one batched run. With `window=1`, the policy has no memory: it encodes only
    the current input.

    The circuit's `n_inputs` must equal `window * F`, where `F` is the number of
    features per step, because each step passes the whole flattened window to the
    encoding gates.

    Attributes:
        window: The number of consecutive inputs to encode at each step. `1` means
            no memory: only the current input is encoded. Whatever the value, the
            circuit's `n_inputs` must equal `window * F`. The default is `1`.
    """

    window: int = 1

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")

    def expand_inputs(self, sequences: np.ndarray) -> np.ndarray:
        """Return the gate inputs of every timestep, with shape `(B, T, window * F)`.

        Step `t` encodes the inputs `t - window + 1` to `t`. The first `window - 1`
        steps are padded with zeros at the front. No quantum state is involved, so this
        is only a reshaping of the input sequence.

        Args:
            sequences (np.ndarray): The whole input batch, of shape `(B, T, F)`.

        Returns:
            The gate inputs for every timestep, of shape `(B, T, window * F)`.
        """
        sequences = np.asarray(sequences, dtype=np.float64)
        B, T, F = sequences.shape
        if self.window == 1:
            return sequences
        pad = np.zeros((B, self.window - 1, F), dtype=np.float64)
        padded = np.concatenate([pad, sequences], axis=1)  # (B, T+window-1, F)
        windows = [padded[:, t : t + self.window, :].reshape(B, -1) for t in range(T)]
        return np.stack(windows, axis=1)  # (B, T, window*F)


@dataclass
class Carry:
    """Policy that applies the circuit to the carried state: `ψ_t = U(x_t) ψ_{t-1}`.

    The quantum state is the memory of the reservoir, and the first step starts
    from `|0…0⟩`. This policy is equivalent to `LeakyCarry(alpha=1.0)`.
    """

    def initial(self, batch: int) -> ReservoirState:
        """Return the state carried into the first timestep.

        Args:
            batch (int): The number of sequences that run in parallel.

        Returns:
            The initial `ReservoirState`.
        """
        return ReservoirState(vector=None)

    def inject(
        self, x_t: np.ndarray, state: ReservoirState
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Turn the input of this timestep into the arguments of the circuit call.

        Args:
            x_t (np.ndarray): The input of this timestep, of shape `(B, F)`.
            state (ReservoirState): The state carried over from the previous step.

        Returns:
            initial_state: The state to evaluate the circuit from, or `None` to start
                from `|0…0⟩`.
            gate_inputs: The values to pass to the gates of the circuit.
        """
        return state.vector, x_t

    def advance(
        self, x_t: np.ndarray, evolved: np.ndarray, state: ReservoirState
    ) -> ReservoirState:
        """Build the state for the next timestep from the result of this step.

        Args:
            x_t (np.ndarray): The input of this timestep, of shape `(B, F)`.
            evolved (np.ndarray): The statevector that the circuit produced.
            state (ReservoirState): The state that went into this step.

        Returns:
            The state for the next timestep.
        """
        return ReservoirState(vector=evolved)


@dataclass
class LeakyCarry:
    """Policy that mixes the carried state with a reset state and renormalises it.

    It approximates a leaky, dissipative reservoir with pure states:
    `ψ_t = normalise(α U(x_t) ψ_{t-1} + (1 − α) reset)`.

    `reset` defaults to `|0…0⟩` and should be a normalised statevector; this is not
    checked. With `alpha=1`, the policy is exactly
    [`Carry`][eigenkit.reservoir.policies.Carry]. This is not a true dissipative
    channel, which would need density matrices that the pure-state backends do not
    provide. It is an approximation that uses pure states only.

    Attributes:
        alpha: The mixing weight of the carried state, in `[0, 1]`. Lower values
            forget faster, and `1` keeps everything, which is exactly `Carry`.
        reset: The state that is mixed in, a normalised statevector. `None` means
            `|0…0⟩`. The default is `None`.
    """

    alpha: float
    reset: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

    def initial(self, batch: int) -> ReservoirState:
        """Return the state carried into the first timestep.

        Args:
            batch (int): The number of sequences that run in parallel.

        Returns:
            The initial `ReservoirState`.
        """
        return ReservoirState(vector=None)

    def inject(
        self, x_t: np.ndarray, state: ReservoirState
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Turn the input of this timestep into the arguments of the circuit call.

        Args:
            x_t (np.ndarray): The input of this timestep, of shape `(B, F)`.
            state (ReservoirState): The state carried over from the previous step.

        Returns:
            initial_state: The state to evaluate the circuit from, or `None` to start
                from `|0…0⟩`.
            gate_inputs: The values to pass to the gates of the circuit.
        """
        return state.vector, x_t

    def advance(
        self, x_t: np.ndarray, evolved: np.ndarray, state: ReservoirState
    ) -> ReservoirState:
        """Build the state for the next timestep from the result of this step.

        Args:
            x_t (np.ndarray): The input of this timestep, of shape `(B, F)`.
            evolved (np.ndarray): The statevector that the circuit produced.
            state (ReservoirState): The state that went into this step.

        Returns:
            The state for the next timestep.
        """
        evolved = np.asarray(evolved, dtype=np.complex128)
        if self.reset is None:
            reset = np.zeros(evolved.shape[1], dtype=np.complex128)
            reset[0] = 1.0
        else:
            reset = np.asarray(self.reset, dtype=np.complex128).reshape(-1)
        mixed = self.alpha * evolved + (1.0 - self.alpha) * reset[None, :]
        norms = np.sqrt(np.sum(np.abs(mixed) ** 2, axis=1, keepdims=True))
        # Guard the degenerate case where the mix cancels to ~0.
        norms = np.where(norms < 1e-12, 1.0, norms)
        return ReservoirState(vector=mixed / norms)
