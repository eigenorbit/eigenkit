r"""Feature extractors: map a ``(B, T, 2**n)`` statevector trajectory to real
features for a classical readout.

The headline extractor is
[`expectation_features()`][eigenkit.reservoir.features.expectation_features],
which measures arbitrary Pauli observables / Hamiltonians given as the library's
``Observable`` / ``PauliExpr`` types or a Hamiltonian string — so you get
\(\langle\psi|H|\psi\rangle\) features without hand-rolling Pauli algebra.
[`probability_features()`][eigenkit.reservoir.features.probability_features]
and [`amplitude_features()`][eigenkit.reservoir.features.amplitude_features]
read out the same trajectory differently.

All extractors are pure numpy and operate on the whole ``(B, T)`` grid at once.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from ..core.ir import Observable


# Single-qubit Pauli matrices (qubit-0-MSB flat order, matching the backends).
_PAULI = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}

ObservableLike = Union[Observable, "object", str, Sequence]


def _normalize_observables(observables: ObservableLike) -> list[Observable]:
    """Coerce the flexible observable spec into a list of ``Observable``.

    Accepts an ``Observable``, a ``PauliExpr``, a Hamiltonian string (e.g.
    ``"0.5 Z0 X1 - 0.3 Y2"``), or a sequence mixing those.
    """
    from ..core.observables import PauliExpr, from_string

    def one(o):
        if isinstance(o, str):
            return from_string(o)
        if isinstance(o, PauliExpr):
            return o.to_observable()
        if isinstance(o, Observable):
            return o
        raise TypeError(f"not an observable: {type(o)}")

    if isinstance(observables, (str, Observable, PauliExpr)):
        return [one(observables)]
    return [one(o) for o in observables]


def _n_qubits(dim: int) -> int:
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"trailing statevector dim {dim} is not a power of 2")
    return n


def _apply_1q(tensor: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    """Apply a 2x2 ``matrix`` to ``axis`` of a ``(M, 2, …, 2)`` batch tensor."""
    out = np.tensordot(matrix, tensor, axes=([1], [axis]))
    return np.moveaxis(out, 0, axis)


def expectation_features(states: np.ndarray, observables: ObservableLike) -> np.ndarray:
    """Expectation values `⟨ψ|H_k|ψ⟩` of each observable, over the whole trajectory.

    `states` is a complex array of shape `(B, T, 2**n)`, and the result is a real
    array of shape `(B, T, n_obs)`. `observables` can be an `Observable`, a
    `PauliExpr` or a Hamiltonian string, or a sequence of these. These are the same
    types that the rest of the library uses, so measuring an arbitrary Hamiltonian,
    as QRC experiments often require, needs no manual handling of Pauli terms. A
    single Hamiltonian string is one observable, and gives one feature.

    Args:
        states (np.ndarray): The trajectory of statevectors, of shape
            `(B, T, 2**n)`.
        observables (Sequence[Observable]): The observables to measure.

    Returns:
        One feature per observable and per timestep.
    """
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim != 3:
        raise ValueError(f"states must be 3-D (B, T, 2**n), got ndim={states.ndim}")
    B, T, dim = states.shape
    n = _n_qubits(dim)
    obs_list = _normalize_observables(observables)

    psi = states.reshape(B * T, *([2] * n))
    conj = np.conjugate(psi)
    columns = []
    for obs in obs_list:
        total = np.zeros(B * T, dtype=np.float64)
        for term in obs.terms:
            acted = psi
            for op, qubit in term.paulis:
                acted = _apply_1q(acted, _PAULI[op], qubit + 1)  # axis 0 is the sample
            inner = np.sum((conj * acted).reshape(B * T, -1), axis=1).real
            total += float(term.coeff) * inner
        columns.append(total)
    return np.stack(columns, axis=1).reshape(B, T, len(obs_list))


def probability_features(states: np.ndarray) -> np.ndarray:
    """The measurement distribution over all `2**n` basis states, `|ψ|²`.

    `states` is a complex array of shape `(B, T, 2**n)`, and the result is a real
    array of the same shape.

    Args:
        states (np.ndarray): The trajectory of statevectors, of shape
            `(B, T, 2**n)`.

    Returns:
        The distribution over the basis states, for each timestep.
    """
    states = np.asarray(states, dtype=np.complex128)
    return np.abs(states) ** 2


def pauli_z_features(states: np.ndarray) -> np.ndarray:
    """The expectation value `⟨Z_q⟩` on each qubit, the usual default readout in QRC.

    `states` has shape `(B, T, 2**n)`, and the result has shape `(B, T, n)`. The
    values are computed with
    [`expectation_features()`][eigenkit.reservoir.features.expectation_features],
    so both functions share the same tested measurement code.

    Args:
        states (np.ndarray): The trajectory of statevectors, of shape
            `(B, T, 2**n)`.

    Returns:
        One feature per qubit and per timestep.
    """
    from ..core.observables import all_z

    dim = np.asarray(states).shape[-1]
    return expectation_features(states, all_z(_n_qubits(dim)))


def amplitude_features(states: np.ndarray) -> np.ndarray:
    """The raw statevector as real features: its real and imaginary parts, concatenated.

    `states` is a complex array of shape `(B, T, 2**n)`, and the result is a real
    array of shape `(B, T, 2 * 2**n)`. It exposes the full state, which only a
    simulator can provide; it cannot be measured on hardware.

    Args:
        states (np.ndarray): The trajectory of statevectors, of shape
            `(B, T, 2**n)`.

    Returns:
        The real and imaginary parts, concatenated, for each timestep.
    """
    states = np.asarray(states, dtype=np.complex128)
    return np.concatenate([states.real, states.imag], axis=-1)
