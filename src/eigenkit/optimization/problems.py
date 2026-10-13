"""Combinatorial problems as Ising and QUBO models, and their qubit Hamiltonians.

QUBO means quadratic unconstrained binary optimisation. The mapping convention is
that the spin `s_i = +1` is the qubit state `|0⟩` and `s_i = -1` is `|1⟩`, which
are the eigenvalues of `Z`. So a bitstring `b`, with qubit values `b_i ∈ {0, 1}`,
corresponds to `s_i = 1 - 2 b_i`. `to_hamiltonian()` returns the pair
`(Observable, offset)`. The Pauli sum holds only the `Z` and `ZZ` terms, and the
constant is kept apart, because the Pauli backends reject terms made only of the
identity. The solver adds `offset` back to `⟨H⟩`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..core.ir import Observable, PauliTerm


def _bits_of(index: int, n: int) -> tuple[int, ...]:
    """Basis-state index -> bitstring in qubit-0-MSB order (matching the
    `probabilities` convention)."""
    return tuple((index >> (n - 1 - k)) & 1 for k in range(n))


def _brute_force(energy, n: int) -> tuple[tuple[int, ...], float]:
    best_bits: tuple[int, ...] | None = None
    best_energy = float("inf")
    for index in range(2**n):
        bits = _bits_of(index, n)
        value = energy(bits)
        if value < best_energy:
            best_energy, best_bits = value, bits
    return best_bits, best_energy


@dataclass
class Ising:
    """An Ising problem, which minimises `Σ h_i s_i + Σ_{i<j} J_ij s_i s_j + constant`.

    The minimum is taken over `s ∈ {+1, -1}^n`. Only the strict upper triangle of `J`
    (`i < j`) is used.

    Attributes:
        h: The linear field, of shape `(n,)`, with one bias per spin.
        J: The coupling matrix, of shape `(n, n)`. Only the strict upper triangle is
            read, so a symmetric matrix and its upper half give the same problem, and
            the entries on or below the diagonal are ignored.
        constant: An energy offset added to every configuration. It shifts the
            reported energy without changing which configuration wins. Default:
            `0.0`.
    """

    h: np.ndarray
    J: np.ndarray
    constant: float = 0.0

    @property
    def num_vars(self) -> int:
        """The number of spins, read from the length of `h`."""
        return len(self.h)

    def to_hamiltonian(self) -> tuple[Observable, float]:
        """Return the qubit Hamiltonian of the problem, with the constant kept apart.

        The Hamiltonian is `Σ h_i Z_i + Σ_{i<j} J_ij Z_i Z_j`.

        Returns:
            observable: The Hamiltonian, as an `Observable`.
            offset: The constant of the problem.

        Raises:
            ValueError: If every coefficient of the model is `0`.
        """
        terms: list[PauliTerm] = []
        n = self.num_vars
        for i in range(n):
            if self.h[i] != 0.0:
                terms.append(PauliTerm(coeff=float(self.h[i]), paulis=(("Z", i),)))
        for i in range(n):
            for j in range(i + 1, n):
                if self.J[i, j] != 0.0:
                    terms.append(
                        PauliTerm(coeff=float(self.J[i, j]), paulis=(("Z", i), ("Z", j)))
                    )
        if not terms:
            raise ValueError("Ising model has no non-zero terms")
        return Observable(terms=tuple(terms)), float(self.constant)

    def energy(self, bits: Iterable[int]) -> float:
        """Return the energy of one assignment.

        The bits are converted to spins with `s = 1 - 2b`, so a `0` is the spin `+1` and
        a `1` is the spin `-1`.

        Args:
            bits (Iterable[int]): One bit per variable, in the order of the variables.

        Returns:
            The energy of the assignment for this problem.
        """
        s = 1 - 2 * np.asarray(list(bits), dtype=np.float64)
        # Strict upper triangle only, which is what the class documents and
        # what `to_hamiltonian` emits. `s @ J @ s` reads the whole matrix, so a
        # symmetric J -- explicitly allowed above -- would count every pair
        # twice and this method would report double the energy of the operator
        # QAOA actually minimises, while a non-zero diagonal would add a
        # constant the Hamiltonian never sees.
        coupling = float(s @ np.triu(self.J, k=1) @ s)
        return float(self.h @ s) + coupling + self.constant

    def brute_force(self) -> tuple[tuple[int, ...], float]:
        """Find the exact minimum by trying every assignment.

        This is practical only for a small `n`.

        Returns:
            bitstring: The assignment with the lowest energy.
            energy: Its energy.
        """
        return _brute_force(self.energy, self.num_vars)


@dataclass
class QUBO:
    """A quadratic unconstrained binary optimisation (QUBO) problem.

    It minimises `xᵀQx + constant` over `x ∈ {0, 1}^n`.

    Attributes:
        Q: The cost matrix, of shape `(n, n)`. The diagonal entries are the linear
            terms, because `x_i² = x_i` for binary variables.
        constant: An energy offset added to every configuration. Default: `0.0`.
    """

    Q: np.ndarray
    constant: float = 0.0

    @property
    def num_vars(self) -> int:
        """The number of binary variables, read from the shape of `Q`."""
        return self.Q.shape[0]

    def to_ising(self) -> Ising:
        """Convert the problem exactly to an `Ising` model, with `x_i = (1 - s_i) / 2`.

        Returns:
            The equivalent `Ising` model.
        """
        Q = np.asarray(self.Q, dtype=np.float64)
        n = Q.shape[0]
        row = Q.sum(axis=1)
        col = Q.sum(axis=0)
        h = -(row + col) / 4.0
        J = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] = (Q[i, j] + Q[j, i]) / 4.0
        constant = self.constant + (float(Q.sum()) + float(np.trace(Q))) / 4.0
        return Ising(h=h, J=J, constant=constant)

    def to_hamiltonian(self) -> tuple[Observable, float]:
        """Return the qubit Hamiltonian of the problem, through `to_ising()`.

        Returns:
            observable: The Hamiltonian, as an `Observable`.
            offset: The constant of the problem.
        """
        return self.to_ising().to_hamiltonian()

    def energy(self, bits: Iterable[int]) -> float:
        """Return the energy of one assignment.

        Args:
            bits (Iterable[int]): One bit per variable, in the order of the variables.

        Returns:
            The energy of the assignment for this problem.
        """
        x = np.asarray(list(bits), dtype=np.float64)
        return float(x @ self.Q @ x) + self.constant

    def brute_force(self) -> tuple[tuple[int, ...], float]:
        """Find the exact minimum by trying every assignment.

        This is practical only for a small `n`.

        Returns:
            bitstring: The assignment with the lowest energy.
            energy: Its energy.
        """
        return _brute_force(self.energy, self.num_vars)


def max_cut(
    edges: Iterable[tuple[int, int]],
    weights: Iterable[float] | None = None,
    num_nodes: int | None = None,
) -> Ising:
    """Build the Ising model of the MaxCut problem for a weighted graph.

    The problem is written as an energy to minimise. The spin assignment with the
    lowest energy is a maximum cut, and the cut value of a bitstring is
    `-ising.energy(bits)`.

    Args:
        edges (Sequence[tuple[int, int]]): The edges of the graph, as pairs `(i, j)`.
        weights (Sequence[float] | None): One weight per edge. Default: `None`, which
            gives every edge the weight `1`.
        num_nodes (int | None): The number of nodes. Default: `None`, which takes it
            from the edges.

    Returns:
        The Ising problem whose minimum is a maximum cut.
    """
    edge_list = [tuple(e) for e in edges]
    if num_nodes is None:
        num_nodes = max((max(i, j) for i, j in edge_list), default=-1) + 1
    weight_list = (
        [1.0] * len(edge_list) if weights is None else [float(w) for w in weights]
    )

    h = np.zeros(num_nodes, dtype=np.float64)
    J = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    total_weight = 0.0
    for (i, j), w in zip(edge_list, weight_list):
        a, b = (i, j) if i < j else (j, i)
        J[a, b] += w / 2.0
        total_weight += w
    return Ising(h=h, J=J, constant=-total_weight / 2.0)
