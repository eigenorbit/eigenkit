"""Exact classical reference: dense matrix and ground-state energy of a qubit
`Observable`. For validating VQE on small systems (exponential cost — keep
``n_qubits`` small).
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from ..core.ir import Observable


_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def dense_hamiltonian(
    observable: Observable, n_qubits: int, offset: float = 0.0
) -> np.ndarray:
    """The `2**n_qubits × 2**n_qubits` matrix of `observable + offset * I`.

    Qubit 0 is the most significant factor of the tensor product, which matches the
    ordering of basis states in the simulators.

    Args:
        observable (Observable): The qubit operator.
        n_qubits (int): The number of qubits.
        offset (float): The constant (identity) term that the mapping returned
            separately, added back to the energy. For a molecule, it includes the
            nuclear repulsion, in Hartree.

    Returns:
        The matrix, as a complex NumPy array.
    """
    dim = 2**n_qubits
    matrix = offset * np.eye(dim, dtype=np.complex128)
    for term in observable.terms:
        ops = ["I"] * n_qubits
        for pauli, qubit in term.paulis:
            ops[qubit] = pauli
        factor = reduce(np.kron, [_PAULI[op] for op in ops])
        matrix = matrix + term.coeff * factor
    return matrix


def exact_ground_state_energy(
    observable: Observable, n_qubits: int, offset: float = 0.0
) -> float:
    """The lowest eigenvalue of `observable + offset * I`, by dense diagonalisation.

    Args:
        observable (Observable): The qubit operator.
        n_qubits (int): The number of qubits.
        offset (float): The constant (identity) term that the mapping returned
            separately, added back to the energy. For a molecule, it includes the
            nuclear repulsion, in Hartree.

    Returns:
        The ground-state energy.
    """
    matrix = dense_hamiltonian(observable, n_qubits, offset)
    return float(np.linalg.eigvalsh(matrix)[0])
