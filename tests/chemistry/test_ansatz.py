"""Hartree-Fock reference-state preparation."""

from __future__ import annotations

import numpy as np

from eigenkit.chemistry import hartree_fock_state
from eigenkit.core.circuit import Circuit
from eigenkit.core.observables import Z, all_z


def test_hartree_fock_prepares_occupied_string():
    # 2 electrons in 4 spin-orbitals -> |1100> (qubits 0,1 occupied).
    spec = hartree_fock_state(n_qubits=4, n_electrons=2)
    circuit = Circuit(spec, all_z(4), "numpy", "numpy")
    probabilities = circuit.probabilities(np.empty(0))
    # |1100> in qubit-0-MSB ordering = index 0b1100 = 12.
    assert np.isclose(probabilities[12], 1.0)


def test_hartree_fock_occupations():
    spec = hartree_fock_state(n_qubits=4, n_electrons=2)
    circuit = Circuit(spec, [Z(q).to_observable() for q in range(4)], "numpy", "numpy")
    out = circuit(np.empty(0))
    # Occupied (qubits 0,1) -> <Z> = -1; empty (2,3) -> <Z> = +1.
    assert np.allclose(out, [-1.0, -1.0, 1.0, 1.0])


def test_invalid_electron_count():
    import pytest

    with pytest.raises(ValueError, match="n_electrons"):
        hartree_fock_state(n_qubits=2, n_electrons=3)
