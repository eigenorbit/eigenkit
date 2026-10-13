"""Quantum chemistry with the core of the library.

Build a fermionic Hamiltonian (`FermionOperator`), map it to a qubit
`Observable` (`jordan_wigner`), and find its ground-state energy with the
Variational Quantum Eigensolver (`VQE`), which uses the variational optimisation
tools of the core. The package uses only NumPy and SciPy, needs no PyTorch, and
works with any backend. For small systems, `exact_ground_state_energy` gives
the classical reference value.

Example:
    ```python
    from eigenkit.chemistry import (
        fermi_hubbard_dimer, jordan_wigner, VQE, exact_ground_state_energy,
    )

    hamiltonian, offset = jordan_wigner(fermi_hubbard_dimer(t=1.0, u=2.0))
    result = VQE(layers=3, restarts=5).solve(hamiltonian, n_qubits=4, offset=offset)
    exact = exact_ground_state_energy(hamiltonian, n_qubits=4, offset=offset)
    print(result.energy, exact)
    ```
"""

from .ansatz import (
    apply_hartree_fock,
    hartree_fock_state,
    uccsd_ansatz,
    uccsd_excitation_generators,
)
from .exact import dense_hamiltonian, exact_ground_state_energy
from .fermion import FermionOperator
from .interop import (
    fermion_operator_from_openfermion,
    observable_from_openfermion,
    to_openfermion_fermion_operator,
)
from .mappings import bravyi_kitaev, jordan_wigner
from .models import fermi_hubbard_dimer, fermi_hubbard_dimer_ground_energy
from .molecule import Molecule, hydrogen_molecule, molecular_hamiltonian
from .vqe import VQE, VQEResult, hardware_efficient_ansatz


__all__ = [
    "FermionOperator",
    "jordan_wigner",
    "bravyi_kitaev",
    "fermi_hubbard_dimer",
    "fermi_hubbard_dimer_ground_energy",
    "Molecule",
    "molecular_hamiltonian",
    "hydrogen_molecule",
    "fermion_operator_from_openfermion",
    "observable_from_openfermion",
    "to_openfermion_fermion_operator",
    "VQE",
    "VQEResult",
    "hardware_efficient_ansatz",
    "apply_hartree_fock",
    "hartree_fock_state",
    "uccsd_ansatz",
    "uccsd_excitation_generators",
    "exact_ground_state_energy",
    "dense_hamiltonian",
]
