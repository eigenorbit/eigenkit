"""Real-molecule Hamiltonians via PySCF + OpenFermion (the ``chemistry`` extra).

We use OpenFermion + openfermion-pyscf for the *classical* chemistry — molecular
integrals and the fermionic Hamiltonian — and keep the quantum mapping ours:
the returned `Molecule.fermion_operator` is our `FermionOperator`, ready for
`jordan_wigner` (or `bravyi_kitaev`). The constant (nuclear repulsion + core
energy) is included as the identity term, so `jordan_wigner`'s ``offset`` is the
full energy shift and ``min<H> + offset`` is the total molecular energy.

Optional deps are imported lazily; install with
``pip install eigenkit[chemistry]``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .fermion import FermionOperator
from .interop import fermion_operator_from_openfermion


def _require_chemistry_extra() -> None:
    try:
        import openfermion  # noqa: F401
        import openfermionpyscf  # noqa: F401
    except ImportError as exc:
        if sys.platform == "win32":
            # The extra leaves PySCF out on Windows, so naming it would not help.
            raise ImportError(
                "real-molecule support needs PySCF, which does not support Windows: "
                "run it under WSL. The rest of eigenkit.chemistry (fermionic "
                "operators, the mappings, the built-in models, VQE) works here."
            ) from exc
        raise ImportError(
            "real-molecule support requires the chemistry extra: "
            "`pip install eigenkit[chemistry]` (PySCF + OpenFermion)."
        ) from exc


@dataclass
class Molecule:
    """A molecular Hamiltonian in second quantisation.

    `fermion_operator` already includes the constant term (the nuclear repulsion),
    so `jordan_wigner(fermion_operator)` returns it as the offset. `hf_energy` and
    `fci_energy`, when computed, are the reference energies from PySCF, which are
    useful targets for the Variational Quantum Eigensolver (VQE).

    Attributes:
        fermion_operator: The Hamiltonian in second quantisation, including the
            constant term.
        n_qubits: The number of qubits needed after the mapping, one per
            spin-orbital with Jordan-Wigner.
        n_electrons: The number of electrons, which sets the Hartree-Fock reference
            state and the excitations that a unitary coupled-cluster singles and
            doubles (UCCSD) ansatz can use.
        nuclear_repulsion: The constant nuclear term, in Hartree. It is already
            included in `fermion_operator`, and it is kept separately so that it can
            be reported.
        hf_energy: The Hartree-Fock energy from PySCF, in Hartree. A VQE run should
            reach a lower energy.
        fci_energy: The full configuration interaction (FCI) energy, in Hartree,
            which is the exact answer in this basis set. It is `None` when it was
            not computed. The difference from `hf_energy` is the correlation energy
            that a VQE run tries to recover.
    """

    fermion_operator: FermionOperator
    n_qubits: int
    n_electrons: int
    nuclear_repulsion: float
    hf_energy: float
    fci_energy: float | None = None


def molecular_hamiltonian(
    geometry: list[tuple[str, tuple[float, float, float]]],
    *,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    run_fci: bool = False,
) -> Molecule:
    """Build a `Molecule` from a geometry, with PySCF and OpenFermion.

    Args:
        geometry: A list of `(element, (x, y, z))` entries, with the coordinates in
            angstroms.
        basis: The name of the PySCF basis set. The default is the minimal basis
            set `"sto-3g"`.
        charge: The total charge of the molecule.
        multiplicity: The spin multiplicity, `2S + 1`.
        run_fci: Whether to also compute the FCI energy, which is exact in this
            basis set and a useful target for VQE. It needs an exact
            diagonalisation, so it is only possible for small systems.

    Returns:
        The `Molecule`, with its Hamiltonian and reference energies.
    """
    _require_chemistry_extra()
    from openfermion import MolecularData, get_fermion_operator
    from openfermionpyscf import run_pyscf

    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=run_fci)
    fermion = fermion_operator_from_openfermion(
        get_fermion_operator(molecule.get_molecular_hamiltonian())
    )
    return Molecule(
        fermion_operator=fermion,
        n_qubits=molecule.n_qubits,
        n_electrons=molecule.n_electrons,
        nuclear_repulsion=float(molecule.nuclear_repulsion),
        hf_energy=float(molecule.hf_energy),
        fci_energy=float(molecule.fci_energy) if run_fci else None,
    )


def hydrogen_molecule(
    bond_length: float = 0.7414, *, basis: str = "sto-3g", run_fci: bool = False
) -> Molecule:
    """The hydrogen molecule `H₂` at the given bond length, in angstroms.

    In the minimal STO-3G basis set, this is the standard 4-qubit benchmark for VQE.
    Its FCI ground-state energy is about `-1.137` Hartree at equilibrium.

    Args:
        bond_length (float): The distance between the nuclei, in angstroms. The
            default, `0.7414`, is close to equilibrium.
        basis (str): The name of the PySCF basis set.
        run_fci (bool): Whether to also compute the exact FCI energy, the target
            against which a VQE run is measured.

    Returns:
        The molecule, with its Hamiltonian and reference energies.
    """
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_length))]
    return molecular_hamiltonian(geometry, basis=basis, run_fci=run_fci)
