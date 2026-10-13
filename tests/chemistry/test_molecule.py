"""Real molecules (H2) via PySCF + OpenFermion, mapped with our Jordan-Wigner.

Skipped unless the chemistry extra (PySCF + OpenFermion + openfermion-pyscf) is
installed; runs in CI where it is. These validate the whole real-molecule path
against PySCF's own FCI energy — no hardcoded constants.
"""

from __future__ import annotations

import pytest


pytest.importorskip("openfermionpyscf")  # implies openfermion + pyscf

from eigenkit.chemistry import (  # noqa: E402
    VQE,
    bravyi_kitaev,
    exact_ground_state_energy,
    hydrogen_molecule,
    jordan_wigner,
    uccsd_ansatz,
)


def test_h2_metadata():
    molecule = hydrogen_molecule()
    assert molecule.n_qubits == 4  # STO-3G: 2 spatial -> 4 spin-orbitals
    assert molecule.n_electrons == 2


def test_h2_jordan_wigner_ground_matches_pyscf_fci():
    # Our JW Hamiltonian's exact ground energy must match PySCF's FCI energy
    # (and be the canonical ~ -1.137 Ha).
    molecule = hydrogen_molecule(run_fci=True)
    observable, offset = jordan_wigner(molecule.fermion_operator)
    ground = exact_ground_state_energy(observable, molecule.n_qubits, offset)
    assert ground == pytest.approx(molecule.fci_energy, abs=1e-6)
    assert ground == pytest.approx(-1.137, abs=1e-2)


def test_h2_vqe_uccsd_reaches_fci():
    molecule = hydrogen_molecule(run_fci=True)
    observable, offset = jordan_wigner(molecule.fermion_operator)
    ansatz = uccsd_ansatz(molecule.n_qubits, molecule.n_electrons, reps=2)
    result = VQE(restarts=6, random_state=0).solve(
        observable, molecule.n_qubits, offset=offset, ansatz=ansatz
    )
    # Chemical accuracy (1.6 mHa) against PySCF FCI.
    assert result.energy == pytest.approx(molecule.fci_energy, abs=1.6e-3)


def test_bravyi_kitaev_same_spectrum_as_jordan_wigner():
    # Different encodings of the same operator -> identical ground energy.
    molecule = hydrogen_molecule(run_fci=True)
    jw_observable, jw_offset = jordan_wigner(molecule.fermion_operator)
    bk_observable, bk_offset = bravyi_kitaev(
        molecule.fermion_operator, n_qubits=molecule.n_qubits
    )
    jw_ground = exact_ground_state_energy(jw_observable, molecule.n_qubits, jw_offset)
    bk_ground = exact_ground_state_energy(bk_observable, molecule.n_qubits, bk_offset)
    assert jw_ground == pytest.approx(bk_ground, abs=1e-6)
    assert jw_ground == pytest.approx(molecule.fci_energy, abs=1e-6)
