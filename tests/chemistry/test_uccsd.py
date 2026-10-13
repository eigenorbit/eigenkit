"""UCCSD ansatz on the Fermi-Hubbard dimer.

Unlike the hardware-efficient ansatz, UCCSD conserves particle number, so it
targets the half-filling sector directly — including at U > 3t, where the
*global* ground (single electron, -t) is unreachable from the N=2 reference.
"""

from __future__ import annotations

import pytest

from eigenkit.chemistry import (
    VQE,
    FermionOperator,
    fermi_hubbard_dimer,
    fermi_hubbard_dimer_ground_energy,
    jordan_wigner,
    uccsd_ansatz,
    uccsd_excitation_generators,
)
from eigenkit.core.circuit import Circuit


@pytest.fixture(params=["numpy", "pennylane"])
def provider(request):
    if request.param == "pennylane":
        pytest.importorskip("pennylane")
    return request.param


def _number_observable(n_modes: int):
    total = FermionOperator.zero()
    for mode in range(n_modes):
        total = total + FermionOperator.number(mode)
    return jordan_wigner(total)  # (Observable, offset)


class TestUCCSD:
    def test_excitation_and_parameter_count(self):
        # 2 electrons, 4 spin-orbitals: 4 singles (occ {0,1} x virt {2,3}) + 1
        # double -> 5 generators / weights.
        assert len(uccsd_excitation_generators(4, 2)) == 5
        assert uccsd_ansatz(4, 2).n_weights == 5
        assert uccsd_ansatz(4, 2, reps=3).n_weights == 15  # reps scales the weights

    def test_reaches_half_filling_ground(self, provider):
        # Two Trotter reps make S+D exact for two electrons (S+D = FCI there).
        t, u = 1.0, 2.0
        hamiltonian, offset = jordan_wigner(fermi_hubbard_dimer(t, u))
        ansatz = uccsd_ansatz(4, 2, reps=2)
        result = VQE(provider=provider, restarts=6, random_state=0).solve(
            hamiltonian, n_qubits=4, offset=offset, ansatz=ansatz
        )
        assert result.energy == pytest.approx(
            fermi_hubbard_dimer_ground_energy(t, u), abs=1e-3
        )

    def test_conserves_particle_number(self):
        t, u = 1.0, 2.0
        hamiltonian, offset = jordan_wigner(fermi_hubbard_dimer(t, u))
        ansatz = uccsd_ansatz(4, 2)
        result = VQE(provider="numpy", restarts=4, random_state=0).solve(
            hamiltonian, n_qubits=4, offset=offset, ansatz=ansatz
        )
        number_observable, number_offset = _number_observable(4)
        expectation = Circuit(ansatz, number_observable, "numpy", "numpy")
        particle_number = float(expectation(result.optimal_parameters)) + number_offset
        assert particle_number == pytest.approx(2.0, abs=1e-9)

    def test_stays_in_sector_at_large_u(self):
        # U > 3t: the global ground is the single-electron state (energy -t),
        # but UCCSD stays at half-filling and reaches its (higher) ground.
        t, u = 0.5, 4.0
        hamiltonian, offset = jordan_wigner(fermi_hubbard_dimer(t, u))
        ansatz = uccsd_ansatz(4, 2, reps=2)
        result = VQE(provider="numpy", restarts=6, random_state=0).solve(
            hamiltonian, n_qubits=4, offset=offset, ansatz=ansatz
        )
        assert result.energy == pytest.approx(
            fermi_hubbard_dimer_ground_energy(t, u), abs=1e-3
        )
        assert result.energy > -t + 1e-3  # well above the global single-electron -t


class TestElectronCountValidation:
    """An ansatz with no excitations is not an ansatz.

    `n_electrons` of 0, or equal to `n_qubits`, would produce a spec with zero
    weights: VQE would then have nothing to optimise, converge immediately and
    report success on a circuit that cannot represent anything. A negative
    count would do the same, and a count above `n_qubits` would fail later with
    a message about a qubit index being out of range. All four are refused up
    front.
    """

    @pytest.mark.parametrize("n_electrons", [0, 4, -1, 5])
    def test_a_configuration_with_no_excitations_is_rejected(self, n_electrons):
        with pytest.raises(ValueError, match="n_electrons must be between"):
            uccsd_ansatz(n_qubits=4, n_electrons=n_electrons)

    @pytest.mark.parametrize("n_electrons", [1, 2, 3])
    def test_valid_counts_still_build_a_trainable_ansatz(self, n_electrons):
        spec = uccsd_ansatz(n_qubits=4, n_electrons=n_electrons)
        assert spec.n_weights > 0
        assert len(spec.gates) > 0
