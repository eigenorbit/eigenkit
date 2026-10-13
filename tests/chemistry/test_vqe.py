"""VQE on the Fermi-Hubbard dimer, validated against the closed-form ground
energy ``(U - sqrt(U^2 + 16 t^2)) / 2``.

The eigenvalue test is the rigorous correctness check on Jordan-Wigner (it
compares the *mapped* Hamiltonian's exact ground energy to the analytic
fermionic result); the VQE test confirms the solver reaches it on each backend.
"""

from __future__ import annotations

import pytest

from eigenkit.chemistry import (
    VQE,
    exact_ground_state_energy,
    fermi_hubbard_dimer,
    fermi_hubbard_dimer_ground_energy,
    jordan_wigner,
)


@pytest.fixture(params=["numpy", "pennylane"])
def provider(request):
    if request.param == "pennylane":
        pytest.importorskip("pennylane")
    return request.param


@pytest.mark.parametrize("restarts", [0, -1])
def test_vqe_rejects_non_positive_restarts(restarts):
    with pytest.raises(ValueError, match="restarts must be >= 1"):
        VQE(restarts=restarts)


class TestHubbardDimer:
    # Half-filling is the global ground only for U < 3t — these cases satisfy it.
    @pytest.mark.parametrize("t,u", [(1.0, 0.0), (1.0, 2.0), (1.0, 2.5)])
    def test_jw_ground_energy_matches_closed_form(self, t, u):
        observable, offset = jordan_wigner(fermi_hubbard_dimer(t, u))
        energy = exact_ground_state_energy(observable, n_qubits=4, offset=offset)
        assert energy == pytest.approx(fermi_hubbard_dimer_ground_energy(t, u), abs=1e-9)

    def test_large_u_global_ground_is_single_electron(self):
        # At U > 3t the global ground over all particle numbers is the
        # single-electron state (energy -t), not the half-filled singlet. This
        # confirms JW reproduces the correct full-Fock-space spectrum.
        t, u = 0.5, 4.0
        observable, offset = jordan_wigner(fermi_hubbard_dimer(t, u))
        energy = exact_ground_state_energy(observable, n_qubits=4, offset=offset)
        assert energy == pytest.approx(-t, abs=1e-9)
        assert energy < fermi_hubbard_dimer_ground_energy(t, u)

    def test_vqe_reaches_ground_state(self, provider):
        t, u = 1.0, 2.0
        observable, offset = jordan_wigner(fermi_hubbard_dimer(t, u))
        exact = fermi_hubbard_dimer_ground_energy(t, u)
        result = VQE(layers=4, restarts=8, provider=provider, random_state=0).solve(
            observable, n_qubits=4, offset=offset
        )
        assert result.energy == pytest.approx(exact, abs=1e-3)
        assert result.energy >= exact - 1e-9  # variational: never below the true ground


class TestVQEApi:
    def test_custom_ansatz_and_result_fields(self):
        from eigenkit.chemistry import hardware_efficient_ansatz

        observable, offset = jordan_wigner(fermi_hubbard_dimer(1.0, 1.0))
        ansatz = hardware_efficient_ansatz(n_qubits=4, layers=2)
        result = VQE(provider="numpy", restarts=2).solve(
            observable, n_qubits=4, offset=offset, ansatz=ansatz
        )
        assert isinstance(result.energy, float)
        assert result.optimal_parameters.shape == (4 + 2 * 2 * 4,)
        assert result.n_iterations >= 1
