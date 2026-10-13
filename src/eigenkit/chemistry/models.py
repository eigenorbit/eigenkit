"""Built-in fermionic models with known properties — for validation and demos
without any external chemistry dependency.
"""

from __future__ import annotations

import math

from .fermion import FermionOperator


def fermi_hubbard_dimer(t: float, u: float) -> FermionOperator:
    """The two-site Fermi-Hubbard model with spin `1/2`.

    It has four spin-orbital modes, indexed by `mode = 2 * site + spin`, with spin
    `0` for up and `1` for down. Site `0` has modes `0` and `1`, and site `1` has
    modes `2` and `3`. The Hamiltonian is:

    ```text
    H = -t Σ_σ (a†_{0σ} a_{1σ} + a†_{1σ} a_{0σ})
        + U (n_{0↑} n_{0↓} + n_{1↑} n_{1↓})
    ```

    The ground-state energy at half filling has the closed form
    [`fermi_hubbard_dimer_ground_energy()`][eigenkit.chemistry.models.fermi_hubbard_dimer_ground_energy].
    This makes the model an exact check of the fermion-to-qubit mapping and of
    VQE, with no external dependency.

    Args:
        t (float): The hopping amplitude between the two sites.
        u (float): The on-site Coulomb repulsion.

    Returns:
        The Hamiltonian, as a `FermionOperator`.
    """
    hamiltonian = FermionOperator.zero()
    for spin in (0, 1):
        site0, site1 = spin, 2 + spin  # 2*0+spin, 2*1+spin
        hop = FermionOperator.creation(site0) * FermionOperator.annihilation(site1)
        hop = hop + FermionOperator.creation(site1) * FermionOperator.annihilation(site0)
        hamiltonian = hamiltonian + (-t) * hop
    for site in (0, 1):
        up, down = 2 * site, 2 * site + 1
        hamiltonian = hamiltonian + u * (
            FermionOperator.number(up) * FermionOperator.number(down)
        )
    return hamiltonian


def fermi_hubbard_dimer_ground_energy(t: float, u: float) -> float:
    """The ground-state energy of the Fermi-Hubbard dimer at half filling.

    The energy of [`fermi_hubbard_dimer()`][eigenkit.chemistry.models.fermi_hubbard_dimer]
    with two electrons in a singlet state is `(U - sqrt(U² + 16t²)) / 2`. This value
    is the ground state over all particle numbers only when `U < 3t`. For a larger
    `U`, double occupancy costs so much energy that the state with a single
    electron, of energy `-t`, is lower, and the ground state over all particle
    numbers is `min(-t, (U - sqrt(U² + 16t²)) / 2)`. An ansatz that conserves the
    number of particles, such as UCCSD, targets the half-filling sector directly.
    The hardware-efficient ansatz explores all sectors, so use it only where the
    half-filling state is the overall ground state.

    Args:
        t (float): The hopping amplitude, as in the model.
        u (float): The on-site repulsion, as in the model.

    Returns:
        The ground-state energy at half filling, from the closed form.
    """
    return (u - math.sqrt(u**2 + 16.0 * t**2)) / 2.0
