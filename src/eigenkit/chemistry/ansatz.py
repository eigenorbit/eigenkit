"""Ansatz building blocks for chemistry circuits: Hartree-Fock reference-state
preparation and the UCCSD ansatz.
"""

from __future__ import annotations

from itertools import combinations

from ..core.builder import CircuitBuilder
from ..core.ir import CircuitSpec
from .fermion import FermionOperator
from .mappings import jordan_wigner


def apply_hartree_fock(circuit: CircuitBuilder, n_electrons: int) -> CircuitBuilder:
    """Prepare the Hartree-Fock reference state on an existing builder.

    It applies an `X` gate to each of the lowest `n_electrons` spin-orbitals, which
    marks them as occupied (`|1⟩`).

    The convention is the one of the Jordan-Wigner mapping: mode `j` is qubit `j`,
    and the spin-orbitals are ordered so that the lowest `n_electrons` of them are
    occupied in the reference state, for example `|1…1 0…0⟩`.

    Args:
        circuit (CircuitBuilder): The builder to add the gates to.
        n_electrons (int): The number of electrons, which sets the Hartree-Fock
            reference state.

    Returns:
        The same builder, so that calls can be chained.
    """
    for qubit in range(n_electrons):
        circuit.x(qubit)
    return circuit


def hartree_fock_state(n_qubits: int, n_electrons: int) -> CircuitSpec:
    """A circuit that prepares the Hartree-Fock reference state.

    The state is `|1…1 0…0⟩` on `n_qubits` qubits, with `n_electrons` occupied
    spin-orbitals.

    Args:
        n_qubits (int): The number of qubits.
        n_electrons (int): The number of electrons, which sets the Hartree-Fock
            reference state.

    Returns:
        The circuit, as a `CircuitSpec` with no trainable weights.
    """
    if not 0 <= n_electrons <= n_qubits:
        raise ValueError(
            f"n_electrons ({n_electrons}) must be between 0 and n_qubits ({n_qubits})"
        )
    builder = CircuitBuilder(n_qubits=n_qubits)
    apply_hartree_fock(builder, n_electrons)
    return builder.build()


def _single_excitation_generator(occupied: int, virtual: int) -> FermionOperator:
    r"""The anti-Hermitian single-excitation generator \(a_v^\dagger a_o - a_o^\dagger a_v\)."""
    excitation = FermionOperator.creation(virtual) * FermionOperator.annihilation(
        occupied
    )
    return excitation - excitation.hermitian_conjugate()


def _double_excitation_generator(
    occ_i: int, occ_j: int, virt_a: int, virt_b: int
) -> FermionOperator:
    r"""The anti-Hermitian double-excitation generator
    \(a_a^\dagger a_b^\dagger a_j a_i - \text{h.c.}\)."""
    excitation = (
        FermionOperator.creation(virt_a)
        * FermionOperator.creation(virt_b)
        * FermionOperator.annihilation(occ_j)
        * FermionOperator.annihilation(occ_i)
    )
    return excitation - excitation.hermitian_conjugate()


def uccsd_excitation_generators(n_qubits: int, n_electrons: int) -> list[FermionOperator]:
    """The excitation generators `T - T†` of the UCCSD ansatz.

    UCCSD is the unitary coupled-cluster ansatz with single and double excitations.
    The excitations go from the occupied spin-orbitals (the lowest `n_electrons`)
    to the unoccupied ones. Single and double excitations are taken over all choices
    of occupied and unoccupied indices, with no restriction from spin symmetry,
    which keeps the construction simple and correct. A variant restricted by spin
    symmetry, which would have fewer parameters, is not implemented.

    Args:
        n_qubits (int): The number of qubits.
        n_electrons (int): The number of electrons, which sets the Hartree-Fock
            reference state.

    Returns:
        The generators, as a list of `FermionOperator` objects.
    """
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    generators: list[FermionOperator] = []
    for occ in occupied:
        for virt in virtual:
            generators.append(_single_excitation_generator(occ, virt))
    for occ_i, occ_j in combinations(occupied, 2):
        for virt_a, virt_b in combinations(virtual, 2):
            generators.append(_double_excitation_generator(occ_i, occ_j, virt_a, virt_b))
    return generators


def uccsd_ansatz(
    n_qubits: int,
    n_electrons: int,
    reps: int = 1,
) -> CircuitSpec:
    """The UCCSD ansatz on the Hartree-Fock reference, with a Trotter approximation.

    UCCSD is the unitary coupled-cluster ansatz with single and double excitations.
    The circuit prepares `(∏_k exp(θ_k (T_k − T_k†)))^reps |HF⟩`, where `|HF⟩` is
    the Hartree-Fock state.

    Each generator `G = T − T†` is anti-Hermitian. The Jordan-Wigner mapping turns
    `H = −iG` into a Hermitian qubit operator, and `exp(θG) = exp(iθH)` is then a
    product of `pauli_exponential` gates. The Pauli terms of one generator commute,
    so that product is exact. The product over different generators is a
    first-order Trotter approximation. `reps` repeats the whole sequence with
    independent weights, which reduces the Trotter error: more repetitions bring
    the circuit closer to the exact UCCSD unitary. The ansatz has
    `reps * n_excitations` weights in total.

    Args:
        n_qubits (int): The number of qubits, one per spin-orbital.
        n_electrons (int): The number of electrons, which sets the Hartree-Fock
            reference and the possible excitations.
        reps (int): The number of Trotter repetitions of the product of
            excitations. The default is `1`.

    Returns:
        The ansatz as a `CircuitSpec`, with one trainable angle per generator and
            per repetition.
    """
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    if not 0 < n_electrons < n_qubits:
        # Outside this range there are no excitations, and the "ansatz" would
        # come back with zero weights: VQE would then have nothing to optimise,
        # converge immediately and report success on a circuit that cannot
        # represent anything. n_electrons > n_qubits would fail later and less
        # clearly, with a message about a qubit index being out of range.
        raise ValueError(
            f"n_electrons must be between 1 and n_qubits - 1 for UCCSD to have "
            f"any excitations; got n_electrons={n_electrons} with "
            f"n_qubits={n_qubits}. Zero electrons and a fully occupied shell "
            f"both leave nothing to excite."
        )
    generators = uccsd_excitation_generators(n_qubits, n_electrons)
    # -1j * G is Hermitian; cache its real Pauli terms once.
    hermitian_terms = [jordan_wigner(-1j * g)[0].terms for g in generators]
    builder = CircuitBuilder(n_qubits=n_qubits)
    thetas = builder.add_weights(len(generators) * reps)
    apply_hartree_fock(builder, n_electrons)
    index = 0
    for _ in range(reps):
        for terms in hermitian_terms:
            theta = thetas[index]
            index += 1
            for term in terms:
                # exp(i theta c_k P_k) = pauli_exponential(P_k, theta, -c_k).
                builder.pauli_exponential(term.paulis, theta, coefficient=-term.coeff)
    return builder.build()
