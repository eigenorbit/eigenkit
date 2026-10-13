# VQE: the variational quantum eigensolver

## The problem

Chemistry asks for the lowest energy that a molecule can have, which is the energy of
its ground state. This energy is the smallest eigenvalue of a Hamiltonian. The matrix
of this Hamiltonian grows as `2ⁿ` with the number `n` of spin-orbitals, so exact
diagonalisation runs out of memory quickly.

The Variational Quantum Eigensolver (VQE) replaces this diagonalisation with a search.
A circuit prepares a trial state, the quantum side measures its energy, a classical
optimiser adjusts the angles of the circuit, and the loop repeats. The variational
principle guarantees that the measured energy can never fall below the true energy of
the ground state, so a lower number is always a better answer. The method is due to
[Peruzzo et al.](https://arxiv.org/abs/1304.3061), and the hybrid framework was set
out by [McClean et al.](https://arxiv.org/abs/1509.04279).

## Input and output

`VQE(...).solve(hamiltonian, n_qubits, offset=..., ansatz=...)` takes a qubit
`Observable`, and returns `energy`, the tuned `optimal_parameters`, `n_iterations`
and `success`.

Going from a molecule to this `Observable` takes two steps:

```python
from eigenkit.chemistry import (
    VQE,
    exact_ground_state_energy,
    fermi_hubbard_dimer,
    jordan_wigner,
)

hamiltonian = fermi_hubbard_dimer(t=1.0, u=2.0)  # 6 fermionic terms
observable, offset = jordan_wigner(hamiltonian)  # 10 Pauli terms, offset 1.0

result = VQE(layers=3, restarts=5).solve(observable, n_qubits=4, offset=offset)
print(result.energy)  # -1.236068
print(exact_ground_state_energy(observable, 4, offset=offset))  # -1.236068
```

`offset` is the constant term. The Pauli backends reject a term that holds only the
identity, so the mapping returns the constant separately, and the solver adds it to
`⟨H⟩` at the end. If you forget to pass it, every energy is wrong by this amount.

## How it works

### Step 1: a fermionic Hamiltonian

Electrons are fermions: when you swap two of them, the sign of the wavefunction
changes. A `FermionOperator` stores the Hamiltonian in terms of creation and
annihilation operators, which follow this rule.

### Step 2: map fermions onto qubits

Qubits do not antisymmetrise on their own, so the mapping must impose this rule.
**Jordan-Wigner** does it with a string of `Z` operators:

```text
a_j  = (Π_{k<j} Z_k) (X_j + iY_j)/2
a_j† = (Π_{k<j} Z_k) (X_j - iY_j)/2
```

There is one qubit per spin-orbital, and `|1⟩` means that the spin-orbital is
occupied, so the number operator becomes `(I - Z_j)/2`. The parity string
`Π_{k<j} Z_k` is what gives the change of sign, and it is also the cost: an operator
that acts on orbital `j` touches every qubit below it, so the Pauli strings become
long in large molecules.

**Bravyi-Kitaev** ([Bravyi and Kitaev](https://arxiv.org/abs/quant-ph/0003137),
applied to electronic structure by
[Seeley, Richard and Love](https://arxiv.org/abs/1208.5986)) stores the occupations
and the parities in a balanced tree instead, which shortens these strings to
`O(log n)`. In this library, `jordan_wigner` is written in the library itself, while
`bravyi_kitaev` calls OpenFermion, and therefore needs the `chemistry` extra.

### Step 3: choose an ansatz

The ansatz is the trial circuit whose angles the optimiser tunes. The library
provides two, which balance physical accuracy against the depth of the circuit.

**Hardware-efficient.** This is the default. It has layers of rotations and
entangling gates, chosen because a device can run them, not because chemistry
suggests them ([Kandala et al.](https://arxiv.org/abs/1704.05018)).
`hardware_efficient_ansatz(n_qubits=4, layers=3)` has `37` gates and `28`
parameters.

**UCCSD.** Unitary coupled cluster with single and double excitations (UCCSD) is the
quantum version of the classical chemistry method
([Romero et al.](https://arxiv.org/abs/1701.02691)). It starts from the
**Hartree-Fock** reference state, which is the simplest guess of the configuration of
the electrons: it fills the lowest `n_electrons` orbitals and nothing else. This state
is one `X` gate per filled orbital. `hartree_fock_state(n_qubits=4, n_electrons=2)`
is exactly two `X` gates and no parameters, and `uccsd_ansatz` adds it at the start
for you. On top of this state, UCCSD applies parametrised excitations that move
electrons out of the filled orbitals. `uccsd_ansatz(n_qubits=4, n_electrons=2,
reps=1)` has `194` gates but only `5` parameters: it is much deeper than the
hardware-efficient circuit, and much cheaper to optimise.

### Step 4: optimise

The loop is the variational loop of the library, which uses NumPy: it evaluates
`⟨H⟩`, takes the gradient with the backend's own method, lets SciPy take a step, and
repeats. `restarts` runs the loop from several random starting points and keeps the
best result. This is how the solver avoids stopping at the nearest local minimum
instead of the global one.

## Guarantees and limits

The variational principle is the one strict guarantee: the answer is an upper bound on
the true energy of the ground state. Everything else is empirical.

An upper bound is not necessarily a tight bound, and a loose bound looks exactly like
a good result: the optimiser converges, `success` is `True`, and the energy is simply
too high. The usual cause is an ansatz that is too shallow, and such an ansatz can
settle on a true eigenvalue rather than on an approximation of the ground state. On
the dimer above, a single repetition of UCCSD converges to the *first excited level*,
and not to an approximation of the ground state. Compare against
`exact_ground_state_energy` while the system is small enough to allow it (dense
diagonalisation becomes impractical at about `14` qubits), and raise `reps` or
`layers` when the gap is large.

There are other limits. Deep circuits with many parameters run into
[barren plateaus](https://arxiv.org/abs/1803.11173), where the gradients vanish as the
circuit grows, so results on a few qubits say little about results on dozens of
qubits. Real molecules (`molecular_hamiltonian`, `hydrogen_molecule`) need PySCF from
the `chemistry` extra, while the model Hamiltonians and Jordan-Wigner run on a basic
install.

## Next

The interface of the solver is in the [chemistry reference](../api/chemistry/vqe.md),
and the [Chemistry guide](../user-guide/chemistry.md) covers the whole pipeline with
working code. The papers are listed in
[References](references.md#vqe-the-variational-quantum-eigensolver).
