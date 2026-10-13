# Ground-State Energy with VQE

This example finds the ground-state energy of a small model with the variational
quantum eigensolver (VQE). The model is the Fermi-Hubbard dimer: two sites, two
electrons, a hopping term `t` and an on-site repulsion `u`. Its ground-state energy
is known in closed form, so the result can be checked exactly.

The example runs on `provider="numpy"`, which is the default of `VQE`, and it uses
only the fermionic tools built into the library, so it does not need `openfermion`
or `pyscf`. The whole page runs in under two seconds. See the
[VQE chapter](../algorithms/vqe.md) for the algorithm, and the
[Quantum Chemistry guide](../user-guide/chemistry.md) for the interface.

## The model

```python
import numpy as np
from eigenkit.chemistry import (
    VQE,
    exact_ground_state_energy,
    fermi_hubbard_dimer,
    fermi_hubbard_dimer_ground_energy,
    jordan_wigner,
    uccsd_ansatz,
)

T, U, N_QUBITS = 1.0, 2.0, 4

fermionic = fermi_hubbard_dimer(t=T, u=U)
analytic = fermi_hubbard_dimer_ground_energy(t=T, u=U)

print("operator type  :", type(fermionic).__name__)
print("analytic ground:", analytic)
```

```text
operator type  : FermionOperator
analytic ground: -1.2360679774997898
```

The model has four spin-orbitals (two sites, each with two spins), so it needs
four qubits after the mapping.

## Mapping fermions to qubits

`jordan_wigner()` returns two values: the qubit `Observable`, and a scalar offset
that holds the identity part of the Hamiltonian.

```python
hamiltonian, offset = jordan_wigner(fermionic)

print("pauli terms:", len(hamiltonian.terms))
print("offset     :", offset)
for term in hamiltonian.terms[:4]:
    print("   ", term.coeff, term.paulis)
```

```text
pauli terms: 10
offset     : 1.0
    -0.5 (('Y', 0), ('Z', 1), ('Y', 2))
    -0.5 (('X', 0), ('Z', 1), ('X', 2))
    -0.5 (('Y', 1), ('Z', 2), ('Y', 3))
    -0.5 (('X', 1), ('Z', 2), ('X', 3))
```

The first four terms are the hopping terms. Each one moves an electron from one
site to the other, and the `Z` between the two ends gives the sign that fermions
need. The `Z` and `ZZ` terms that follow are the on-site repulsion.

The offset must be included in every energy calculation. Here it is exactly `1.0`,
and if you leave it out, every result is shifted by this amount.

```python
without = exact_ground_state_energy(hamiltonian, N_QUBITS)
with_off = exact_ground_state_energy(hamiltonian, N_QUBITS, offset)

print("without the offset:", without)
print("with the offset   :", with_off)
print("analytic          :", analytic)
```

```text
without the offset: -2.23606797749979
with the offset   : -1.2360679774997898
analytic          : -1.2360679774997898
```

`exact_ground_state_energy()` diagonalises the Hamiltonian directly. At four
qubits, this is a `16 × 16` matrix.

## Running VQE

```python
result = VQE(layers=2, random_state=0).solve(hamiltonian, N_QUBITS, offset=offset)

print("energy        ", result.energy)
print("error vs exact", abs(result.energy - analytic))
print("n_iterations  ", result.n_iterations)
print("success       ", result.success)
print("n parameters  ", len(result.optimal_parameters))
```

```text
energy         -0.9999999991069795
error vs exact 0.23606797839281035
n_iterations   20
success        True
n parameters   20
```

`success` only reports that the classical optimiser converged. Here it converged
to `-1.0000`, which is `0.236` above the ground-state energy. `-1.0` is itself an
eigenvalue of this Hamiltonian. It is the lowest energy with one electron, and the
default hardware-efficient ansatz with two layers cannot reach the lower
two-electron state.

## Depth of the default ansatz

`layers` sets the depth of the hardware-efficient ansatz that `VQE` builds when no
`ansatz` is given.

```python
print("layers  params  energy          error")
for layers in (1, 2, 3, 4):
    r = VQE(layers=layers, random_state=0).solve(hamiltonian, N_QUBITS, offset=offset)
    print(
        f"{layers:6d}  {len(r.optimal_parameters):6d}  {r.energy:.10f}  {abs(r.energy - analytic):.3e}"
    )
```

```text
layers  params  energy          error
     1      12  -0.8535533903  3.825e-01
     2      20  -0.9999999991  2.361e-01
     3      28  -0.9999999893  2.361e-01
     4      36  -1.2360679394  3.806e-08
```

Two and three layers both stop at `-1.0`. Four layers, with `36` parameters, reach
the ground-state energy with an error of `3.8e-8`.

## A chemistry-aware ansatz

`uccsd_ansatz()` builds the unitary coupled cluster circuit with single and double
excitations (UCCSD), approximated by Trotterisation, on the Hartree-Fock reference
state for a given number of electrons. `reps` is the number of Trotter repetitions
of the product of excitations.

```python
print("reps  params  energy          error")
for reps in (1, 2, 3):
    ansatz = uccsd_ansatz(n_qubits=N_QUBITS, n_electrons=2, reps=reps)
    r = VQE(random_state=0).solve(hamiltonian, N_QUBITS, offset=offset, ansatz=ansatz)
    print(
        f"{reps:4d}  {len(r.optimal_parameters):6d}  {r.energy:.10f}  {abs(r.energy - analytic):.3e}"
    )
```

```text
reps  params  energy          error
   1       5  -0.9999999971  2.361e-01
   2      10  -1.2360679758  1.657e-09
   3      15  -1.2360679775  2.675e-12
```

One repetition stops at `-1.0`, the same value where the shallow
hardware-efficient circuits stopped. The first-order Trotter error is the limit
here. Two repetitions, with `10` parameters, give an error of `1.7e-9`, compared
with `3.8e-8` for the hardware-efficient ansatz with `36` parameters. Three
repetitions reach an error of `2.7e-12`.

## Next

- [MaxCut with QAOA](qaoa-max-cut.md) shows the other variational solver, the
  Quantum Approximate Optimization Algorithm (QAOA), on a combinatorial problem.
- The [VQE chapter](../algorithms/vqe.md) explains the ansätze and the mappings.
- The [Quantum Chemistry guide](../user-guide/chemistry.md) covers `Molecule`,
  `bravyi_kitaev()` and the rest of the package. Real molecules need the
  `chemistry` extra.
