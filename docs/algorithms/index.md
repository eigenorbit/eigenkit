# Algorithms explained

These pages explain how each algorithm in eigenkit works: what happens between the
data that you put in and the result that it gives back. For the interface itself
(the arguments of the constructors, the return types, saving a model), see the
[Estimators guide](../user-guide/estimators.md) and the API Reference, which is
generated from the code.

| Page | What it computes | What it gives back |
|---|---|---|
| [IQAE](iqae.md) | An unknown probability amplitude `a`, to a target precision. | A number and a confidence interval. |
| [HHL](hhl.md) | The solution `x` of a linear system `Ax = b`, with a fast `"reference"` mode and an `"honest"` mode that is faithful to hardware. | A classical solution vector. |
| [QFT](qft.md) | The Fourier transform of the amplitudes of a register. | A change of basis that other algorithms read from. |
| [QAOA](qaoa.md) | A good bitstring for a combinatorial problem. | A bitstring and its cost. |
| [VQE](vqe.md) | The lowest energy that a molecule or a model Hamiltonian can have. | An energy, and the angles that reached it. |
| [QSVC / QSVR](qsvc-qsvr.md) | A similarity (kernel) matrix for a support vector model. | scikit-learn predictions. |
| [HQNN](hqnn.md) | A trained hybrid quantum-classical neural network. | Labels, probabilities or values. |
| [Reservoir computing](reservoir.md) | Features from a circuit that nobody trains. | A linear readout over a trajectory of states. |

The names in the first column are acronyms, and each page defines its own: iterative
quantum amplitude estimation (IQAE), the Harrow-Hassidim-Lloyd algorithm (HHL), the
quantum Fourier transform (QFT), the Quantum Approximate Optimization Algorithm
(QAOA), the Variational Quantum Eigensolver (VQE), the quantum support vector
classifier and regressor (QSVC and QSVR), and hybrid quantum neural networks (HQNN).

## Glossary

These terms come up on several pages. Each is also defined where it first appears.

- **Qubit**: the quantum version of a bit. `n` qubits describe a state with `2ⁿ`
  complex amplitudes.
- **Statevector**: the full list of these `2ⁿ` amplitudes.
- **Observable**: a quantity that can be measured, here a Hermitian sum of the Pauli
  operators `X`, `Y` and `Z`. Its **expectation value** `⟨O⟩` is the average outcome
  over many measurements, a real number that is often in `[-1, 1]`.
- **Ansatz**: a circuit template with free (trainable) parameters, such as a
  `Variational` block. "Variational" means that these parameters are tuned by
  optimisation.
- **Amplitude amplification, or Grover operator**: a unitary that repeatedly rotates
  probability towards a chosen "good" subspace, so that a property becomes easier to
  read. It is the engine of [IQAE](iqae.md).
- **Quantum phase estimation (QPE)**: a part of a circuit that reads the eigenvalue of
  a unitary into a register of "clock" qubits. It is the engine of [HHL](hhl.md).
- **Hamiltonian simulation, or Trotterisation**: building the evolution `e^{iAt}` from
  elementary gates, by splitting `A` into a sum of simple terms and interleaving their
  evolutions over short times. The `"honest"` mode of HHL uses it. More
  `trotter_steps` lower the approximation error, and `trotter_steps="auto"` chooses a
  number of steps that keeps this error below the resolution of the clock register.
  The error is not only in the eigenvalues: a coarse grid mostly rotates the
  *eigenvectors*, which changes the direction of the answer while the spectrum still
  looks correct.
- **Gershgorin bound**: a cheap upper bound on the largest magnitude of the
  eigenvalues of a matrix, which costs `O(nnz)`: the largest absolute row sum. The
  honest mode of HHL uses it to choose the evolution time without diagonalising `A`.
- **Condition number** `κ`: for a matrix, the ratio of its largest to its smallest
  eigenvalue, in absolute value. A large `κ` means that the system is hard to solve
  accurately.
- **Feature map**: a circuit that embeds a classical sample `x` into a quantum state
  `|φ(x)⟩`. It is the basis of [quantum kernels](qsvc-qsvr.md).
- **Angle encoding**: the usual feature map here. One input value becomes one rotation
  angle, so `n` features need `n` qubits. This is what the `Encoding` block does.
- **Amplitude encoding**: the other way to load data. A vector of `2ⁿ` numbers is
  written directly into the amplitudes of a state of `n` qubits, so `n` qubits hold
  exponentially more values than with angle encoding. The vector is padded to the next
  power of two and scaled to a norm of `1`, because a statevector must have a norm of
  `1`. It is cheap in qubits and expensive in gates: preparing an arbitrary state is
  not an operation of fixed depth, which is why the `AmplitudeEncoding` block adds one
  gate to the intermediate representation and leaves each backend to use its own
  primitive for state preparation.
- **Kernel, or Gram matrix**: a table of the similarities `K(x_i, x_j)` between pairs
  of samples. Classical support vector machines train directly on this table.
- **Barren plateau**: a failure where the training gradients of a circuit shrink
  exponentially as the circuit gets wider or deeper, so it stops learning. It is a risk
  for the [HQNN](hqnn.md) family.
