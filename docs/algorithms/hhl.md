# HHL: solving a linear system

## The problem

Given a square matrix `A` and a vector `b`, find `x` such that `Ax = b`. This is a
standard numerical task.

The original [Harrow-Hassidim-Lloyd paper](https://arxiv.org/abs/0811.3171), which gives
the algorithm its name (HHL), prepares a quantum state `|x⟩` proportional to `A⁻¹|b⟩`.
Reading the full vector back as numbers would remove the speedup. The speedup also
assumes that the matrix is sparse and well conditioned, and that it can be accessed
efficiently through an oracle. The [primer by Dervovic et al.](https://arxiv.org/abs/1802.08227)
is a short reference for these assumptions.

In eigenkit, HHL is a solver for an exact simulator. It builds the structure of the HHL
circuit, reads the full statevector, and returns a classical NumPy vector.

## Input and output

`HHL(...).solve(A, b)` takes a square invertible matrix and a vector that is not zero.
The settings are:

- `n_eigenvalue_qubits` is the number of clock qubits used to estimate eigenvalues. It
  is the main setting for resolution.
- `mode` is `"reference"` (the default) or `"honest"` (see below).
- `trotter_steps`, `trotter_order` and `trotter_tolerance` set the accuracy of the
  Trotter product, in `"honest"` mode only. The defaults (`"auto"`, `2` and `0.25`) are
  safe. See [Choosing the Trotter grid](#choosing-the-trotter-grid).

It returns a classical solution vector, its norm, the residual, and two post-selection
probabilities. The first, `success_probability`, is the probability that the ancilla is
in `|1⟩`, summed over all states of the clock. The second, `readout_probability`, is the
probability of the joint event `ancilla = 1 and clock = 0`, which is the event that the
solution is actually read from. Use the second one to plan the number of shots. It is
never the larger of the two. Across the systems in the
[worked example](../examples/hhl-honest.md), the ratio between them goes from `1.0` to
`1.8`. It is exactly `1.0` when every eigenvalue lies on the grid of buckets, because
then no clock state other than `|0…0⟩` carries any weight.

## How it works

HHL works in the **eigenbasis** of `A`. If `A` has eigenvalues `λ` and eigenvectors
`|u⟩`, solving `Ax = b` means multiplying each eigen-component of `b` by `1/λ`. The
circuit does this in five steps:

1. It makes `A` Hermitian. A non-Hermitian system is converted to `A†Ax = A†b`.
2. It loads the normalised vector `b` as amplitudes.
3. It uses quantum phase estimation (QPE) to write information about the eigenvalues
   into the clock qubits. This is the step where the Hamiltonian evolution `e^{iAt}`
   is applied.
4. It rotates an ancilla by an angle that encodes the reciprocal `1/λ`. This is a
   controlled `RY` rotation whose angle is a fixed function of the clock register. No
   classical division happens in this step, in either mode.
5. It uncomputes the clock, post-selects the ancilla on `1`, and rescales the readout to
   physical units.

Step 3 is the step that differs between the two modes.

## Reference and honest mode

The `mode` argument selects how the circuit is built. Both modes solve the same system,
but they do not return the same numbers: `"reference"` mode is exact up to the
resolution of the clock, and `"honest"` mode adds a Trotter error on top of that. The
difference between them is whether the circuit could run on hardware.

| | `"reference"` (default) | `"honest"` |
|---|---|---|
| Eigenvalues | They are found classically with `eigvalsh`, at a cost of `O(N³)`. | They are not computed. |
| Evolution `e^{iAt}` | It is a dense matrix from `expm`, applied as one `QubitUnitary` block. | It is Trotterised into elementary gates (`HamiltonianQPEBlock`). |
| Evolution time `t` | It is tuned to the known spectrum. | It is fixed from a Gershgorin bound on `‖A‖`. |
| Rotation constant `C` | It is the true `λ_min`. | It is the resolution of the clock register, `2π/(t · 2^n_l)`, where `n_l` is the number of clock qubits. |
| The evolution can be built from elementary gates | No, it is a unitary given only as a matrix. | Yes, it uses elementary gates only. |
| Dense unitary gate in the circuit | There is one. | There is none. |
| Extra settings | There are none. | They are `trotter_steps`, `trotter_order` and `trotter_tolerance`. |

`"reference"` mode is fast and exact. However, it does `O(N³)` classical work, and it
applies a `2ⁿ × 2ⁿ` unitary given only as a matrix, with no decomposition into
elementary gates, so it gives no quantum speedup. Use it to validate results and for
quick experiments.

`"honest"` mode removes the three shortcuts that set the parameters of the circuit. It
decomposes `A` into a sum of Pauli strings, and simulates `e^{iAt}` with a Trotter
product built only from `H`, `RX`, `RZ`, controlled-NOT (`CNOT`), `CPHASE`, `SWAP` and
`PhaseShift` gates. The evolution time comes from a Gershgorin bound, which costs
`O(nnz)`, where `nnz` is the number of non-zero entries of `A`. The rotation constant
comes from the resolution of the clock. The condition number `κ` is never an input. It
only bounds the success probability that can be reached. `"honest"` mode needs
`n_eigenvalue_qubits >= 2`, so that the clock register is two-sided and can resolve the
sign of an eigenvalue. It provides `trotter_steps`, `trotter_order` and
`trotter_tolerance` to trade circuit depth for accuracy.

Two things stay classical in `"honest"` mode:

- **Input validation** runs one singular value decomposition (SVD) of `A`, at a cost of
  `O(N³)`. It rejects a singular matrix with a clear message, and it gives a condition
  number for the warnings. No spectral quantity from it reaches `t`, `C`, the rotation
  angles or the number of Trotter steps, and changing it leaves the circuit
  byte-identical. Even so, `solve()` itself still does `O(N³)` work. The SVD takes
  1.2 ms at `N = 128`, compared with 20 ms for the Pauli decomposition.
- **The inversion of the eigenvalues uses a table of angles**, with one entry for each
  clock bucket. The gates are elementary: a uniformly controlled rotation
  ([Möttönen et al.](https://arxiv.org/abs/quant-ph/0407010)) adds exactly `2^n_l`
  `RY` gates and `2^n_l` `CNOT` gates, with no ancillas and no multi-controlled gate.
  The assembled circuit in `"honest"` mode therefore contains no unitary-matrix gate of
  any kind. The `2^n_l` angles are listed bucket by bucket, so the classical description
  stays of size `Θ(2^n_l)`. The circuit can be built from elementary gates, but its
  description is not compact.

```python
import numpy as np
from eigenkit import HHL

# A = I + X + Z. X and Z anticommute, so the Trotter product is really
# approximate and "auto" has real work to do. (The tempting [[2,1],[1,2]] is 2I + X:
# one non-identity term, so its Trotterisation is exact and trotter_steps is a no-op.)
A = np.array([[2.0, 1.0], [1.0, 0.0]])
b = np.array([1.0, 1.0])

# hardware-faithful path; trotter_steps/order default to "auto" and 2
result = HHL(n_eigenvalue_qubits=4, mode="honest", provider="numpy").solve(A, b)

print(result.solution)  # x ≈ A⁻¹b
print(result.residual)  # ‖Ax − b‖ / ‖b‖
print(result.readout_probability)  # the post-selection rate a device would see
```

This example uses four clock qubits, and not more. This system reaches a residual of
`0.012` at `n_l = 4` and `0.020` at `n_l = 6`, while the post-selection rate falls from
one preparation in six to one in 99. Each extra clock qubit costs about four times more
shots, for a gain that the residual does not always deliver. The register to choose is
therefore the smallest one that passes the resolution check. See
[Linear systems with HHL](../examples/hhl-honest.md), which chooses it from the residual
instead of by hand, and explains the rest.

## Choosing the Trotter grid

The Trotter step size `τ = t / trotter_steps` is **fixed** for every controlled power.
QPE is therefore an *exact* phase estimation of the unitary `W` of a single step, and
the whole circuit is ideal HHL for an effective Hamiltonian
`A_eff := log(W) / (iτ)`. The whole Trotter error is this one change of spectrum,
`A → A_eff`, and it falls mostly on the **eigenvectors**, not on the eigenvalues.

This is why a coarse grid is dangerous: an error in the eigenvectors cannot be seen by
the alignment of buckets, by the Nyquist guard bands, or by the Gershgorin bound. Take
the Heisenberg Hamiltonian `XX + YY + ZZ + ZI + IZ`, with `κ = 3` and five Pauli terms,
which is the favourable case for this method. A single first-order Trotter step
reproduces its spectrum to `1e-15` at every `n_l`, but rotates the eigenvectors by 59°.
At some register sizes this puts two of the four signs of the answer wrong. The second
way to fail is a drift of an eigenvalue across the Nyquist bucket, whose sign is
ambiguous. This flips the sign of `1/λ` for the extreme eigenvalue.

`trotter_steps="auto"`, the default, prevents both. It bounds `‖A_eff − A‖` from the
Pauli coefficients alone. It uses the standard leading-order bounds for product
formulas, in the commutator-scaling form of
[Childs et al. 2021](https://arxiv.org/abs/1912.08854): `τ Γ₁` at order 1 and
`τ² Γ₂` at order 2, written for Pauli terms, where every commutator norm is `0` or `2`.
It then picks the smallest number of steps that keeps the bound below
`trotter_tolerance × C`, which is a quarter of the eigenvalue resolution of the clock
register. The bound costs `O(L²)`, where `L` is the number of Pauli terms, and the test
suite checks it against `‖A_eff − A‖` computed exactly. On the Heisenberg example above,
with `b` a vector of ones, the default grid returns a solution whose cosine with the
exact answer is `0.9999` or more.

Several things follow from this:

- **The cost is real.** To keep the drift below the resolution, the number of steps
  must grow as `steps ∝ √(2^n_l)` at `trotter_order=2`, so the Trotter part of the
  circuit grows as `Θ(2^(1.5 n_l))`, that is, `Θ(κ^1.5)`. At `trotter_order=1`, the
  number of steps grows as `steps ∝ 2^n_l` and the cost as `Θ(4^n_l)`. The automatic
  choice is limited by a gate budget. When the limit applies, the solver warns and gives
  the number of steps that the criterion asked for.
- **`trotter_order=2` is the default for a structural reason.** The symmetric product is
  a palindrome, so the first-order commutator cancels, and `A_eff` stays real for a real
  `A`. At order 1, `A_eff` picks up `Y`-type terms, and a real system returns a complex
  state. Both orders uncompute exactly: the inverse QPE ladder applies the first-order
  factors in reverse order. This is what makes it the adjoint of the forward pass, and
  not only its product for a negative time.
- **An explicit integer turns off the automatic choice.** The solver then warns when the
  drift bound exceeds `C/2`, the threshold at which the Nyquist bucket can capture an
  eigenvalue.
- **Real inputs converge one order faster at `trotter_order=1`.** The readout discards
  the imaginary part, and at order 1 the leading error in the eigenvectors is exactly
  what this step removes. A real `A` and `b` therefore converge as `steps⁻²`, while a
  complex Hermitian `A` or a complex `b` keeps that error and converges as `steps⁻¹`. At
  `trotter_order=2`, the first-order commutator has already cancelled, so discarding the
  imaginary part has nothing left to remove, and both converge as `steps⁻²`. The default
  is therefore the order where this difference does not appear, not the order that
  causes it.
- **The order of the terms matters, and it is fixed.** The first-order Trotter error
  depends on the order in which the Pauli factors are applied, which is the order that
  `pauli_decompose` returns. A regression test fixes this order, because changing it
  would change every result in `"honest"` mode at a fixed `trotter_steps`.

## Performance notes

The timings come from one developer workstation and are only indicative. The ratios
between them are the part that lasts.

The Pauli decomposition runs once per call to `solve()` and costs `O(N² log N)`. A Pauli
string has exactly one non-zero entry per row, at column `r ⊕ x`, so the
Hilbert-Schmidt trace reduces to a single sum over rows. That sum, taken over every `z`
at once for a fixed `x`, is a Walsh-Hadamard transform of the diagonal of `A` shifted by
`x`. On a dense complex Hermitian matrix, it takes 0.3, 1.0, 4.3, 20 and 119 ms at
dimensions 16, 32, 64, 128 and 256, and 1.6 s at `N = 1024`. The `O(N³)` SVD in input
validation is of the same order, 1.2 ms at `N = 128`.

The cost that limits the method is the circuit depth,
`#terms × steps × 2^n_l`. This is the real exponential cost, and no optimisation can
remove it.

Because the exponential cost is in the circuit *depth*, and not in the dimension of the
Hilbert space, `lightning.gpu` runs about 20 % slower than `lightning.qubit` at every
size tried. The number of qubits, `1 + n_l + n_b`, stays small (13 qubits, or 8192
amplitudes, in the largest useful run), while the number of gates reaches hundreds of
thousands. The time is therefore spent on the overhead of launching kernels, not on
arithmetic. A graphics processing unit (GPU) starts to be faster at about 20 qubits,
which needs `n_b ≈ 10`. At that size the decomposition costs 1.6 s, and the circuit
depth is the real obstacle.

## Guarantees and limits

For an invertible `A` on a statevector backend, you get a classical `x` read from the
post-selected state of the circuit. The accuracy is measured, not proven: increasing
`n_eigenvalue_qubits` lowers the residual **provided that the Trotter grid keeps up**,
which is what `trotter_steps="auto"` is for. On a fixed coarse grid the pattern can
reverse, because `t` grows with `N`, and so does `τ = t / steps`. A small irregular
variation in the residual remains by construction, even on a fine grid. The projection
on `clock = |0⟩` weights each eigenvalue by how well it fits the grid of buckets, and
this pattern changes as eigenvalues move on and off the grid when `n_l` changes.

Two limits are separate, and both must be respected. The **register** must satisfy
`2^n_l ≳ 2κ + 2`, because a two-sided register resolves `κ` only up to `(N - 2)/2`, and
a one-sided register up to `N - 1`. Below that, the smallest eigenvalue falls into
bucket `0`, which has no rotation, and its component, which carries the largest weight
`1/λ`, is reduced or lost. The **Trotter grid** must keep the drift below the
resolution. Raising `n_eigenvalue_qubits` alone fixes the first limit and makes the
second worse. `"auto"` handles the second limit for you.

The readout is the main caveat. HHL natively produces the quantum *state* `|x⟩`. The
full classical vector is recovered from the statevector of the simulator, which is only
possible on a simulator. On real hardware, only expectation values `⟨x|M|x⟩` are
available, and reading all `N` components would take `O(N)` measurements, which cancels
the speedup. On hardware, HHL is used to pass `|x⟩` to a later quantum routine, or to
estimate a single observable.

In addition, the returned vector is **completed by two classical steps** that a device
cannot perform. For real inputs, the imaginary part is discarded. The magnitude *and the
sign* come from a least-squares projection, `x = r ⟨Ar, b⟩ / ‖Ar‖²`, which uses the full
classical `A` and `b`. Only the *direction* of the raw post-selected vector comes from
the quantum circuit. The rescaling can improve the reported residual by more than an
order of magnitude, and for an indefinite `A` its correction of the sign is not
reliable, because `⟨Ax, b⟩ ≥ 0` does not imply `⟨x, A⁻¹b⟩ ≥ 0`. Read `residual` as the
quality of the best classical rescaling of a quantum direction. The unit-norm complex
vector that the circuit actually produced, before either classical step, is returned as
`raw_solution`, so you can check all of this yourself.

**Scope of the speedup.** HHL is efficient only for an `A` that is **a short sum of
Pauli terms and well conditioned**, with an efficient way to prepare the state for `b`.
The cost that matters is the number of Pauli terms, and this number depends on sparsity
**in the Pauli basis**, which is *not* the same property as sparsity of the matrix:

| `A` | Non-zero entries per row | Terms at `N = 128` |
|---|---|---|
| `diag(1..N)`, the Ising model `Σ ZZ + Σ X`, and local spin chains. | From 1 to 3. | `O(n)` ✅. |
| A Toeplitz tridiagonal matrix. | 3. | `N = 128`. |
| A real symmetric tridiagonal matrix, with variable coefficients. | 3. | `576 = Θ(n 2ⁿ)`. |
| **A random diagonal matrix.** | **1.** | `128 = Θ(N)`. |
| **The tridiagonal cycle, with its vertices relabelled at random.** | **3.** | **5440.** |
| A random dense real symmetric matrix. | 128. | 8256. |
| A random dense complex Hermitian matrix. | 128. | `16384 = 4ⁿ`. |

Two rows show the main point. A random *diagonal* matrix, which is as sparse as an
invertible matrix can be, needs `N` terms, because the coefficients of the `Z` strings
are the Walsh-Hadamard transform of the diagonal, and this transform is dense for a
generic diagonal. Also, the number of non-zero entries is not the right variable: a
tridiagonal matrix and a randomly relabelled cycle graph both have three non-zero
entries per row, but one needs 128 terms and the other needs 5440, which is 66 % of a
fully dense matrix. A non-zero entry at `(i, j)` contributes to the Pauli strings whose
`X`/`Y` support is the bit pattern `i ⊕ j`, so the number of terms follows the number
of *distinct* offsets `i ⊕ j`. A band has only the `log₂ N` patterns `2^k - 1`, which is
why the real tridiagonal row with variable coefficients above costs `576 = 64 · 7 + 128` at
`n = 7`, while a scattered pattern uses them all. For generic coefficients, the number
of terms is exactly `2^(n-1)` for each distinct non-zero offset when `A` is real
symmetric, or `2ⁿ` when `A` is complex Hermitian, plus the number of non-zero
Walsh-Hadamard coefficients of the diagonal. This predicts every row of the
table.

So "sparse means cheap" is false here. What holds is "built as a short Pauli sum means
cheap". The efficiency assumption of the 2009 HHL paper uses a **different input
model**: the matrix is `s`-sparse, its rows can be computed efficiently, and it is
simulated through a sparse-access oracle. That cost analysis does not carry over to a
decomposition into Pauli terms or into a linear combination of unitaries.

`"reference"` mode hides this cost inside `expm`, so it stays fast in simulation but
does not represent hardware. Non-Hermitian inputs go through the normal equations, which
square the condition number and can reduce accuracy. `"honest"` mode knows that the
converted system is positive semidefinite, and uses the whole phase window for it. This
is worth about one extra bit of eigenvalue resolution, and was measured as a clear gain
in accuracy (38 of 75 A/B pairs better, and 4 worse). The cost is a post-selection rate
2 to 4 times lower, and about twice the Trotter depth, because the ancilla amplitude is
`C/λ` and this choice halves `C`. Eigen-directions below the `epsilon` cutoff are removed
before the circuit is built, and `"honest"` mode warns when this happens. With the
default `ε = 0.01`, it starts to remove buckets at `n_l ≥ 8`, and these are the buckets
with the largest weights `1/λ`. If post-selection is almost impossible, the solver
returns a zero vector with `residual = 1.0`. A rough answer
(`‖Ax - b‖ / ‖b‖ > 0.25`) comes back with a warning.

**Noisy backends are not supported.** The solution is read from a slice of the
amplitudes, which a mixed state does not provide, so density-matrix devices (for
example PennyLane `default.mixed`) raise `NotImplementedError`. Coherent and calibration
errors can be studied by perturbing the gate angles, and the circuit tolerates them
well: a random error of 1 milliradian on each of 20 000 gates costs `2e-5` in the cosine
of the solution, and 10 milliradians cost 0.2 %. Compared with a coarse Trotter grid,
calibration noise is negligible.

## Next

The details of the interface are in the [API reference](../api/algorithms.md). If you
change only one parameter while you experiment, change `n_eigenvalue_qubits`: it
controls the resolution of the phase estimation. The papers are listed in
[References](references.md#hhl-linear-systems).
