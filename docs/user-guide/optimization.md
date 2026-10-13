# Optimization

`eigenkit.optimization` solves combinatorial problems. In these problems, the answer
is an assignment of bits, and the cost is a quadratic function of the bits. You write
the problem as a `QUBO` (quadratic unconstrained binary optimisation) or an `Ising`
model, and pass it to `QAOA` (the Quantum Approximate Optimization Algorithm). You
get back the best bitstring that it found, together with the distribution that it
sampled from.

Like `chemistry` and `reservoir`, this is a domain package of the core. It uses only
NumPy and SciPy, without PyTorch or the package of any backend. `QAOA` also uses
`provider="numpy"` by default, so nothing here needs an optional dependency.

The theory is on the [QAOA](../algorithms/qaoa.md) page of Algorithms Explained.

## MaxCut in five lines

MaxCut looks for the split of the nodes of a graph that cuts the most edges.
`max_cut` builds the Ising model for you, written as an energy to minimise:

```python
from eigenkit.optimization import QAOA, max_cut

problem = max_cut([(0, 1), (1, 2), (2, 0), (2, 3)])
result = QAOA(layers=2, random_state=0).solve(problem)

print(result.best_bitstring)  # (1, 1, 0, 1)
print(result.best_energy)  # -3.0
print(result.success)  # True
```

The graph has four nodes and four edges, and the best cut has three edges. On a
problem this small, you can check the answer by trying every bitstring. It is worth
doing once, to see what "the answer" means here:

```python
import itertools

best = min(itertools.product([0, 1], repeat=4), key=problem.energy)

print(best)  # (0, 0, 1, 0)
print(problem.energy(best))  # -3.0
```

The bitstring is different, but the energy is the same, and each bit of one
bitstring is the opposite of the same bit in the other. Moving every node to the
other side of the cut describes the same partition, so the solutions of MaxCut
always come in pairs. Compare energies, not bitstrings.

## Stating the problem

`QUBO` minimises `xᵀQx + constant` over `x ∈ {0,1}ⁿ`. `energy` gives the cost of any
candidate. This is the quickest way to check a formulation before you spend time on
a solve:

```python
import numpy as np
from eigenkit.optimization import QUBO

Q = np.array([[-2.0, 1.0], [0.0, -2.0]])
qubo = QUBO(Q, constant=0.5)

print(qubo.energy([0, 0]))  # 0.5
print(qubo.energy([1, 0]))  # -1.5
print(qubo.energy([1, 1]))  # -2.5   the minimum

print(QAOA(layers=2, random_state=0).solve(qubo).best_bitstring)  # (1, 1)
```

`Ising` is the same problem in spin variables. It minimises
`Σᵢhᵢsᵢ + Σᵢ<ⱼJᵢⱼsᵢsⱼ + constant` over `s ∈ {−1,+1}ⁿ`. The two forms are
equivalent, and `QUBO.to_ising()` converts a `QUBO` to an `Ising` model:

```python
from eigenkit.optimization import Ising

ising = qubo.to_ising()
print(ising.h)  # [0.75 0.75]
print(ising.constant)  # -1.25

direct = Ising(np.array([0.5, -0.5]), np.array([[0.0, 1.0], [0.0, 0.0]]))
print(direct.energy([0, 0]))  # 1.0
print(direct.energy([1, 0]))  # -2.0
```

Note that `Ising.energy` takes **bits, not spins**: a `0` means the spin `+1`, and a
`1` means the spin `−1`. This keeps `energy` the same for both types of problem, but
it will surprise you if you pass `-1` directly.

`max_cut(edges, weights=None, num_nodes=None)` is a shortcut that builds the Ising
model of a weighted graph. Pass `weights` for weighted edges, and `num_nodes` when
the graph has isolated nodes that the list of edges never mentions.

## Reading the result

`solve` returns an `OptimizationResult`, which holds more than the best bitstring:

| Field | What it holds |
|---|---|
| `best_bitstring` | The best assignment found, as a tuple of `0` and `1` values. |
| `best_energy` | Its exact classical cost. This is the answer you would act on. |
| `expectation` | The mean cost over the whole distribution. This is the quantity that QAOA minimises. |
| `probabilities` | The whole distribution over all `2**n` bitstrings. |
| `optimal_parameters` | The optimised angles. There are `2 * layers` of them. |
| `n_iterations` | The number of steps of the optimiser. |
| `success` | Whether the optimiser reported convergence. |

```python
result = QAOA(layers=2, random_state=0).solve(problem)

print(result.best_energy)  # -3.0        the best sample
print(result.expectation)  # -2.903547   the mean over the distribution
print(result.probabilities.shape)  # (16,)       one entry per bitstring
print(result.n_iterations)  # 18
print(result.optimal_parameters)  # 4 angles: gamma and beta per layer
```

The two energies answer different questions. `best_energy` is the cost of the best
single bitstring among those that the optimised state makes most likely. This is the
answer you would act on. `expectation` is the average cost over the whole
distribution, and it is the quantity that QAOA actually minimises. The difference
between them comes from the circuit not putting all its amplitude on one solution.
More `layers` make it smaller.

`probabilities` is the whole distribution over all `2ⁿ` bitstrings, so you can also
look at the solutions that come after the best one. To choose `best_bitstring`,
`solve` scores the most probable bitstrings with the `energy` of the problem. By
default, it scores the `64` most probable ones, or all of them when there are fewer.
Pass `top_k=` to `solve` to change how many bitstrings are scored, which matters once
`n` is large.

## The circuit behind it

`QAOA` builds the cost Hamiltonian and the ansatz itself, but you can also get both
separately, to inspect or reuse them. `to_hamiltonian()` turns a problem into the
pair `(Observable, offset)`, with the same convention as the chemistry mappings.
`qaoa_ansatz` builds the standard circuit: `|+⟩ⁿ`, followed by `layers` repetitions
of the cost unitary and the mixer.

```python
from eigenkit import Circuit
from eigenkit.optimization import qaoa_ansatz

hamiltonian, offset = problem.to_hamiltonian()
print(len(hamiltonian.terms))  # 4
print(offset)  # -2.0
print(hamiltonian.terms[0])  # PauliTerm(coeff=0.5, paulis=(('Z', 0), ('Z', 1)))

spec = qaoa_ansatz(hamiltonian, 4, layers=2)
print(spec.n_weights)  # 4

circuit = Circuit(spec, observables=hamiltonian, provider="numpy")
print(float(circuit(result.optimal_parameters, [])) + offset)  # -2.903547
```

The last number is `result.expectation`, computed again from the parts. It is a
useful check when you start to use your own mixer or a different schedule of
parameters.

One limitation: `qaoa_ansatz` expects an `Observable`, which is the type that
`to_hamiltonian` returns. It does not accept a `PauliExpr` built with the `X`, `Y`
and `Z` helpers, although `Circuit(observables=...)` does. Passing one raises an
`AttributeError` about `terms`.

## API reference

::: eigenkit.optimization
