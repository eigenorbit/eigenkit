# MaxCut with QAOA

MaxCut asks for the partition of the nodes of a graph into two sets that cuts the
most edges. In general, it is NP-hard, where NP stands for nondeterministic
polynomial time: no known algorithm solves every case in polynomial time. With six
nodes, it is small enough to check by hand, and it is the standard first problem for
the Quantum Approximate Optimization Algorithm (QAOA).
QAOA is a variational algorithm: its circuit alternates a cost layer and a mixer
layer, and a classical optimiser then tunes the two families of angles.

This example runs on `provider="numpy"`, the default of `QAOA`, so it needs no
optional extra. The whole page runs in about two seconds. See the
[QAOA chapter](../algorithms/qaoa.md) for the construction, and the
[Optimization guide](../user-guide/optimization.md) for the application programming
interface (API).

## The graph, and the answer we are looking for

The graph has six nodes in a ring, plus one chord across the ring. It is small
enough to list all `64` partitions, so we know the true optimum before we run
anything quantum.

```python
import itertools
import numpy as np
from eigenkit.optimization import QAOA, max_cut

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
N_NODES = 6


def cut_size(bits):
    """Edges whose two endpoints land on opposite sides of the partition."""
    return sum(1 for u, v in EDGES if bits[u] != bits[v])


best = max(itertools.product([0, 1], repeat=N_NODES), key=cut_size)
print("edges         ", EDGES)
print("best partition", best)
print("its cut       ", cut_size(best))
```

```text
edges          [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
best partition (0, 1, 0, 1, 0, 1)
its cut        7
```

All seven edges can be cut at once. Flipping every bit gives the same cut, so
`(1, 0, 1, 0, 1, 0)` is also optimal. The solutions of MaxCut always come in such
complementary pairs.

## From a graph to an Ising model

`max_cut()` turns a list of edges into the `Ising` model whose ground state is the
maximum cut.

```python
problem = max_cut(EDGES)

print("type    ", type(problem).__name__)
print("h       ", problem.h)
print("J       ")
print(problem.J)
print("constant", problem.constant)
```

```text
type     Ising
h        [0. 0. 0. 0. 0. 0.]
J
[[0.  0.5 0.  0.5 0.  0.5]
 [0.  0.  0.5 0.  0.  0. ]
 [0.  0.  0.  0.5 0.  0. ]
 [0.  0.  0.  0.  0.5 0. ]
 [0.  0.  0.  0.  0.  0.5]
 [0.  0.  0.  0.  0.  0. ]]
constant -3.5
```

The numbers show how the mapping works. There is no local field, so `h` is all
zeros: MaxCut does not prefer either side for a node on its own. Each edge adds `0.5`
to the upper triangle of `J`, and the constant is `-|E|/2 = -3.5`, where `|E|` is the
number of edges. Together, they make the Ising energy equal to minus the size of the
cut, so *minimising* the energy *maximises* the cut.

`QAOA` also accepts a `QUBO` (quadratic unconstrained binary optimisation problem)
if the problem is stated with binary variables instead of spin variables. See the
[Optimization guide](../user-guide/optimization.md).

## Solve it

```python
result = QAOA(layers=1, random_state=0).solve(problem)

print("best_bitstring    ", result.best_bitstring)
print("best_energy       ", result.best_energy)
print("expectation       ", round(result.expectation, 6))
print("n_iterations      ", result.n_iterations)
print("success           ", result.success)
print("optimal_parameters", np.round(result.optimal_parameters, 6))
```

```text
best_bitstring     (1, 0, 1, 0, 1, 0)
best_energy        -7.0
expectation        -5.050979
n_iterations       5
success            True
optimal_parameters [ 0.708303 -0.392699]
```

`OptimizationResult` has seven fields. These are the ones that you read first:

- **`best_bitstring`** is the partition with the lowest energy among the most
  probable bitstrings of the final state.
- **`best_energy`** is its Ising energy, `-7.0`, so it is a cut of `7`.
- **`expectation`** is the mean energy of the final state, `-5.05`. It is the
  quantity that QAOA minimises.
- **`optimal_parameters`** holds one `γ` and one `β` for `layers=1`. With more
  layers, the array grows to `2 * layers` entries.
- **`n_iterations`** and **`success`** come from the classical optimiser that tuned
  the angles.

## Check the answer

```python
print("cut of best_bitstring:", cut_size(result.best_bitstring))
print("optimal cut          :", cut_size(best))
```

```text
cut of best_bitstring: 7
optimal cut          : 7
```

One layer returns an optimal partition. To choose `best_bitstring`, `solve` scores
the `64` most probable bitstrings with the energy of the problem (or all of them,
when there are fewer), and keeps the best one. Six nodes give only `64` bitstrings,
so here every partition is scored, and `best_bitstring` does not show the average
quality of the state. That quality shows in `expectation`, the quantity that QAOA
minimises.

## What depth buys

The approximation ratio is the mean cut that the circuit produces, divided by the
best cut that exists.

```python
optimum = cut_size(best)
print("layers  expectation  approx ratio  iterations")
for p in (1, 2, 3, 4, 5):
    r = QAOA(layers=p, random_state=0).solve(problem)
    print(
        f"{p:6d}  {r.expectation:11.4f}  {-r.expectation / optimum:12.4f}  {r.n_iterations:10d}"
    )
```

```text
layers  expectation  approx ratio  iterations
     1      -5.0510        0.7216           5
     2      -5.9967        0.8567          11
     3      -6.5026        0.9289          25
     4      -6.7074        0.9582          69
     5      -6.9087        0.9870          61
```

With one layer, the cuts sampled are `5.05` on average, which is 72 % of the
optimum. Five layers reach 98.7 %. The classical optimiser works on `2 * layers`
angles, and its number of iterations rises from `5` to about `60` over the same
range.

## Reading the distribution

`probabilities` is the full distribution over the `2 ** n_nodes` partitions. Its
index is the bitstring read as a binary integer, with node `0` as the most
significant bit.

```python
r3 = QAOA(layers=3, random_state=0).solve(problem)
probs = r3.probabilities

print("bitstring        probability  cut")
for idx in np.argsort(probs)[::-1][:6]:
    bits = tuple(int(b) for b in format(int(idx), f"0{N_NODES}b"))
    print(f"{str(bits):16s} {probs[idx]:11.4f}  {cut_size(bits)}")
```

```text
bitstring        probability  cut
(0, 1, 0, 1, 0, 1)      0.4155  7
(1, 0, 1, 0, 1, 0)      0.4155  7
(1, 0, 1, 0, 0, 1)      0.0146  5
(0, 1, 0, 1, 1, 0)      0.0146  5
(0, 0, 1, 1, 0, 1)      0.0146  5
(1, 1, 0, 0, 1, 0)      0.0146  5
```

The two optimal partitions have a probability of `0.4155` each. Each one is the
other with every bit flipped. The QAOA circuit does not change when every qubit is
flipped, so their amplitudes are equal by construction. Together, they hold 83 % of
the distribution at three layers.

The next partitions form pairs for the same reason, and all of them cut `5` edges:
this graph has no partition that cuts exactly `6`.

## Next

- [Ground-state energy with VQE](vqe-fermi-hubbard.md) shows the other variational
  solver, the Variational Quantum Eigensolver (VQE), on a chemistry problem with an
  exact reference.
- The [QAOA chapter](../algorithms/qaoa.md) explains the cost and mixer layers
  behind `solve()`.
- The [Optimization guide](../user-guide/optimization.md) covers `QUBO`, `Ising`,
  `max_cut()` and the full `OptimizationResult`.
