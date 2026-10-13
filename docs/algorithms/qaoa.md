# QAOA: quantum approximate optimization algorithm

## The problem

Many optimisation problems ask for a bitstring `b ∈ {0, 1}ⁿ` with a low cost. MaxCut
is the standard example: you split the nodes of a graph into two groups, so that many
edges cross from one group to the other.

The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid version of this
search. A quantum circuit proposes bitstrings, and a classical optimiser tunes the
angles of the circuit. The method alternates a cost layer and a mixer layer, as in
the construction of [Farhi, Goldstone, and Gutmann](https://arxiv.org/abs/1411.4028).

## Input and output

`QAOA(...).solve(problem)` accepts an `Ising`, a `QUBO`, or the output of
`max_cut(...)`. `layers` is the depth `p` of QAOA. The method returns the decoded
bitstring, its exact classical cost, the optimised expectation value, the tuned
angles, and the probability distribution.

## How it works

First, the problem is rewritten as a diagonal **cost Hamiltonian**, whose energy is
equal to the cost of the bitstring:

```text
H_C = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + constant offset
```

`Z_i` reads the value of qubit `i`, and the constant offset is kept separately. For a
worked derivation of this mapping, see the
[Sturm QAOA tutorial](https://arxiv.org/abs/2301.09535).

QAOA then prepares a state that it can tune, and optimises it:

1. It prepares `|+⟩^{⊗n}`, the equal superposition of all bitstrings.
2. It repeats the following `p` times: it applies a cost layer `exp(-iγH_C)` and a
   mixer layer `exp(-iβ Σ_i X_i)`.
3. A classical optimiser (by default, the SciPy method `"L-BFGS-B"`) adjusts the
   `2p` angles to lower `⟨H_C⟩`.
4. It decodes the optimised state: it scores the `top_k` most probable bitstrings,
   and returns the one with the lowest cost.

The cost layer scores the candidates: a lower energy means a better bitstring for the
encoded objective. The mixer moves probability between bitstrings, and the optimiser
tunes the angles so that the final distribution puts more weight on the candidates
with a low energy. Decoding turns this distribution into one classical answer.

## Guarantees and limits

The library builds the standard QAOA ansatz, runs a classical optimisation loop, and
returns the decoded candidate with the lowest cost, together with its exact classical
cost. On the `numpy` backend, `probabilities` is the exact distribution
`|amplitude|²`.

QAOA returns a candidate solution, not a proof that the solution is the global
optimum. The optimiser can get stuck, `top_k` can miss an optimum that has a low
probability, and `expectation` is different from `best_energy`. By default, `top_k`
scores the `64` most probable bitstrings, or all of them when there are fewer. The
simulator reads the full distribution over `2ⁿ` bitstrings, so the examples stay
small. The current mixer is the standard transverse field `Σ_i X_i`.

## Next

The interface of the solver is in the [QAOA reference](../api/optimization/qaoa.md).
QAOA also appears in the [architecture notes](../architecture-diagrams.md). The papers
are listed in [References](references.md#qaoa-combinatorial-optimisation).
