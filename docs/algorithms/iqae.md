# IQAE: iterative quantum amplitude estimation

## The problem

Suppose that a circuit `A` prepares a state where the last qubit, the **ancilla**,
reads `1` with an unknown probability `a`. Amplitude estimation is the task of
estimating this `a`.

Formally, `A` acts on `n` qubits so that

```text
A|0…0⟩ = √(1 - a) |ψ_0⟩|0⟩ + √a |ψ_1⟩|1⟩
```

where the last qubit marks the "good" outcome, and `a = sin²θ`. When you count shots
directly, the error shrinks like `1/√N`. Iterative quantum amplitude estimation (IQAE)
amplifies the angle before counting, following the construction of
[Grinko et al.](https://arxiv.org/abs/1912.05559), which does not need quantum phase
estimation.

The number `a` is useful when the circuit marks an event that has a meaning outside
the circuit. A risk model can mark "the loss of the portfolio is above this
threshold", and use IQAE to estimate the probability of this tail. The output is then
a probability at the level of the application: a number that the caller can compare,
use as a threshold, or pass into a larger estimate of risk.

## Input and output

`IQAE(...).solve(oracle)` takes an oracle `A` that prepares a state and has no
parameters, either as a PennyLane callable or as a `CircuitSpec`. The last qubit is
the ancilla. The result is classical: an estimate, a confidence interval, and the
numbers of rounds and of oracle calls.

## How it works

Applying the Grover operator `Q` exactly `m` times changes the probability that the
ancilla reads `1` to

```text
P(ancilla = 1 after A then Q^m) = sin²((2m + 1)θ)
```

So `Q` multiplies the angle `θ` by the odd factor `2m + 1`, and a small probability
becomes easier to distinguish.

IQAE then runs an adaptive loop:

1. It chooses a Grover power `m` that keeps `(2m + 1)θ` within a half-period where it
   is not ambiguous.
2. It runs `A` followed by `Q^m`, and counts the outcomes where the ancilla reads `1`.
3. It builds a Clopper-Pearson confidence interval for this amplified probability.
4. It maps the interval back through `P = sin²((2m + 1)θ)`, to narrow the interval
   for `a`.

## Guarantees and limits

With its adaptive schedule, the interval is built to contain the true `a`
with a probability of at least `1 - α`. If the loop converges within its budget of
rounds, the width of the interval is at most `2ε`. In theory, the number of queries
grows like `1/ε`, up to logarithmic factors, instead of `1/ε²` for a direct Monte
Carlo estimate ([Grinko et al.](https://arxiv.org/abs/1912.05559);
[Brassard et al.](https://arxiv.org/abs/quant-ph/0005055)).

In the default path, the library computes the exact probability of the ancilla from
the statevector, and then draws a seeded binomial sample to imitate shots. Use
`shot_mode="provider"` with `provider="pennylane"` for real sampling by the backend.
If too few shots prevent convergence, IQAE returns the wider interval that it has,
and issues a warning. `epsilon` must be in `(0, π/8)`.

## Next

The details of the interface are in the [API reference](../api/algorithms.md). The
same idea also appears in `PortfolioRiskEstimator`, where the ancilla marks a tail
loss. The papers are listed in [References](references.md#iqae-amplitude-estimation).
