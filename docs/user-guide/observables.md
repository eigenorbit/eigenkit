# Observables

What to measure is set with the `observables=` argument of `Circuit`. It is not a
special block. There are three ways to build an observable, each suited to a
different level of detail.

## Default: `Z` on every qubit

If you do not pass `observables=`, the circuit measures `Z` on every qubit:

```python
from eigenkit import Circuit, Encoding, Variational

circuit = Circuit(
    blocks=[Encoding(in_features=4), Variational(depth=2)],
    # observables= defaults to all-Z; output is shape (..., n_qubits).
)
```

## Typed constructors

`Observable.x(qubit)`, `Observable.y(qubit)` and `Observable.z(qubit)` build
observables made of a single Pauli operator:

```python
from eigenkit import Observable, all_z

# Measure Z on every qubit.
observables = all_z(4)

# Measure X on qubit 0 only.
observables = [Observable.x(0)]
```

## Pauli algebra

The `X`, `Y` and `Z` constructors return a `PauliExpr`, which you can combine with
arithmetic. Use this to write Hamiltonians:

```python
from eigenkit import Circuit, Encoding, Variational, X, Y, Z

H = 0.7 * Z(0) + 0.3 * X(1) * Z(2) - 0.5 * Y(0) * Y(1)

# H acts on qubits 0-2, so the circuit needs at least three wires.
circuit = Circuit(
    blocks=[Encoding(in_features=3), Variational(depth=2)],
    observables=H,
)
```

A `PauliExpr` passed as a single observable gives a scalar output, which is one
expectation value. Pass a list of `PauliExpr` to get a vector output.

A measured observable holds only Pauli terms. So an expression with a constant term,
such as the cost of an edge in MaxCut, `0.5 * (1 - Z(0) * Z(1))`, raises a
`ValueError`, instead of losing the constant without an error. Measure the Pauli
part, and add the constant back:

```python
cost = 0.5 * (1 - Z(0) * Z(1))
edge = Circuit(
    blocks=[Encoding(in_features=2), Variational(depth=1)],
    observables=cost - cost.constant,
)
value = edge(edge.spec.n_weights * [0.1], [0.3, 0.4]) + cost.constant
```

## Text format

For quick experiments, you can write the observable as a string:

```python
from eigenkit import from_string

H = from_string("0.5 Z0 X1 + 0.3 Y2")
# Equivalent to 0.5 * Z(0) * X(1) + 0.3 * Y(2)
```

## Output shape

The shape of the output depends on what you pass, **and on the number of dimensions
of the input**. A 1-D `inputs` is one sample and has no batch axis. A 2-D input, of
shape `(batch, n_inputs)`, has one.

| `observables=` | 1-D input | 2-D `(batch, n_inputs)` |
|---|---|---|
| `None` | `(n_qubits,)` | `(batch, n_qubits)` |
| A single `Observable` or `PauliExpr` | `()`, a scalar | `(batch,)` |
| A list of `Observable` or `PauliExpr` | `(len(list),)` | `(batch, len(list))` |

These shapes were measured on a circuit with four qubits and a batch of shape
`(32, 4)`. In both cases, `weights` stays 1-D, because the weights are shared across
the batch.

## Measurements that are not Pauli operators

For observables based on fidelity (quantum kernels, swap tests), use
`circuit.zero_state_probability(weights, inputs)` instead. It returns the probability
of measuring the all-zeros state, without going through the Pauli machinery. See
[Quantum Kernels](kernels.md).

## API reference

::: eigenkit.core.ir.Observable
::: eigenkit.core.observables
