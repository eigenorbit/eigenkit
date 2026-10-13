# Circuit Builder

`CircuitBuilder` builds a circuit one gate at a time. Its style follows Qiskit,
Braket and CUDA Quantum: `qc.rx(angle, qubit=0)` adds a gate and returns the
builder, so that calls can be chained. Use it when no ready-made block fits, or when
you want to write a part of a circuit and then use it in a larger composition.

```python
from eigenkit import CircuitBuilder

qc = CircuitBuilder(n_qubits=3)
weights = qc.add_weights(6)
inputs = qc.add_inputs(3)

# Encoding.
for q in range(3):
    qc.ry(inputs[q], qubit=q)

# Two variational layers + CNOT ladder.
for depth in range(2):
    for q in range(3):
        qc.ry(weights[depth * 3 + q], qubit=q)
    for q in range(2):
        qc.cnot(q, q + 1)

spec = qc.build()
```

`spec` is a `CircuitSpec`, the same data type, independent of any provider, that
`assemble()` returns. Pass it directly to `Circuit`:

```python
from eigenkit import Circuit, all_z

circuit = Circuit(blocks=spec, observables=all_z(3))  # numpy interface + sim by default
```

`Circuit` also accepts the builder itself and calls `build()` for you, so
`Circuit(blocks=qc, observables=all_z(3))` is the same circuit. In both cases, the
circuit gets the gates added so far: gates added to `qc` afterwards are not part of
it.

## Declaring weights and inputs

`add_weights(n)` and `add_inputs(n)` return lists of `WeightRef` and `InputRef`.
These are symbolic handles to the trainable parameters and to the input vector. The
gate methods also accept constants, which are wrapped as `Constant` automatically:

```python
qc = CircuitBuilder(n_qubits=2)
weights = qc.add_weights(2)

# Trainable rotation.
qc.rz(weights[0], qubit=0)

# Constant rotation: 0.5 is wrapped as Constant(0.5).
qc.rx(0.5, qubit=1)
```

## Available gates

| Method | Gate | Parameters |
|---|---|---|
| `rx(angle, qubit)` | `RX(angle)` | One `Parameter`. |
| `ry(angle, qubit)` | `RY(angle)` | One `Parameter`. |
| `rz(angle, qubit)` | `RZ(angle)` | One `Parameter`. |
| `h(qubit)` | The Hadamard gate. | None. |
| `cnot(control, target)` | The controlled-NOT (CNOT) gate. | None. |
| `x(qubit)` | The Pauli `X` gate. | None. |
| `z(qubit)` | The Pauli `Z` gate. | None. |
| `mcx(*qubits)` | A multi-controlled `X` gate. Every qubit except the last is a control. | None. |
| `mcz(*qubits)` | A multi-controlled `Z` gate. Every qubit except the last is a control. | None. |
| `phase_shift(angle, qubit)` | A phase shift, which applies `e^(iφ)` to `\|1⟩` and leaves `\|0⟩` unchanged. | One `Parameter`. |
| `swap(qubit0, qubit1)` | The SWAP gate. | None. |
| `cry(angle, control, target)` | A controlled `RY` gate. | One `Parameter`. |
| `s(qubit)` | The `S` gate, a quarter turn about `Z`. | None. |
| `sdg(qubit)` | The `S†` gate, the inverse of `s()`. | None. |
| `cphase(angle, control, target)` | A controlled phase shift, which is symmetric in its two qubits. | One `Parameter`. |
| `pauli_exponential(paulis, angle, coefficient=1.0)` | `exp(-i · coefficient · angle · P)` for a Pauli product `P`, given as `(operator, qubit)` pairs on different qubits. | One `Parameter`. |

`pauli_exponential` is the Trotter building block that the cost and mixer layers of
the Quantum Approximate Optimization Algorithm (QAOA) and the excitation operators of
unitary coupled cluster with singles and doubles (UCCSD) are made of. It is
decomposed into the core set of gates.

`to_block()` is not a gate. It wraps what has been built so far as a `CustomBlock`,
so that a part of a circuit built gate by gate can be placed in a list of blocks.
`add_weights(count)`, `add_inputs(count)` and `build()` are the other three methods
that are not gates.

## Parameter expressions

Gate angles are not raw values. They are expression trees that the backends evaluate
when the circuit runs. The most common nodes are these:

| Node | Meaning |
|---|---|
| `Constant(0.5)` | A literal value. An `int` or a `float` is wrapped automatically. |
| `WeightRef(i)` | The trainable weight number `i`. `add_weights()` returns these. |
| `InputRef(i)` | The input number `i`. `add_inputs()` returns these. |
| `Scale(expr, factor)` | The product `factor * expr`. |
| `Negate(expr)` | The negation `-expr`. |
| `Sum((a, b, ...))` | The sum of the expressions. |
| `Multiply(a, b)` | The product of two expressions, for example `lambda * x` for trainable feature scaling. |

You rarely need to name these nodes yourself. The handles from `add_weights()` and
`add_inputs()` support ordinary Python arithmetic, and each operator builds exactly
the node that its constructor would build. So the two ways of writing give the same
intermediate representation, and you can mix them freely:

```python
import math

qc = CircuitBuilder(n_qubits=2)
weights = qc.add_weights(2)
inputs = qc.add_inputs(2)

# 2*pi * x_0.
qc.rx(2 * math.pi * inputs[0], qubit=0)

# -w_0 (useful for adjoints).
qc.rz(-weights[0], qubit=1)

# w_0 + x_1.
qc.ry(weights[0] + inputs[1], qubit=0)

# Trainable feature scaling, w_1 * x_0.
qc.rx(weights[1] * inputs[0], qubit=1)

# A ZZ-style feature-map angle, 2 * (pi - x_0) * (pi - x_1).
qc.cnot(0, 1)
qc.phase_shift(2 * ((math.pi - inputs[0]) * (math.pi - inputs[1])), qubit=1)
qc.cnot(0, 1)
```

`+`, `-`, `*`, `/` and the unary `-` work on every node, with plain numbers allowed
on either side. Two details are worth knowing:

- **Multiplication chooses the node for you.** A constant factor becomes a `Scale`,
  which a backend folds into the angle of the gate. Only a symbol times a symbol
  becomes a `Multiply`. It is easy to get this wrong when you build the nodes by
  hand, and nothing warns you when you do.
- **Addition is flattened.** `a + b + c` is one `Sum` with three terms, not a `Sum`
  inside another `Sum`.

You can divide only by a number (`x / 2` is `Scale(x, 0.5)`). The intermediate
representation has no division node, so `x / y` raises a `TypeError`. The explicit
constructors are still available from `eigenkit.core.ir`, for code that builds
expression trees programmatically:

```python
from eigenkit.core.ir import Multiply, Negate, Scale, Sum

qc.rx(Scale(inputs[0], factor=2 * math.pi), qubit=0)  # same as 2*pi * inputs[0]
qc.rz(Negate(weights[0]), qubit=1)  # same as -weights[0]
qc.ry(Sum((weights[0], inputs[1])), qubit=0)  # same as weights[0] + inputs[1]
qc.rx(Multiply(weights[1], inputs[0]), qubit=1)  # same as weights[1] * inputs[0]
```

On backends without their own automatic differentiation, a `Multiply` makes the
gradient use the parameter-shift rule together with the chain rule. The gradients
match the automatic differentiation of PennyLane to machine precision on PennyLane,
Qiskit and Braket. CUDA Quantum does not support products of two symbols at the
moment.

## Building an inversion circuit

A common pattern is to build `U(x_1)` followed by the adjoint of `U(x_2)`, to read
out the fidelity `|<0...0| U^dag(x_2) U(x_1) |0...0>|^2`. This is what `QuantumKernel`
uses, and it serves any fidelity measure in the style of a swap test.

```python
from eigenkit import Circuit, CircuitBuilder
from eigenkit.core.ir import Negate

n_features = 3
qc = CircuitBuilder(n_qubits=n_features)
x1 = qc.add_inputs(n_features)
x2 = qc.add_inputs(n_features)

# Forward U(x_1).
for q in range(n_features):
    qc.rx(x1[q], qubit=q)

# Adjoint U^dag(x_2): reversed order, negated angles.
for q in reversed(range(n_features)):
    qc.rx(Negate(x2[q]), qubit=q)

circuit = Circuit(blocks=qc.build())
# No trainable weights here, so pass an empty weight vector; the data is inputs.
print(circuit.zero_state_probability([], [0.4, -0.2, 0.6, 0.4, -0.2, 0.6]))  # ~1.0
```

See [Quantum Kernels](kernels.md) for the full implementation of the kernel.

## Inspecting the gates

`circuit.draw()` returns a drawing, as text over several lines, of the circuit that
you have just built. It is the easiest way to check a circuit written by hand before
training:

```python
print(circuit.draw())
```

```text
q_0: ──RX(x_0)──RX(-x_3)──

q_1: ──RX(x_1)──RX(-x_4)──

q_2: ──RX(x_2)──RX(-x_5)──
```

The drawing works in the same way for circuits assembled from blocks. See
[Basic Circuit](../examples/basic-circuit.md) for an example with `Encoding` and
`Variational`.

## Composing with blocks

`qc.to_block()` wraps the current state of the builder as a `CustomBlock`, which you
can use anywhere a `Block` is accepted. The assembler moves the local `WeightRef` and
`InputRef` indices of the block into the numbering of the whole circuit. In a list of
blocks, you can also leave out `to_block()` and put `qc` in the list directly. The
assembler then makes the same call for you.

```python
from eigenkit import Circuit, CircuitBuilder, Variational, all_z
from eigenkit.core.ir import Scale

# Hand-written encoding: scale every input by 2*pi.
qc = CircuitBuilder(n_qubits=4)
inputs = qc.add_inputs(4)
for q in range(4):
    qc.ry(Scale(inputs[q], factor=2 * 3.14159), qubit=q)
my_encoding = qc.to_block()

# Compose with a standard variational block.
circuit = Circuit(
    blocks=[my_encoding, Variational(depth=2)],
    observables=all_z(4),
)
```

## When to use the builder

- You need a sequence of gates that no ready-made block expresses.
- You want full control over the parameter expressions, with your own patterns of
  `Multiply`, `Sum` and `Scale`.
- You want to write a part of a circuit gate by gate, and then use it in a list of
  blocks through `.to_block()`.
- You are following a paper that uses a specific decomposition into gates.

For the usual shape of an encoding followed by a variational block, prefer the block
interface ([Encoding](encoding.md), [Variational](variational.md),
[Reuploading](reuploading.md)). It is shorter, and it stays readable as the circuit
grows.

## API reference

::: eigenkit.core.builder.CircuitBuilder
