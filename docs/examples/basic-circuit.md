# Basic Circuit

This example goes through the whole process: it builds a quantum circuit from
blocks, chooses a provider, and runs the circuit on data.

## Setup

```python
import numpy as np
from eigenkit import Circuit, Encoding, Variational, all_z

encoding = Encoding(in_features=6, rotation="RX")
variational = Variational(depth=2, rotation="RY")
observables = all_z(6)
```

## Choose a provider

```python
from eigenkit import PennyLaneConfig

config = PennyLaneConfig(device_name="lightning.qubit", shots=None)
```

You can also pass a registered name, as a string (`"pennylane"`, `"qiskit"` or
`"braket"`), or a backend instance. See [Providers](../user-guide/providers.md).

## Build and run

```python
circuit = Circuit(
    blocks=[encoding, variational],
    observables=observables,
    provider=config,
)

# 6 qubits, 6 features. The numpy interface (default) takes weights per call;
# output has one expectation value per qubit.
weights = np.zeros(circuit.spec.n_weights)  # or a trained vector
inputs = np.random.uniform(0, np.pi, size=(3, 6))  # batch of 3 samples
outputs = circuit(weights, inputs)
print(outputs.shape)
```

```text
(3, 6)
```

To get a differentiable `torch.nn.Module` that holds its own weights and fits into
a PyTorch model, build the circuit with `interface="torch"` (this needs the `torch`
extra), and call `circuit(inputs)`. See [Getting Started](../getting-started.md).

## Inspecting the gate structure

`circuit.draw()` returns a drawing of the circuit, as text over several lines. The
drawing is built from the `CircuitSpec`, so it is the same for every backend:

```python
print(circuit.draw())
```

```text
q_0: ──RX(x_0)──RY(w_0)──●──RY(w_6)──...
                         │
q_1: ──RX(x_1)──RY(w_1)──X──...
...
```

Parametrised angles are drawn as symbols: `x_i` for the input slots, `w_i` for the
trainable weights, and the numbers themselves for `Constant` angles. Combined
expressions, such as `Scale`, `Sum` and `Multiply`, are written as ordinary formulas
(for example `RY(2*x_0)` and `RX(w_0*x_1)`).

## Scalar output

The size of the output equals the number of observables. Pass a single observable,
or a `PauliExpr`, to get one scalar output for each sample:

```python
from eigenkit import Z

scalar_circuit = Circuit(
    blocks=[encoding, variational],
    observables=Z(0),  # only measure Z on qubit 0
    provider=config,
)
print(scalar_circuit(weights, inputs).shape)  # (3,)
```

## Adding trainable feature scaling

Replace the separate `Encoding` and `Variational` blocks with a `Reuploading`
block. It repeats the encoding, and it can give each pair of layer and feature a
trainable scale:

```python
from eigenkit import Reuploading

reuploading = Reuploading(
    in_features=6,
    depth=3,
    encoding_rotation="RY",
    variational_rotation="RY",
    trainable_scale=True,
)

circuit = Circuit(blocks=[reuploading], observables=all_z(6))
```

See [Reuploading](../user-guide/reuploading.md) for all the options.

## Imperative construction

When no ready-made block fits, use `CircuitBuilder`. It is the interface for
adding one gate at a time, in the style of Qiskit, Braket and CUDA Quantum:

```python
from eigenkit import Circuit, CircuitBuilder, all_z

qc = CircuitBuilder(n_qubits=4)
weights = qc.add_weights(8)
inputs = qc.add_inputs(4)

# Encoding.
for q in range(4):
    qc.ry(inputs[q], qubit=q)

# Two variational layers + CNOT ladder.
for d in range(2):
    for q in range(4):
        qc.ry(weights[d * 4 + q], qubit=q)
    for q in range(3):
        qc.cnot(q, q + 1)

spec = qc.build()
circuit = Circuit(blocks=spec, observables=all_z(4))
```

`qc.build()` returns a `CircuitSpec`, which you pass directly to `Circuit`. For a
sub-circuit that you want to combine with the standard blocks, use `qc.to_block()`
instead. See the [Circuit Builder guide](../user-guide/circuit-builder.md) for the
full set of gates, the parameter expressions, and an example of an inversion
circuit.
