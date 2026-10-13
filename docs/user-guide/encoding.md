# Encoding

The `Encoding` block maps classical features to quantum states, with one
single-qubit rotation per feature.

```python
from eigenkit import Encoding

encoding = Encoding(in_features=4, rotation="RX")
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_features` | `int` | Required | The number of features to encode, one per qubit. It must be positive. |
| `rotation` | `"RX"` \| `"RY"` \| `"RZ"` | `"RY"` | The rotation gate applied for each feature. |
| `scale` | `float` | `1.0` | A factor that multiplies every input angle. It is useful when the data is normalised to `[0, 1]` and you want angles in `[0, 2π]`. It must be finite. |

!!! note "The arguments are checked when the block is created"

    A non-positive `in_features` is rejected immediately:
    `Encoding(in_features=0)` raises
    `ValueError: in_features must be positive, got 0`. A `rotation` other than
    `RX`, `RY` or `RZ`, and a `scale` that is not finite, are rejected in the same
    way. Every block in the library checks its arguments like this.

`Encoding` is the name exported from the top level and from `eigenkit.configs`. The
class itself is [`AngleEncoding`][eigenkit.core.blocks.AngleEncoding], which is the
name that appears in tracebacks and in the heading of the API reference below.

## Methods

| Method | What it does |
|---|---|
| `required_qubits()` | Returns the number of qubits that the block needs, which is `in_features`. |
| `build(n_qubits)` | Builds the gates of the block for a circuit with `n_qubits` qubits, as an `IRFragment`. `assemble()` calls it, so you rarely call it yourself. |
| `to_dict()` | Returns a description of the block that can be saved as JSON, including a `type` key. |
| `from_dict(d)` | A class method that rebuilds the block from the output of `to_dict()`. |

The `.eigenkit` bundle format relies on `to_dict()` and `from_dict()` to save and
load blocks. See [Model persistence](model-persistence.md).

## How the number of qubits is found

The number of qubits comes from `Encoding.in_features`. Every block reports the
number of qubits that it needs through `required_qubits()`. `assemble()` combines a
list of blocks into one `CircuitSpec`: it takes the largest of these numbers, and it
concatenates the weights and the inputs of the blocks in the order of the blocks.
`Circuit` calls `assemble()` for you when you pass `blocks=[...]`. See
[Circuit Builder](circuit-builder.md) to work with a `CircuitSpec` directly.

```python
encoding = Encoding(in_features=6, rotation="RZ")
# This circuit uses 6 qubits.
```

## Rotation gates

- `"RX"` applies `RX(feature_value)` to each qubit.
- `"RY"` applies `RY(feature_value)` to each qubit.
- `"RZ"` applies `RZ(feature_value)` to each qubit.

For data in a bounded range that does not cover `[0, 2π]`, use the `scale` argument,
or normalise the data first with `MinMaxNorm`:

```python
from eigenkit import MinMaxNorm
import torch

norm = MinMaxNorm(feature_dim=4, min_val=0.0, max_val=torch.pi)
```

## API reference

::: eigenkit.configs.Encoding
