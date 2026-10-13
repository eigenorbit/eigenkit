# Reuploading

The `Reuploading` block alternates encodings of the data with variational blocks, and
repeats this pattern `depth` times within a single circuit. For non-trivial
classification and regression tasks, reuploading is usually more expressive than
encoding the data only once.

```python
from eigenkit import Reuploading

reuploading = Reuploading(
    in_features=4,
    depth=3,
    encoding_rotation="RY",
    variational_rotation="RY",
)
```

## Two patterns

`Reuploading` covers two known patterns:

1. **Standard reuploading** (the default; Pérez-Salinas et al. 2020). The same input
   is encoded `depth` times, with new variational weights between the encodings.
2. **Trainable feature scaling** (`trainable_scale=True`; Schuld et al. 2021). Each
   encoding rotation takes the angle `λ * x`, where `λ` is a trainable weight for
   each pair of layer and feature. At the same depth, this is strictly more
   expressive than pattern 1.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_features` | `int` | Required | The total number of input features. |
| `depth` | `int` | Required | The number of full repetitions. |
| `encoding_rotation` | `"RX"` \| `"RY"` \| `"RZ"` | `"RY"` | The rotation of the encoding. |
| `variational_rotation` | `"RX"` \| `"RY"` \| `"RZ"` | `"RY"` | The rotation of the variational block. It is used only when `variational_type="basic"`. |
| `variational_type` | `"basic"` \| `"strong"` | `"basic"` | The kind of variational block. |
| `trainable_scale` | `bool` | `False` | Whether each encoding rotation gets a trainable scale for each pair of layer and feature. |

## Number of weights

| Setup | Trainable weights |
|---|---|
| `"basic"`, without a scale | `depth * in_features` |
| `"strong"`, without a scale | `3 * depth * in_features` |
| With `trainable_scale=True` | `depth * in_features` more, for the scales |

## Trainable scale

`trainable_scale=True` adds the parameter expression `Multiply(weight, input)`. The
default NumPy simulator differentiates it with its analytic adjoint method. Backends
without their own automatic differentiation use the parameter-shift jacobian, which
takes the chain rule into account. The gradients agree to machine precision on every
backend that supports `Multiply`.

CUDA Quantum does not support `Multiply` at the moment. Use a different backend, or
set `trainable_scale=False` if you need CUDA Quantum.

## API reference

::: eigenkit.configs.Reuploading
