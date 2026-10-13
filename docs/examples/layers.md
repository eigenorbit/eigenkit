# Pre-built Layers

The pre-built layers wrap `Circuit` with reasonable defaults. They are all
subclasses of `torch.nn.Module`, and they can be used directly inside
`nn.Sequential`.

## `Basic`: one encoding and one variational block

```python
import torch
from eigenkit import Basic

layer = Basic(n_qubits=4, depth=2)
out = layer(torch.randn(8, 4))
print(out.shape)  # torch.Size([8, 4])
```

## `Reupload`: standard data reuploading

```python
import torch
from eigenkit import Reupload

layer = Reupload(n_qubits=4, depth=3, trainable_scale=True)
out = layer(torch.randn(8, 4))
print(out.shape)  # torch.Size([8, 4])
```

Here, `in_features == n_qubits`. Set `trainable_scale=True` for the trainable
feature scaling of Schuld et al. (2021).

## `Ensemble`: `N` independent circuits over chunks

```python
import torch
from eigenkit import Ensemble

layer = Ensemble(in_features=20, n_qubits=4, depth=5)
out = layer(torch.randn(128, 20))
print(out.shape)  # torch.Size([128, 20]): 5 circuits * 4 qubits
```

## Inspecting the underlying circuit

The `Basic`, `Reupload` and `Ensemble` layers expose `.circuit`
(or `.circuits` for `Ensemble`), so that you can inspect the circuit or save its
weights:

```python
layer = Basic(n_qubits=4, depth=2)
print(layer.circuit.spec.n_weights)  # 8 = depth * n_qubits
print(list(layer.parameters()))  # one Parameter of shape (8,), float64,
# randomly initialised unless you seed torch
```

To draw the circuit, call `layer.circuit.draw()`, which returns a drawing as text.
For a drawing from a backend, build a circuit for a single call, without a batch, and
use the drawing method of that backend (for example `qml.draw_mpl` for PennyLane).
