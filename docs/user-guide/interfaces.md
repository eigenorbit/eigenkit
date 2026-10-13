# Interfaces

`Circuit(...)` returns a different type of circuit depending on its `interface=`
argument, which you set explicitly, as in PennyLane. Both interfaces run the **same**
`CircuitSpec` on the **same** providers. They differ only in *how you pass the
weights* and *how you get gradients*:

- **`"numpy"`**, the default, is the core circuit, which does not depend on any
  framework. You pass the `weights` at every call, and the gradients come from
  `circuit.jacobian(...)`. It does not use PyTorch, and `pip install eigenkit` with
  no extra is enough.
- **`"torch"`** is a `torch.nn.Module` that stores its weights as an `nn.Parameter`
  and computes gradients through `torch.autograd`. It is part of the optional PyTorch
  package (the `torch` extra).

| | `numpy` (default) | `torch` |
|---|---|---|
| Returns | A `core.circuit.Circuit`. | A `TorchCircuit`, which is an `nn.Module`. |
| Weights | They are passed at each call: `circuit(weights, x)`. | The circuit stores them as an `nn.Parameter`: `circuit(x)`. |
| Gradients | `circuit.jacobian(weights, x)`. | `loss.backward()`, through autograd. |
| Dependency | The core (NumPy and SciPy). | The `torch` extra. |
| Best for | Chemistry and optimisation, variational loops, and custom training without a framework. | Hybrid quantum-classical models trained end to end in PyTorch. |

Both accept the same `provider=` argument, in any of its forms. See
[Providers](providers.md).

!!! note "`Circuit` is a function, not a class"
    The top-level `Circuit(...)` is a **factory function**. It returns one of two
    concrete types, depending on `interface=`: the NumPy circuit
    `eigenkit.core.circuit.Circuit`, or the PyTorch circuit `TorchCircuit`, which is
    an `nn.Module`. So it is not a class: `isinstance(obj, Circuit)`,
    `issubclass(...)`, subclassing `Circuit`, and using `Circuit` as a type
    annotation do not work. Check the type against the concrete class, or subclass
    it, instead (`from eigenkit.core.circuit import Circuit as NumpyCircuit`, or
    `from eigenkit.torch import TorchCircuit`).

---

## The NumPy interface (default)

`Circuit(blocks, observables)` returns the core NumPy circuit. It does **not** store
parameters: you pass a 1-D `weights` vector, of length `circuit.spec.n_weights`, and
an `inputs` vector at each call.

```python
import numpy as np
from eigenkit import Circuit, Encoding, Variational, all_z

circuit = Circuit(
    blocks=[Encoding(in_features=4, rotation="RX"), Variational(depth=2)],
    observables=all_z(4),
)  # provider="numpy", interface="numpy"

weights = np.zeros(circuit.spec.n_weights)
x = np.array([0.1, 0.2, 0.3, 0.4])
y = circuit(weights, x)  # shape (4,): one expectation per observable
```

### Forward calls and batching

`inputs` can be 1-D, for a single sample, or 2-D, with shape `(batch, n_inputs)`.
`weights` stays 1-D, because the weights are shared across the batch. The output has
shape `(n_observables,)` or `(batch, n_observables)`. It becomes a scalar, or shape
`(batch,)`, when a single observable is measured.

```python
x_batch = np.random.uniform(0, np.pi, size=(32, 4))
circuit(weights, x_batch)  # shape (32, 4)
```

### Gradients with `circuit.jacobian`

The NumPy interface is fully differentiable without PyTorch:

```python
g = circuit.jacobian(weights, x)  # d<O>/d(weights)
gw, gx = circuit.jacobian(weights, x, input_grad=True)  # also d<O>/d(inputs)
```

`jacobian` returns shape `(n_weights,)` for a single observable, and
`(n_observables, n_weights)` for several, in the same way as `__call__`. It uses the
differentiation of the active backend when there is one, such as the analytic
**adjoint** method of the NumPy simulator, or the automatic differentiation of
PennyLane. Otherwise, it uses the **parameter-shift** rule, which works on any
backend. Pass one 1-D sample. For a batch, loop over the rows. On a hardware backend,
it raises an error, like `.backward()`, unless the configuration of the device sets
`enable_gradients=True`.

The variational optimisation tools use this gradient, so a loop that finds a ground
state takes a few lines and does not need PyTorch. These tools work on a **purely**
variational circuit, which has no `inputs`, so the example takes an ansatz in place
of the encoding circuit above:

```python
from eigenkit import Variational, Z, minimize_expectation

ansatz = Circuit([Variational(depth=2)], observables=Z(0))  # n_inputs == 0

rng = np.random.default_rng(0)
x0 = rng.uniform(0, 2 * np.pi, size=ansatz.spec.n_weights)
result = minimize_expectation(ansatz, initial_parameters=x0)
print(result.optimal_value)
```

### Other primitives

`zero_state_probability(weights, inputs)`, which gives the fidelity used to invert
kernels, and `probabilities(weights, inputs)`, which gives the whole distribution of
measurement outcomes, take the same `(weights, inputs)` arguments:

```python
circuit.zero_state_probability(weights, x)
circuit.probabilities(weights, x)  # shape (2**n_qubits,)
```

A circuit with no trainable weights, such as the inversion circuit of a kernel that
uses only data, takes an empty weight vector: `inversion.zero_state_probability([], x)`.

### When to use it

The NumPy interface is the right default for the domains that do not use a framework
(`chemistry`, `optimization`), for variational loops driven by SciPy, for any custom
training where you control the optimiser, and for light installs that should not
install PyTorch.

---

## The PyTorch interface

Pass `interface="torch"` to get a `torch.nn.Module`. It stores its weights as an
`nn.Parameter`, so you call it with the inputs only, and you train it with any
PyTorch optimiser. It needs the `torch` extra, and it runs on the built-in NumPy
simulator, which is differentiable:

```python
import torch
from eigenkit import Circuit, Encoding, Variational, all_z

layer = Circuit(
    blocks=[Encoding(in_features=4), Variational(depth=2)],
    observables=all_z(4),
    interface="torch",
    init=None,  # optional initial weights, shape (n_weights,)
)

layer.weights  # nn.Parameter of shape (n_weights,)
list(layer.parameters())  # plugs into torch.optim like any module
```

`init` sets the initial weight tensor. When it is not given, the weights are drawn
at random with `torch.randn`. Both the top-level `Circuit(...)` factory and a
`TorchCircuit(...)` built directly default to `provider="numpy"`.

### What the layer exposes

| Member | What it does |
|---|---|
| `layer(x)` | The forward pass. `x` can be 1-D (one sample) or 2-D, with shape `(batch, n_inputs)`. |
| `weights` | The trainable `nn.Parameter`, of shape `(n_weights,)`. |
| `spec`, `n_qubits`, `out_features` | The assembled `CircuitSpec`, and the two sizes derived from it. |
| `draw(decimals=3)` | A drawing of the circuit as text, over several lines. It takes no input, because the layer stores its weights. |
| `draw_mpl(**kwargs)` | The same drawing as a matplotlib `(fig, ax)`. It needs the `viz` extra. |
| `zero_state_probability(x)` | The probability of the all-zeros outcome, which is the fidelity that the kernels use. |
| `submit(x)` | Submits the circuit without waiting, and returns a `JobHandle`. It works on Braket, Qiskit Runtime and IQM only. |
| `dry_run_info()` | Gives, without submitting anything, the gate counts, the shots and the number of circuits that a call runs. It works on Braket, Qiskit Runtime and IQM, and the other backends raise `NotImplementedError`. |
| `with_provider(provider, ...)` | Builds the same circuit again on another backend. See below. |

`probabilities()` and `jacobian()` are **not** available on the PyTorch circuit. The
whole distribution of measurement outcomes is a primitive of the NumPy interface,
and the gradients come from `loss.backward()` instead of a call to a jacobian.
`submit()` and `dry_run_info()` are explained in [QPU and Cloud](qpu.md).

### Forward calls and batching

```python
y = layer(torch.randn(8, 4))  # (8, 4): batch of 8, one value per observable
```

A 1-D input is a single sample, and a 2-D input is a batch. The weights are shared
across the batch, exactly as in the NumPy interface.

### Gradients with `loss.backward()`

The gradients flow through `torch.autograd`:

```python
x = torch.randn(8, 4)
target = torch.zeros(8, 4)
opt = torch.optim.Adam(layer.parameters(), lr=0.05)
loss = (layer(x) - target).pow(2).mean()
loss.backward()  # weights.grad is populated
opt.step()
```

Backends with their own automatic differentiation, such as PennyLane, differentiate
through their own engine. For the other backends, the forward pass is wrapped in a
`torch.autograd.Function`, whose backward pass uses the same parameter-shift or
adjoint jacobian as the NumPy interface. Backends without gradients
(`gradient_method="none"`, for example quantum hardware and Qiskit Runtime) raise a
clear error on `.backward()`. See [QPU and Cloud](qpu.md) for the pattern of training
on a simulator and then running on the device, and for `enable_gradients=True`, which
lets a hardware configuration use the parameter-shift rule on the device.

Training also works on the default **NumPy** provider, without PennyLane. The
analytic adjoint method of the NumPy backend differentiates a whole batch in one
sweep, so `Circuit(..., interface="torch")`, with the NumPy provider by default,
trains hybrid models efficiently. Switch to `provider="pennylane"` if you prefer its
simulator. Both process batches natively: the forward and backward passes of a
`(B, …)` batch run in one call, not in a Python loop over the samples.

### Moving a circuit to another provider with `with_provider`

`with_provider` builds the **same** `CircuitSpec` and observables again on a
different backend, without listing the blocks again:

```python
fast = layer.with_provider("qiskit")  # clones weights
fresh = layer.with_provider("qiskit", copy_weights=False)  # re-initialises
exact = layer.with_provider(
    "qiskit",
    init=torch.zeros(layer.spec.n_weights, dtype=torch.float64),
)
```

The weights are cloned by default. `copy_weights=False` draws new random weights, and
`init=` sets them explicitly (it takes precedence over `copy_weights`). The new layer
takes its data type from `init`, so pass `dtype=torch.float64` to keep the double
precision that a layer built from blocks starts with. A plain `torch.zeros(n)` is
`float32`, and it would halve the precision without any warning.

### Composing and saving

Because it is a plain `nn.Module`, a PyTorch circuit can be used inside
`nn.Sequential` and next to the ready-made layers (see [Layers](layers.md)). The scikit-learn compatible estimators (`HQNC`, `HQNR` and
others) are built on it. Its state can be saved and restored with the usual PyTorch
methods (`state_dict()` and `load_state_dict()`), and fitted estimators are saved as
`.eigenkit` bundles. See [Model persistence](model-persistence.md).

### When to use it

Use the PyTorch interface when the quantum circuit is one layer in a larger PyTorch
model, or whenever you want to train end to end with automatic differentiation and
the optimisers and tools of PyTorch.

---

## Same circuit, both interfaces

The interface is a choice made *when the code runs*, over one definition of the
circuit. The blocks, the observables and the numerical results of the provider are
the same:

```python
import numpy as np, torch
from eigenkit import Circuit, Encoding, Variational, all_z

blocks = [Encoding(in_features=2), Variational(depth=1)]

np_circuit = Circuit(blocks, all_z(2))  # numpy
tc = Circuit(blocks, all_z(2), interface="torch")  # torch

w = tc.weights.detach().numpy()
x = np.array([0.3, -0.2])
np.allclose(np_circuit(w, x), tc(torch.tensor(x)).detach().numpy())  # True
```

## API reference

The factory function that chooses the type from `interface=`, and the two concrete
types that it returns.

::: eigenkit.circuit.Circuit
::: eigenkit.core.circuit.Circuit
::: eigenkit.torch.circuit.TorchCircuit
