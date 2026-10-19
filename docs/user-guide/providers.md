# Providers

Every entry point that builds a circuit accepts a `provider=` argument, in the same
three forms: a string (a registered name), a configuration dataclass, or a backend
instance. The same circuit can run on any backend without changes.

```python
from eigenkit import Circuit, PennyLaneConfig

# String: registered name lookup.
Circuit(blocks=..., provider="pennylane")

# Config: routes to the matching backend via `make_backend()`.
Circuit(blocks=..., provider=PennyLaneConfig(device_name="lightning.qubit"))

# Backend instance: used directly.
from eigenkit import CudaQuantumBackend, register

Circuit(blocks=..., provider=CudaQuantumBackend())

# Or register it once under a name, then use the name anywhere.
register("cudaq", CudaQuantumBackend())
Circuit(blocks=..., provider="cudaq")
```

The built-in **NumPy** simulator is the default (`provider="numpy"`), and it is
always available, with no extra. PennyLane, Qiskit, Braket, IQM, TensorCircuit,
Qulacs and qsim register their names automatically once their extras are installed.
CUDA Quantum and Qiskit Runtime do not register a name by themselves: pass an
instance as `provider=`, or import the backend and call `register()`. A
`QiskitRuntimeConfig` passed as `provider=` does not need registration either.

Naming a backend whose package is not installed raises an `ImportError`, which tells
you which extra to install. The name stays registered, so once the extra is
installed, the same process finds it, without a restart:

```text
ImportError: provider 'pennylane' is registered but its package is not
installed (install with `pip install eigenkit[pennylane]`).
```

A name that the library does not know at all raises a `KeyError` instead, and lists
the names that it knows. These are two different problems, and the two messages keep
them apart.

## What "faster" is measured against

When these docs call a backend *fast* or *faster* without naming a baseline, the
comparison is with the built-in **NumPy** reference simulator. This is the default
`provider="numpy"`, run on the central processing unit (CPU), analytically
(`shots=None`) and in double precision. That simulator is written to be readable,
and it is not tuned, which is what makes it the zero point. "Faster than NumPy" says
that a backend is worth installing for a given workload. It does not say that the
backend beats every other backend.

The batching table further down uses a **different** baseline, and its heading says
so. It compares the native batched call of a backend with the library's own Python
loop over the samples, *on that same backend*. The speedup of about `30×` given for
NumPy there is that same comparison between the loop and the batch, not a
comparison between backends.

All these figures are rough expectations from development runs, not published
benchmarks with fixed hardware and versions. Time your own circuit before you
choose a backend for its speed.

## Backend matrix

| Backend | Registration | Gradients | Native batching | Notes |
|---|---|---|---|---|
| NumPy | Built in (`"numpy"`). | Adjoint. | Yes, vectorised. | **The default.** The reference statevector simulator, with no extra dependencies. |
| PennyLane | Lazy (`"pennylane"`). | Native. | Yes. | Any PennyLane device, through `PennyLaneConfig`. It needs the `pennylane` extra. |
| Qiskit | Lazy (`"qiskit"`). | Parameter shift. | No, it loops. | A statevector simulator. Finite shots are not supported yet, and any value other than `shots=None` raises an error. |
| Braket (local) | Lazy (`"braket"`). | Parameter shift. | No, it loops. | `LocalSimulator`. |
| Braket (cloud or QPU) | The same backend, chosen by the configuration. | Parameter shift, or none. | No, it loops. | `BraketConfig(device_arn=...)`. On a quantum processing unit (QPU), gradients are turned off unless `enable_gradients=True`, and `shots > 0` is required. |
| Qiskit Runtime | On request (`register("qiskit-runtime", ...)`). | None, or parameter shift. | Yes. | IBM Eagle and Heron devices, and the fake backends. Gradients only with `enable_gradients=True`. |
| IQM | Lazy (`"iqm"`). | None, or parameter shift. | No, one job per sample. | IQM Resonance through iqm-client, and the fake backends of IQM, which model its noise. The hardware returns only counts, so every value is computed from histograms of bitstrings. Gradients only with `enable_gradients=True`. It needs the `iqm` extra. |
| CUDA Quantum | On request (`register("cudaq", ...)`). | Parameter shift. | Partly. | The optional `cudaq` dependency. `Sum` and `Multiply` expressions are not supported. |
| TensorCircuit | Lazy (`"tensorcircuit"`). | Automatic differentiation with JAX. | Yes, with `jax.vmap`. | A tensor-network simulator, through tensorcircuit-ng on JAX. It needs the `tensorcircuit` extra. It is best for workloads that are run many times (see the note below). |
| Qulacs | Lazy (`"qulacs"`). | Adjoint. | No, it loops. | A fast statevector simulator written in C++. It has a native reverse-mode gradient through `backprop`, and it supports the full set of gates. It needs the `qulacs` extra. |
| Qiskit Aer | Lazy (`"qiskit-aer"`). | Parameter shift. | No, it loops. | The dense statevector core of Aer. It comes with the `qiskit` extra, so it adds no dependency. |
| Qiskit Aer MPS | Lazy (`"qiskit-aer-mps"`). | Parameter shift. | No, it loops. | The same backend, with `method="matrix_product_state"`. It is exact when the bond dimension is not limited. It is the natural point of comparison for a simulation based on the time-evolving block decimation (TEBD) algorithm. |
| qsim | Lazy (`"qsim"`). | Parameter shift. | No, it loops. | The qsim simulator of Google, through Cirq. It works in **single precision** (`float32`), so it is fast but has a residual error of about `1e-6`. It needs the `qsim` extra. |
| EigenOrbit Router | Lazy (`"router"`). | None on a QPU, parameter shift on a simulator. | No, one job per sample, submitted in parallel. | Real QPUs and cloud simulators through the hosted EigenOrbit Router, with one EigenOrbit API key and a prepaid balance. Pass `RouterConfig(device=...)` from eigenorbit-router-client: the device has no default, so a bare `provider="router"` raises an error that says so. It needs the `router` extra. |

## NumPy (built in)

This is the default: a reference statevector simulator in pure NumPy, with no extra
dependencies, so that an install with no extra can run circuits immediately:

```python
import numpy as np
from eigenkit import Circuit, Encoding, Variational, all_z

circuit = Circuit(
    blocks=[Encoding(in_features=2), Variational(depth=1)],
    observables=all_z(2),
)  # provider="numpy", interface="numpy" by default
circuit(np.zeros(circuit.spec.n_weights), np.array([0.3, -0.2]))
```

Gradients use the analytic **adjoint** method, which is exact and costs `O(1)` in the
number of parameters on statevector simulators. The simulator takes the simple speed
gains that NumPy allows. A batch of inputs evolves in one vectorised sweep, with a
leading batch axis on the state tensor, which is about `30×` faster than a loop over
the samples with a batch of `64`. A CNOT is applied as a swap of amplitudes
(`np.flip`), not as a matrix product. It is still the readable reference and not a
tuned simulator, and it is the baseline of every "faster" claim in these docs.
Install `pennylane` for the fast path. The forward values and the gradients agree
with PennyLane to machine precision.

## PennyLane

```python
from eigenkit import Circuit, PennyLaneConfig

config = PennyLaneConfig(device_name="lightning.qubit", shots=None)
Circuit(blocks=..., provider=config)
```

`device_name` is the identifier of any PennyLane device. `shots=None` evaluates the
circuit analytically. The PyTorch interface of PennyLane computes gradients
natively.

## Qiskit

```python
from eigenkit import Circuit, QiskitConfig

config = QiskitConfig(shots=None)
Circuit(blocks=..., provider=config)
```

The backend builds a parametric `QuantumCircuit` once. Each forward pass binds the
parameters with `assign_parameters` and computes the expectation values with
`Statevector`. Gradients use the parameter-shift rule.

## Braket

`BraketConfig` chooses between the local simulator, a cloud simulator and a QPU,
based on `device_arn`:

```python
from eigenkit import BraketConfig

# Local (default).
local = BraketConfig()

# Cloud simulator (SV1).
sv1 = BraketConfig(
    device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    shots=1000,
)

# QPU (gradients disabled unless enable_gradients=True, shots > 0 enforced).
qpu = BraketConfig(
    device_arn="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    shots=1000,
)
```

Runs on the cloud and on a QPU issue a single `CostWarning` when the circuit is
compiled. See [QPU and cloud workflows](qpu.md) for submitting without waiting and
for dry runs.

Braket runs `run`, `sample`, `probabilities` and `zero_state_probability`, either
waiting for the result or through `submit()`. `sample()` needs `shots > 0`. With
`shots=0`, `probabilities()` and `zero_state_probability()` are exact on the local
simulator, SV1 and DM1. The result type that each device uses, and the largest
number of qubits for the exact results, are in the
[table of Braket result types](qpu.md#braket-shot-rules-and-result-types).

## Qiskit Runtime

This backend targets real IBM hardware, and the fake backends of
`qiskit_ibm_runtime`, which model the noise of the devices. The token is looked up in
this order: an explicit token, then the `QISKIT_IBM_TOKEN` environment variable, then
the saved account file. `channel` defaults to
`"ibm_quantum_platform"`, because the older `"ibm_quantum"` channel was retired.

```python
from eigenkit import Circuit, QiskitRuntimeConfig

config = QiskitRuntimeConfig(backend_name="ibm_fez", shots=1024)
Circuit(blocks=..., provider=config)

# Or let the service choose its least busy operational device.
config = QiskitRuntimeConfig(backend_name="least_busy", shots=1024)
```

`backend_name` is required unless `backend_instance` is given. There is no default
device, because IBM retires devices. Passing the configuration does not need a
`register()` call.

For tests, pass a `FakeBackendV2` directly through `backend_instance`. No
authentication is needed.

`primitive="sampler"` uses `SamplerV2` in place of `EstimatorV2`, and computes the
expectation values from the raw counts of bitstrings. This also makes `sample()` and
`probabilities()` available. See [QPU and cloud workflows](qpu.md).

## IQM

This backend targets IQM hardware through the software of IQM itself, and not through
Braket. The difference is between a bill from Amazon Web Services (AWS) and the plans
of IQM, and this is the only way to use the fake backends of IQM and its Star
architectures, which use MOVE gates. The token is looked up in this order: an
explicit token, then the `IQM_TOKEN` environment variable, then whatever iqm-client
finds by itself.

```python
from eigenkit import Circuit, IQMConfig

config = IQMConfig(
    url="https://resonance.iqm.tech/",
    quantum_computer="garnet",
    shots=2048,
)
Circuit(blocks=..., provider=config)
```

The service of IQM returns only histograms of bitstrings, so every expectation value
is computed from counts, and `shots > 0` is required. For tests, pass an
`IQMFakeAdonis()` or an `IQMFakeDeneb()` directly through `backend_instance`. They run
locally on Aer, with a model of the noise of IQM, and they need no token and no
network.

## CUDA Quantum

```python
from eigenkit import (
    Circuit,
    CudaQuantumConfig,
    CudaQuantumBackend,
    register,
)

register("cudaq", CudaQuantumBackend())
config = CudaQuantumConfig(target="nvidia", shots=None)
Circuit(blocks=..., provider=config)
```

Common targets are `"qpp-cpu"` (a statevector on the CPU, the default), `"nvidia"`
(one graphics processing unit, or GPU), `"nvidia-mgpu"`, `"nvidia-mqpu"` and
`"tensornet"`.

The arithmetic of CUDA Quantum on parameters (`QuakeValue`) is limited: `Sum` and
`Multiply` expressions are not supported. This rules out
`Reuploading(trainable_scale=True)` on CUDA Quantum.

## TensorCircuit

```python
from eigenkit import Circuit, TensorCircuitConfig

# Auto-registered once the `tensorcircuit` extra is installed.
Circuit(blocks=..., provider="tensorcircuit")

# Equivalently, via the config.
Circuit(blocks=..., provider=TensorCircuitConfig())
```

This backend simulates circuits as tensor networks, with
[tensorcircuit-ng](https://github.com/tensorcircuit/tensorcircuit-ng) on the JAX
backend (the `tensorcircuit` extra). Expectation values are exact contractions of the
tensor network. The batch of parameters is vectorised with `jax.vmap`, and the whole
forward pass and gradient are compiled with `jax.jit`. Gradients use the native
reverse-mode automatic differentiation of JAX (`jax.jacrev`), exposed as the
`jacobian` of the provider. So both the NumPy variational optimisation tools (VQE and
QAOA through `minimize_expectation`) and the PyTorch interface use them, without the
parameter-shift rule. The software is imported only when it is needed, so
`import eigenkit` does not import JAX until this provider is resolved. The supported
gates are `RX`, `RY`, `RZ`, `H`, `X`, `Z`, `CNOT` and `SWAP`.

**When it pays off.** The simulation is still an *exact* contraction, so what this
backend gives is a constant factor, not access to larger circuits. It has a one-time
compilation cost with `jax.jit` of a few seconds, which grows with the number of
qubits and the size of the Hamiltonian. So it suits workloads that are **run many
times** (training loops, searches over hyperparameters, VQE and QAOA with an ansatz of
fixed shape), where this cost is spread over many evaluations. For single runs or
exploration, the NumPy simulator is faster. The forward values and the gradients
match the NumPy reference to machine precision.

## Qulacs

```python
from eigenkit import Circuit, QulacsConfig, QulacsBackend, register

register("qulacs", QulacsBackend())  # auto-registers when the extra is installed
Circuit(blocks=..., provider="qulacs")  # or provider=QulacsConfig()
```

[qulacs](https://github.com/qulacs/qulacs) is a statevector simulator with a
compiled C++ core, where the NumPy reference is written in Python and NumPy. It has a
**native adjoint gradient** (`ParametricQuantumCircuit.backprop`), so it declares
`gradient_method="adjoint"`, like the NumPy reference simulator. One `backprop` sweep
per observable gives the gradient of every gate angle in reverse mode, at a cost of
`O(1)` in the number of parameters. These gradients are mapped onto the weights and
the inputs with the chain factors of the intermediate representation, so
`Reuploading(trainable_scale=True)` and other angles with `Scale` or `Multiply` are
differentiated correctly. The forward values, the gradients,
`zero_state_probability` (the primitive of quantum kernels) and the reading and
writing of `statevector` all agree with the NumPy reference to machine precision.

The full set of gates is supported. Gates for which qulacs has no parametric
primitive are decomposed onto parametric rotations, so that they stay differentiable:
`PhaseShift` becomes `RZ`, `CRY` becomes `RY·CX·RY·CX`, and `CPHASE` becomes three
`RZ` and `CX` gates. `adjoint(AmplitudeEncoding)` is not supported, as with Qiskit,
because a state preparation cannot be inverted as a gate. Install it with
`pip install eigenkit[qulacs]`.

## Qiskit Aer

```python
from eigenkit import Circuit, QiskitAerConfig

Circuit(blocks=..., provider="qiskit-aer")  # dense statevector core
Circuit(blocks=..., provider="qiskit-aer-mps")  # matrix-product-state core
Circuit(blocks=..., provider=QiskitAerConfig(method="matrix_product_state"))
```

[Aer](https://github.com/Qiskit/qiskit-aer) is the high-performance simulator of
Qiskit. It already comes with the `qiskit` extra, so this backend adds **no new
dependency**. It reuses the translation of the plain Qiskit backend without changes
(the lowering of gates, the binding of symbolic `Parameter` values, and the
`SparsePauliOp` observables). Only the way the circuit is executed changes: the
expectation values are read through `save_expectation_value` of Aer, in place of
`Statevector.from_instruction`.

Two methods are available, as two provider names. `qiskit-aer` is the dense
statevector core, which is exact and matches the NumPy reference to about `1e-12`.
`qiskit-aer-mps` is the matrix product state (MPS) core. It is exact when the bond
dimension is not limited (the default of Aer), it is a fast approximation for
circuits with little entanglement, and it is the natural point of comparison for a
simulation with tensor networks. It agrees with NumPy to about `1e-9`, and the
difference comes from the truncation of the singular value decomposition.

The full set of gates is supported by both. Both methods translate the circuit to the
native gates of Aer first (`optimization_level=0`, with no coupling map, so no
routing and no permutation of the qubits). This is needed because the MPS core
rejects `cry`, and because the statevector core cannot run the controlled unitary
that a `ControlledQubitUnitary` becomes. Sampling with finite shots is not supported
yet, so `shots` must be `None`. Install it with `pip install eigenkit[qiskit]`.

## qsim

```python
from eigenkit import Circuit, QsimConfig

Circuit(blocks=..., provider="qsim")  # or provider=QsimConfig()
Circuit(blocks=..., provider=QsimConfig(threads=8))
```

[qsim](https://github.com/quantumlib/qsim) is the high-performance Schrödinger
statevector simulator of Google, used through its `qsimcirq` bridge to Cirq. The
intermediate representation is translated once into a parametrised `cirq.Circuit`,
whose angles are SymPy symbols, and each call resolves the symbols and runs the
circuit.

**qsim works in single precision (`float32`)**, because the pip package has no build
in double precision. The values are returned as `complex128`, but they carry an
accumulated error of about `1e-6`, so qsim does not pass a parity test at `1e-9`.
This residual comes from the precision and is not a bug, as with the single-precision
path of CUDA Quantum. The parity tests check qsim at `1e-4`. What qsim gives is speed
over the NumPy reference simulator, and the residual should be reported together with
any timing, not hidden.

The full set of gates of the circuit algebra is supported. The primitives that carry
data (`AmplitudeEncoding`, `QubitUnitary` and `ControlledQubitUnitary`) are not, and
they are rejected before the run. An `initial_state` is not supported, and passing one
raises a `NotImplementedError`. The number of threads follows `OMP_NUM_THREADS`
unless `QsimConfig(threads=...)` sets it, so qsim respects a benchmark setup that fixes
the threads. Install it with `pip install eigenkit[qsim]`.

## Batched evaluation

Batching works on **both interfaces** and on every backend. Pass a 2-D `inputs` array
of shape `(B, n_inputs)`, and you get an output of shape `(B, n_obs)`, or `(B,)` when
there is a single observable. `weights` stays 1-D, because the weights are shared
across the batch.

```python
import numpy as np
from eigenkit import Circuit, Encoding, Variational, all_z

# numpy interface (the default): weights are passed per call.
circuit = Circuit(
    blocks=[Encoding(in_features=4), Variational(rotation="RY", depth=2)],
    observables=all_z(4),
)
weights = np.zeros(circuit.spec.n_weights)  # or a trained vector
x_batch = np.random.uniform(0, np.pi, size=(128, 4))
y_batch = circuit(weights, x_batch)  # shape (128, 4)
```

On the PyTorch interface, the module stores the weights, so you pass only the batch:

```python
import torch
from eigenkit import Circuit, Encoding, Variational, all_z

circuit = Circuit(
    blocks=[Encoding(in_features=4), Variational(rotation="RY", depth=2)],
    observables=all_z(4),
    interface="torch",
    provider="pennylane",
)
y_batch = circuit(torch.randn(128, 4, dtype=torch.float64))  # shape (128, 4)
```

### What "native batching" means here

When a call has the shape of the **fast path** (`weights.ndim == 1` and
`inputs.ndim == 2`), the backends with native batching send the whole batch through
one batched call of the provider. The Qiskit statevector backend and the local Braket
backend use the shared Python loop `batch_call`. Any other combination of shapes
(batched weights, a single sample of inputs in a 2-D wrapper and so on) falls back to
a loop in Python, which gives the index of a sample that fails.

The speedups below are **within one backend**: the native batched call of the backend
against the library's own loop over the samples, on the same simulator. They are not
comparisons between backends, and they are not measured against the NumPy reference.

| Backend | How the fast path works | Speedup over the loop on the same backend |
|---|---|---|
| **NumPy** | A leading batch axis on the state tensor, so the whole batch evolves in one sweep. The inputs are transposed to `(n_inputs, B)`, so an angle that depends on the inputs becomes a `(B,)` vector (a stack of `(B, 2, 2)` rotations), while angles that depend only on weights stay a single shared `(2, 2)` matrix. A CNOT is a swap of amplitudes with `np.flip`, not a matrix product. | About `30×` over the loop with a batch of `64`. |
| **PennyLane** | Parameter broadcasting in the QNode. The `inputs` are transposed to `(n_inputs, B)`, so each `InputRef(j)` becomes a vector of shape `(B,)`, and every gate angle is broadcast through a single evaluation of the tape. | About `B×` over the loop. On `default.qubit` with inputs of shape `(128, 4)`, the overhead of the library is at most `20%` of the plain PennyLane QNode call. |
| **Qiskit Runtime** | The `EstimatorV2` primitive accepts a `(B, n_params)` array of parameters in each Primitive Unified Bloc (PUB), so the whole batch is sent as one job, with one PUB per observable, in place of one job per sample. | Large. IBM Runtime is dominated by the overhead of each job, so collapsing one job per sample into one job is a big change. Check it on your own backend. |
| **CUDA Quantum** | `cudaq.observe(kernel, ham, *2D_args)` uses the broadcast helper of CUDA Quantum. **Limitation**: this helper is itself a Python loop for kernels built with `cudaq.make_kernel(...)`, so the cost of just-in-time compilation is not shared. The behaviour is correct, and it is the usual way to batch in CUDA Quantum, but in practice the run time is currently within about `10%` of the explicit loop. Full vectorisation would require moving the backend to `@cudaq.kernel` declarations. | About the same as the loop. |
| **TensorCircuit** | `jax.vmap` over the contraction of one sample, compiled with `jax.jit`, so the batch of parameters is a vectorised extra index of the tensor-network contraction. | About `B×` once it is compiled. The one-time compilation takes seconds, grows with the number of qubits, and is spread over the loop. |
| **Qiskit (statevector)** | None. `Statevector.expectation_value` works on one state at a time, so the backend loops through the shared helper. | None. |
| **Braket (local)** | None. It calls `LocalSimulator.run` once per task. Braket has `run_batch` to submit independent tasks in parallel, but it is not used yet. | None. |
| **Qulacs** | None. `Observable.get_expectation_value` works on one state at a time, so the backend loops through the shared helper. The gradients also loop over `backprop`, one sample at a time, and each call is fast because it runs in C++. | None. |

### Gradients through batched inputs

For the **backends with native automatic differentiation** (PennyLane), gradients
through a 2-D `inputs` batch go through the same broadcasting as the forward pass, so
the library does no extra work.

For the **backends that use the parameter-shift rule** (Qiskit, local Braket, CUDA
Quantum), the backward pass of the autograd function loops over the samples. It
computes the jacobian of each sample, adds them into the shared gradient of the
weights, and stacks the gradients of the inputs of each sample. This loop is part of
the parameter-shift rule, because each sample has its own input vector, and therefore
its own shifted expectation values. The forward pass still uses the batched path of
the backend. Qiskit Runtime and IQM refuse gradients (`gradient_method="none"`)
unless their configuration sets `enable_gradients=True`. Then they use the same loop,
and every shifted evaluation is a paid forward call.


## Registering custom configs

Use `register()` to give a configuration a name, so that you can reuse it:

```python
from eigenkit import PennyLaneConfig, register

register("lightning", PennyLaneConfig(device_name="lightning.qubit"))

Circuit(blocks=..., provider="lightning")
```

## Reusing the same circuit on another provider

`with_provider` is a method of the **PyTorch** circuit (`interface="torch"`). If you
already have one, and you want the **same** `CircuitSpec` and observables on a
different backend, call it instead of building the list of blocks again. The
`provider` argument accepts the same three forms as `Circuit(..., provider=...)`: a
registered name, a configuration with `make_backend()`, or a backend instance.

```python
import torch

from eigenkit import Circuit, Encoding, Variational, all_z, PennyLaneConfig

layer = Circuit(
    blocks=[Encoding(in_features=2), Variational(depth=1)],
    observables=all_z(2),
    provider="pennylane",
    interface="torch",
)
x = torch.randn(2, dtype=torch.float64)

layer_qiskit = layer.with_provider("qiskit")
layer_lightning = layer.with_provider(
    PennyLaneConfig(device_name="lightning.qubit"),
)
```

By default, the weights are **cloned** from the original circuit. Pass
`copy_weights=False` for new random weights, or `init=...` to set the weights
explicitly (it takes precedence over `copy_weights`).

## API reference

::: eigenkit.core.provider.register
::: eigenkit.core.provider.registered_names
::: eigenkit.torch.circuit.TorchCircuit.with_provider
::: eigenkit.core.provider.CostWarning

### Config dataclasses

::: eigenkit.core.backends.numpy_backend.NumpyConfig
::: eigenkit.core.backends.pennylane_backend.PennyLaneConfig
::: eigenkit.core.backends.qiskit_backend.QiskitConfig
::: eigenkit.core.backends.qiskit_aer_backend.QiskitAerConfig
::: eigenkit.core.backends.braket_backend.BraketConfig
::: eigenkit.core.backends.qiskit_runtime_backend.QiskitRuntimeConfig
::: eigenkit.core.backends.iqm_backend.IQMConfig
::: eigenkit.core.backends.cudaquantum_backend.CudaQuantumConfig
::: eigenkit.core.backends.tensorcircuit_backend.TensorCircuitConfig
::: eigenkit.core.backends.qulacs_backend.QulacsConfig
::: eigenkit.core.backends.qsim_backend.QsimConfig
