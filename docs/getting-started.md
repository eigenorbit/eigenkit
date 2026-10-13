# Getting Started

This guide walks you through the basics of `eigenkit`, from installation to
running your first quantum circuit. You will learn how to build circuits using
different approaches, integrate them with PyTorch for automatic
differentiation, run variational algorithms without PyTorch, and use
ready-made quantum layers. Finally, you will see how to execute the same
circuits on different backends.

For more advanced topics, explore the [User Guide](user-guide/encoding.md),
which covers each feature in more detail.

## Installation

`eigenkit` supports Python 3.12, 3.13, and 3.14. You can install it using pip
or [uv](https://docs.astral.sh/uv/):

```bash
pip install eigenkit
```

Or, if you are using a uv-managed project:

```bash
uv add eigenkit
```

The core installation is lightweight, requiring only NumPy, SciPy, and
Pydantic. It includes everything needed to build and execute quantum circuits
on the built-in NumPy simulator, as well as the framework-independent
optimization and quantum chemistry modules.

Additional integrations, including PyTorch, scikit-learn, and external quantum
computing backends, are available as optional dependencies. This allows you to
install only the components you need.

### Optional extras

Extras go in square brackets, and several can be combined
(`uv add` takes the same form):

```bash
pip install "eigenkit[qiskit,braket]"
pip install "eigenkit[qml]"     # torch-free kernel QML (QSVC/QSVR)
pip install "eigenkit[torch]"   # differentiable hybrid models (HQNC/HQNR)
pip install "eigenkit[all]"
```

| Extra         | Adds                                | Unlocks                                                              |
| ------------- | ----------------------------------- | ------------------------------------------------------------------- |
| `torch`       | PyTorch + scikit-learn              | `interface="torch"`, the layers, and the hybrid estimators (HQNC/…) |
| `pennylane`   | PennyLane + Lightning               | `provider="pennylane"`                                              |
| `qiskit`      | Qiskit + Aer + IBM Runtime          | `provider="qiskit"`, `"qiskit-aer"`, `"qiskit-aer-mps"`; IBM Runtime / QPU |
| `braket`      | Amazon Braket SDK                   | `provider="braket"` (local, cloud, AWS QPUs)                        |
| `iqm`         | iqm-client + its Qiskit adapter     | `provider="iqm"` (IQM Resonance and IQM's fake backends)             |
| `cudaquantum` | NVIDIA CUDA-Q                       | `CudaQuantumBackend` — opt-in, see below (Linux CPU + GPU; macOS Apple Silicon CPU) |
| `tensorcircuit` | tensorcircuit-ng + JAX            | `provider="tensorcircuit"`                                          |
| `qulacs`      | Qulacs                              | `provider="qulacs"`                                                 |
| `qsim`        | qsimcirq + Cirq                     | `provider="qsim"` (single-precision)                                |
| `perceval`    | Perceval (Quandela's photonic SDK)  | `eigenkit.photonics` (local simulation, Quandela Cloud)   |
| `qml`         | scikit-learn (no PyTorch)           | `QuantumKernel`, `QSVC`/`QSVR`                                      |
| `datasets`    | scikit-learn                        | `eigenkit.datasets`                                       |
| `benchmark`   | matplotlib + pandas + scikit-learn  | the comparison harness                                              |
| `chemistry`   | PySCF + OpenFermion (no PySCF on Windows) | real molecules (`molecular_hamiltonian`, `hydrogen_molecule`)  |
| `viz`         | matplotlib                          | `Circuit.draw_mpl()` (the text drawer needs nothing)                |
| `all`         | every extra above except `perceval` | —                                                                   |

Backend classes and their configurations are always available for import,
even when their optional dependencies are not installed. However, attempting to
use a backend without its required dependencies, such as
`Circuit(provider="qiskit")`, will raise an error with instructions on how to
install them. The same applies to PyTorch and scikit-learn features.

Two backends are **not registered automatically**: NVIDIA CUDA-Q, which
selects a process-wide execution target, and Qiskit Runtime, which can submit
billable jobs to IBM quantum hardware. You can pass either one's configuration
or a backend instance directly to `provider=`, for example
`provider=QiskitRuntimeConfig(...)`, with no registration. To use them by name
like any other backend, register an instance once:

```python
from eigenkit import CudaQuantumBackend, QiskitRuntimeBackend, register

register("cudaq", CudaQuantumBackend())
register("qiskit-runtime", QiskitRuntimeBackend())
```

All other backends are registered automatically when their dependencies are
available.

### Platform compatibility

Some backends have additional compatibility restrictions depending on the
Python version and operating system:

- **Python 3.14:** The `qulacs`, `qsim`, and `iqm` extras can be installed, but
  their corresponding SDKs are not included due to compatibility limitations.
  Qulacs and qsim do not yet provide Python 3.14 wheels, while IQM requires a
  Pydantic version that conflicts with Braket. To use these providers directly,
  we recommend Python 3.12 or 3.13. IQM hardware remains accessible through
  Braket on Python 3.14.
- **Windows:** The `chemistry` extra installs OpenFermion without PySCF,
  because PySCF does not support Windows. Real molecules
  (`molecular_hamiltonian`, `hydrogen_molecule`) need PySCF, so run them under
  WSL. The rest of `eigenkit.chemistry` works on Windows.
- **Linux ARM:** The `qsim` extra does not include `qsimcirq`, as compatible
  wheels are unavailable.
- **Linux ARM:** The `perceval` extra needs the system library `libatomic.so.1`,
  which minimal images such as Ubuntu Server leave out. Install it with
  `sudo apt install libatomic1`.
- **Linux ARM with PyTorch and Qiskit:** PyTorch, scikit-learn, CUDA-Q and
  PennyLane Lightning each bring their own copy of the OpenMP runtime. Loaded
  in one process together with Qiskit, for example with the `torch` and
  `qiskit` extras or with `all`, the last library to load fails with `cannot
  allocate memory in static TLS block`, whatever the import order. Set this
  variable before you start Python:

    ```bash
    export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=262144
    ```

- **Linux ARM with Python 3.14:** The `qiskit` extra does not include Qiskit
  Aer because compatible wheels are unavailable.

These limitations affect specific optional backends and do not prevent you from
using the built-in simulator or other compatible providers.

### Choosing a PyTorch build

The `torch` extra asks for any PyTorch release in `torch>=2.5,<3`, and installs
whichever build your package index serves by default. To get a specific build,
CPU-only on Linux for example, install PyTorch from its own index first;
eigenkit then uses the PyTorch that is already there:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "eigenkit[torch]"
```

In a uv project, point `torch` at PyTorch's index in your own `pyproject.toml`,
as [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
describes. Working on eigenkit itself is different: the repository routes PyTorch
to a CUDA index on Linux and Windows, and
[CONTRIBUTING.md](https://github.com/eigenorbit/eigenkit/blob/main/CONTRIBUTING.md)
explains how to swap in a CPU build.

## Your first circuit

A `Circuit` is built from an ordered list of *blocks* (or gate by gate, as
shown below) and an *observables* spec. By default it runs on the **numpy** interface and the built-in
**numpy** simulator — no extra required. The numpy circuit doesn't own weights;
you pass them at call time:

```python
import numpy as np
from eigenkit import Circuit, Encoding, Variational, all_z

# 1. Define blocks. Encoding maps classical features to rotations;
#    Variational adds depth * n_qubits trainable parameters.
# 2. Choose what to measure — all_z(4) measures Z on every qubit.
circuit = Circuit(
    blocks=[Encoding(in_features=4, rotation="RX"), Variational(depth=2, rotation="RY")],
    observables=all_z(4),
)

# 3. Run: pass weights and inputs explicitly.
weights = np.zeros(circuit.spec.n_weights)
x = np.array([0.1, 0.2, 0.3, 0.4])
y = circuit(weights, x)
print(y.shape)  # (4,)
```

The qubit count is inferred from `Encoding.in_features` (or from the
`in_features` of a `Reuploading` block). The output dimension equals the
number of observables.

Both blocks above pass `rotation` explicitly; left out, it is `"RY"` on each.

!!! note "Where the names come from"

    Everything in the top-level `__all__` is imported from `eigenkit`
    itself: `Circuit`, the blocks, the observable helpers, the backend
    configs, the estimators, `CircuitBuilder`, `QuantumKernel`, and the
    solvers `HHL`, `IQAE` and `PortfolioRiskEstimator`. The domain packages
    keep their own names, so those are imported from the package they belong
    to — `eigenkit.optimization` for `QUBO`, `Ising`, `max_cut` and `QAOA`,
    `eigenkit.chemistry` for `VQE`, `eigenkit.benchmark` for `compare`,
    `eigenkit.datasets` for the datasets and their EDA reports, and
    `eigenkit.core.ir` for the parameter-expression nodes `Scale`, `Negate`,
    `Sum` and `Multiply`. Every page follows that.

### The same circuit, gate by gate

Instead of using predefined blocks, you can construct circuits gate by gate
with `CircuitBuilder`, following a similar approach to Qiskit, Braket, and
CUDA-Q.

The `add_weights()` and `add_inputs()` methods define symbolic parameters for
trainable weights and input data, respectively. You can then use these
parameters when adding gates to the circuit. Each gate method returns the
builder, allowing you to chain operations together.

Once the circuit is complete, `build()` converts it into a `CircuitSpec`, a
provider-neutral representation that can be executed by `Circuit` on any
supported backend:

```python
from eigenkit import CircuitBuilder

qc = CircuitBuilder(n_qubits=4)
params = qc.add_weights(8)
features = qc.add_inputs(4)

# Encoding(in_features=4, rotation="RX"): one RX per input.
for q in range(4):
    qc.rx(features[q], qubit=q)

# Variational(depth=2, rotation="RY"): per layer, an RY on every qubit,
# then a chain of CNOTs.
for layer in range(2):
    for q in range(4):
        qc.ry(params[layer * 4 + q], qubit=q)
    for q in range(3):
        qc.cnot(q, q + 1)

by_hand = Circuit(qc.build(), observables=all_z(4))
print(by_hand.spec == circuit.spec)  # True - the very same circuit
print(by_hand.draw())
```

```text
q_0: ──RX(x_0)──RY(w_0)──●──RY(w_4)──────────────●───────────
                         │                       │
q_1: ──RX(x_1)──RY(w_1)──X─────●─────RY(w_5)─────X─────●─────
                               │                       │
q_2: ──RX(x_2)──RY(w_2)────────X────────●─────RY(w_6)──X──●──
                                        │                 │
q_3: ──RX(x_3)──RY(w_3)─────────────────X─────RY(w_7)─────X──
```

Both approaches produce equivalent circuits and can be used interchangeably in
the examples that follow. Predefined blocks offer a simpler way to build common
circuit architectures, particularly those combining data encoding and
variational layers. `CircuitBuilder`, on the other hand, provides more
flexibility by allowing you to define arbitrary gate sequences.

The builder also supports arithmetic expressions involving symbolic parameters,
such as `2 * math.pi * features[0]` or `params[0] * features[1]`, making it
easy to define parameterised gates.

You can also combine both approaches using `qc.to_block()`, which converts a
circuit built with `CircuitBuilder` into a reusable block.

For a complete list of supported gates and additional examples, see the
[Circuit Builder](user-guide/circuit-builder.md) guide.

### A differentiable PyTorch circuit

To integrate a quantum circuit into a PyTorch model, set `interface="torch"`.
This creates a `torch.nn.Module` with trainable parameters managed as
`nn.Parameter` objects.

The resulting module supports batched inputs and automatic differentiation
through backpropagation, allowing quantum circuits to be trained alongside
classical neural networks in hybrid models.

This functionality requires the optional `torch` dependencies, and it runs on the
built-in NumPy simulator, which is differentiable:

```python
import torch
from eigenkit import Circuit, Encoding, Variational, all_z

circuit = Circuit(
    blocks=[Encoding(in_features=4, rotation="RX"), Variational(depth=2)],
    observables=all_z(4),
    interface="torch",
)
y = circuit(torch.randn(3, 4))  # batch=3, features=4 -> torch.Size([3, 4])
```

### A variational ground-state loop (no PyTorch)

The numpy circuit plugs straight into the scipy-based variational spine:

```python
import numpy as np
from eigenkit import Circuit, Variational, Z, minimize_expectation

# Variational alone declares a single wire, and minimize_expectation takes
# exactly one observable — the Hamiltonian.
circuit = Circuit([Variational(depth=2, rotation="RY")], observables=Z(0))

# Start from a small random point — an all-zeros init sits on a symmetric
# saddle where many gradients vanish, which stalls gradient-based optimizers.
rng = np.random.default_rng(0)
x0 = rng.uniform(0, 2 * np.pi, size=circuit.spec.n_weights)
result = minimize_expectation(circuit, initial_parameters=x0)
print(result.optimal_value)
```

The `minimize_expectation` function also accepts several optional keyword
arguments to customise the optimization process:

- `optimizer`: The optimization method to use. Defaults to `"L-BFGS-B"`, but
  supports any method available in `scipy.optimize.minimize`, `"spsa"`, or a
  custom callable.
- `max_iterations` and `tol`: Control the maximum number of iterations and
  convergence tolerance. Both default to `None`, allowing the optimizer to use
  its own settings.
- `random_state`: Controls the random initialisation of parameters when
  `initial_parameters` is not provided. Defaults to `42` for reproducibility.
- `callback`: An optional function called during optimization. Defaults to
  `None`.

### Computing gradients without PyTorch

Quantum circuits can also be differentiated directly through the NumPy
interface, without requiring PyTorch. The `circuit.jacobian(weights, inputs)`
method computes the derivatives of expectation values with respect to the
trainable parameters, using analytic adjoint differentiation or
parameter-shift methods, depending on the backend.

The NumPy interface also provides methods for inspecting and executing
circuits, including `probabilities()`, `statevector()`,
`zero_state_probability()`, `zero_state_probabilities()`, `draw()`,
`draw_mpl()`, `submit()`, and `dry_run_info()`.

For more information about these methods, see the
[Interfaces](user-guide/interfaces.md) guide.

## Your first quantum layer

`eigenkit` provides ready-to-use quantum layers that simplify the integration
of quantum circuits into PyTorch models. These layers wrap differentiable
quantum circuits and can be used alongside classical layers to build hybrid
neural networks.

They require the optional `torch` dependencies, and they run on the built-in NumPy
simulator unless you pass `provider=`:

```python
import torch
from eigenkit import Basic

layer = Basic(n_qubits=4, depth=2)
out = layer(torch.randn(8, 4))
print(out.shape)  # torch.Size([8, 4])
```

The available layers are:

- `Basic` — one Encoding + one Variational + measurement.
- `Reupload` — standard data reuploading (optionally with trainable scale).
- `Ensemble` — N independent circuits over input chunks.
- `MinMaxNorm` — feature-wise min/max normalisation.

See the [User Guide](user-guide/encoding.md) for block-level control or
the [Examples](examples/basic-circuit.md) section for end-to-end models.

## Switching backends

The `provider` argument accepts three shapes:

```python
from eigenkit import Circuit, Encoding, Variational, PennyLaneConfig

blocks = [Encoding(in_features=2), Variational(depth=1)]

# 1. A registered name string ("numpy" is the default).
Circuit(blocks=blocks, provider="numpy")

# 2. A config dataclass — needs the matching extra installed.
Circuit(blocks=blocks, provider=PennyLaneConfig(device_name="lightning.qubit"))

# 3. A Backend instance (advanced) — used as it is.
from eigenkit import NumpySimulator, register

Circuit(blocks=blocks, provider=NumpySimulator())

# Registering an instance gives it a name you can reuse anywhere.
register("numpy-copy", NumpySimulator())
Circuit(blocks=blocks, provider="numpy-copy")
```

The built-in NumPy simulator is the default backend and is always available
without additional dependencies. Other backends become available automatically
when their corresponding extras are installed.

The only exceptions are CUDA-Q and Qiskit Runtime, which require explicit
configuration. You can use either backend by passing an instance directly to
`provider=`, or by registering it with `register()` under a custom name for
reuse.
