# eigenkit

[![PyPI](https://img.shields.io/pypi/v/eigenkit)](https://pypi.org/project/eigenkit/)
[![Python](https://img.shields.io/pypi/pyversions/eigenkit)](https://pypi.org/project/eigenkit/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/eigenorbit/eigenkit/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-docs.kit.eigenorbit.com-informational)](https://docs.kit.eigenorbit.com)

**Build quantum algorithms once. Run them anywhere.**

eigenkit is an open-source Python library for quantum computing by
[EigenOrbit](https://eigenorbit.com). It separates the definition of a quantum
algorithm from the backend that executes it. A circuit is described once as
data: the gates, the qubits they act on, and parameters that are constants or
references to trainable weights and data inputs. Each provider compiles that
description into its own SDK's objects and runs it on a simulator or a quantum
processor, so moving an algorithm to another backend means changing the
`provider=` argument.

## Why move between backends

No backend suits every workload, so it is often worth trying several.

Simulators differ in what they are fast at. Which one finishes first depends on
the number of qubits, the structure of the circuit and how much entanglement it
builds, the batch size, and whether the job evaluates a circuit once or trains
a model over many batches. They also differ in which gates and parameter
expressions they accept, and in whether they sample shots or run on a GPU.

Quantum processors differ in qubit count, connectivity, native gates, coherence
and error rates, as well as in price and queue times. A circuit whose
interactions match a device's coupling map needs fewer routing gates there, so
the better device can change from one circuit to the next.

eigenkit keeps the algorithm the same on all of them, so trying another backend
means changing one argument.

## What's in it

- **One circuit, many backends.** Simulators: the built-in numpy simulator,
  PennyLane, Qiskit, Qiskit Aer (statevector and matrix-product-state), Qulacs,
  qsim, TensorCircuit, CUDA-Q and Amazon Braket's simulators. Hardware: IBM
  through Qiskit Runtime, IQM through IQM Resonance, and the QPUs on Amazon
  Braket. Photonic circuits run on Quandela's Perceval.
- **Gradients on every backend.** A provider's own differentiation is used
  where it has one: the adjoint method on the built-in simulator, Qulacs and
  TensorCircuit, and PennyLane's autodiff. The other simulators and the quantum
  hardware get the library's parameter-shift rule through the same interface.
  On hardware this is opt-in (`enable_gradients=True` in the device config),
  since one backward pass runs at least two jobs per weight.
- **Two ways to build a circuit.** Gate by gate with `CircuitBuilder`, or from
  ready-made blocks: encodings, variational layers, data re-uploading. Both
  produce the same circuit representation.
- **Fits the classical stack.** A circuit is a numpy function or a PyTorch
  `nn.Module`, and the ready-made models follow scikit-learn's `fit`/`predict`.
- **Algorithms.** Machine learning: hybrid neural networks, quantum kernels with
  QSVC/QSVR, reservoir computing. Optimization: QUBO/Ising, Max-Cut, QAOA.
  Chemistry: fermionic operators with the Jordan-Wigner and Bravyi-Kitaev
  mappings, UCCSD, VQE, and molecular Hamiltonians through PySCF and
  OpenFermion. Also HHL, iterative amplitude estimation (IQAE) and portfolio
  risk (VaR/CVaR).
- **Benchmarking.** `compare()` runs quantum and classical models on the same
  data over several seeds; `eigenkit.datasets` has loaders and synthetic
  generators.
- **A light core.** numpy, scipy and pydantic. PyTorch, scikit-learn and every
  quantum SDK are optional extras, imported on first use.
- **Guards on billable hardware.** Compiling for a paid device raises a
  `CostWarning`, and `dry_run_info()` previews a job before it is submitted.

The abstraction adds a few tens of microseconds per call against hand-written
code for the same SDK (measured on PennyLane, Qiskit and Qulacs). The expensive
native objects, such as a PennyLane QNode or a parametric Qiskit circuit, are
built once per circuit and reused on every call.

## Installation

eigenkit needs Python 3.12, 3.13 or 3.14.

```bash
pip install eigenkit
```

That alone builds circuits and runs them on the built-in numpy simulator
(`provider="numpy"`, the default), and includes the framework-free
`optimization` (QAOA) and `chemistry` (VQE, Jordan-Wigner / Bravyi-Kitaev)
domains. PyTorch, scikit-learn and the quantum-SDK backends are opt-in
**extras**:

```bash
pip install "eigenkit[qiskit,braket]"
pip install "eigenkit[qml]"     # torch-free kernel QML (QSVC/QSVR)
pip install "eigenkit[torch]"   # differentiable hybrid models (HQNC/HQNR)
pip install "eigenkit[all]"
```

| Extra         | Adds                                              | Unlocks                                                                |
| ------------- | ------------------------------------------------- | ---------------------------------------------------------------------- |
| `torch`       | PyTorch + scikit-learn (+ tqdm)                   | `interface="torch"`, the layers, and the hybrid estimators (HQNC/HQNR/...) |
| `pennylane`   | PennyLane + Lightning                             | `provider="pennylane"`                                                 |
| `qiskit`      | Qiskit + Aer + IBM Runtime                        | `provider="qiskit"`, `"qiskit-aer"`, `"qiskit-aer-mps"`; IBM Runtime / QPU |
| `braket`      | Amazon Braket SDK                                 | `provider="braket"` (local simulator, cloud, AWS QPUs)                 |
| `iqm`         | iqm-client + its Qiskit adapter                   | `provider="iqm"` (IQM Resonance direct, plus IQM's fake backends)      |
| `cudaquantum` | NVIDIA CUDA-Q                                     | `CudaQuantumBackend`, opt-in (Linux CPU + GPU; macOS Apple Silicon CPU) |
| `tensorcircuit` | tensorcircuit-ng + JAX                          | `provider="tensorcircuit"`                                             |
| `qulacs`      | Qulacs                                            | `provider="qulacs"`                                                    |
| `qsim`        | qsimcirq + Cirq                                   | `provider="qsim"` (single-precision, ~1e-6 residual)                   |
| `perceval`    | Perceval (Quandela's photonic SDK)                | `eigenkit.photonics` (local simulation, Quandela Cloud)                |
| `router`      | eigenorbit-router-client                          | `provider="router"`, `RouterConfig` (EigenOrbit Router: real QPUs with one EigenOrbit API key) |
| `qml`         | scikit-learn (torch-free)                         | `QuantumKernel`, `QSVC`/`QSVR`                                         |
| `datasets`    | scikit-learn                                      | `eigenkit.datasets`                                                    |
| `benchmark`   | matplotlib + pandas + scikit-learn                | the model comparison harness                                           |
| `chemistry`   | PySCF + OpenFermion (no PySCF on Windows)         | real molecules (`molecular_hamiltonian`, `hydrogen_molecule`)          |
| `viz`         | matplotlib                                        | `Circuit.draw_mpl()` (the text drawer needs nothing)                   |
| `all`         | every extra above except `perceval`               | —                                                                      |

A backend class and its config are always importable, but
`Circuit(provider="qiskit")` (and so on), or a torch-only or sklearn-only
symbol, fails with an install hint until the matching extra is present.

Two backends are **opt-in**: installing the package does not register a
provider name for them, because each needs a deliberate choice from you —
CUDA-Q sets a process-global target, and Qiskit Runtime submits billable jobs
to real IBM hardware. Register either yourself once, then use the name
anywhere:

```python
from eigenkit import CudaQuantumBackend, QiskitRuntimeBackend, register

register("cudaq", CudaQuantumBackend())
register("qiskit-runtime", QiskitRuntimeBackend())
```

Every other backend registers itself as soon as its package is importable.

CUDA-Q ships wheels for Linux (x86_64 and ARM64) and macOS Apple Silicon; the
latter runs the CPU-only `qpp-cpu` simulator. On Windows and macOS x86_64 there
are no wheels, so the extra installs nothing there and the backend raises a
clear error if you try to use it.

PySCF does not support Windows, so there the `chemistry` extra installs
OpenFermion alone, and real molecules need WSL. Linux ARM has a few notes of
its own; the Getting started guide lists them under "Platform compatibility".

## Quick start

A Bell pair, built gate by gate and measured on the built-in simulator:

```python
from eigenkit import Circuit, CircuitBuilder, X, Z

qc = CircuitBuilder(n_qubits=2)
qc.h(0).cnot(0, 1)

bell = Circuit(qc.build(), observables=[Z(0) * Z(1), X(0) * X(1)])
print(bell([]))  # [1. 1.] - no trainable weights, so an empty weight vector
```

Running it elsewhere means passing a `provider`, for example
`Circuit(qc.build(), observables=..., provider="qulacs")`.

A larger circuit follows the same pattern. This IQP circuit (Hadamards, a layer
of ZZ rotations, Hadamards) is built by a helper function and run on three
simulators:

```python
from eigenkit import Circuit, CircuitBuilder, Z


def iqp(n, edges, angles):
    """H on every qubit, a ZZ rotation per edge, H on every qubit."""
    qc = CircuitBuilder(n_qubits=n)
    for q in range(n):
        qc.h(q)
    for (i, j), angle in zip(edges, angles):
        qc.pauli_exponential((("Z", i), ("Z", j)), angle)
    for q in range(n):
        qc.h(q)
    return qc.build()


spec = iqp(4, [(0, 1), (1, 2), (2, 3), (3, 0)], [0.7, 1.1, 0.4, 1.9])
observables = [Z(0) * Z(1), Z(1) * Z(2), Z(0) * Z(1) * Z(2) * Z(3)]

for provider in ["numpy", "qulacs", "qiskit-aer"]:
    circuit = Circuit(spec, observables=observables, provider=provider)
    print(provider, circuit([]))
```

```text
numpy [0.46548538 0.11841725 1.        ]
qulacs [0.46548538 0.11841725 1.        ]
qiskit-aer [0.46548538 0.11841725 1.        ]
```

`qulacs` and `qiskit-aer` need the `qulacs` and `qiskit` extras. On hardware the
provider is a device config, and the rest of the program stays the same. This
needs an IBM Quantum or IQM Resonance account, and compiling for a paid device
warns with a `CostWarning` first:

```python
from eigenkit import IQMConfig, QiskitRuntimeConfig

devices = {
    "IBM Heron": QiskitRuntimeConfig(
        backend_name="ibm_kingston", shots=8192, primitive="sampler"
    ),
    "IQM Emerald": IQMConfig(quantum_computer="emerald", shots=8192),
}
for name, device in devices.items():
    circuit = Circuit(spec, observables=observables, provider=device)
    print(name, circuit([]))
```

Ready-made blocks cover the common shapes, and `jacobian()` returns the
gradient with respect to the weights:

```python
import numpy as np
from eigenkit import Circuit, Encoding, Variational, all_z

circuit = Circuit(
    blocks=[Encoding(in_features=4), Variational(depth=2)],
    observables=all_z(4),
)

weights = np.zeros(circuit.spec.n_weights)
x = np.random.uniform(0, np.pi, size=(8, 4))  # a batch of 8 samples
y = circuit(weights, x)  # shape (8, 4): one expectation per qubit
grad = circuit.jacobian(weights, x[0])  # d<O>/d(weights), no torch needed
```

With `interface="torch"` (the `torch` extra) the same circuit is a
`torch.nn.Module` that owns its weights and trains inside a larger model.

## Documentation

The full documentation is at **[docs.kit.eigenorbit.com](https://docs.kit.eigenorbit.com)**:
a user guide per topic, the algorithms explained, worked examples, and the API
reference. Every example page also ships as a notebook in
[`examples/notebooks/`](https://github.com/eigenorbit/eigenkit/tree/main/examples/notebooks).

## Contributing

Bug reports, ideas and pull requests are welcome.
[CONTRIBUTING.md](https://github.com/eigenorbit/eigenkit/blob/main/CONTRIBUTING.md)
covers the development setup, tests and conventions, and
[ARCHITECTURE.md](https://github.com/eigenorbit/eigenkit/blob/main/ARCHITECTURE.md)
is a tour of the codebase. Please report security issues privately, as
described in [SECURITY.md](https://github.com/eigenorbit/eigenkit/blob/main/SECURITY.md).

## Citing eigenkit

If eigenkit helps your research, please cite it. GitHub's **Cite this
repository** button gives the reference from
[CITATION.cff](https://github.com/eigenorbit/eigenkit/blob/main/CITATION.cff).

## License

eigenkit is licensed under the
[Apache License 2.0](https://github.com/eigenorbit/eigenkit/blob/main/LICENSE).
