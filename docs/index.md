# eigenkit

**Build quantum algorithms once. Run them anywhere.**

`eigenkit` is an open-source Python library for quantum computing that allows
users to define quantum algorithms once and execute them across different
simulators and quantum hardware through a common interface. Switching between
providers requires changing only the `provider` argument, without modifying the
algorithm itself. The library also provides a consistent interface for computing
gradients across backends.

`eigenkit` supports a wide range of applications, including quantum machine
learning, optimization, and quantum chemistry, with ready-to-use algorithms and
models. It integrates naturally with the classical scientific computing
ecosystem, including NumPy, PyTorch, and scikit-learn, making it easy to
incorporate quantum components into existing workflows.

The library includes a built-in simulator and supports multiple external
execution providers, including PennyLane, Qiskit, Amazon Braket, IQM, NVIDIA
CUDA-Q, TensorCircuit, Qulacs, and qsim, plus Quandela's Perceval for photonic
circuits. External dependencies are optional, allowing users to install only
the backends they need. For cloud-based quantum hardware,
`eigenkit` also provides cost warnings and execution previews to help users
estimate resource requirements before submitting jobs.

---

## Quick look

A circuit is written in one of two ways: from ready-made blocks, or gate by
gate with `CircuitBuilder`. Both produce the same provider-neutral
`CircuitSpec`, so everything after construction (backends, gradients, the
torch interface) works the same either way. By default a circuit runs on the
numpy interface and the built-in numpy simulator, with no extra dependencies.

=== "From blocks"

    Blocks cover the common shapes: an encoding of the input features followed
    by trainable layers. The numpy circuit owns no weights; you pass them per
    call:

    ```python
    import numpy as np
    from eigenkit import Circuit, Encoding, Variational, all_z

    circuit = Circuit(
        blocks=[
            Encoding(in_features=4, rotation="RX"),
            Variational(depth=2, rotation="RY"),
        ],
        observables=all_z(4),
    )

    weights = np.zeros(circuit.spec.n_weights)
    x = np.random.uniform(0, np.pi, size=(8, 4))  # batch of 8 samples
    y = circuit(weights, x)  # shape (8, 4) - one expectation per qubit
    grad = circuit.jacobian(weights, x[0])  # d<O>/d(weights), no torch
    ```

=== "Gate by gate"

    `CircuitBuilder` appends one gate per call, in the style of Qiskit, Braket
    and CUDA-Q, and returns itself so calls can be chained:

    ```python
    from eigenkit import Circuit, CircuitBuilder, X, Z

    qc = CircuitBuilder(n_qubits=2)
    qc.h(0).cnot(0, 1)  # a Bell pair

    bell = Circuit(qc.build(), observables=[Z(0) * Z(1), X(0) * X(1)])
    print(bell([]))  # [1. 1.] - no trainable weights, so an empty weight vector
    print(bell.probabilities([]))  # [0.5 0.  0.  0.5]
    ```

    Trainable angles and input slots come from `qc.add_weights(n)` and
    `qc.add_inputs(n)`, and gate angles accept arithmetic on them, such as
    `weights[0] * inputs[1]`. See [Circuit Builder](user-guide/circuit-builder.md).

For a differentiable `torch.nn.Module` that owns its weights, pass
`interface="torch"` (the `torch` extra):

```python
import torch
from eigenkit import Circuit, Encoding, Variational, all_z

layer = Circuit(
    blocks=[Encoding(in_features=4), Variational(depth=2)],
    observables=all_z(4),
    interface="torch",
)
y = layer(torch.randn(8, 4))  # shape (8, 4); backprops into a torch model
```

The `provider` argument is polymorphic: pass a registered name (`"numpy"`,
`"pennylane"`, `"qiskit"`, `"braket"`, … — see
[Providers](user-guide/providers.md) for the full list), a config
(`PennyLaneConfig(...)`, `QiskitConfig(...)`, `BraketConfig(...)`), or a Backend
instance.

---

## What's in the box

- **Circuit construction.** Define quantum circuits using reusable blocks, build
  them gate by gate with `CircuitBuilder`, or combine both approaches using
  `to_block()`. Add observables and execute circuits on any supported backend.
- **High-level quantum layers.** Use ready-made layers such as `Basic`,
  `Reupload`, `Ensemble`, and `MinMaxNorm`, which can be integrated directly into PyTorch's `nn.Sequential`.
- **Scikit-learn estimators.** Ready-to-use classifiers and regressors with a
  familiar scikit-learn interface, including `HQNC`, `HQNR`, `RecurrentHQNC`,
  `RecurrentHQNR`, `QSVC`, `QSVR`, `CustomQuantumClassifier`,
  `CustomQuantumRegressor`, `GenericClassifier`, `GenericRegressor`,
  `QuantumTimeSeriesClassifier`, `QuantumTimeSeriesRegressor`,
  `QuantumReservoirClassifier`, and `QuantumReservoirRegressor`.
- **Quantum algorithms.** Implementations of `HHL` for linear systems, `IQAE`
  for amplitude estimation, and `PortfolioRiskEstimator` for Value at Risk (VaR)
  and Conditional Value at Risk (CVaR). Available in `eigenkit.algorithms`,
  these solvers follow a simple configure-and-run workflow and return immutable
  results rather than using a `fit`/`predict` interface.
- **Optimization.** Tools for formulating `QUBO` and `Ising` problems, solving
  Max-Cut problems with `max_cut`, and running `QAOA`. Available in
  `eigenkit.optimization`, with NumPy and SciPy as the only numerical
  dependencies.
- **Quantum chemistry.** Variational ground-state calculations with `VQE`,
  fermionic operators, Jordan-Wigner and Bravyi-Kitaev mappings, and UCCSD
  ansätze. Available in `eigenkit.chemistry` without requiring PyTorch.
  Molecular Hamiltonians (`molecular_hamiltonian`, `hydrogen_molecule`) require
  the optional `chemistry` dependencies.
- **Model persistence.** Save and load trained estimators as `.eigenkit`
  bundles, and export benchmark results for individual random seeds.
- **Datasets and benchmarking.** Built-in datasets, exploratory data analysis
  (EDA) reports, and multi-seed benchmarking tools for comparing quantum models
  against classical baselines.
- **Backend support.** Execute circuits using the built-in NumPy simulator or
  external providers, including PennyLane, Qiskit, Qiskit Aer (dense and MPS),
  Amazon Braket (local simulators, cloud simulators, and QPUs), Qiskit Runtime,
  IQM, NVIDIA CUDA-Q, TensorCircuit, Qulacs, and qsim. All backends use the same
  `provider=` interface. CUDA-Q and Qiskit Runtime require explicit
  configuration through a backend instance or a registered provider name.
- **Photonic computing.** Build and execute photonic circuits using Quandela's
  Perceval, either locally or through Quandela Cloud. Available in
  `eigenkit.photonics` with the optional `perceval` dependencies.

---

## Navigation

- [Getting Started](getting-started.md) — install and run your first circuit.
- User Guide — blocks and the gate-by-gate circuit builder, the numpy &
  torch interfaces, layers, observables, providers, quantum kernels,
  algorithms, QPU/cloud workflows.
- [Examples](examples/basic-circuit.md) — end-to-end worked examples.
- API Reference — auto-generated from source docstrings.
