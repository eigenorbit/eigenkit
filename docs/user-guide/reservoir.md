# Quantum Reservoir Computing

Quantum reservoir computing (QRC) uses a quantum circuit as a fixed dynamical
system, called the **reservoir**, that is never trained. Only a classical model
on top of the features it produces, called the **readout**, is trained, and it
is inexpensive to train. The weights of the reservoir are sampled once and then
frozen, so all of the learning happens in the readout.

This page covers `eigenkit.reservoir`, the part of the library that uses only
NumPy and needs no other framework. It contains the driver, the state policies,
the feature extractors and a ridge-regression readout. Use these building blocks
when you want full control, for example to process the features in your own way
or to use your own readout from scikit-learn or PyTorch. The scikit-learn-style
estimators built on them, `QuantumReservoirClassifier` and
`QuantumReservoirRegressor`, are described on the
[quantum machine learning (QML)](qml.md) page.

Everything relies on `Circuit.statevector(weights, inputs, initial_state=)`.
The state is carried from one timestep to the next by reading the statevector
out and injecting it back as the initial state of the following step.

## The driver

`run_reservoir(circuit, policy, sequences, weights)` evolves the circuit over a
batch of sequences under a policy, and returns the trajectory of raw
statevectors. The driver does not compute features or fit a readout, so the
three layers can be combined freely: the driver first, then a feature
extractor, then a readout.

```python
import numpy as np
from eigenkit import Circuit, Encoding, Variational, all_z
from eigenkit.reservoir import (
    run_reservoir,
    LeakyCarry,
    expectation_features,
    LinearReadout,
)

circuit = Circuit(
    blocks=[Encoding(in_features=3, rotation="RY"), Variational(depth=3, rotation="RY")],
    observables=all_z(3),
    provider="numpy",
)
weights = np.random.default_rng(0).standard_normal(circuit.spec.n_weights)
sequences = np.random.default_rng(1).uniform(-1, 1, size=(8, 40, 3))

states = run_reservoir(
    circuit, LeakyCarry(alpha=0.5), sequences, weights
)  # (8, 40, 8) complex
feats = expectation_features(states, "Z0 + Z1 + 0.5 X0 X1")  # (8, 40, 1)
readout = LinearReadout(ridge=1e-4)
```

## State policies

A *policy* decides, at each timestep, which state the circuit starts from and
which inputs drive it. Two policies carry the quantum state from one timestep
to the next, so the timesteps run one after another. The third policy carries
no state, so all timesteps can run in parallel.

| Policy | Constructor | What it does | Memory | Execution |
|---|---|---|---|---|
| `Stateless` | `Stateless(window=1)` | Starts each step from `|0…0⟩` and encodes the last `window` inputs as rotation angles. | classical (the window) | **parallel**: all timesteps in one batched run |
| `Carry` | `Carry()` | Applies the circuit to the previous state, `ψ_t = U(x_t) ψ_{t-1}`. | quantum state | sequential |
| `LeakyCarry` | `LeakyCarry(alpha, reset=None)` | Mixes the carried state with a reset state and renormalises the result. | quantum, fading | sequential |

With the carry policies, the circuit takes `n_inputs = F` inputs per step,
where `F` is the number of features per step. With `Stateless`, it takes the
whole window, so the circuit must encode `n_inputs = window * F`. The driver
checks this and raises a clear error otherwise.

!!! note "Fading memory matters"
    A `Carry` policy preserves the norm of the state, but it tends to mix the
    history into a form that a linear readout cannot recover, so it often has
    little usable memory. A memory that fades over time is what makes a
    reservoir work in practice, so `LeakyCarry` with `alpha` between `0.3` and
    `0.7` is usually the right starting point for temporal tasks. The
    [memory-capacity example](../examples/reservoir-memory.md) shows this.

## Feature extractors

The readout trains on real-valued features extracted from the trajectory of
statevectors. Each extractor takes the trajectory returned by the driver and
returns real-valued features for each timestep.

| Extractor | Features per step | Notes |
|---|---|---|
| `pauli_z_features(states)` | `n_qubits` | The expectation value of `Z` on each qubit. This is the usual default in QRC; it is inexpensive and can be measured on hardware. |
| `probability_features(states)` | `2**n_qubits` | The measurement distribution over all basis states. It carries more information, and it is quadratic in the amplitudes. |
| `amplitude_features(states)` | `2 * 2**n_qubits` | The real and imaginary parts of the state. Only a simulator can provide them; they cannot be measured on hardware. |
| `expectation_features(states, observables)` | one per observable | The expectation values of any Pauli observables. Pass the same `Observable` or `PauliExpr` objects, or Hamiltonian strings, that the rest of the library uses. A single Hamiltonian string is one observable. |

## Readout

`LinearReadout(ridge=1e-06)` performs ridge regression in closed form, with a
bias term and no iterative optimiser. It is only a convenience: the features
are a plain NumPy array, so any scikit-learn or PyTorch model can replace it.

## Backends and limitations

- Quantum reservoir computing runs on any backend that provides a statevector
  **and** accepts an injected initial state: NumPy, PennyLane, the Qiskit
  statevector simulator and the CUDA-Q simulators. Backends without access to
  the statevector, such as the Braket cloud services and Qiskit Runtime, raise a
  clear `NotImplementedError`. The default `provider="numpy"` needs no extra
  dependencies.
- The carry policies use state injection, which cannot be differentiated, so
  the reservoir is **frozen** and only the classical readout is trained. This
  is the standard setup in QRC, and it avoids computing gradients through the
  carried state.
- `LeakyCarry` approximates a dissipative reservoir with pure states. True
  dissipation needs density matrices, which the pure-state backends do not
  provide. The renormalised mixing used here approximates that behaviour, but
  it is not a physical quantum channel.

See the [short-term memory example](../examples/reservoir-memory.md) for a
worked benchmark that compares the policies.

## API reference

::: eigenkit.reservoir
