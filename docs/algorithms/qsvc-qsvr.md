# QSVC / QSVR: quantum kernel machines

## Start with the kernel trick

A support vector machine (SVM) trains from a table of similarities between pairs of
samples, `K(x_i, x_j)`, which is also called a kernel or Gram matrix. The quantum
support vector classifier (QSVC) and the quantum support vector regressor (QSVR)
change one thing: a quantum circuit computes this similarity. The fit step is still
the training of an SVM by scikit-learn. This is the idea of a quantum feature space
from [Havlíček et al.](https://arxiv.org/abs/1804.11326).

- `QSVC` does classification, and wraps `SVC` from scikit-learn.
- `QSVR` does regression, and wraps `SVR` from scikit-learn.

## Input and output

With the built-in feature maps, `QSVC.fit(X, y)` and `QSVR.fit(X, y)` use one qubit
per feature. A custom map passed with `kernel=QuantumKernel(...)` sets its own number
of qubits. `QSVC` returns class labels, and also probabilities when
`probability=True`. `QSVR` returns floats. You can also call
`QuantumKernel.compute_kernel_matrix(X)` directly, to inspect the values of the
fidelity.

## How it works

The circuit computes a **fidelity kernel**. A feature map `U(x)` encodes a sample as
the state `|φ(x)⟩ = U(x)|0⟩`. The kernel is the squared overlap of two encoded
states:

```text
K(x_i, x_j) = |⟨φ(x_j)|φ(x_i)⟩|²
```

On hardware and on the other backends that use shots, this value is read from an
inversion circuit:

1. It applies `U(x_i)`.
2. It applies `U(x_j)†`.
3. It reads the probability of returning to `|0…0⟩`, which is the value of the
   kernel.
4. It fills the Gram matrix, and passes it to `SVC` or `SVR` with
   `kernel="precomputed"`.

Simulators that give access to the statevector do not use the inversion circuit.
They prepare each `|φ(x)⟩` once, and take every overlap in one matrix product.

The two built-in feature maps are simple on purpose. `"Angle"` is an angle embedding
(`RX` rotations, with `RZ` rotations on the extra repetitions) with **no
entanglement**. `"AngleCNOT"` adds a fixed ladder of controlled-NOT (CNOT) gates
between neighbouring qubits. Any other map goes in through
`QSVC(kernel=QuantumKernel(feature_map=...))`: for example a ZZ map, an entangling
pattern that is native to the hardware, or a map with trainable weights tuned by
kernel alignment ([Glick et al.](https://arxiv.org/abs/2105.03406)). See the
[Quantum Kernels guide](../user-guide/kernels.md).

## Guarantees and limits

On the simulator, each entry of the kernel is the exact fidelity without noise,
`|⟨φ(x_j)|φ(x_i)⟩|²`. The matrix `K(X, X)` is symmetric, and its diagonal entries
equal `1.0` within floating-point precision. After that, the guarantee is the one
from scikit-learn: `SVC` and `SVR` solve their usual convex problem on the
precomputed matrix. This matches the view of kernel methods in
[Schuld and Killoran](https://arxiv.org/abs/1803.07128).

The learning is classical. The circuit only produces similarities. Real hardware
that uses shots would add sampling noise, which the exact path of the simulator
leaves out. The original paper uses a `ZZ` feature map. The built-in `"Angle"` map
is a plain angle embedding, and a ZZ map takes a few lines with
`QuantumKernel(feature_map=...)`. Very expressive maps can also suffer from the
concentration of kernel values ([Thanasilp et al.](https://arxiv.org/abs/2208.11060)).
On a statevector simulator, a Gram matrix costs `O(n_samples)` state evolutions and
one matrix product, and the state grows exponentially with the number of qubits. On
hardware, it costs `n_samples(n_samples - 1)/2` inversion circuits. Qiskit Runtime
and IQM send all of them in one job, and the other backends send them in smaller
calls.

## Next

The interfaces of the estimators are in the [QSVC reference](../api/qml/qsvc.md) and
the [QSVR reference](../api/qml/qsvr.md).

See [Quantum Classification](../examples/algorithms-classification.md) for `QSVC`,
[Quantum Regression & Time Series](../examples/algorithms-regression.md) for `QSVR`,
and the [Quantum Kernels guide](../user-guide/kernels.md) to inspect the kernel
matrix. The Iris example of hyperparameter optimisation uses `QSVC` inside
[GridSearchCV](../examples/hpo-qsvc-iris.md). The papers are listed in
[References](references.md#qsvc-qsvr-quantum-kernels).
