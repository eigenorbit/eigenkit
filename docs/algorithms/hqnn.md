# HQNN: hybrid quantum neural networks

## A family of estimators

In this library, a hybrid quantum neural network (HQNN) is an estimator in the style
of sklearn, built around a hybrid neural model. You call `fit`, `predict` and, for
classifiers, `predict_proba`. Inside the estimator, the data passes through
classical layers, a trainable quantum circuit, and a classical head. The name covers
several designs.

> An HQNN encodes the data, runs a trainable quantum circuit, measures it, and passes
> the result to a classical head. All of it is trained end-to-end, with one
> optimiser and a standard loss.

The quantum circuit gives the model a different way to shape the features. The
accuracy still depends on the data, on the training setup, and on a comparison with
classical baselines.

## The shared pipeline

The base estimators use the same structure: an optional linear adapter, a
`MinMaxNorm` layer that rescales the inputs into `[0, π]`, the quantum layer, a cast
to float32, and a linear head. The quantum layer encodes the data as rotation
angles, runs a variational circuit, and measures one Pauli expectation value per
qubit. The training is the usual PyTorch loop, with one optimiser that updates the
classical and the quantum parameters together.

## Input and output

`fit(X, y)` trains the model. `predict(X)` returns classical NumPy labels or floats.
`predict_proba(X)` returns a softmax over the logits of the classical head.
`train_losses_`, and `validation_losses_` when you give validation data, store the
loss curves.

## The catalogue

All of these estimators share the pipeline above. They differ in what surrounds the
quantum layer.

| Estimator | Task | What makes it different |
|---|---|---|
| `HQNC` / `HQNR` | They classify or regress tabular data. | They are the base hybrid models: the hybrid quantum neural network classifier (HQNC) and regressor (HQNR). |
| `RecurrentHQNC` / `RecurrentHQNR` | They handle sequences (3-D input). | A classical recurrent neural network (an RNN, a gated recurrent unit (GRU) or a long short-term memory (LSTM) network) encodes the sequence first. By default, its last time step goes to the quantum layer. |
| `CustomQuantumClassifier` / `CustomQuantumRegressor` | They classify or regress. | You give the circuit as an explicit list of blocks, together with the observables. |
| `GenericClassifier` / `GenericRegressor` | They classify or regress. | They wrap any `torch.nn.Module` in the sklearn interface. They are quantum only if you put a quantum layer inside. |
| `QuantumTimeSeriesClassifier` / `QuantumTimeSeriesRegressor` | They handle time series. | They are a purely classical wrapper with sliding windows around an inner estimator. |

Inside `HQNC` and `HQNR`, `layer_type` chooses the quantum feature extractor:

| `layer_type` | Quantum layer |
|---|---|
| `"Basic"` (default) | It is one `Encoding` block and one variational block. |
| `"Reupload"` | It uses data reuploading: the input is encoded again `depth` times, which gives a richer class of (Fourier) functions. |
| `"Ensemble"` | It splits the features into chunks the size of the number of qubits, runs one independent circuit on each chunk, and concatenates their outputs. |

`variational_type` sets the ansatz. `"basic"` uses one rotation and a ladder of
controlled-NOT (CNOT) gates per layer. `"strong"` uses `RZ`, `RY` and `RZ` rotations
and a ladder of CNOT gates per layer, and has more parameters.

## Guarantees and limits

You get an estimator that is compatible with scikit-learn, and whose classical and
quantum parameters are trained end-to-end by one optimiser. The output is classical.
On a statevector simulator evaluated analytically (with `shots=None` in the
configuration of the provider, which is the default), the expectation values of each
qubit are exact. The randomness comes from the initialisation of the weights and from the
order of the mini-batches.

Any claimed advantage has to be shown against a baseline that is a classical
network. Deep or wide variational circuits can reach barren plateaus, where the
gradients shrink and the learning stalls
([McClean et al.](https://arxiv.org/abs/1803.11173)). The encoding also limits what
the model can represent ([Schuld, Sweke, and Meyer](https://arxiv.org/abs/2008.08605)),
and `MinMaxNorm` clips the values that fall outside the fitted range.
`GenericClassifier` and `GenericRegressor` are quantum only if the module that you
pass contains a quantum layer. The simulation stays small, because the number of
features sets the number of qubits.

## Next

The interfaces of the estimators are in the
[HQNC reference](../api/torch/estimators/hqnc.md) and the
[HQNR reference](../api/torch/estimators/hqnr.md).

See [Quantum Classification](../examples/algorithms-classification.md) for `HQNC`,
[Quantum Regression & Time Series](../examples/algorithms-regression.md) for `HQNR`,
and [Generic Estimators](../examples/generic-estimators.md) for custom PyTorch
modules. Other HQNN examples focus on one dataset each:
[breast cancer](../examples/breast-cancer-hqnc.md),
[circles](../examples/circles-hqnc.md), and
[concrete strength](../examples/cement-hqnr.md). The papers are listed in
[References](references.md#hqnn-hybrid-quantum-neural-networks).
