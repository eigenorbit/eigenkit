# Layers

Five `nn.Module` layers cover the usual patterns, and they can all be used inside
`nn.Sequential`. Most of these layers expose `in_features` and `out_features`.
`MinMaxNorm` uses `feature_dim`.

The four layers that build a circuit, `Basic`, `Reupload`, `Ensemble` and
`OrthogonalQNN`, take the same `provider=` argument as `Circuit`, and they pass it to the
circuit unchanged.
`MinMaxNorm` is classical arithmetic and has no circuit to run, so it does not take
the argument, and passing it raises a `TypeError`.

The same four layers return the provider on `.provider`, as the resolved backend,
whichever form was passed in. `.provider.name` is its registered name:

```python
from eigenkit import Basic, PennyLaneConfig

layer = Basic(n_qubits=4, depth=2, provider=PennyLaneConfig())
layer.provider.name  # "pennylane"
```

| Layer | Pattern |
|---|---|
| `Basic` | One encoding, one variational block, and a measurement of each qubit. |
| `Reupload` | Standard data reuploading, optionally with a trainable scale. |
| `Ensemble` | `N` independent circuits, one for each chunk of the input. |
| `OrthogonalQNN` | A unary loader, an orthogonal network of RBS gates, and the probability of each qubit. |
| `MinMaxNorm` | Min/max normalisation of each feature, in the style of batch normalisation. |

## `Basic`

`Basic` is the thinnest wrapper around `Circuit`: one encoding, one variational
block, and a measurement of each qubit. Use it for textbook variational classifiers
and regressors, where `n_features == n_qubits`.

```python
import torch
from eigenkit import Basic

# Defaults: RX encoding, RY variational, Z basis, which are three different axes.
layer = Basic(n_qubits=4, depth=2)
out = layer(torch.randn(8, 4))  # shape (8, 4)
```

![The circuit a Basic layer builds (light theme)](images/basic_light.svg){ .theme-image-light }
![The circuit a Basic layer builds (dark theme)](images/basic_dark.svg){ .theme-image-dark }

The circuit that this call builds. `RX(x_i)` encodes one feature per qubit. Then the
`basic` variational block, which is one `RY(θ)` per qubit followed by the ladder of
controlled-NOT (CNOT) gates, repeats `depth` times, and every qubit is measured in
`Z`. These three axes are the defaults. With `variational_type="strong"`, each
`RY(θ)` box becomes the three rotations `RZ`, `RY` and `RZ`.
{: .figure-caption }

!!! warning "Some combinations of axes make the circuit inert"

    Some choices of `enc_rotation`, `var_rotation` and `basis` build a circuit whose
    weights, or whose input, provably cannot reach the output. All three layers check
    for this and issue a `UserWarning` that names the reason. The layer is still
    built, because you may want to build a degenerate circuit on purpose, as a
    baseline.

    There are three ways to get one:

    - **The weights commute with the observable.** With `variational_type="basic"`
      and `var_rotation` on the measurement axis (`RZ` with `basis="Z"`, or `RX` with
      `basis="X"`), the rotations and the CNOT ladder both preserve that group of
      Pauli operators. The weights then cancel out of the expectation value, and
      every gradient is exactly `0`. Rotate away from the measurement axis, or use
      `variational_type="strong"`, whose three rotations `RZ`, `RY` and `RZ` always
      include one away from the axis.
    - **`RZ` encoding.** The circuit starts in `|0...0⟩`, which every `RZ` leaves
      unchanged up to a phase, so the input never enters. `Reupload` escapes this
      once it encodes again, because the second encoding acts on a state that has
      already moved. `Basic` and `Ensemble` encode only once, so they do not escape
      it.
    - **A real circuit measured in `Y`.** `RY` and CNOT are real matrices, so an `RY`
      encoding with `RY` rotations keeps the statevector real, and `⟨Y⟩` is always
      `0` on a real state.

    Sharing an axis is not a problem in itself. `RY` rotations with `basis="Y"` train
    well on two qubits or more, because the CNOT ladder does not preserve strings of
    `Y` operators.

## `Reupload`

`Reupload` is standard data reuploading in a single circuit. The input is encoded
again `depth` times, with new variational weights between the encodings. Set
`trainable_scale=True` to add a trainable scale for each pair of layer and feature
(Schuld et al. 2021).

Here, `in_features == n_qubits`.

```python
from eigenkit import Reupload

# Encoding on RX, variational on RY: same rule as Basic.
layer = Reupload(n_qubits=4, depth=3, enc_rotation="RX", trainable_scale=True)
out = layer(torch.randn(8, 4))  # shape (8, 4)
```

![The circuit a Reupload layer builds (light theme)](images/reupload_light.svg){ .theme-image-light }
![The circuit a Reupload layer builds (dark theme)](images/reupload_dark.svg){ .theme-image-dark }

What distinguishes this layer from `Basic` is where the outer bracket starts. It
covers the encoding as well as the variational block, so the input is encoded again
at each repetition, with new weights. The circuit is drawn with `enc_rotation="RX"`,
as in the call above, and without the trainable scale. With `trainable_scale=True`,
each `RX(x_i)` becomes `RX(λ·x_i)`, with one trainable `λ` for each pair of layer and
feature.
{: .figure-caption }

Keep `enc_rotation` and `var_rotation` on different axes here too. When they share
one, each `R(x)` and the `R(θ)` that follows it combine into a single `R(x + θ)`, and
the variational weights of that layer become a plain offset of the data angle. The
defaults of `Reupload` put both rotations on `RY`, so pass `enc_rotation`
explicitly, as in the call above.

## `Ensemble`

`Ensemble` runs `N` independent circuits, one for each chunk of the input, each with
its own weights. The outputs are concatenated along the feature dimension.

```python
from eigenkit import Ensemble

layer = Ensemble(in_features=20, n_qubits=4, depth=5)
out = layer(torch.randn(128, 20))  # shape (128, 20): 5 circuits * 4 qubits
```

![How an Ensemble layer routes its input (light theme)](images/ensemble_light.svg){ .theme-image-light }
![How an Ensemble layer routes its input (dark theme)](images/ensemble_dark.svg){ .theme-image-dark }

The input is split into chunks of `n_qubits` features, and each chunk gets its own
circuit, with its own weights and its own measurement. The outputs of the circuits
are concatenated back into one feature vector. Nothing is shared between the
branches: no statevector and no parameters. Four branches are drawn, while the call
above has five. Note also that on this layer, the variational block defaults to
`variational_type="strong"`, not `"basic"`.
{: .figure-caption }

## `OrthogonalQNN`

`OrthogonalQNN` is a quantum orthogonal layer. It loads each input vector \(x\)
into the unary basis, where exactly one qubit is 1, applies an orthogonal network
\(W(\theta)\) of RBS gates, and returns the probability of each qubit,
\((Wx)_i^2 / \lVert x \rVert^2\). Each output row sums to 1. The weights are the
angles of the network, one per gate, and its layout is a `topology` of
[`eigenkit.orthogonal`](orthogonal.md): `"pyramid"` (the default), `"butterfly"`,
`"x"`, `"diagonal"`, or a `Topology` of your own. `init="random"` (the default)
draws standard normal angles, and `init="identity"` sets them to zero, so that
\(W = I\).

```python
import torch
from eigenkit import OrthogonalQNN

layer = OrthogonalQNN(n_qubits=4, topology="pyramid")
out = layer(torch.randn(8, 4))  # shape (8, 4), each row sums to 1
```

The values and the gradient with respect to the weights come from the circuit.
The gradient with respect to the input is the exact derivative of
\((Wx)_i^2 / \lVert x \rVert^2\), also where \(x\) has zeros, as after a ReLU or
for a one-hot input, where the loader angles themselves are not differentiable.
The `loader` argument picks the layout of the loader: `"diagonal"`, the default,
works for every width and uses adjacent qubits only; `"semi_diagonal"` needs an
even width and `"parallel"` a power of two. A row of zeros has no direction to
load, so it gives the uniform distribution \(1/n\) on all \(n\) qubits, and no
gradient flows back to it; the other rows of the batch are not affected. The
circuit simulates \(2^n\) amplitudes, so the layer is practical up to about 20 qubits;
`eigenkit.orthogonal.torch.Orthogonal` computes the same matrix classically at any
width.

## `MinMaxNorm`

`MinMaxNorm` normalises each feature with its minimum and maximum, and keeps running
statistics. It is useful before encoding rotations, which expect bounded angles.

```python
import torch
from eigenkit import MinMaxNorm

norm = MinMaxNorm(feature_dim=16, min_val=0.0, max_val=torch.pi)
```

In training mode, it uses the minimum and maximum of each batch, and it updates an
exponential moving average of them (`momentum=0.1` by default). In evaluation mode, it
uses the running statistics. The output is clamped to `[min_val, max_val]`, to guard
against outliers that were not seen in training.

## API reference

::: eigenkit.torch.layers
