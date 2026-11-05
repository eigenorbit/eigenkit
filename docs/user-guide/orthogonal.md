# Orthogonal Networks

An **orthogonal network** is a circuit of RBS gates. An RBS gate acts on two
wires \((i, j)\), \(i < j\): it rotates the states \(|01\rangle\) and \(|10\rangle\)
by an angle \(\theta\) and leaves \(|00\rangle\) and \(|11\rangle\) alone. On those
two states it is the Givens rotation

\[
\mathrm{RBS}(\theta) =
\begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}
\]

(Cherrat et al., Quantum 2024, Eq. 1). RBS gates keep the number of 1s fixed, so
on the **unary basis**, where exactly one wire is 1, a network of them acts as an
\(n \times n\) orthogonal matrix. The matrix stays orthogonal for every value of
the angles: there is no constraint to enforce, and the angles train with any
optimiser.

`eigenkit.orthogonal` uses only NumPy and needs no other framework. One layout
gives two views that agree exactly:

- the **matrix**, `orthogonal_matrix(topology, theta)`, a dense NumPy array;
- the **circuit**, the `RBSNetwork` block, which runs on every provider.

## Topologies

A `Topology` lists the pairs of wires that each layer of gates acts on. The
pairs of one layer are disjoint, so their gates run in parallel. The angles are
laid out layer by layer and, inside a layer, in the order of the pairs.

```python
from eigenkit.orthogonal import butterfly, pyramid, topology, x

t = butterfly(8)
t.n_angles, t.depth  # (12, 3)
pyramid(8).n_angles  # 28
topology("x", 8).depth  # 7
```

| Factory | Angles | Depth | Width \(n\) | Wires used |
|---|---|---|---|---|
| `butterfly(n)` | \((n/2)\log_2 n\) | \(\log_2 n\) | a power of two | any pair |
| `pyramid(n)` | \(n(n-1)/2\) | \(2n-3\) | any, at least 2 | adjacent |
| `x(n)` | \(2n-3\) | \(n-1\) | **even**, at least 2 | adjacent |
| `diagonal(n)` | \(n-1\) | \(n-1\) | any, at least 2 | adjacent |
| `brick(n, depth)` | about \((n/2)\cdot\mathrm{depth}\) | \(\mathrm{depth}\) | any, at least 2 | adjacent |

The butterfly, pyramid and X rows are the three quantum orthogonal layers of
Cherrat et al. (Table 1, Figs. 7-9). The pyramid comes from Landman et al.
(Quantum 2022) and reaches every rotation in \(\mathrm{SO}(n)\);
`pyramid(n, reflection=True)` adds a final Z gate on the last wire, which reaches
the matrices with determinant \(-1\) too, so all of \(\mathrm{O}(n)\). The brick
layout alternates the even pairs \((0, 1), (2, 3), \dots\) and the odd pairs
\((1, 2), (3, 4), \dots\), so its layers hold \(\lfloor n/2 \rfloor\) and
\(\lfloor (n-1)/2 \rfloor\) gates in turn; for \(n = 2\) every layer is the pair
\((0, 1)\). The diagonal layout is one diagonal of
X, which Cherrat et al. ran on hardware as the "\\ circuit". You can also build a
`Topology` from your own layers.

!!! note "X needs an even width"
    `x(n)` raises a `ValueError` for an odd \(n\). Cherrat et al. define the X
    layout only for even widths, and for an odd width no layout of adjacent
    gates with depth \(n - 1\) connects every input wire to every output wire:
    the two diagonals would need the middle wire in the same layer. For an odd
    width, use `pyramid(n)` or `diagonal(n)`, which accept any \(n\), or pad the
    width to an even number.

The butterfly needs a power-of-two width and never pads it: any other width
raises a `ValueError`.

## The matrix

`orthogonal_matrix(topology, theta)` starts from the identity and applies the
layers in order, updating the two rows of every pair of a layer at once. The
cost is \(O(\mathrm{depth}\cdot n^2)\). Zero angles give the identity, which
`identity_angles(topology)` returns.

```python
import numpy as np

from eigenkit.orthogonal import orthogonal_matrix, orthogonal_matrix_jacobian, x

t = x(6)
theta = np.random.default_rng(0).normal(size=t.n_angles)
W = orthogonal_matrix(t, theta)
np.allclose(W.T @ W, np.eye(6))  # True
dW = orthogonal_matrix_jacobian(t, theta)  # shape (n_angles, 6, 6)
```

## The circuit

`RBSNetwork(topology)` is a block, so it combines with the other blocks through
`Circuit` and `assemble`. It takes one trainable weight per gate. Each RBS gate
on wires \((i, j)\) is built from gates every backend runs:

\[
\mathrm{RBS}_{ij}(\theta) =
H_i \, \mathrm{CNOT}_{ij} \, H_j \, R_Y^{(i)}(-\theta) \, R_Y^{(j)}(\theta) \,
H_j \, \mathrm{CNOT}_{ij} \, H_i
\]

Each angle enters only `RY` rotations, so the two-term parameter-shift rule
gives its exact gradient.

Qubit 0 is the most significant bit of the statevector, so the unary state with
wire \(k\) set is the basis state \(2^{\,n-1-k}\). Started on wire \(c\), the
network leaves the amplitude \(W_{rc}\) on wire \(r\), on every provider.

To feed a vector in, put a `UnaryLoader` in front. It applies an X gate and then
\(n - 1\) RBS gates, and loads \(x / \lVert x \rVert\) into the unary basis (Johri
et al., 2021). Its angles are non-linear functions of \(x\), so the block takes
them as inputs, and `unary_loader_angles(x, layout)` computes them. The three
layouts are those of Cherrat et al., Fig. 14:

| `layout` | Depth | Width \(n\) | Wires used |
|---|---|---|---|
| `"parallel"` | \(\log_2 n\) | a power of two | any pair |
| `"diagonal"` | \(n - 1\) | any | adjacent |
| `"semi_diagonal"` | \(n / 2\) | even | adjacent |

`unary_readout(n)` gives the observables \(Z_0, \dots, Z_{n-1}\), and
`unary_probabilities` turns their values into the probability of each wire,

\[
p_i = \frac{1 - \langle Z_i \rangle}{2} = \frac{(Wx)_i^2}{\lVert x \rVert^2}.
\]

With zero angles \(W = I\), so this circuit returns \(x_i^2 / \lVert x \rVert^2\):

```python
import numpy as np

from eigenkit import Circuit
from eigenkit.orthogonal import (
    RBSNetwork,
    UnaryLoader,
    butterfly,
    unary_loader_angles,
    unary_probabilities,
    unary_readout,
)

t = butterfly(8)
circuit = Circuit(
    [UnaryLoader(8, layout="parallel"), RBSNetwork(t)],
    observables=unary_readout(8),
)
x = np.random.default_rng(1).normal(size=8)
theta = np.zeros(t.n_angles)
p = unary_probabilities(circuit(theta, unary_loader_angles(x, layout="parallel")))
```

### From trained angles to a circuit

`to_block(topology, theta)` fixes the angles as constants, and
`to_circuit(topology, theta, loader="diagonal", provider=...)` builds the whole
circuit: loader, network and readout. Its inputs are the loader angles, and it
runs on any provider, a simulator or hardware. To train the angles of the circuit
itself, as a layer of a PyTorch model, use `OrthogonalQNN` (see
[Layers](layers.md#orthogonalqnn)).

```python
import numpy as np

from eigenkit.orthogonal import (
    orthogonal_matrix,
    pyramid,
    to_circuit,
    unary_loader_angles,
    unary_probabilities,
)

t = pyramid(4)
theta = np.random.default_rng(0).normal(size=t.n_angles)
x = np.array([1.0, 2.0, -0.5, 0.3])
circuit = to_circuit(t, theta)
p = unary_probabilities(circuit(np.zeros(0), unary_loader_angles(x)))
np.allclose(p, (orthogonal_matrix(t, theta) @ x) ** 2 / (x @ x))  # True
```

## PyTorch modules

`eigenkit.orthogonal.torch` needs the `torch` extra. Its modules hold the angles
as an `nn.Parameter` and rebuild the matrix from them, so it is orthogonal at every
training step, with no projection and no penalty:

- `Orthogonal(n, topology="butterfly", init="identity")` provides `matrix()` and
  rotates an input along one axis, `forward(x, dim=-1)`: a batch `(B, n)`, or a
  feature map `(B, n, H, W)` with `dim=1`.
- `OrthogonalConv1x1(n, bias=True)` is an orthogonal 1x1 convolution.
- `PartitionedOrthogonal(width, stages=2)` takes any width: it splits it into
  powers of two (\(588 = 512 + 64 + 8 + 4\)), rotates each part, and mixes the
  parts with fixed permutations between stages.

`init="identity"` starts at the identity, and `init="random", std=0.1` draws
`torch.randn(n_angles) * std`. In eval mode with autograd off the matrix is
cached, and `freeze_matrix()`, or the `frozen()` context, fixes it so that
gradients reach only the input, as an input-space attack needs.
`singular_values()` and `orthogonality_error()` check the matrix in float64.

```python
import torch

from eigenkit.orthogonal.torch import Orthogonal

rotation = Orthogonal(8, "x", init="random", std=0.5)
x = torch.randn(4, 8)
torch.allclose(rotation(x).norm(dim=1), x.norm(dim=1))  # True
rotation.singular_values()  # (1.0, 1.0) to float32 precision, about 1e-7
```

The default `mode="materialise"` builds the dense matrix, which costs
\(O(\mathrm{depth}\cdot n^2)\), and it is refused with a `ValueError` when
\(\mathrm{depth}\cdot n^2 > 2^{31}\): a pyramid wider than 1024 needs
`mode="apply"`, which applies the rotations to the input layer by layer.

Each module documents the exact torch operations it runs, and the tests pin them
bit for bit. Changing one of them is a breaking change.

### Orthogonal fine-tuning

`OrthogonalAdapter(layer)` wraps a pretrained `nn.Linear` or `nn.Conv2d`, freezes
it, and trains only the angles of a rotation \(R\) of its weight: \(W' = RW\) on
the output channels (`side="out"`) or \(W' = WR^\top\) on the input channels
(`side="in"`). \(R\) starts at the identity, so the adapted model starts at the
pretrained function, and \(RW\) keeps every singular value of \(W\): the
Lipschitz constant of the layer never changes. With the butterfly layout, a width
that is not a power of two gets a `PartitionedOrthogonal` automatically.

```python
import torch
import torch.nn as nn

from eigenkit.orthogonal.torch import (
    apply_orthogonal_adapters,
    merge_adapters,
    trainable_parameters,
)

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10))
apply_orthogonal_adapters(model, ["0", "2"], topology="pyramid")
optimiser = torch.optim.Adam(trainable_parameters(model), lr=1e-2)
# ... train ...
merge_adapters(model)  # plain nn.Linear layers again
```

Train an adapted model with a training loop of your own, as above. The
estimators `GenericClassifier` and `GenericRegressor` re-initialise every layer
in `fit()` (see [Estimators](estimators.md)), which would replace the pretrained
weights with random ones, so an adapter refuses to be re-initialised and `fit()`
raises a `RuntimeError`.

`target_modules` takes qualified names or a predicate `(name, module) -> bool`.
`fold_bn(conv, bn)` folds an eval-mode BatchNorm into the convolution before it,
so that a convolution and its BatchNorm can be adapted as one layer.

### Certified robustness

An orthogonal network keeps the Lipschitz constant of the model known, and a
known Lipschitz constant gives a certificate. If the features in front of a
linear head with rows \(w_k\) are \(L\)-Lipschitz, and the logits are \(z\), no
input perturbation of L2 norm below

\[
r = \min_{j \ne p} \frac{z_p - z_j}{L \, \lVert w_p - w_j \rVert}
\]

can change the predicted class \(p\). `eigenkit.orthogonal.torch` has the pieces:

- `LLNHead(in_features, out_features)`, a linear head with unit-norm rows, and
  `MaxMin(pairing="adjacent" | "halves")`, a 1-Lipschitz activation that keeps the
  norm. The two pairings are different functions: `"adjacent"` sorts channels
  \(2k\) and \(2k + 1\) (GroupSort), `"halves"` pairs channel \(k\) with
  \(k + C/2\) (the MinMax of BRONet).
- `lipschitz_bound(model)`, an upper bound on \(L\) for an `nn.Sequential` of
  `Linear`, `Conv2d`, orthogonal and 1-Lipschitz layers. An orthogonal adapter
  leaves it unchanged. A convolution must pad with `"zeros"`, or with
  `"circular"` no wider than \(d\,(k - 1)/2\) on each axis, for dilation \(d\)
  and kernel size \(k\): wider circular padding, `"reflect"` and `"replicate"`
  feed a pixel to one kernel offset more than once and raise a `ValueError`.
- `certified_radius(logits, head_weight, L)`, which returns the prediction, the
  margin and the radius of every row, in float64; `certified_accuracy(correct,
  radius, eps)`, the fraction correct and certified beyond `eps`; and
  `cert_ce(logits, y, head_weight, L, eps)`, a cross-entropy that trains for a
  margin beyond `eps`.

```python
import torch
import torch.nn as nn

from eigenkit.orthogonal.torch import (
    LLNHead,
    MaxMin,
    OrthogonalConv1x1,
    cert_ce,
    certified_accuracy,
    certified_radius,
    lipschitz_bound,
)

features = nn.Sequential(OrthogonalConv1x1(16, init="random"), MaxMin(), nn.Flatten())
head = LLNHead(16 * 4 * 4, 10)
L = lipschitz_bound(features)  # 1.0: an orthogonal layer and MaxMin

x, y = torch.randn(32, 16, 4, 4), torch.randint(0, 10, (32,))
logits = head(features(x))
loss = cert_ce(logits, y, head.get_weight(), L, eps=0.1)
cert = certified_radius(logits, head.get_weight(), L)
certified_accuracy(cert.pred == y, cert.radius, eps=0.1)
```

## Limits

- The circuit simulates a statevector of \(2^n\) amplitudes, so it is practical
  up to about 20 wires. The matrix is the same map restricted to the unary basis
  and scales to thousands of wires.
- The matrix costs \(O(\mathrm{depth}\cdot n^2)\). For a pyramid the depth is
  \(2n - 3\), so a dense pyramid matrix grows as \(n^3\).
- A circuit measures the probabilities \((Wx)_i^2 / \lVert x \rVert^2\), not the
  signed values \((Wx)_i\).

## API reference

::: eigenkit.orthogonal
