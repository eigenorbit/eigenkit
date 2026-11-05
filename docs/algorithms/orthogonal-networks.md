# Orthogonal networks: matrices that stay orthogonal

## The problem

Many models need a weight matrix that is **orthogonal**, \(W^\top W = I\). An orthogonal
matrix keeps the length of every vector, so a layer built on it neither amplifies
nor shrinks its input: gradients do not explode or vanish through it, and its
Lipschitz constant is exactly 1. That last property is what certified robustness
needs, because a network whose Lipschitz constant is known can prove that no small
perturbation of an input changes its prediction.

Keeping a matrix orthogonal while it trains is the hard part. A gradient step on
\(W\) leaves the set of orthogonal matrices, so the usual fixes project \(W\) back
after every step, or add a penalty that only pulls it towards orthogonality.

An orthogonal network avoids the problem by construction. It writes \(W\) as a product
of **Givens rotations**, each of which turns two coordinates by an angle. Every
rotation is orthogonal for every angle, so the product is too, at every step, with
no projection and no penalty. The angles train with any optimiser.

The same product has a quantum meaning. A Givens rotation on two coordinates is the
**RBS gate** (reconfigurable beam splitter) on two qubits, restricted to the states
with exactly one qubit set. So an orthogonal network is also a quantum circuit, the
quantum orthogonal layer of Landman et al. and Cherrat et al., and the matrix is the
exact classical simulation of that circuit.

## Input and output

A **topology** fixes which pairs of wires the rotations act on. It gives the matrix,
a circuit and a PyTorch module, and the three agree exactly:

```python
import numpy as np

from eigenkit.orthogonal import pyramid, orthogonal_matrix, to_circuit

t = pyramid(4)
theta = np.random.default_rng(0).normal(size=t.n_angles)
W = orthogonal_matrix(t, theta)  # 4 x 4, orthogonal
circuit = to_circuit(t, theta)  # the same map as a circuit on 4 qubits
```

In PyTorch, `eigenkit.orthogonal.torch.Orthogonal` holds the angles as a parameter,
`OrthogonalAdapter` fine-tunes a pretrained layer by rotating its weight, and
`OrthogonalQNN` is the circuit as a trainable layer. The
[user guide](../user-guide/orthogonal.md) covers the interface.

## How it works

### Step 1: one rotation is one gate

On two wires \((i, j)\), \(i < j\), the RBS gate leaves \(|00\rangle\) and
\(|11\rangle\) alone and rotates the other two states. On \(|01\rangle\), \(|10\rangle\)
it is

\[
\mathrm{RBS}(\theta) =
\begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}
\]

(Cherrat et al., Eq. 1). Every RBS gate keeps the number of qubits set to 1. A
circuit of them therefore never leaves the **unary basis**, the \(n\) states
\(e_k\) with exactly qubit \(k\) set, and on that basis it acts as an
\(n \times n\) real matrix. The matrix is orthogonal because the circuit is
unitary. The amplitude of \(e_r\) after the circuit, started from \(e_c\), is
\(W_{rc}\).

eigenkit has no native RBS gate, so each one is built from gates that every backend
runs:

\[
\mathrm{RBS}_{ij}(\theta) =
H_i \, \mathrm{CNOT}_{ij} \, H_j \, R_Y^{(i)}(-\theta) \, R_Y^{(j)}(\theta) \,
H_j \, \mathrm{CNOT}_{ij} \, H_i
\]

Each angle enters only plain `RY` rotations, so the two-term parameter-shift rule
gives its exact gradient on hardware.

### Step 2: arrange the rotations

A layer of the network is a set of disjoint pairs, whose rotations commute and run
in parallel. The matrix is built from the identity, one layer at a time: each layer
replaces the two rows of every pair by their rotation. The arrangement decides the
cost, the hardware it needs, and which matrices it can reach:

| Topology | Angles | Depth | Width | Wires | Reaches |
|---|---|---|---|---|---|
| butterfly | \((n/2)\log_2 n\) | \(\log_2 n\) | a power of two | any pair | a subset of \(\mathrm{SO}(n)\) |
| pyramid | \(n(n-1)/2\) | \(2n-3\) | any | adjacent | all of \(\mathrm{SO}(n)\) |
| X | \(2n-3\) | \(n-1\) | even | adjacent | a subset |
| diagonal | \(n-1\) | \(n-1\) | any | adjacent | a subset |

The pyramid is the layer of Landman et al.; it is exactly as expressive as
\(\mathrm{SO}(n)\), and a final Z gate on the last qubit adds the matrices with
determinant \(-1\), so all of \(\mathrm{O}(n)\). The butterfly and the X layouts
are those of Cherrat et al. The butterfly follows the pattern of the Cooley-Tukey
fast Fourier transform: it reaches every output from every input in \(\log_2 n\)
layers, with only \((n/2)\log_2 n\) angles, but its gates join distant qubits. The
X layout is the two diagonals of the pyramid, which cross in the middle and share
one gate there. They share a gate only for an even width, and for an odd width no
layout of adjacent gates with depth \(n - 1\) connects every input to every
output, so `x(n)` takes even widths only. The diagonal layout, one diagonal of X,
is the "\\ circuit" that Cherrat et al. ran on hardware.

A width that is not a power of two can still use butterflies: `PartitionedOrthogonal`
splits it into powers of two (\(588 = 512 + 64 + 8 + 4\)), rotates each part, and
mixes the parts with fixed permutations between stages.

### Step 3: load the data and read it out

To compute \(Wx\) on a quantum computer, the vector has to be loaded into the
unary basis first. The **unary loader** of Johri et al. does this with one X gate
and \(n - 1\) RBS gates, whose angles are computed classically from \(x\). It
prepares \(x / \lVert x \rVert\), so the length of \(x\) is lost and only its
direction is loaded. Three layouts trade depth for connectivity: parallel (depth
\(\log_2 n\)), diagonal (depth \(n - 1\), adjacent qubits only) and semi-diagonal
(depth \(n / 2\)).

After the network, the state is \(\sum_i a_i e_i\) with \(a = Wx / \lVert x \rVert\),
and measuring \(Z\) on qubit \(i\) gives \(\langle Z_i \rangle = 1 - 2 a_i^2\). So
the probability of wire \(i\) is

\[
p_i = \frac{1 - \langle Z_i \rangle}{2} = \frac{(Wx)_i^2}{\lVert x \rVert^2}
\]

A circuit returns these squares, not the signed values \((Wx)_i\).

### Step 4: train classically, keep the guarantee

The matrix and the circuit are the same map, so the angles can be trained in the
matrix form, in \(O(\mathrm{depth}\cdot n^2)\) per step at any width, and then run
as a circuit with the trained angles fixed. The matrix form is what a classical
network uses as a layer.

An orthogonal rotation also fine-tunes a pretrained layer without touching its
spectrum. \(W' = RW\) has exactly the singular values of \(W\) for every
orthogonal \(R\), so an `OrthogonalAdapter` keeps the operator norm of every layer
it adapts. \(R\) starts at the identity, so fine-tuning starts at the pretrained
function, and `merge()` folds \(R\) into the weight when training ends.

Together with a 1-Lipschitz activation (MaxMin) and a head with unit-norm rows
(LLN), a known Lipschitz constant \(L\) of the features gives a certificate. For a
prediction \(p\) with logits \(z\) and head rows \(w_k\), no perturbation of L2
norm below

\[
r = \min_{j \ne p} \frac{z_p - z_j}{L \, \lVert w_p - w_j \rVert}
\]

can change the prediction (Tsuzuku et al.; Singla et al.). Because the adapter
keeps \(L\), the Lipschitz bound computed for the pretrained network still holds
after orthogonal fine-tuning, and the certificates of the fine-tuned network need
only its new logits. A free update such as LoRA changes \(L\), and the bound has to
be computed again.

## Guarantees and limits

The matrix is orthogonal for every value of the angles; the tests check
\(\max \lvert W^\top W - I \rvert\) to float64 precision for every layout.
Restricted to the unary basis
the circuit is the matrix exactly, and the tests check this on several backends. An
orthogonal adapter keeps every singular value of the weight to floating-point
rounding.

The limits come from the same structure:

- The circuit simulates \(2^n\) amplitudes, so it is practical up to about 20
  qubits. The matrix form scales to thousands of wires.
- The dense matrix of a pyramid costs \(O(n^3)\) to build. Above
  \(\mathrm{depth}\cdot n^2 = 2^{31}\) the modules refuse to build it with a
  `ValueError`; `mode="apply"` applies the rotations to the input instead.
- The loader loses the length of \(x\), and the readout loses the signs of \(Wx\).
- Every layout except the pyramid reaches only part of \(\mathrm{SO}(n)\), and none
  of them reaches a matrix with determinant \(-1\) without the final Z gate.
- The certificate is only as tight as \(L\). For a convolution with a
  \(k_h \times k_w\) kernel, `lipschitz_bound` multiplies the norm of the weight
  matrix by \(\sqrt{k_h k_w}\). That is an upper bound for zero padding and for
  circular padding up to \(d\,(k - 1)/2\) on each axis, for dilation \(d\) and
  kernel size \(k\), and it can be loose; padding that feeds a pixel to one kernel
  offset twice (wider circular, reflect, replicate) is refused.

## Next

The interface is in [Orthogonal Networks](../user-guide/orthogonal.md) and in the
`OrthogonalQNN` section of [Layers](../user-guide/layers.md). The papers are listed
in [References](references.md#orthogonal-networks).
