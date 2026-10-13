# QFT: the quantum Fourier transform

## The problem

Several algorithms need to read a number that is not stored as bits, but as a
*phase*: an angle spread across a register. Quantum phase estimation (QPE) is the
standard case. A unitary has left the answer as `e^{2πiφ}` on the clock qubits, and
`φ` must come out as an ordinary binary fraction.

The quantum Fourier transform (QFT) is the change of basis that does this. It is the
discrete Fourier transform applied to the `2ⁿ` amplitudes of a register, and it maps
a periodic pattern of phases onto the basis state that names its frequency. The
circuit below is the one by [Coppersmith](https://arxiv.org/abs/quant-ph/0201067), and
it is the core of [Shor's factoring algorithm](https://arxiv.org/abs/quant-ph/9508027),
which made it well known.

## Input and output

`QFTBlock(n_qubits, inverse=False)` is a block, not a solver. It takes no weights and
no inputs, and it only adds gates. You put it in a list of blocks like any other
block. `n_qubits` must be positive, and a width that is not positive raises
`ValueError` when the block is created.

```python
from eigenkit import assemble
from eigenkit.core.blocks import QFTBlock

spec = assemble([QFTBlock(n_qubits=3)])
print(spec.n_weights, spec.n_inputs)  # 0 0
```

`inverse=True` builds the adjoint, `QFT†`. This is the direction that phase
estimation actually uses: the forward transform writes a frequency into a phase, and
the inverse transform reads it back.

## How it works

On the computational basis, the transform is

```text
|k⟩  ↦  (1/√(2ⁿ)) Σ_{j=0}^{2ⁿ-1} e^{2πi jk/2ⁿ} |j⟩
```

As a circuit, it needs no unusual gates. For each qubit `k` in turn:

1. It applies `H` to qubit `k`, which turns its bit into a phase.
2. For every later qubit `j > k`, it applies a controlled phase rotation `CPHASE` of
   angle `2π/2^{j-k+1}`, so that each bit of lower order shifts the phase by its own
   weight.
3. When every qubit has been through step 1, it reverses the register with `SWAP`
   gates, because the construction produces the output bits in the opposite order.

This gives `n` Hadamard gates, `n(n-1)/2` controlled phases and `⌊n/2⌋` swaps. A
transform on three qubits has seven gates, and `assemble([QFTBlock(n_qubits=3)])`
builds exactly `H, CPHASE, CPHASE, H, CPHASE, H, SWAP`. The inverse runs the same
construction backwards, with the phase angles negated.

The cost is what makes the transform useful: `O(n²)` gates, against the
`O(2ⁿ log 2ⁿ)` operations that a classical fast Fourier transform (FFT) needs on the
same `2ⁿ` amplitudes. This speedup can only be used when the answer can be read from
a few measurements rather than from the whole transformed vector, which is exactly
the case of phase estimation. This construction comes from
[Kitaev 1995](https://arxiv.org/abs/quant-ph/9511026) and
[Cleve, Ekert, Macchiavello and Mosca 1998](https://arxiv.org/abs/quant-ph/9708016).

## Guarantees and limits

The transform is exact. It is a unitary built from exact gates, with no approximation
and no setting to tune. Every gate in it (`H`, `CPHASE` and `SWAP`) is a gate that
the intermediate representation (IR) of the circuit defines, so the transform
compiles on any backend whose gate set includes these three gates.

The first limit is that the QFT alone answers no question. It is a change of basis,
and what makes it useful is the algorithm around it, which here means `QPEBlock` and
the [Harrow-Hassidim-Lloyd algorithm (HHL)](hhl.md). `QPEBlock` uses `QFTBlock`
directly. The `"honest"` mode of HHL reaches the same gates through the helper
function on which `QFTBlock` is built, so the transform is shared and not written
twice.

The other limit is hardware. The number of gates grows quadratically with the width
of the register, while the phase angles shrink exponentially, so on a real device
the smallest rotations eventually fall below the resolution of the device.
[Barenco, Ekert, Suominen and Törmä](https://arxiv.org/abs/quant-ph/9601018) showed
that removing them costs little and can help under decoherence, which is why
approximate QFTs exist. This implementation keeps every rotation.

## Next

The block is in the [blocks reference](../api/core/blocks.md), together with
`QPEBlock`, which is its main user. [HHL](hhl.md) shows phase estimation in use. The
papers are listed in [References](references.md#cross-cutting-foundations).
