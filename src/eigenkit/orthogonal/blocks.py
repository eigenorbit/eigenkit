r"""Circuit blocks for orthogonal networks: the RBS network, the unary loader and the
unary readout.

An RBS gate is not in the gate vocabulary of the core, so each one is decomposed
into gates every backend runs (Landman et al., Quantum 2022):

\[
\mathrm{RBS}_{ij}(\theta) =
H_i \, \mathrm{CNOT}_{ij} \, H_j \, R_Y^{(i)}(-\theta) \, R_Y^{(j)}(\theta) \,
H_j \, \mathrm{CNOT}_{ij} \, H_i
\]

On the states \(|01\rangle\) and \(|10\rangle\) of the wires \((i, j)\), \(i < j\),
it is the Givens rotation of `orthogonal_matrix`, and every other basis state is
left as it is. Each angle enters only plain `RY` rotations, so the two-term
parameter-shift rule is exact for it, unlike a decomposition with `CRY`.

Qubit 0 is the most significant bit of the statevector, so the unary state
\(e_k\), with only wire \(k\) set, is the basis state \(2^{\,n-1-k}\). Started in
\(e_c\), an `RBSNetwork` leaves the amplitude \(W_{rc}\) on \(e_r\), where
`W = orthogonal_matrix(topology, theta)`.

Example:
    ```python
    import numpy as np

    from eigenkit import Circuit
    from eigenkit.orthogonal import (
        RBSNetwork,
        UnaryLoader,
        butterfly,
        unary_loader_angles,
        unary_readout,
    )

    t = butterfly(4)
    circuit = Circuit(
        [UnaryLoader(4, layout="parallel"), RBSNetwork(t)],
        observables=unary_readout(4),
    )
    x = np.array([0.5, -1.0, 2.0, 0.25])
    z = circuit(np.zeros(t.n_angles), unary_loader_angles(x, layout="parallel"))
    # (1 - z) / 2 == x**2 / |x|**2, because the zero angles give the identity.
    ```
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core.blocks import IRFragment
from ..core.builder import CircuitBuilder
from ..core.ir import Constant, Negate, Observable, Parameter
from .topology import Topology, _as_int


LoaderLayout = Literal["parallel", "diagonal", "semi_diagonal"]

_LAYOUTS = ("parallel", "diagonal", "semi_diagonal")


def _rbs(builder: CircuitBuilder, angle: Parameter, i: int, j: int) -> None:
    negated = Constant(-angle.value) if isinstance(angle, Constant) else Negate(angle)
    builder.h(i)
    builder.cnot(i, j)
    builder.h(j)
    builder.ry(negated, i)
    builder.ry(angle, j)
    builder.h(j)
    builder.cnot(i, j)
    builder.h(i)


def _fragment(builder: CircuitBuilder) -> IRFragment:
    spec = builder.build()
    return IRFragment(gates=spec.gates, n_weights=spec.n_weights, n_inputs=spec.n_inputs)


def _check_wires(name: str, needed: int, n_qubits: int) -> None:
    if n_qubits < needed:
        raise ValueError(f"{name} needs {needed} qubits, circuit has {n_qubits}")


@dataclass
class RBSNetwork:
    """An orthogonal network of RBS gates, as a circuit block on the wires `0` to
    `topology.n - 1`.

    The gates follow `topology.pairs()`, layer by layer, and gate `k` takes angle
    `k`. With `reflection=True` in the topology, a final Z gate acts on the last
    wire. Restricted to the unary basis the block is the matrix
    `orthogonal_matrix(topology, theta)`.

    Without `angles` the block takes `topology.n_angles` trainable weights. With
    `angles` it takes none, and the angles are fixed `Constant`s, which is how
    classically trained angles become a circuit (see `to_block`).

    Args:
        topology (Topology): The gate layout.
        angles (tuple[float, ...] | None): Fixed angles, one per gate, or `None` for
            trainable weights. Default: `None`.
    """

    topology: Topology
    angles: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.topology, Topology):
            raise TypeError(f"topology must be a Topology, got {type(self.topology)}")
        if self.angles is not None:
            self.angles = tuple(float(a) for a in self.angles)
            if len(self.angles) != self.topology.n_angles:
                raise ValueError(
                    f"{self.topology.name} on {self.topology.n} wires needs "
                    f"{self.topology.n_angles} angles, got {len(self.angles)}"
                )
            if not all(math.isfinite(a) for a in self.angles):
                raise ValueError("angles must be finite")

    def required_qubits(self) -> int:
        """Return `topology.n`, one qubit per wire."""
        return self.topology.n

    def build(self, n_qubits: int) -> IRFragment:
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled. The
                qubits from `topology.n` on are left unchanged.

        Returns:
            IRFragment: The gates, with local weight indices starting at `0`.
        """
        _check_wires("RBSNetwork", self.topology.n, n_qubits)
        builder = CircuitBuilder(n_qubits=n_qubits)
        if self.angles is None:
            params: list[Parameter] = list(builder.add_weights(self.topology.n_angles))
        else:
            params = [Constant(a) for a in self.angles]
        for angle, (i, j) in zip(params, self.topology.pairs(), strict=True):
            _rbs(builder, angle, i, j)
        if self.topology.reflection:
            builder.z(self.topology.n - 1)
        return _fragment(builder)

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        Returns:
            dict: The topology, the fixed angles (or `None`) and the `type`.
        """
        t = self.topology
        return {
            "type": "RBSNetwork",
            "n": t.n,
            "layers": [[list(pair) for pair in layer] for layer in t.layers],
            "name": t.name,
            "reflection": t.reflection,
            "angles": None if self.angles is None else list(self.angles),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RBSNetwork:
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            RBSNetwork: The rebuilt block.
        """
        topology = Topology(d["n"], d["layers"], d["name"], d["reflection"])
        angles = d.get("angles")
        return cls(topology, None if angles is None else tuple(angles))


@dataclass(frozen=True)
class _LoaderStep:
    owner: int
    receiver: int
    keep: tuple[int, ...]
    passed: tuple[int, ...]


def _loader_plan(n: int, layout: str) -> tuple[int, tuple[tuple[_LoaderStep, ...], ...]]:
    if layout == "diagonal":
        steps = tuple(
            (_LoaderStep(k, k + 1, (k,), tuple(range(k + 1, n))),) for k in range(n - 1)
        )
        return 0, steps
    if layout == "parallel":
        layers = []
        size = n
        while size > 1:
            half = size // 2
            layers.append(
                tuple(
                    _LoaderStep(
                        o,
                        o + half,
                        tuple(range(o, o + half)),
                        tuple(range(o + half, o + size)),
                    )
                    for o in range(0, n, size)
                )
            )
            size = half
        return 0, tuple(layers)
    m = n // 2 - 1
    layers = [(_LoaderStep(m, m + 1, tuple(range(m + 1)), tuple(range(m + 1, n))),)]
    for k in range(1, m + 1):
        up = _LoaderStep(m - k + 1, m - k, (m - k + 1,), tuple(range(m - k + 1)))
        down = _LoaderStep(m + k, m + k + 1, (m + k,), tuple(range(m + k + 1, n)))
        layers.append((up, down))
    return m, tuple(layers)


def _check_loader(n: int, layout: str) -> int:
    n = _as_int(n, "n", 2)
    if layout not in _LAYOUTS:
        raise ValueError(f"unknown loader layout {layout!r}; choose one of {_LAYOUTS}")
    if layout == "parallel" and n & (n - 1):
        raise ValueError(
            f"the parallel loader needs a power-of-two width, got n={n}; "
            "use layout='diagonal', which accepts any n"
        )
    if layout == "semi_diagonal" and n % 2:
        raise ValueError(
            f"the semi-diagonal loader needs an even width, got n={n}; "
            "use layout='diagonal', which accepts any n"
        )
    return n


def loader_depth(n: int, layout: LoaderLayout = "diagonal") -> int:
    r"""The number of layers of RBS gates of a unary loader, the X gate not counted.

    It is \(\log_2 n\) for `"parallel"`, \(n - 1\) for `"diagonal"` and \(n / 2\) for
    `"semi_diagonal"` (Cherrat et al., Fig. 14, count the X gate too).

    Args:
        n (int): The width.
        layout (LoaderLayout): The layout. Default: `"diagonal"`.

    Returns:
        int: The depth.
    """
    n = _check_loader(n, layout)
    return len(_loader_plan(n, layout)[1])


@dataclass
class UnaryLoader:
    r"""Load a vector \(x \in \mathbb{R}^n\) into the unary basis:

    \[
    |x\rangle = \sum_i \frac{x_i}{\lVert x \rVert} \, e_i
    \]

    The loader of Johri et al. (2021): an X gate, then \(n - 1\) RBS gates whose
    angles depend on \(x\). The three layouts are those of Cherrat et al., Fig. 14:

    - `"parallel"`: a binary tree from wire 0, depth \(\log_2 n\); `n` must be a
      power of two, and the gates act on distant wires;
    - `"diagonal"`: a chain from wire 0 to wire \(n - 1\), depth \(n - 1\); any
      `n`, adjacent wires only;
    - `"semi_diagonal"`: two chains from the middle, depth \(n / 2\); `n` must be
      even, adjacent wires only.

    The angles are non-linear functions of \(x\) that the `Parameter` tree cannot
    express, so the block takes them as \(n - 1\) inputs, computed by
    `unary_loader_angles(x, layout)`. It takes no weights.

    The estimators `CustomQuantumClassifier` and `CustomQuantumRegressor` do not
    take this block yet: they read the number of circuit inputs from an
    `Encoding` or `Reuploading` block only, so a circuit that loads its inputs
    with a `UnaryLoader` raises a `ValueError` in `fit()`, even with
    `in_features` given. Run such a circuit with `Circuit` or `TorchCircuit`, or
    use the `OrthogonalQNN` layer, which computes the loader angles itself.

    Args:
        n (int): The width, at least 2.
        layout (LoaderLayout): The gate layout. Default: `"diagonal"`.
    """

    n: int
    layout: LoaderLayout = "diagonal"

    def __post_init__(self) -> None:
        self.n = _check_loader(self.n, self.layout)

    def required_qubits(self) -> int:
        """Return `n`, one qubit per component of `x`."""
        return self.n

    def build(self, n_qubits: int) -> IRFragment:
        """Build the gates of this block for a circuit with `n_qubits` qubits.

        Args:
            n_qubits (int): The number of qubits of the circuit being assembled. The
                qubits from `n` on are left unchanged.

        Returns:
            IRFragment: The gates, with local input indices starting at `0`.
        """
        _check_wires("UnaryLoader", self.n, n_qubits)
        builder = CircuitBuilder(n_qubits=n_qubits)
        inputs = iter(builder.add_inputs(self.n - 1))
        start, layers = _loader_plan(self.n, self.layout)
        builder.x(start)
        for layer in layers:
            for step in layer:
                i, j = sorted((step.owner, step.receiver))
                _rbs(builder, next(inputs), i, j)
        return _fragment(builder)

    def to_dict(self) -> dict:
        """Return a description of this block that can be saved as JSON.

        Returns:
            dict: The width, the layout and the `type`.
        """
        return {"type": "UnaryLoader", "n": self.n, "layout": self.layout}

    @classmethod
    def from_dict(cls, d: dict) -> UnaryLoader:
        """Rebuild a block from the output of `to_dict()`.

        Args:
            d (dict): A dictionary returned by `to_dict()`.

        Returns:
            UnaryLoader: The rebuilt block.
        """
        return cls(n=d["n"], layout=d["layout"])


def unary_loader_angles(x, layout: LoaderLayout = "diagonal") -> np.ndarray:
    r"""The input angles of `UnaryLoader` that load \(x / \lVert x \rVert\).

    Each gate splits the amplitude on its owner wire between the wires it keeps
    and the wires it passes on. With \(v_S\) the signed value \(x_s\) when \(S\) is
    one wire \(s\) and the norm \(\lVert x_S \rVert\) otherwise, the angle is
    \(\operatorname{atan2}(-v_{\mathrm{pass}}, v_{\mathrm{keep}})\) when the amplitude
    moves to a higher wire and
    \(\operatorname{atan2}(v_{\mathrm{pass}}, v_{\mathrm{keep}})\) when it moves to a
    lower one.

    Args:
        x (array-like): One vector of shape `(n,)` or a batch of shape `(B, n)`. It
            does not need to be normalised, and its scale does not change the
            angles.
        layout (LoaderLayout): The loader layout. Default: `"diagonal"`.

    Returns:
        np.ndarray: The angles, of shape `(n - 1,)` or `(B, n - 1)`.

    Raises:
        ValueError: If a vector is zero or not finite, or the layout does not
            accept the width.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")
    batch = np.atleast_2d(x)
    n = batch.shape[1]
    _check_loader(n, layout)
    if not np.all(np.isfinite(batch)):
        raise ValueError("cannot load a vector with NaN or infinite entries")
    # Scale each row by a power of two that brings its largest entry into
    # [0.5, 1): exact, so the norm cannot overflow or underflow at any finite scale.
    _, exponent = np.frexp(np.max(np.abs(batch), axis=1, keepdims=True))
    batch = np.ldexp(batch, -exponent)
    norms = np.linalg.norm(batch, axis=1)
    if np.any(norms == 0):
        raise ValueError("cannot load a zero vector")
    unit = batch / norms[:, None]

    def value(wires):
        if len(wires) == 1:
            return unit[:, wires[0]]
        return np.linalg.norm(unit[:, list(wires)], axis=1)

    angles = []
    for layer in _loader_plan(n, layout)[1]:
        for step in layer:
            sign = -1.0 if step.receiver > step.owner else 1.0
            angles.append(np.arctan2(sign * value(step.passed), value(step.keep)))
    out = np.stack(angles, axis=1)
    return out[0] if x.ndim == 1 else out


def unary_readout(n: int) -> list[Observable]:
    r"""The observables \(Z_0, \dots, Z_{n-1}\) that read a unary state out.

    On a unary state \(\sum_i a_i e_i\), \(\langle Z_i \rangle = 1 - 2 a_i^2\), so the
    probability of wire \(i\) is \(p_i = (1 - \langle Z_i \rangle) / 2\) (see
    `unary_probabilities`). After a `UnaryLoader` and an `RBSNetwork`,
    \(p_i = (Wx)_i^2 / \lVert x \rVert^2\).

    Args:
        n (int): The number of wires.

    Returns:
        list[Observable]: One `Z` observable per wire.
    """
    n = _as_int(n, "n", 1)
    return [Observable.z(q) for q in range(n)]


def unary_probabilities(expectations) -> np.ndarray:
    r"""Convert the expectation values of `unary_readout` into probabilities.

    Args:
        expectations (array-like): The values \(\langle Z_i \rangle\), of any shape.

    Returns:
        np.ndarray: \((1 - \langle Z_i \rangle) / 2\), of the same shape.
    """
    return (1.0 - np.asarray(expectations, dtype=np.float64)) / 2.0
