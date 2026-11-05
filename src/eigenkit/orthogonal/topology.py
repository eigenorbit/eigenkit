"""Gate layouts of orthogonal networks built from RBS (Givens) rotations.

A `Topology` says which pairs of wires each layer of RBS gates acts on. It holds
only data and uses no numerical library. The same topology gives a dense
orthogonal matrix (`orthogonal_matrix`), a quantum circuit and a PyTorch module,
so the three views always agree on the order of the gates and of the angles.

The pyramid, butterfly and X layouts are the three quantum orthogonal layers of
Cherrat et al., "Quantum Vision Transformers", Quantum 8, 1265 (2024), Figs. 7-9
and Table 1. The diagonal layout is one diagonal of the X circuit, which the same
paper ran on hardware as the "\\ circuit". The brick layout alternates even and
odd nearest-neighbour pairs.

Example:
    ```python
    from eigenkit.orthogonal import butterfly, pyramid, topology

    t = butterfly(8)
    t.n_angles, t.depth  # (12, 3)
    pyramid(8).n_angles  # 28
    topology("x", 8).depth  # 7
    ```
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Literal


TopologyName = Literal["butterfly", "pyramid", "x", "diagonal", "brick"]

Pair = tuple[int, int]
Layer = tuple[Pair, ...]


@dataclass(frozen=True)
class Topology:
    """The gate layout of an orthogonal network on `n` wires.

    Each layer is a tuple of wire pairs `(i, j)` with `i < j`. The pairs of one
    layer are disjoint, so their RBS gates commute and run in parallel. Within a
    layer the pairs are sorted by `i`. The angles of the network are laid out
    layer by layer and, inside a layer, in the order of the pairs, which is the
    order `orthogonal_matrix` and the circuit blocks read them in.

    `reflection=True` adds a final Z gate on the last wire. On the unary basis it
    negates the last row of the matrix, so the network reaches orthogonal
    matrices with determinant -1 (Cherrat et al., Appendix B).

    Attributes:
        n (int): The number of wires: qubits for the circuit, rows and columns of
            the matrix.
        layers (tuple[tuple[tuple[int, int], ...], ...]): The pairs of each layer,
            in the order the layers are applied. A wire may be any integer type and
            is stored as `int`; a float raises a `ValueError`.
        name (str): A label, used in `repr` and error messages.
        reflection (bool): Whether a final Z gate acts on the last wire.
            Default: `False`.
    """

    n: int
    layers: tuple[Layer, ...]
    name: str = "custom"
    reflection: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", _check_n(self.n))
        layers = tuple(
            tuple((_as_wire(i), _as_wire(j)) for i, j in layer) for layer in self.layers
        )
        object.__setattr__(self, "layers", layers)
        for depth, layer in enumerate(layers):
            if not layer:
                raise ValueError(f"layer {depth} is empty")
            seen: set[int] = set()
            for i, j in layer:
                if not 0 <= i < j < self.n:
                    raise ValueError(
                        f"layer {depth}: pair {(i, j)} needs 0 <= i < j < n={self.n}"
                    )
                if i in seen or j in seen:
                    raise ValueError(f"layer {depth}: pair {(i, j)} reuses a wire")
                seen.update((i, j))
            if list(layer) != sorted(layer):
                raise ValueError(
                    f"layer {depth}: pairs must be sorted by their first wire"
                )

    @property
    def n_angles(self) -> int:
        """The number of RBS gates, which is the number of angles."""
        return sum(len(layer) for layer in self.layers)

    @property
    def depth(self) -> int:
        """The number of layers of RBS gates. The final Z gate is not counted."""
        return len(self.layers)

    @property
    def nearest_neighbour(self) -> bool:
        """Whether every gate acts on adjacent wires `(i, i + 1)`."""
        return all(j == i + 1 for layer in self.layers for i, j in layer)

    def pairs(self) -> tuple[Pair, ...]:
        """Every pair, in the order of the angles."""
        return tuple(pair for layer in self.layers for pair in layer)


def _as_wire(value) -> int:
    # A wire is an integer: 0.5 is not wire 0, so a float is refused, not cut.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"a wire must be an integer, got {value!r}")
    return int(value)


def _as_int(value, name: str, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer of at least {minimum}, got {value!r}"
        )
    return int(value)


def _check_n(n: int) -> int:
    return _as_int(n, "n", 2)


def butterfly_pairs(n: int, level: int) -> Layer:
    r"""The pairs of one butterfly level: `(i, i | 2**level)` for every `i` without
    that bit, sorted by `i`.

    Args:
        n (int): The width, a power of two of at least 2.
        level (int): The level, from 0 to \(\log_2 n - 1\).

    Returns:
        tuple[tuple[int, int], ...]: The `n // 2` disjoint pairs of the level.

    Raises:
        ValueError: If `n` is not a power of two of at least 2, or `level` is out
            of range.
    """
    n = _check_power_of_two(n)
    m = n.bit_length() - 1
    if isinstance(level, bool) or not isinstance(level, numbers.Integral):
        raise ValueError(f"level must be an integer, got {level!r}")
    level = int(level)
    if not 0 <= level < m:
        raise ValueError(f"level must be in [0, {m}), got {level}")
    bit = 1 << level
    return tuple((i, i | bit) for i in range(n) if not i & bit)


def _check_power_of_two(n: int) -> int:
    if (
        isinstance(n, bool)
        or not isinstance(n, numbers.Integral)
        or n < 2
        or int(n) & (int(n) - 1)
    ):
        raise ValueError(
            f"butterfly width must be a power of two and at least 2, got {n!r}; "
            "widths are never padded"
        )
    return int(n)


def butterfly(n: int) -> Topology:
    r"""The butterfly layout of Cherrat et al. (Fig. 8), the Cooley-Tukey FFT pattern.

    Level \(l\) pairs every wire \(i\) with \(i \oplus 2^l\) (bitwise XOR). The
    levels run from 0 to \(\log_2 n - 1\), so the first level couples neighbours
    and the last one couples wires \(n / 2\) apart. Fig. 8 draws the levels in the
    opposite order; the two are the same network up to reversing the bits of the
    wire index.

    It has \((n / 2) \log_2 n\) angles and depth \(\log_2 n\), and needs gates between
    distant wires.

    Args:
        n (int): The width, a power of two of at least 2.

    Returns:
        Topology: The layout.

    Raises:
        ValueError: If `n` is not a power of two of at least 2.

    Example:
        ```python
        from eigenkit.orthogonal import butterfly

        butterfly(4).layers  # (((0, 1), (2, 3)), ((0, 2), (1, 3)))
        ```
    """
    n = _check_power_of_two(n)
    levels = n.bit_length() - 1
    return Topology(
        n, tuple(butterfly_pairs(n, level) for level in range(levels)), "butterfly"
    )


def pyramid(n: int, reflection: bool = False) -> Topology:
    r"""The pyramid layout of Landman et al. (Cherrat et al., Fig. 7).

    Layer \(t\) holds the pairs \((i, i + 1)\) with \(i \equiv t \pmod 2\) and
    \(i \le \min(t, 2n - 4 - t)\). The first gate acts on wires 0 and 1, and the
    apex of the pyramid is the gate on the last two wires.

    It has \(n(n - 1)/2\) angles and depth \(2n - 3\), uses only adjacent wires,
    and reaches every rotation in \(\mathrm{SO}(n)\). With `reflection=True` a
    final Z gate on the last wire extends it to all of \(\mathrm{O}(n)\).

    Args:
        n (int): The width, at least 2.
        reflection (bool): Whether to add the final Z gate on the last wire.
            Default: `False`.

    Returns:
        Topology: The layout.

    Raises:
        ValueError: If `n` is smaller than 2.
    """
    n = _check_n(n)
    layers = tuple(
        tuple((i, i + 1) for i in range(t % 2, min(t, 2 * n - 4 - t) + 1, 2))
        for t in range(2 * n - 3)
    )
    return Topology(n, layers, "pyramid", reflection=bool(reflection))


def x(n: int) -> Topology:
    r"""The X layout of Cherrat et al. (Fig. 9): the two diagonals of the pyramid.

    Layer \(t\) holds the pairs \((t, t + 1)\) and \((n - 2 - t, n - 1 - t)\). The
    two diagonals cross in the middle layer, where they share one gate.

    It has \(2n - 3\) angles and depth \(n - 1\), and uses only adjacent wires.

    `n` must be even. Cherrat et al. define the X layout only for even widths,
    and for odd `n` no nearest-neighbour layout of depth \(n - 1\) connects every
    input wire to every output wire: the two diagonals would need the middle wire
    in the same layer. For an odd width, use `pyramid` or `diagonal`, which
    accept any `n`, or pad the width to an even number yourself.

    Args:
        n (int): The width, an even number of at least 2.

    Returns:
        Topology: The layout.

    Raises:
        ValueError: If `n` is odd or smaller than 2.
    """
    n = _check_n(n)
    if n % 2:
        raise ValueError(
            f"the X layout is defined only for an even width, got n={n}. "
            "For an odd width use pyramid(n) or diagonal(n), which accept any n, "
            "or pad the width to an even number"
        )
    layers = tuple(
        tuple(sorted({(t, t + 1), (n - 2 - t, n - 1 - t)})) for t in range(n - 1)
    )
    return Topology(n, layers, "x")


def diagonal(n: int) -> Topology:
    r"""One diagonal of the X layout, the "\ circuit" of Cherrat et al.

    Layer \(t\) holds the single pair \((t, t + 1)\). It has \(n - 1\) angles and
    depth \(n - 1\), and uses only adjacent wires.

    Args:
        n (int): The width, at least 2.

    Returns:
        Topology: The layout.

    Raises:
        ValueError: If `n` is smaller than 2.
    """
    n = _check_n(n)
    return Topology(n, tuple(((t, t + 1),) for t in range(n - 1)), "diagonal")


def brick(n: int, depth: int) -> Topology:
    r"""The brick-wall layout: nearest-neighbour pairs, even and odd layers in turn.

    Layer \(t\) holds the pairs \((i, i + 1)\) with \(i \equiv t \pmod 2\). For
    \(n = 2\), which has no odd pair, every layer is the pair \((0, 1)\).

    Args:
        n (int): The width, at least 2.
        depth (int): The number of layers, at least 1.

    Returns:
        Topology: The layout.

    Raises:
        ValueError: If `n` is smaller than 2 or `depth` is smaller than 1.
    """
    n = _check_n(n)
    depth = _as_int(depth, "depth", 1)
    if n == 2:
        return Topology(n, (((0, 1),),) * depth, "brick")
    layers = tuple(
        tuple((i, i + 1) for i in range(t % 2, n - 1, 2)) for t in range(depth)
    )
    return Topology(n, layers, "brick")


_FACTORIES = {
    "butterfly": butterfly,
    "pyramid": pyramid,
    "x": x,
    "diagonal": diagonal,
    "brick": brick,
}


def topology(name: TopologyName, n: int, **kwargs) -> Topology:
    """Build a named layout.

    Args:
        name (TopologyName): One of `"butterfly"`, `"pyramid"`, `"x"`,
            `"diagonal"` and `"brick"`.
        n (int): The width.
        **kwargs (Any): Passed to the factory, such as `reflection` for the pyramid or
            `depth` for the brick layout.

    Returns:
        Topology: The layout.

    Raises:
        ValueError: If `name` is not a known layout, or `"brick"` comes without a
            `depth`.
    """
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"unknown topology {name!r}; choose one of {sorted(_FACTORIES)}"
        ) from None
    if name == "brick" and "depth" not in kwargs:
        raise ValueError(
            "the brick layout needs a depth: call topology('brick', n, depth=...), "
            "or pass the Topology that brick(n, depth) returns where a module takes "
            "a topology"
        )
    return factory(n, **kwargs)
