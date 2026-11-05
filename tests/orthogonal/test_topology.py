"""Gate layouts: counts against Cherrat et al. (Quantum 2024), Table 1, and validation."""

import math

import numpy as np
import pytest

from eigenkit.orthogonal import (
    Topology,
    brick,
    butterfly,
    butterfly_pairs,
    diagonal,
    orthogonal_matrix,
    pyramid,
    topology,
    x,
)


@pytest.mark.parametrize("n", [2, 4, 8, 64, 2048])
def test_butterfly_matches_table_1(n):
    t = butterfly(n)
    assert t.n_angles == (n // 2) * int(math.log2(n))
    assert t.depth == int(math.log2(n))
    assert not t.nearest_neighbour or n == 2


@pytest.mark.parametrize("n", [2, 3, 5, 8, 13])
def test_pyramid_matches_table_1(n):
    t = pyramid(n)
    assert t.n_angles == n * (n - 1) // 2
    assert t.depth == 2 * n - 3
    assert t.nearest_neighbour


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_x_matches_table_1(n):
    t = x(n)
    assert t.n_angles == 2 * n - 3
    assert t.depth == n - 1
    assert t.nearest_neighbour


@pytest.mark.parametrize("n", [2, 3, 8, 9])
def test_diagonal_is_one_gate_per_layer(n):
    t = diagonal(n)
    assert t.n_angles == n - 1
    assert t.depth == n - 1
    assert t.layers == tuple(((i, i + 1),) for i in range(n - 1))


@pytest.mark.parametrize(("n", "depth"), [(2, 3), (5, 4), (8, 6)])
def test_brick_alternates_even_and_odd_pairs(n, depth):
    t = brick(n, depth)
    assert t.depth == depth
    assert t.nearest_neighbour
    if n > 2:
        assert t.layers[0][0] == (0, 1)
        assert t.layers[1][0] == (1, 2)


def test_pyramid_layout_is_figure_7():
    t = pyramid(8)
    assert t.layers[0] == ((0, 1),)
    assert t.layers[1] == ((1, 2),)
    assert t.layers[6] == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert t.layers[7] == ((1, 2), (3, 4), (5, 6))
    assert t.layers[-1] == ((0, 1),)


def test_x_layout_is_figure_9():
    t = x(8)
    assert t.layers[0] == ((0, 1), (6, 7))
    assert t.layers[2] == ((2, 3), (4, 5))
    assert t.layers[3] == ((3, 4),)
    assert t.layers[6] == ((0, 1), (6, 7))


def test_x_layout_is_the_two_diagonals_of_the_pyramid():
    n = 8
    down = {(t, (t, t + 1)) for t in range(n - 1)}
    up = {(t, (n - 2 - t, n - 1 - t)) for t in range(n - 1)}
    gates = {(t, pair) for t, layer in enumerate(x(n).layers) for pair in layer}
    assert gates == down | up


@pytest.mark.parametrize(
    "t",
    [butterfly(16), pyramid(7), x(10), diagonal(6), brick(7, 5)],
    ids=lambda t: t.name,
)
def test_pairs_within_a_layer_are_disjoint_and_sorted(t):
    for layer in t.layers:
        wires = [w for pair in layer for w in pair]
        assert len(wires) == len(set(wires))
        assert list(layer) == sorted(layer)
    assert len(t.pairs()) == t.n_angles


@pytest.mark.parametrize("n", [2, 4, 8, 32])
def test_butterfly_levels_follow_the_reference(n):
    torch = pytest.importorskip("torch")
    from ._reference import butterfly_pairs as reference_pairs

    for level in range(int(math.log2(n))):
        i_idx, j_idx = reference_pairs(n, level)
        expected = tuple(zip(i_idx.tolist(), j_idx.tolist(), strict=True))
        assert butterfly_pairs(n, level) == expected
        assert i_idx.dtype == torch.long


@pytest.mark.parametrize("n", [0, 1, 3, 6, 12, 2.0, True])
def test_butterfly_rejects_widths_that_are_not_a_power_of_two(n):
    with pytest.raises(ValueError, match="power of two"):
        butterfly(n)


def test_butterfly_pairs_rejects_an_out_of_range_level():
    with pytest.raises(ValueError, match="level"):
        butterfly_pairs(8, 3)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_x_rejects_odd_widths_and_names_the_alternatives(n):
    with pytest.raises(ValueError, match="even width") as info:
        x(n)
    assert f"n={n}" in str(info.value)
    assert "pyramid(n)" in str(info.value)
    assert "diagonal(n)" in str(info.value)


@pytest.mark.parametrize("n", [3, 5])
def test_the_registry_rejects_an_odd_x_with_the_same_message(n):
    with pytest.raises(ValueError, match="pyramid"):
        topology("x", n)


@pytest.mark.parametrize("factory", [pyramid, diagonal, x])
def test_factories_reject_widths_below_two(factory):
    with pytest.raises(ValueError, match="at least 2"):
        factory(1)


def test_brick_rejects_a_depth_below_one():
    with pytest.raises(ValueError, match="depth"):
        brick(4, 0)


@pytest.mark.parametrize(
    ("layers", "message"),
    [
        ((((0, 1), (1, 2)),), "reuses a wire"),
        ((((1, 0),),), "0 <= i < j"),
        ((((0, 4),),), "0 <= i < j"),
        ((((2, 3), (0, 1)),), "sorted"),
        (((),), "empty"),
    ],
)
def test_a_custom_topology_is_validated(layers, message):
    with pytest.raises(ValueError, match=message):
        Topology(4, layers)


def test_a_custom_topology_accepts_lists():
    t = Topology(3, [[(0, 2)], [[0, 1]]], name="mine")
    assert t.layers == (((0, 2),), ((0, 1),))
    assert hash(t) == hash(Topology(3, (((0, 2),), ((0, 1),)), name="mine"))
    assert not t.nearest_neighbour


def test_the_registry_builds_every_named_layout():
    assert topology("butterfly", 8) == butterfly(8)
    assert topology("pyramid", 5, reflection=True) == pyramid(5, reflection=True)
    assert topology("x", 6) == x(6)
    assert topology("diagonal", 6) == diagonal(6)
    assert topology("brick", 6, depth=3) == brick(6, 3)


def test_the_registry_lists_the_known_layouts_for_an_unknown_name():
    with pytest.raises(ValueError, match="butterfly"):
        topology("spiral", 8)


@pytest.mark.parametrize("n", [2, 3, 6])
def test_pyramid_with_reflection_reaches_determinant_minus_one(n):
    rng = np.random.default_rng(n)
    plain, reflected = pyramid(n), pyramid(n, reflection=True)
    assert reflected.n_angles == plain.n_angles
    assert reflected.depth == plain.depth
    theta = rng.normal(size=plain.n_angles)
    assert np.linalg.det(orthogonal_matrix(plain, theta)) == pytest.approx(1.0)
    assert np.linalg.det(orthogonal_matrix(reflected, theta)) == pytest.approx(-1.0)


def test_numpy_integers_are_accepted_and_stored_as_int():
    w = np.int64(8)
    for t in (
        butterfly(w),
        pyramid(w),
        x(w),
        diagonal(np.int32(5)),
        brick(w, np.int64(3)),
    ):
        assert type(t.n) is int
    assert butterfly(w) == butterfly(8)
    assert brick(w, np.int64(3)) == brick(8, 3)
    assert butterfly_pairs(w, np.int64(1)) == butterfly_pairs(8, 1)
    assert Topology(np.int64(3), (((0, 1),),)).n == 3


@pytest.mark.parametrize("bad", [8.0, "8", np.bool_(True)])
def test_non_integers_are_rejected(bad):
    with pytest.raises(ValueError):
        pyramid(bad)


def test_brick_by_name_without_a_depth_says_what_is_missing():
    with pytest.raises(ValueError, match="depth"):
        topology("brick", 8)
    assert topology("brick", 8, depth=2) == brick(8, 2)


@pytest.mark.parametrize(
    "pair", [(0.5, 1.7), (0.0, 1.0), (np.float64(1.2), 2), (True, 2)]
)
def test_a_wire_that_is_not_an_integer_is_rejected(pair):
    with pytest.raises(ValueError, match="wire must be an integer"):
        Topology(3, ((pair,),))
