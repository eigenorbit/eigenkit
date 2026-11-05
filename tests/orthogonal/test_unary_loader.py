"""The unary loader: the three layouts of Cherrat et al., Fig. 14, load x / |x| with
n - 1 RBS gates, and the depths are log2(n), n - 1 and n / 2."""

import math

import numpy as np
import pytest

from eigenkit import Circuit
from eigenkit.core.blocks import assemble
from eigenkit.orthogonal import (
    UnaryLoader,
    loader_depth,
    unary_loader_angles,
    unary_probabilities,
    unary_readout,
)


CASES = [
    ("diagonal", 2),
    ("diagonal", 3),
    ("diagonal", 7),
    ("parallel", 2),
    ("parallel", 4),
    ("parallel", 8),
    ("semi_diagonal", 2),
    ("semi_diagonal", 6),
    ("semi_diagonal", 8),
]


def _unary(sv, n):
    return np.array([sv[2 ** (n - 1 - wire)] for wire in range(n)])


@pytest.mark.parametrize(("layout", "n"), CASES)
def test_the_loader_reproduces_the_normalised_vector(layout, n):
    rng = np.random.default_rng(n)
    circuit = Circuit(assemble([UnaryLoader(n, layout)]))
    for x in (rng.normal(size=n), -np.abs(rng.normal(size=n)), np.eye(n)[n - 1] * 3):
        sv = circuit.statevector(np.zeros(0), unary_loader_angles(x, layout))
        np.testing.assert_allclose(_unary(sv, n), x / np.linalg.norm(x), atol=1e-12)


@pytest.mark.parametrize(("layout", "n"), CASES)
def test_the_loader_uses_n_minus_1_rbs_gates(layout, n):
    spec = assemble([UnaryLoader(n, layout)])
    assert spec.n_inputs == n - 1
    assert spec.n_weights == 0
    assert sum(g.name == "RY" for g in spec.gates) == 2 * (n - 1)
    assert sum(g.name == "X" for g in spec.gates) == 1
    assert not any(g.name == "CRY" for g in spec.gates)


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_the_depths_follow_figure_14(n):
    assert loader_depth(n, "parallel") == int(math.log2(n))
    assert loader_depth(n, "diagonal") == n - 1
    assert loader_depth(n, "semi_diagonal") == n // 2


def test_the_layouts_place_the_gates_of_figure_14():
    def pairs(layout):
        spec = assemble([UnaryLoader(8, layout)])
        cnots = [g.qubits for g in spec.gates if g.name == "CNOT"]
        return [tuple(q) for q in cnots[::2]]

    assert pairs("parallel") == [(0, 4), (0, 2), (4, 6), (0, 1), (2, 3), (4, 5), (6, 7)]
    assert pairs("diagonal") == [(k, k + 1) for k in range(7)]
    assert pairs("semi_diagonal") == [
        (3, 4),
        (2, 3),
        (4, 5),
        (1, 2),
        (5, 6),
        (0, 1),
        (6, 7),
    ]


def test_a_batch_of_vectors_gives_a_batch_of_angles():
    xs = np.random.default_rng(0).normal(size=(5, 8))
    batch = unary_loader_angles(xs, "parallel")
    assert batch.shape == (5, 7)
    for row, x in zip(batch, xs, strict=True):
        np.testing.assert_array_equal(row, unary_loader_angles(x, "parallel"))


def test_the_readout_gives_the_squared_amplitudes():
    x = np.array([0.5, -1.0, 2.0, 0.25])
    circuit = Circuit(assemble([UnaryLoader(4)]), observables=unary_readout(4))
    p = unary_probabilities(circuit(np.zeros(0), unary_loader_angles(x)))
    np.testing.assert_allclose(p, x**2 / (x @ x), atol=1e-12)


def test_a_zero_vector_is_rejected():
    with pytest.raises(ValueError, match="zero vector"):
        unary_loader_angles(np.zeros(4))


@pytest.mark.parametrize(
    ("layout", "n", "message"),
    [
        ("parallel", 6, "power-of-two"),
        ("semi_diagonal", 5, "even width"),
        ("spiral", 4, "unknown loader layout"),
        ("diagonal", 1, "at least 2"),
    ],
)
def test_a_layout_that_does_not_fit_the_width_is_rejected(layout, n, message):
    with pytest.raises(ValueError, match=message):
        UnaryLoader(n, layout)
    with pytest.raises(ValueError, match=message):
        unary_loader_angles(np.ones(n), layout)


def test_the_loader_round_trips_through_a_dict():
    loader = UnaryLoader(8, "semi_diagonal")
    assert UnaryLoader.from_dict(loader.to_dict()) == loader


def test_the_torch_loader_accepts_an_integer_tensor_like_numpy():
    torch = pytest.importorskip("torch")
    from eigenkit.orthogonal.torch import unary_loader_angles as torch_angles

    angles = torch_angles(torch.tensor([1, 2, 3, 4]))
    assert angles.dtype == torch.float64
    np.testing.assert_allclose(
        angles.numpy(), unary_loader_angles([1, 2, 3, 4]), atol=1e-12
    )


@pytest.mark.parametrize("bad", [[1.0, np.nan, 3.0, 4.0], [1.0, np.inf, 0.0, 0.0]])
def test_non_finite_vectors_are_rejected(bad):
    with pytest.raises(ValueError, match="NaN or infinite"):
        unary_loader_angles(np.array(bad))
    torch = pytest.importorskip("torch")
    from eigenkit.orthogonal.torch import unary_loader_angles as torch_angles

    with pytest.raises(ValueError, match="NaN or infinite"):
        torch_angles(torch.tensor(bad))


X_SCALE = np.array([0.3, -1.2, 0.7, 2.0, -0.4, 0.9, 1.1, -0.6])


@pytest.mark.parametrize("layout", ["diagonal", "parallel", "semi_diagonal"])
@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e200, 1e300])
def test_the_angles_do_not_depend_on_the_scale(layout, scale):
    # Squaring these entries would overflow or underflow float64.
    expected = unary_loader_angles(X_SCALE, layout)
    got = unary_loader_angles(X_SCALE * scale, layout)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    torch = pytest.importorskip("torch")
    from eigenkit.orthogonal.torch import unary_loader_angles as torch_angles

    got = torch_angles(torch.tensor(X_SCALE * scale), layout).numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-25, 1e25])
def test_the_float32_torch_angles_do_not_depend_on_the_scale(scale):
    torch = pytest.importorskip("torch")
    from eigenkit.orthogonal.torch import unary_loader_angles as torch_angles

    x = torch.tensor(X_SCALE, dtype=torch.float32)
    torch.testing.assert_close(
        torch_angles(x * scale), torch_angles(x), rtol=0, atol=1e-6
    )
