"""The NumPy orthogonal matrix: orthogonality, identity at zero, the jacobian, and the
butterfly build of the reference implementation reproduced bit for bit."""

import math

import numpy as np
import pytest

from eigenkit.orthogonal import (
    Topology,
    brick,
    butterfly,
    diagonal,
    identity_angles,
    orthogonal_matrix,
    orthogonal_matrix_jacobian,
    pyramid,
    x,
)


TOPOLOGIES = [
    butterfly(2),
    butterfly(8),
    butterfly(64),
    pyramid(2),
    pyramid(7),
    pyramid(8, reflection=True),
    x(2),
    x(8),
    diagonal(5),
    brick(6, 4),
    Topology(4, (((0, 3),), ((0, 1), (2, 3)), ((1, 2),))),
]
IDS = [f"{t.name}{t.n}{'r' if t.reflection else ''}" for t in TOPOLOGIES]


@pytest.mark.parametrize("t", TOPOLOGIES, ids=IDS)
def test_the_matrix_is_orthogonal_for_any_angles(t):
    theta = np.random.default_rng(0).normal(scale=2.0, size=t.n_angles)
    W = orthogonal_matrix(t, theta)
    assert W.shape == (t.n, t.n)
    assert W.dtype == np.float64
    assert np.abs(W.T @ W - np.eye(t.n)).max() <= 1e-12


@pytest.mark.parametrize("t", TOPOLOGIES, ids=IDS)
def test_zero_angles_give_the_identity(t):
    W = orthogonal_matrix(t, identity_angles(t))
    expected = np.eye(t.n)
    if t.reflection:
        expected[-1, -1] = -1.0
    assert np.array_equal(W, expected)


def test_one_gate_is_the_rbs_rotation_of_equation_1():
    theta = 0.3
    W = orthogonal_matrix(Topology(2, (((0, 1),),)), [theta])
    c, s = math.cos(theta), math.sin(theta)
    assert np.array_equal(W, np.array([[c, s], [-s, c]]))


def test_layers_multiply_from_the_left():
    t = Topology(3, (((0, 1),), ((1, 2),)))
    theta = np.array([0.4, -1.1])
    first = orthogonal_matrix(Topology(3, (((0, 1),),)), theta[:1])
    second = orthogonal_matrix(Topology(3, (((1, 2),),)), theta[1:])
    np.testing.assert_allclose(orthogonal_matrix(t, theta), second @ first, atol=1e-15)


def test_float32_angles_give_a_float32_matrix():
    t = butterfly(16)
    theta = np.random.default_rng(1).normal(size=t.n_angles).astype(np.float32)
    W = orthogonal_matrix(t, theta)
    assert W.dtype == np.float32
    assert np.abs(W.T @ W - np.eye(16)).max() <= 1e-5


def test_integer_and_list_angles_are_read_as_float64():
    W = orthogonal_matrix(diagonal(3), [0, 1])
    assert W.dtype == np.float64


def test_the_wrong_number_of_angles_is_rejected():
    with pytest.raises(ValueError, match="needs 12 angles"):
        orthogonal_matrix(butterfly(8), np.zeros(11))


@pytest.mark.parametrize("t", TOPOLOGIES[:10], ids=IDS[:10])
def test_the_jacobian_matches_central_differences(t):
    if t.n > 16:
        pytest.skip("finite differences over every angle are slow at this width")
    theta = np.random.default_rng(2).normal(size=t.n_angles)
    jac = orthogonal_matrix_jacobian(t, theta)
    assert jac.shape == (t.n_angles, t.n, t.n)
    h = 1e-6
    for k in range(t.n_angles):
        step = np.zeros(t.n_angles)
        step[k] = h
        fd = (orthogonal_matrix(t, theta + step) - orthogonal_matrix(t, theta - step)) / (
            2 * h
        )
        np.testing.assert_allclose(jac[k], fd, atol=1e-8)


def test_the_jacobian_of_one_gate_is_the_derivative_of_the_rotation():
    theta = 0.7
    jac = orthogonal_matrix_jacobian(Topology(2, (((0, 1),),)), [theta])
    c, s = math.cos(theta), math.sin(theta)
    np.testing.assert_array_equal(jac[0], np.array([[-s, c], [-c, -s]]))


def _reference_butterfly(torch, n, scale):
    from ._reference import build_butterfly_matrix, butterfly_pairs

    generator = torch.Generator().manual_seed(n)
    theta = torch.randn(
        n // 2 * int(math.log2(n)), generator=generator, dtype=torch.float64
    )
    theta = theta * scale
    levels = [butterfly_pairs(n, level) for level in range(int(math.log2(n)))]
    return theta, build_butterfly_matrix(theta, n, levels).numpy()


@pytest.mark.parametrize("n", [2, 8, 64, 256])
@pytest.mark.parametrize("scale", [0.0, 0.1, 3.0])
def test_the_butterfly_build_repeats_the_reference_operation_for_operation(
    n, scale, monkeypatch
):
    # NumPy and torch round cos and sin differently in the last bit, so the order
    # of the arithmetic is checked with torch's cos and sin on the same slices.
    torch = pytest.importorskip("torch")
    theta, expected = _reference_butterfly(torch, n, scale)
    monkeypatch.setattr(np, "cos", lambda a: torch.cos(torch.from_numpy(a)).numpy())
    monkeypatch.setattr(np, "sin", lambda a: torch.sin(torch.from_numpy(a)).numpy())
    assert np.array_equal(orthogonal_matrix(butterfly(n), theta.numpy()), expected)


@pytest.mark.parametrize("n", [2, 8, 64, 256])
@pytest.mark.parametrize("scale", [0.0, 0.1, 3.0])
def test_the_butterfly_matrix_matches_the_reference(n, scale):
    torch = pytest.importorskip("torch")
    theta, expected = _reference_butterfly(torch, n, scale)
    W = orthogonal_matrix(butterfly(n), theta.numpy())
    np.testing.assert_allclose(W, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble, np.int32])
def test_angles_other_than_float32_give_a_float64_matrix(dtype):
    t = pyramid(4)
    theta = np.arange(t.n_angles).astype(dtype) / 7
    W = orthogonal_matrix(t, theta)
    assert W.dtype == np.float64
    assert np.array_equal(W, orthogonal_matrix(t, theta.astype(np.float64)))


def test_complex_angles_are_rejected():
    from eigenkit.orthogonal import to_block

    t = butterfly(4)
    theta = np.array([0.1 + 1j, 0, 0, 0])
    for build in (orthogonal_matrix, orthogonal_matrix_jacobian, to_block):
        with pytest.raises(ValueError, match="real"):
            build(t, theta)
