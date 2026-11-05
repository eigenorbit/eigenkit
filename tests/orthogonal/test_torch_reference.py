"""The torch modules against the reference implementation, in the same
process: every numerical contract (C1-C8, C12) must hold bit for bit."""

import math

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from eigenkit.orthogonal import orthogonal_matrix, topology  # noqa: E402
from eigenkit.orthogonal.torch import (  # noqa: E402
    Orthogonal,
    OrthogonalConv1x1,
    PartitionedOrthogonal,
    binary_parts,
    fold_bn,
    mixing_permutation,
    rotate_weight,
)

from . import _reference as ref  # noqa: E402


DTYPES = [torch.float32, torch.float64]
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.gpu),
]


def _require(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _random_angles(module, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.theta.copy_(torch.randn(module.theta.shape, generator=g) * scale)


def _pair(n, seed, dtype, device):
    old = ref.ButterflyOrthogonal(n, bias=True)
    new = OrthogonalConv1x1(n, bias=True)
    _random_angles(old, seed)
    with torch.no_grad():
        new.theta.copy_(old.theta)
        old.bias.copy_(torch.randn(n, generator=torch.Generator().manual_seed(seed + 1)))
        new.bias.copy_(old.bias)
    return old.to(device, dtype), new.to(device, dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [2, 8, 64, 512])
def test_the_butterfly_matrix_is_the_reference_matrix(n, dtype, device):
    _require(device)
    old, new = _pair(n, n, dtype, device)
    assert torch.equal(new.matrix(), old.matrix())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_conv_forward_is_the_reference_forward(dtype, device):
    _require(device)
    old, new = _pair(64, 3, dtype, device)
    x = torch.randn(3, 64, 5, 5, dtype=dtype, device=device)
    assert torch.equal(new(x), old(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_rotation_along_an_axis_is_the_partitioned_forward(dtype):
    old = ref.PartitionedButterfly(64, stages=1)
    new = Orthogonal(64)
    _random_angles(old.blocks[0][0], 5)
    with torch.no_grad():
        new.theta.copy_(old.blocks[0][0].theta)
    old, new = old.to(dtype), new.to(dtype)
    x2 = torch.randn(7, 64, dtype=dtype)
    x4 = torch.randn(2, 64, 3, 3, dtype=dtype)
    assert torch.equal(new(x2), old(x2))
    assert torch.equal(new(x4, dim=1), old(x4))
    assert torch.equal(new(x2.float()), x2.float() @ new.matrix().float().t())


@pytest.mark.parametrize("n", [8, 64, 2048])
@pytest.mark.parametrize("seed", [0, 7])
def test_random_init_draws_the_reference_stream(n, seed):
    torch.manual_seed(seed)
    old = ref.ButterflyOrthogonal(n)
    after_old = torch.randn(3)
    torch.manual_seed(seed)
    new = Orthogonal(n, init="random", std=0.1)
    after_new = torch.randn(3)
    assert torch.equal(new.theta, old.theta)
    assert torch.equal(after_new, after_old)


def test_identity_init_draws_no_random_numbers():
    state = torch.get_rng_state()
    module = PartitionedOrthogonal(588)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(module.matrix(), torch.eye(588))


def test_the_state_dict_holds_only_the_angles_and_the_bias():
    assert list(Orthogonal(8).state_dict()) == ["theta"]
    assert list(OrthogonalConv1x1(8).state_dict()) == ["theta", "bias"]
    assert list(OrthogonalConv1x1(8, bias=False).state_dict()) == ["theta"]


def test_an_old_checkpoint_loads_without_renaming():
    old = ref.ButterflyOrthogonal(16, bias=True)
    new = OrthogonalConv1x1(16, bias=True)
    new.load_state_dict(old.state_dict())
    assert torch.equal(new.matrix(), old.matrix())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("width", "stages", "seed"), [(588, 2, 0), (12, 2, 3), (13, 3, 1)]
)
def test_the_partitioned_rotation_is_the_reference_one(
    width, stages, seed, dtype, device
):
    _require(device)
    old = ref.PartitionedButterfly(width, stages, perm_seed=seed)
    new = PartitionedOrthogonal(width, stages, perm_seed=seed)
    assert sorted(new.state_dict()) == sorted(old.state_dict())
    g = torch.Generator().manual_seed(width)
    with torch.no_grad():
        for so, sn in zip(old.blocks, new.blocks, strict=True):
            for bo, bn in zip(so, sn, strict=True):
                bo.theta.copy_(torch.randn(bo.theta.shape, generator=g))
                bn.theta.copy_(bo.theta)
    old, new = old.to(device, dtype), new.to(device, dtype)
    for s in range(stages - 1):
        assert torch.equal(getattr(new, f"perm_{s}"), getattr(old, f"perm_{s}"))
    assert torch.equal(new.matrix(), old.matrix())
    x2 = torch.randn(5, width, dtype=dtype, device=device)
    x4 = torch.randn(2, width, 3, 3, dtype=dtype, device=device)
    assert torch.equal(new(x2), old(x2))
    assert torch.equal(new(x4, dim=1), old(x4))
    assert new.orthogonality_error() == old.orthogonality_error()


def test_binary_parts_and_permutations_are_the_reference_ones():
    for width in (1, 12, 13, 588, 2048):
        assert binary_parts(width) == ref.binary_parts(width)
    for seed in (0, 1, 3001):
        assert torch.equal(
            mixing_permutation(588, seed), ref.mixing_permutation(588, seed)
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_rotate_weight_is_the_reference_product(dtype):
    R = ref.ButterflyOrthogonal(64)
    _random_angles(R, 4)
    R = R.matrix().detach().to(dtype)
    w_conv = torch.randn(64, 32, 3, 3, dtype=dtype)
    w_lin = torch.randn(64, 48, dtype=dtype)
    expected_conv = (R @ w_conv.reshape(w_conv.shape[0], -1)).reshape(w_conv.shape)
    assert torch.equal(rotate_weight(R, w_conv), expected_conv)
    assert torch.equal(rotate_weight(R, w_lin), R @ w_lin)


def test_fold_bn_is_the_reference_fold():
    conv = torch.nn.Conv2d(16, 24, 3, bias=False)
    bn = torch.nn.BatchNorm2d(24)
    with torch.no_grad():
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.normal_()
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.1, 1.1)
    w, b = fold_bn(conv, bn.eval())
    w_ref, b_ref = ref.fold_bn(conv, bn)
    assert w.dtype == b.dtype == torch.float64
    assert w.device.type == "cpu"
    assert torch.equal(w, w_ref) and torch.equal(b, b_ref)


def test_the_diagnostics_are_the_reference_ones():
    old, new = _pair(64, 11, torch.float32, "cpu")
    assert new.singular_values() == old.singular_values()
    assert new.orthogonality_error() == old.orthogonality_error()


@pytest.mark.parametrize(
    "t",
    [("butterfly", 16), ("pyramid", 7), ("x", 8), ("diagonal", 5)],
    ids=lambda t: f"{t[0]}{t[1]}",
)
def test_the_torch_matrix_matches_the_numpy_matrix(t):
    module = Orthogonal(t[1], t[0]).double()
    _random_angles(module, 2)
    expected = orthogonal_matrix(topology(*t), module.theta.detach().numpy())
    np.testing.assert_allclose(module.matrix().detach().numpy(), expected, atol=1e-12)


def test_the_angle_count_is_the_reference_count():
    for n in (2, 8, 64, 2048):
        assert Orthogonal(n).n_angles == ref.num_butterfly_angles(n)
        assert Orthogonal(n).n_angles == (n // 2) * int(math.log2(n))
