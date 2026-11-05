"""`Orthogonal`, `OrthogonalConv1x1` and `PartitionedOrthogonal`: angle counts,
orthogonality and the partitioned rotation, plus caching, freezing, modes, shapes
and the lazy import."""

import math
import subprocess
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")

import eigenkit.orthogonal as orthogonal  # noqa: E402
from eigenkit.orthogonal import Topology, pyramid  # noqa: E402
from eigenkit.orthogonal.torch import (  # noqa: E402
    MAX_MATERIALISE,
    Orthogonal,
    OrthogonalAdapter,
    OrthogonalConv1x1,
    PartitionedOrthogonal,
    binary_parts,
)


WIDTHS = [8, 16, 32, 64, 128]
TOL = 1e-5


# ---- angle counts, orthogonality, gradients ----


@pytest.mark.parametrize("c", WIDTHS)
def test_angle_count_is_half_n_log2_n(c):
    assert OrthogonalConv1x1(c, bias=False).theta.numel() == (c // 2) * int(math.log2(c))


@pytest.mark.parametrize("c", WIDTHS)
@pytest.mark.parametrize("draw", range(3))
def test_orthogonal_at_init(c, draw):
    torch.manual_seed(draw)
    layer = OrthogonalConv1x1(c)
    with torch.no_grad():
        layer.theta.normal_(0.0, 2.0)
    assert layer.orthogonality_error() < TOL
    smax, smin = layer.singular_values()
    assert abs(smax - 1.0) < TOL and abs(smin - 1.0) < TOL


def test_orthogonal_after_training():
    torch.manual_seed(0)
    layer = OrthogonalConv1x1(64, init="random")
    x, target = torch.randn(8, 64, 4, 4), torch.randn(8, 64, 4, 4)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        (layer(x) - target).pow(2).mean().backward()
        opt.step()
    assert layer.theta.grad.abs().max() > 0
    assert layer.orthogonality_error() < TOL


@pytest.mark.parametrize("c", WIDTHS)
def test_preserves_norm(c):
    torch.manual_seed(3)
    layer = OrthogonalConv1x1(c)
    with torch.no_grad():
        layer.theta.normal_(0.0, 2.0)
        w, v = layer.matrix(), torch.randn(c, 64)
    assert torch.allclose((w @ v).norm(dim=0), v.norm(dim=0), atol=TOL)


@pytest.mark.parametrize("c", [3, 6, 12, 20, 63, 100])
def test_non_power_of_two_raises(c):
    with pytest.raises(ValueError, match="power of two"):
        OrthogonalConv1x1(c)


def test_freeze_is_transparent():
    layer = OrthogonalConv1x1(16, init="random").eval()
    x = torch.randn(2, 16, 3, 3, requires_grad=True)
    reference = layer(x)
    layer.freeze_matrix()
    frozen = layer(x)
    assert torch.allclose(reference, frozen, atol=1e-6)
    (g,) = torch.autograd.grad(frozen.sum(), x)
    assert g.abs().max() > 0


def test_butterfly_pins_every_singular_value():
    torch.manual_seed(0)
    layer = OrthogonalConv1x1(64)
    with torch.no_grad():
        layer.theta.normal_(0, 3.0)
    smax, smin = layer.singular_values()
    assert abs(smax - 1.0) < TOL and abs(smin - 1.0) < TOL


# ---- the partitioned rotation ----


def test_binary_parts():
    assert binary_parts(588) == [512, 64, 8, 4]
    assert binary_parts(2048) == [2048]
    assert binary_parts(13) == [8, 4, 1]
    with pytest.raises(ValueError):
        binary_parts(0)


@pytest.mark.parametrize(("width", "stages"), [(588, 2), (12, 2), (13, 3), (64, 1)])
def test_partitioned_is_orthogonal_and_starts_at_identity(width, stages):
    r = PartitionedOrthogonal(width, stages, perm_seed=3)
    assert torch.equal(r.matrix(), torch.eye(width))
    with torch.no_grad():
        for stage in r.blocks:
            for b in stage:
                b.theta.normal_(0, 1)
    assert r.orthogonality_error() < 1e-5
    x = torch.randn(5, width, 4, 4)
    assert torch.allclose(r(x, dim=1).norm(dim=1), x.norm(dim=1), atol=1e-4)
    v = torch.randn(7, width)
    assert torch.allclose(r(v), v @ r.matrix().t(), atol=1e-6)


def test_partitioned_mixes_all_parts_and_has_gradients():
    r = PartitionedOrthogonal(588)
    assert r.n_angles == 2 * (2304 + 192 + 12 + 4)
    with torch.no_grad():
        for stage in r.blocks:
            for b in stage:
                b.theta.normal_(0, 1)
    R = r.matrix()
    assert (R.abs() > 1e-9).float().mean() > 0.9
    r.matrix().sum().backward()
    assert all(b.theta.grad is not None for stage in r.blocks for b in stage)
    single = PartitionedOrthogonal(588, stages=1)
    with torch.no_grad():
        for b in single.blocks[0]:
            b.theta.normal_(0, 1)
    assert (single.matrix()[:512, 512:] == 0).all()


def test_partitioned_rejects_bad_arguments():
    with pytest.raises(ValueError, match="width"):
        PartitionedOrthogonal(1)
    with pytest.raises(ValueError, match="stages"):
        PartitionedOrthogonal(12, stages=0)


def test_the_inverse_permutations_are_stored_once_and_not_saved(monkeypatch):
    r = PartitionedOrthogonal(13, stages=3, perm_seed=2)
    with torch.no_grad():
        for stage in r.blocks:
            for b in stage:
                b.theta.normal_(0, 1)
    for s in range(2):
        assert torch.equal(getattr(r, f"inv_{s}"), torch.argsort(getattr(r, f"perm_{s}")))
    assert list(r.state_dict()) == [k for k in r.state_dict() if k.endswith("theta")]
    # The matrix as the contract writes it, sorting each permutation where it is used.
    R = r._stage_matrix(0)
    for s in range(1, 3):
        inv = torch.argsort(getattr(r, f"perm_{s - 1}"))
        R = r._stage_matrix(s)[inv][:, inv] @ R

    def no_sort(*args, **kwargs):
        raise AssertionError("matrix() sorted a permutation")

    monkeypatch.setattr(torch, "argsort", no_sort)
    assert torch.equal(r.matrix(), R)


# ---- caching and freezing (contract C4) ----


def test_the_matrix_is_cached_only_without_autograd_in_eval():
    layer = Orthogonal(16, init="random").eval()
    with torch.no_grad():
        first = layer.matrix()
        assert layer.matrix() is first
        layer.theta.add_(0.1)
        assert layer.matrix() is not first
    assert layer.matrix() is not layer.matrix()
    layer.train()
    with torch.no_grad():
        assert layer.matrix() is not layer.matrix()


def test_the_cache_follows_device_and_dtype():
    layer = Orthogonal(8, init="random").eval()
    with torch.no_grad():
        single = layer.matrix()
        layer.double()
        assert layer.matrix().dtype == torch.float64
        assert layer.matrix() is not single


def test_the_frozen_context_restores_the_live_matrix():
    layer = Orthogonal(8, init="random")
    with layer.frozen() as same:
        assert same is layer
        frozen = layer.matrix()
        assert not frozen.requires_grad
        assert layer.matrix() is frozen
    assert layer.matrix().requires_grad


def test_a_frozen_matrix_gives_no_gradient_to_the_angles():
    layer = Orthogonal(8, init="random")
    x = torch.randn(3, 8, requires_grad=True)
    with layer.frozen():
        layer(x).sum().backward()
    assert layer.theta.grad is None
    assert x.grad is not None


# ---- forward along an axis (contract C5) ----


@pytest.mark.parametrize(
    ("shape", "dim"),
    [((8,), -1), ((3, 8), 0), ((3, 8, 4), -1), ((3, 4, 8), 2)],
)
def test_other_shapes_and_axes_are_rejected(shape, dim):
    with pytest.raises(ValueError, match="along"):
        Orthogonal(8)(torch.randn(shape), dim=dim)


def test_a_2d_input_takes_dim_1_as_its_last_axis():
    layer = Orthogonal(8, init="random")
    x = torch.randn(4, 8)
    assert torch.equal(layer(x, dim=1), layer(x))


def test_the_rotation_follows_the_input_dtype():
    layer = Orthogonal(8, init="random")
    assert layer(torch.randn(2, 8, dtype=torch.float64)).dtype == torch.float64


# ---- topologies and modes ----


@pytest.mark.parametrize("name", ["pyramid", "x", "diagonal"])
def test_every_layout_is_orthogonal(name):
    layer = Orthogonal(10, name, init="random", std=1.0)
    assert layer.orthogonality_error() < TOL


def test_a_topology_object_is_accepted_and_must_match_the_width():
    t = pyramid(5, reflection=True)
    layer = Orthogonal(5, t, init="random")
    assert float(torch.linalg.det(layer.matrix().detach().double())) == pytest.approx(
        -1.0
    )
    with pytest.raises(ValueError, match="wires"):
        Orthogonal(6, t)
    custom = Topology(3, (((0, 2),), ((0, 1),)))
    assert Orthogonal(3, custom).n_angles == 2


@pytest.mark.parametrize(
    "t", ["butterfly", "pyramid", "x", pyramid(8, reflection=True)], ids=str
)
def test_apply_mode_equals_materialise_mode(t):
    torch.manual_seed(0)
    dense = Orthogonal(8, t, init="random", std=1.0)
    lazy = Orthogonal(8, t, mode="apply")
    lazy.load_state_dict(dense.state_dict())
    x2, x4 = torch.randn(5, 8), torch.randn(2, 8, 3, 3)
    torch.testing.assert_close(lazy(x2), dense(x2))
    torch.testing.assert_close(lazy(x4, dim=1), dense(x4, dim=1))
    lazy(x2).sum().backward()
    assert lazy.theta.grad is not None


def test_apply_mode_conv_adds_the_bias():
    torch.manual_seed(0)
    dense = OrthogonalConv1x1(8, init="random", std=1.0)
    lazy = OrthogonalConv1x1(8, mode="apply")
    with torch.no_grad():
        dense.bias.normal_()
    lazy.load_state_dict(dense.state_dict())
    x = torch.randn(2, 8, 3, 3)
    torch.testing.assert_close(lazy(x), dense(x))


def test_materialising_a_wide_pyramid_is_refused():
    n = 1100
    assert (2 * n - 3) * n * n > MAX_MATERIALISE
    with pytest.raises(ValueError, match="mode='apply'"):
        Orthogonal(n, "pyramid")
    layer = Orthogonal(n, "pyramid", mode="apply")
    assert layer(torch.randn(2, n)).shape == (2, n)


@pytest.mark.parametrize(("argument", "value"), [("init", "zeros"), ("mode", "lazy")])
def test_unknown_options_are_rejected(argument, value):
    with pytest.raises(ValueError, match=argument):
        Orthogonal(8, **{argument: value})


def test_the_x_layout_refuses_an_odd_width_here_too():
    with pytest.raises(ValueError, match="pyramid"):
        Orthogonal(7, "x")


def test_the_index_buffers_move_with_the_module_but_are_not_saved():
    layer = Orthogonal(8)
    assert set(dict(layer.named_buffers())) == {"pair_i", "pair_j"}
    assert list(layer.state_dict()) == ["theta"]
    assert layer.double().pair_i.dtype == torch.long


@pytest.mark.gpu
def test_the_module_runs_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    layer = PartitionedOrthogonal(12).cuda()
    x = torch.randn(3, 12, 2, 2, device="cuda")
    assert layer(x, dim=1).device.type == "cuda"


# ---- the lazy import from eigenkit.orthogonal ----


def test_the_torch_names_are_reachable_from_the_numpy_package():
    from eigenkit.orthogonal import torch as ring

    for name in ("Orthogonal", "PartitionedOrthogonal", "OrthogonalAdapter", "fold_bn"):
        assert getattr(orthogonal, name) is getattr(ring, name)
    with pytest.raises(AttributeError):
        orthogonal.NotAName  # noqa: B018


def test_a_missing_torch_gives_the_install_hint():
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import eigenkit.orthogonal as o\n"
        "try:\n"
        "    o.Orthogonal\n"
        "except ModuleNotFoundError as e:\n"
        "    print(e)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "pip install 'eigenkit[torch]'" in proc.stdout


def test_the_numpy_and_torch_matrices_agree_for_a_custom_layout():
    from eigenkit.orthogonal import orthogonal_matrix

    t = Topology(4, (((0, 3),), ((0, 1), (2, 3)), ((1, 2),)))
    layer = Orthogonal(4, t, init="random", std=1.0).double()
    np.testing.assert_allclose(
        layer.matrix().detach().numpy(),
        orthogonal_matrix(t, layer.theta.detach().numpy()),
        atol=1e-12,
    )


@pytest.mark.parametrize("write", ["copy", "assign"])
def test_the_cache_follows_a_write_through_data(write):
    layer = Orthogonal(8).eval()
    with torch.no_grad():
        stale = layer.matrix()
    new = torch.full((layer.n_angles,), 0.3)
    if write == "copy":
        layer.theta.data.copy_(new)
    else:
        layer.theta.data = new
    expected = Orthogonal(8)
    with torch.no_grad():
        expected.theta.copy_(new)
        fresh = layer.matrix()
        assert not torch.equal(fresh, stale)
        assert torch.equal(fresh, expected.matrix())
        assert layer.matrix() is fresh


def test_a_nested_frozen_context_keeps_an_outer_freeze():
    layer = Orthogonal(8, init="random")
    layer.freeze_matrix()
    outer = layer.matrix()
    with layer.frozen():
        assert layer.matrix() is outer
    assert layer.matrix() is outer


@pytest.mark.parametrize("cls", [Orthogonal, PartitionedOrthogonal])
def test_a_frozen_matrix_follows_the_dtype_of_the_module(cls):
    layer = cls(8) if cls is Orthogonal else cls(12)
    layer.freeze_matrix()
    frozen = layer.matrix()
    layer.double()
    assert layer.matrix().dtype == torch.float64
    assert torch.equal(layer.matrix(), frozen.double())


@pytest.mark.gpu
def test_a_frozen_matrix_follows_the_device_of_the_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    layer = Orthogonal(8, init="random")
    layer.freeze_matrix()
    layer.cuda()
    assert layer.matrix().device.type == "cuda"
    assert layer(torch.randn(2, 8, device="cuda")).device.type == "cuda"


@pytest.mark.parametrize(
    "make",
    [
        lambda: Orthogonal(8, "brick"),
        lambda: PartitionedOrthogonal(12, topology="brick"),
    ],
    ids=["Orthogonal", "PartitionedOrthogonal"],
)
def test_brick_by_name_without_a_depth_is_a_value_error(make):
    with pytest.raises(ValueError, match="depth"):
        make()
    from eigenkit.orthogonal import brick

    assert Orthogonal(8, brick(8, 3)).n_angles == brick(8, 3).n_angles


def test_to_empty_keeps_the_frozen_matrix():
    layer = Orthogonal(4, init="random")
    layer.freeze_matrix()
    frozen = layer.matrix().clone()
    layer.to_empty(device="cpu")
    assert torch.equal(layer.matrix(), frozen)


META_MODULES = {
    "butterfly": (lambda: Orthogonal(8, init="random"), (3, 8)),
    "pyramid": (lambda: Orthogonal(6, "pyramid", init="random"), (3, 6)),
    "conv1x1": (lambda: OrthogonalConv1x1(8, init="random"), (2, 8, 3, 3)),
    "partitioned": (lambda: PartitionedOrthogonal(13, stages=3, perm_seed=2), (3, 13)),
    "adapter": (lambda: OrthogonalAdapter(torch.nn.Linear(4, 13)), (3, 4)),
}


@pytest.mark.parametrize("name", list(META_MODULES))
def test_a_meta_build_loads_a_checkpoint(name):
    # Large-model loading builds on the meta device, then to_empty() and
    # load_state_dict(); the unsaved index and permutation buffers must come back.
    make, shape = META_MODULES[name]
    torch.manual_seed(0)
    reference = make()
    with torch.no_grad():
        for p in reference.parameters():
            p.normal_()
    with torch.device("meta"):
        module = make()
    module.to_empty(device="cpu")
    module.load_state_dict(reference.state_dict())
    buffers = dict(reference.named_buffers())
    for key, value in module.named_buffers():
        assert torch.equal(value, buffers[key]), key
    x = torch.randn(shape)
    with torch.no_grad():
        assert torch.equal(module(x), reference(x))


@pytest.mark.parametrize("name", list(META_MODULES))
def test_a_meta_build_runs_on_meta(name):
    # Shape inference and FLOP counting run whole models on the meta device.
    make, shape = META_MODULES[name]
    with torch.device("meta"):
        module = make()
    for key, value in module.named_buffers():
        assert value.device.type == "meta", key
    out = module(torch.empty(shape, device="meta"))
    assert out.device.type == "meta"


@pytest.mark.parametrize("name", list(META_MODULES))
def test_a_meta_build_runs_on_meta_in_eval_twice(name):
    # The eval cache checks theta by value, which a meta tensor does not have;
    # repeated calls and a later freeze_matrix() must still work on meta.
    make, shape = META_MODULES[name]
    with torch.device("meta"):
        module = make()
    module.eval()
    x = torch.empty(shape, device="meta")
    with torch.no_grad():
        for _ in range(2):
            assert module(x).device.type == "meta"
    rotation = module.rotation if name == "adapter" else module
    rotation.freeze_matrix()
    assert rotation.matrix().device.type == "meta"


def _reset_all(model):
    # What the estimators do after seeding (eigenkit.torch.estimators.generic).
    model.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )


def test_a_reset_keeps_the_identity_start():
    model = torch.nn.Sequential(
        Orthogonal(4, "pyramid"), PartitionedOrthogonal(13, stages=2)
    )
    torch.manual_seed(3)
    _reset_all(model)
    assert all(torch.count_nonzero(p) == 0 for p in model.parameters())


def test_the_seed_controls_the_random_angles_through_a_reset():
    model = torch.nn.Sequential(Orthogonal(4, init="random", std=0.5))
    thetas = []
    for seed in (0, 1, 0):
        torch.manual_seed(seed)
        _reset_all(model)
        thetas.append(model[0].theta.detach().clone())
    assert torch.equal(thetas[0], thetas[2])
    assert not torch.equal(thetas[0], thetas[1])
    torch.manual_seed(0)
    assert torch.equal(thetas[0], Orthogonal(4, init="random", std=0.5).theta)


@pytest.mark.parametrize("mode", ["materialise", "apply"])
def test_conv1x1_rotates_the_channels_of_an_unbatched_map(mode):
    torch.manual_seed(0)
    m = OrthogonalConv1x1(4, init="random", std=0.7, mode=mode).double()
    with torch.no_grad():
        m.bias.normal_()
    x = torch.randn(4, 6, 6, dtype=torch.float64)
    expected = m(x.unsqueeze(0)).squeeze(0)
    torch.testing.assert_close(m(x), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", ["materialise", "apply"])
@pytest.mark.parametrize(
    ("module_dtype", "input_dtype"),
    [(torch.float32, torch.float64), (torch.float64, torch.float32)],
)
def test_conv1x1_follows_the_dtype_of_the_input(mode, module_dtype, input_dtype):
    # Orthogonal.forward casts its matrix to the dtype of the input; the 1x1
    # convolution does the same with its matrix and its bias.
    torch.manual_seed(0)
    m = OrthogonalConv1x1(4, init="random", std=0.7, mode=mode).to(module_dtype)
    with torch.no_grad():
        m.bias.normal_()
    x = torch.randn(2, 4, 3, 3, dtype=input_dtype)
    out = m(x)
    assert out.dtype == input_dtype
    reference = OrthogonalConv1x1(4, mode=mode).to(input_dtype)
    reference.load_state_dict({k: v.to(input_dtype) for k, v in m.state_dict().items()})
    torch.testing.assert_close(out, reference(x), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_the_eval_cache_does_not_stay_on_the_gpu_after_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    m = Orthogonal(64, init="random").cuda().eval()
    with torch.no_grad():
        m(torch.randn(2, 64, device="cuda"))
    assert m._cache_W is not None
    m.cpu()
    assert m._cache_W is None
    with torch.no_grad():
        assert m(torch.randn(2, 64)).device.type == "cpu"


def test_a_conv1x1_reset_also_restores_the_zero_bias():
    m = OrthogonalConv1x1(4, init="random", std=0.5)
    with torch.no_grad():
        m.bias.fill_(1.0)
    torch.manual_seed(4)
    m.reset_parameters()
    assert torch.count_nonzero(m.bias) == 0
    torch.manual_seed(4)
    assert torch.equal(m.theta, OrthogonalConv1x1(4, init="random", std=0.5).theta)
    OrthogonalConv1x1(4, bias=False).reset_parameters()
