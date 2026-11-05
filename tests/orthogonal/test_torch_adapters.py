"""Orthogonal fine-tuning: the adapter starts at the pretrained function, keeps every
singular value of the weight, merges into a plain layer, and only its angles train."""

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

from eigenkit.orthogonal.torch import (  # noqa: E402
    Orthogonal,
    OrthogonalAdapter,
    PartitionedOrthogonal,
    apply_orthogonal_adapters,
    fold_bn,
    merge_adapters,
    rotate_weight,
    trainable_parameters,
)


def _move(adapter, std=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in adapter.rotation.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)


def _matrix(w):
    return w.reshape(w.shape[0], -1).double()


LAYERS = [
    lambda: nn.Linear(24, 16),
    lambda: nn.Linear(24, 12),
    lambda: nn.Conv2d(8, 16, 3, padding=1),
    lambda: nn.Conv2d(6, 12, 3, stride=2),
]
IDS = ["linear16", "linear12", "conv16", "conv12"]


def _input(layer):
    if isinstance(layer, nn.Linear):
        return torch.randn(5, layer.in_features)
    return torch.randn(2, layer.in_channels, 7, 7)


@pytest.mark.parametrize("make", LAYERS, ids=IDS)
@pytest.mark.parametrize("side", ["out", "in"])
def test_the_adapter_starts_at_the_pretrained_function(make, side):
    torch.manual_seed(0)
    layer = make()
    x = _input(layer)
    expected = layer(x)
    adapter = OrthogonalAdapter(layer, side=side, topology="pyramid")
    assert torch.equal(adapter(x), expected)


@pytest.mark.parametrize("make", LAYERS, ids=IDS)
@pytest.mark.parametrize("side", ["out", "in"])
def test_the_adapter_keeps_every_singular_value(make, side):
    torch.manual_seed(1)
    layer = make()
    before = torch.linalg.svdvals(_matrix(layer.weight))
    adapter = OrthogonalAdapter(layer, side=side, topology="pyramid")
    _move(adapter)
    after_w = adapter.weight
    assert not torch.allclose(after_w, layer.weight, atol=1e-2)
    torch.testing.assert_close(
        torch.linalg.svdvals(_matrix(after_w)), before, atol=1e-5, rtol=0
    )


def test_a_butterfly_on_a_width_that_is_not_a_power_of_two_is_partitioned():
    assert isinstance(OrthogonalAdapter(nn.Linear(4, 12)).rotation, PartitionedOrthogonal)
    assert isinstance(OrthogonalAdapter(nn.Linear(4, 16)).rotation, Orthogonal)
    assert isinstance(
        OrthogonalAdapter(nn.Linear(4, 12), topology="pyramid").rotation, Orthogonal
    )
    assert isinstance(
        OrthogonalAdapter(nn.Linear(4, 16), partition=True).rotation,
        PartitionedOrthogonal,
    )
    with pytest.raises(ValueError, match="power of two"):
        OrthogonalAdapter(nn.Linear(4, 12), partition=False)


def test_the_x_layout_on_an_odd_width_names_the_alternatives():
    with pytest.raises(ValueError, match="pyramid"):
        OrthogonalAdapter(nn.Linear(4, 7), topology="x")


@pytest.mark.parametrize("make", LAYERS, ids=IDS)
def test_merge_gives_the_same_function(make):
    torch.manual_seed(2)
    layer = make()
    adapter = OrthogonalAdapter(layer)
    _move(adapter, seed=3)
    x = _input(layer)
    merged = adapter.merge()
    assert type(merged) is type(layer)
    torch.testing.assert_close(merged(x), adapter(x))
    assert merged.weight is not layer.weight


def test_the_base_is_frozen_and_only_the_angles_train():
    layer = nn.Linear(8, 16)
    adapter = OrthogonalAdapter(layer)
    names = [n for n, p in adapter.named_parameters() if p.requires_grad]
    assert names == ["rotation.theta"]
    adapter(torch.randn(3, 8)).sum().backward()
    assert layer.weight.grad is None
    assert adapter.rotation.theta.grad is not None


def test_the_angles_follow_the_dtype_of_the_base():
    adapter = OrthogonalAdapter(nn.Linear(8, 16).double())
    assert adapter.rotation.theta.dtype == torch.float64
    assert adapter(torch.randn(2, 8, dtype=torch.float64)).dtype == torch.float64


def test_the_state_dict_keeps_the_base_and_the_angles():
    adapter = OrthogonalAdapter(nn.Linear(8, 16))
    assert list(adapter.state_dict()) == ["base.weight", "base.bias", "rotation.theta"]


def test_rotate_weight_on_the_input_side_is_w_times_r_transposed():
    R = Orthogonal(6, "pyramid", init="random", std=1.0).matrix().detach()
    w = torch.randn(4, 6)
    torch.testing.assert_close(rotate_weight(R, w, side="in"), w @ R.t())
    with pytest.raises(ValueError, match="cannot rotate"):
        rotate_weight(R, torch.randn(6, 4), side="in")
    with pytest.raises(ValueError, match="side"):
        rotate_weight(R, w, side="both")


def _model():
    torch.manual_seed(4)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )


def test_adapters_by_name_start_at_the_model_and_merge_back():
    model = _model()
    x = torch.randn(2, 3, 4, 4)
    expected = model(x)
    apply_orthogonal_adapters(model, ["0", "3"])
    assert isinstance(model[0], OrthogonalAdapter)
    assert isinstance(model[3], OrthogonalAdapter)
    assert torch.equal(model(x), expected)
    params = list(trainable_parameters(model))
    assert {id(p) for p in params} == {
        id(model[0].rotation.theta),
        id(model[3].rotation.theta),
    }
    _move(model[3])
    moved = model(x)
    merge_adapters(model)
    assert not any(isinstance(m, OrthogonalAdapter) for m in model.modules())
    torch.testing.assert_close(model(x), moved)


def test_adapters_by_predicate():
    model = _model()
    apply_orthogonal_adapters(model, lambda name, m: isinstance(m, nn.Linear))
    assert [type(m).__name__ for m in model] == [
        "Conv2d",
        "ReLU",
        "Flatten",
        "OrthogonalAdapter",
        "ReLU",
        "OrthogonalAdapter",
    ]


def test_freeze_rest_false_keeps_the_other_parameters_trainable():
    model = _model()
    apply_orthogonal_adapters(model, ["3"], freeze_rest=False)
    assert model[0].weight.requires_grad
    assert not model[3].base.weight.requires_grad


def test_nested_names_are_replaced_in_their_parent():
    model = nn.Sequential(nn.Sequential(nn.Linear(4, 8)), nn.Linear(8, 2))
    apply_orthogonal_adapters(model, ["0.0"], topology="x")
    assert isinstance(model[0][0], OrthogonalAdapter)
    assert merge_adapters(model[0][0]).__class__ is nn.Linear


@pytest.mark.parametrize(
    ("targets", "error", "match"),
    [
        (["9"], KeyError, "no submodule"),
        (["1"], TypeError, "only nn.Linear"),
        ([], ValueError, "no module"),
        (lambda name, m: False, ValueError, "no module"),
    ],
)
def test_bad_targets_are_rejected(targets, error, match):
    with pytest.raises(error, match=match):
        apply_orthogonal_adapters(_model(), targets)


def test_unsupported_layers_are_rejected():
    with pytest.raises(TypeError, match="nn.Linear"):
        OrthogonalAdapter(nn.Conv1d(4, 8, 3))
    with pytest.raises(ValueError, match="grouped"):
        OrthogonalAdapter(nn.Conv2d(4, 8, 3, groups=2))
    with pytest.raises(ValueError, match="side"):
        OrthogonalAdapter(nn.Linear(4, 8), side="both")


def test_fold_bn_matches_the_eval_mode_pair():
    torch.manual_seed(5)
    for conv_bias in (False, True):
        conv = nn.Conv2d(4, 6, 3, bias=conv_bias)
        bn = nn.BatchNorm2d(6)
        with torch.no_grad():
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.normal_()
            bn.running_mean.normal_()
            bn.running_var.uniform_(0.1, 1.1)
        bn.eval()
        x = torch.randn(2, 4, 5, 5, dtype=torch.float64)
        w, b = fold_bn(conv, bn)
        expected = bn.double()(conv.double()(x))
        torch.testing.assert_close(nn.functional.conv2d(x, w, b), expected)
        conv.float()
        bn.float()


def test_fold_bn_without_affine_parameters():
    torch.manual_seed(6)
    conv = nn.Conv2d(4, 6, 3, bias=False)
    bn = nn.BatchNorm2d(6, affine=False)
    with torch.no_grad():
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.5, 2.0)
    bn.eval()
    x = torch.randn(2, 4, 5, 5, dtype=torch.float64)
    w, b = fold_bn(conv, bn)
    expected = bn.double()(conv.double()(x))
    torch.testing.assert_close(nn.functional.conv2d(x, w, b), expected)


def test_fold_bn_needs_running_statistics():
    bn = nn.BatchNorm2d(6, track_running_stats=False)
    with pytest.raises(ValueError, match="running statistics"):
        fold_bn(nn.Conv2d(4, 6, 3), bn)


def test_an_adapter_with_brick_by_name_needs_a_depth():
    with pytest.raises(ValueError, match="depth"):
        OrthogonalAdapter(nn.Linear(4, 8), topology="brick")


def test_the_adapted_weight_and_bias_read_like_the_layer_they_replace():
    base = nn.Linear(4, 6)
    adapter = OrthogonalAdapter(base, topology="pyramid")
    with torch.no_grad():
        adapter.rotation.theta.normal_()
    expected = rotate_weight(adapter.rotation.matrix(), base.weight)
    assert torch.equal(adapter.weight, expected)
    assert adapter.bias is base.bias
    assert OrthogonalAdapter(nn.Linear(4, 6, bias=False)).bias is None


def test_adapters_inside_attention_start_at_the_pretrained_function_and_train():
    # nn.MultiheadAttention reads out_proj.weight and out_proj.bias directly.
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(
        8, 2, dim_feedforward=16, dropout=0.0, batch_first=True
    ).eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        before = layer(x)
    apply_orthogonal_adapters(
        layer, lambda name, m: isinstance(m, nn.Linear), topology="pyramid"
    )
    assert isinstance(layer.self_attn.out_proj, OrthogonalAdapter)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), before)
    layer.train()
    layer(x).sum().backward()
    assert layer.self_attn.out_proj.rotation.theta.grad.abs().sum() > 0


def test_a_layer_inside_an_adapter_cannot_be_named():
    model = nn.Sequential(nn.Linear(4, 4))
    apply_orthogonal_adapters(model, ["0"])
    with pytest.raises(ValueError, match="inside an OrthogonalAdapter"):
        apply_orthogonal_adapters(model, ["0.base"])


def test_fold_bn_refuses_a_batchnorm_of_another_width():
    bn = nn.BatchNorm2d(8).eval()
    with pytest.raises(ValueError, match="8 features"):
        fold_bn(nn.Conv2d(3, 1, 3), bn)


def test_fold_bn_refuses_a_transposed_convolution():
    with pytest.raises(TypeError, match="Conv1d, nn.Conv2d or nn.Conv3d"):
        fold_bn(nn.ConvTranspose2d(3, 4, 3), nn.BatchNorm2d(4).eval())


@pytest.mark.parametrize(
    ("conv", "bn", "shape", "functional"),
    [
        (nn.Conv1d(3, 4, 3), nn.BatchNorm1d(4), (2, 3, 9), "conv1d"),
        (nn.Conv3d(3, 4, 3), nn.BatchNorm3d(4), (2, 3, 5, 5, 5), "conv3d"),
    ],
)
def test_fold_bn_is_exact_for_one_and_three_dimensional_convolutions(
    conv, bn, shape, functional
):
    torch.manual_seed(7)
    with torch.no_grad():
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.normal_()
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.1, 1.1)
    bn.eval()
    x = torch.randn(shape, dtype=torch.float64)
    w, b = fold_bn(conv, bn)
    assert w.shape == conv.weight.shape
    expected = bn.double()(conv.double()(x))
    torch.testing.assert_close(getattr(nn.functional, functional)(x, w, b), expected)


def test_an_adapter_refuses_to_be_re_initialised():
    model = apply_orthogonal_adapters(nn.Sequential(nn.Linear(4, 4)), ["0"])
    with pytest.raises(RuntimeError, match="cannot be re-initialised"):
        model.apply(
            lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
        )


def test_an_estimator_refuses_an_adapted_model_and_leaves_it_intact():
    # fit() re-initialises every layer of a copy of the model, which would train
    # a rotation of a random weight; it must fail instead, and the model passed
    # in keeps its pretrained weight.
    np = pytest.importorskip("numpy")
    from eigenkit.torch.estimators.generic import GenericClassifier

    torch.manual_seed(0)
    model = apply_orthogonal_adapters(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)), ["0"]
    )
    pretrained = model[0].base.weight.detach().clone()
    X = np.random.default_rng(0).normal(size=(12, 4)).astype(np.float32)
    clf = GenericClassifier(
        model, n_classes=3, epochs=1, show_progress=False, random_state=0
    )
    with pytest.raises(RuntimeError, match="cannot be re-initialised"):
        clf.fit(X, np.arange(12) % 3)
    assert torch.equal(model[0].base.weight, pretrained)
