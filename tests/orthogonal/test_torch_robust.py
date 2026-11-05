"""Certification: MaxMin, LLNHead, the certified radius, cert_ce and sigma_max against
the reference implementation (contracts C9-C11, bit for bit), and the
properties lipschitz_bound promises."""

import math

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

from eigenkit.orthogonal.torch import (  # noqa: E402
    LLNHead,
    MaxMin,
    OrthogonalConv1x1,
    apply_orthogonal_adapters,
    cert_ce,
    certified_accuracy,
    certified_radius,
    head_distances,
    lipschitz_bound,
    merge_adapters,
    sigma_max,
)

from . import _reference as ref  # noqa: E402


DTYPES = [torch.float32, torch.float64]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def _require(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


# ---- C9: MaxMin ----


def _with_ties(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1))
    x[0, 1] = x[0, 0]
    x[0, shape[1] // 2] = x[0, 0]
    return x.to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(5, 6), (4, 8, 3, 3)])
def test_adjacent_pairing_is_groupsort(shape, dtype, device):
    _require(device)
    x = _with_ties(shape, dtype, device)
    assert torch.equal(MaxMin("adjacent")(x), ref.GroupSort()(x))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(5, 6), (4, 8, 3, 3)])
def test_halves_pairing_is_bronet_minmax(shape, dtype, device):
    _require(device)
    x = _with_ties(shape, dtype, device)
    assert torch.equal(MaxMin("halves")(x), ref.minmax(x))


@pytest.mark.parametrize("pairing", ["adjacent", "halves"])
def test_maxmin_is_1_lipschitz_and_keeps_the_norm(pairing):
    torch.manual_seed(0)
    act = MaxMin(pairing)
    for _ in range(20):
        a, b = torch.randn(8, 16, 5, 5), torch.randn(8, 16, 5, 5)
        num = (act(a) - act(b)).flatten(1).norm(dim=1)
        den = (a - b).flatten(1).norm(dim=1)
        assert float((num / den).max()) <= 1.0 + 1e-6
        torch.testing.assert_close(
            act(a).flatten(1).norm(dim=1), a.flatten(1).norm(dim=1)
        )


def test_the_two_pairings_are_different_functions():
    x = torch.tensor([[3.0, 1.0, 2.0, 4.0]])
    assert MaxMin("adjacent")(x).tolist() == [[1.0, 3.0, 2.0, 4.0]]
    assert MaxMin("halves")(x).tolist() == [[3.0, 4.0, 2.0, 1.0]]


@pytest.mark.parametrize("pairing", ["adjacent", "halves"])
def test_an_odd_channel_count_is_rejected(pairing):
    with pytest.raises(ValueError, match="even"):
        MaxMin(pairing)(torch.randn(2, 5))


def test_an_unknown_pairing_is_rejected():
    with pytest.raises(ValueError, match="pairing"):
        MaxMin("random")


# ---- C10: LLNHead ----


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_head_is_the_reference_head(dtype):
    torch.manual_seed(0)
    old = ref.LLNHead(16, 5).to(dtype)
    torch.manual_seed(0)
    new = LLNHead(16, 5).to(dtype)
    assert list(new.state_dict()) == ["linear.weight", "linear.bias"]
    assert torch.equal(new.linear.weight, old.linear.weight)
    h = torch.randn(7, 16, dtype=dtype)
    assert torch.equal(new(h), old(h))
    assert torch.equal(new.get_weight(), old.get_weight())
    torch.testing.assert_close(
        new.get_weight().norm(dim=1), torch.ones(5, dtype=dtype), atol=1e-6, rtol=0
    )


# ---- C11: certificate and loss ----


def _certify_inputs(dtype=torch.float32):
    g = torch.Generator().manual_seed(7)
    head = nn.functional.normalize(torch.randn(5, 16, generator=g), dim=1)
    head[4] = head[3]
    logits = torch.randn(40, 5, generator=g)
    logits[0, 1] = logits[0, 0] = logits[0].max() + 1
    y = torch.randint(0, 5, (40,), generator=g)
    return head.to(dtype), logits.to(dtype), y


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L", [1.0, 1.7, 13.0])
def test_the_radius_is_the_reference_radius(dtype, L):
    head, logits, _ = _certify_inputs(dtype)
    new = certified_radius(logits, head, L)
    old = ref.radii(logits, head, L)
    assert torch.equal(new.pred, old["pred"])
    assert torch.equal(new.margin, old["margin"])
    assert torch.equal(new.radius, old["radius"])
    assert torch.equal(head_distances(head), ref.head_distances(head))
    assert new.radius.dtype == torch.float64


@pytest.mark.parametrize("eps", [0.0, 0.01, 0.1, 1.0])
def test_certified_accuracy_is_the_reference_one(eps):
    head, logits, y = _certify_inputs()
    cert = certified_radius(logits, head, 1.7)
    correct = cert.pred == y
    assert certified_accuracy(correct, cert.radius, eps) == ref.certified_accuracy(
        correct, cert.radius, eps
    )


def test_eps_zero_is_the_plain_accuracy():
    correct = torch.tensor([True, False, True, True])
    radius = torch.tensor([0.0, 1.0, 0.0, 2.0], dtype=torch.float64)
    assert certified_accuracy(correct, radius, 0) == 0.75
    assert certified_accuracy(correct, radius, 1e-12) == 0.25


def test_a_class_with_the_same_head_row_never_counts():
    head = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    cert = certified_radius(logits, head, L=1.0)
    expected = 2.0 / (1.0 * math.sqrt(2.0))
    assert cert.radius.item() == pytest.approx(expected)
    assert cert.margin.item() == 1.0


def test_no_perturbation_inside_the_radius_changes_the_prediction():
    torch.manual_seed(3)
    head = LLNHead(8, 4)
    features = torch.randn(64, 8)
    cert = certified_radius(head(features), head.get_weight(), L=1.0)
    for _ in range(50):
        delta = torch.randn(64, 8)
        delta = (
            delta / delta.norm(dim=1, keepdim=True) * cert.radius.float()[:, None] * 0.999
        )
        moved = head(features + delta).argmax(1)
        assert torch.equal(moved, cert.pred)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", [0.0, 0.25])
def test_cert_ce_is_the_reference_loss(dtype, eps):
    head, logits, y = _certify_inputs(dtype)
    head.requires_grad_(True)
    new = cert_ce(logits, y, head, 1.7, eps, tau=4.0)
    old = ref.cert_ce(logits, y, head, 1.7, eps, tau=4.0)
    assert torch.equal(new, old)
    (g_new,) = torch.autograd.grad(new, head)
    (g_old,) = torch.autograd.grad(old, head)
    assert torch.equal(g_new, g_old)


def test_cert_ce_at_eps_zero_is_the_tempered_cross_entropy():
    head, logits, y = _certify_inputs()
    torch.testing.assert_close(
        cert_ce(logits, y, head, 1.0, 0.0, tau=2.0),
        nn.functional.cross_entropy(2.0 * logits, y),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sigma_max_is_the_reference_operation(dtype):
    w = torch.randn(32, 48, dtype=dtype, generator=torch.Generator().manual_seed(2))
    assert torch.equal(sigma_max(w), ref.sigma_max(w))
    assert float(sigma_max(w)) == pytest.approx(
        float(torch.linalg.svdvals(w).max()), rel=1e-4
    )


# ---- lipschitz_bound ----


def _chain():
    torch.manual_seed(4)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        MaxMin(),
        nn.Conv2d(8, 8, 1),
        nn.ReLU(),
        nn.Sequential(nn.Flatten(), nn.Linear(8 * 4 * 4, 16)),
        MaxMin("halves"),
        LLNHead(16, 3),
    )


def test_the_bound_is_the_product_of_the_layer_bounds():
    model = _chain().double()
    expected = 1.0
    for layer in (model[0], model[2], model[4][1]):
        w = layer.weight.detach().reshape(layer.weight.shape[0], -1)
        factor = (
            math.sqrt(layer.kernel_size[0] * layer.kernel_size[1])
            if isinstance(layer, nn.Conv2d)
            else 1
        )
        expected *= float(torch.linalg.svdvals(w).max()) * factor
    expected *= float(torch.linalg.svdvals(model[6].get_weight().detach()).max())
    assert lipschitz_bound(model) == pytest.approx(expected, rel=1e-10)


def test_the_bound_is_an_upper_bound():
    model = _chain().eval()
    bound = lipschitz_bound(model)
    torch.manual_seed(5)
    for _ in range(20):
        a, b = torch.randn(16, 3, 4, 4), torch.randn(16, 3, 4, 4)
        ratio = (model(a) - model(b)).norm(dim=1) / (a - b).flatten(1).norm(dim=1)
        assert float(ratio.max().detach()) <= bound


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adapting_a_chain_keeps_its_bound(dtype):
    model = _chain().to(dtype)
    before = lipschitz_bound(model)
    apply_orthogonal_adapters(model, ["0", "2", "4.1"], topology="pyramid")
    g = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.copy_(torch.randn(p.shape, generator=g) * 2.0)
    assert lipschitz_bound(model) == pytest.approx(
        before, rel=1e-6 if dtype == torch.float32 else 1e-12
    )
    merge_adapters(model)
    assert lipschitz_bound(model) == pytest.approx(
        before, rel=1e-6 if dtype == torch.float32 else 1e-12
    )


def test_an_orthogonal_layer_counts_as_one():
    layer = OrthogonalConv1x1(8, init="random", std=1.0)
    assert lipschitz_bound(nn.Sequential(layer, MaxMin())) == 1.0


@pytest.mark.parametrize(
    "layer", [nn.BatchNorm1d(4), nn.LeakyReLU(2.0), nn.Conv2d(4, 4, 3, groups=2)]
)
def test_layers_without_a_known_bound_are_rejected(layer):
    with pytest.raises(ValueError):
        lipschitz_bound(nn.Sequential(layer))


@pytest.mark.parametrize("mode", ["reflect", "replicate"])
def test_padding_that_copies_border_pixels_is_refused(mode):
    # Reflect and replicate padding feed a border pixel into one kernel offset
    # more than once, which the sqrt(kh*kw) factor does not cover.
    conv = nn.Conv2d(3, 4, 3, padding=1, padding_mode=mode)
    with pytest.raises(ValueError, match="padding_mode"):
        lipschitz_bound(conv)
    adapted = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, padding_mode=mode))
    apply_orthogonal_adapters(adapted, ["0"], topology="pyramid")
    with pytest.raises(ValueError, match="padding_mode"):
        lipschitz_bound(adapted)


@pytest.mark.parametrize("mode", ["zeros", "circular"])
def test_zero_and_circular_padding_are_bounded(mode):
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 4, 3, padding=1, padding_mode=mode)
    w = conv.weight.detach().double().reshape(4, -1)
    expected = float(torch.linalg.svdvals(w).max()) * 3.0
    assert lipschitz_bound(conv) == pytest.approx(expected, rel=1e-10)


def _exact_conv_norm(conv, shape):
    conv = conv.double()

    def apply(x):
        return conv._conv_forward(x, conv.weight, None)

    jac = torch.autograd.functional.jacobian(
        apply, torch.zeros(shape, dtype=torch.float64)
    )
    return float(torch.linalg.svdvals(jac.reshape(-1, math.prod(shape))).max())


@pytest.mark.parametrize(
    ("kernel", "padding", "dilation"),
    [(3, 1, 1), (5, 2, 1), (3, 2, 2), (3, 0, 1), (3, "same", 1), ((3, 5), (1, 2), 1)],
)
@pytest.mark.parametrize("seed", range(4))
def test_circular_padding_within_the_kernel_reach_is_bounded(
    kernel, padding, dilation, seed
):
    torch.manual_seed(seed)
    conv = nn.Conv2d(
        3, 4, kernel, padding=padding, dilation=dilation, padding_mode="circular"
    )
    if seed % 2:
        with torch.no_grad():
            conv.weight.abs_()
    exact = _exact_conv_norm(conv, (1, 3, 8, 8))
    assert exact <= lipschitz_bound(conv) * (1 + 1e-12)


@pytest.mark.parametrize(
    ("kernel", "padding", "dilation"),
    [(3, 2, 1), (5, 4, 1), (3, 3, 2), ((3, 5), (1, 3), 1)],
)
def test_circular_padding_beyond_the_kernel_reach_is_refused(kernel, padding, dilation):
    # Wider circular padding wraps a pixel under one kernel offset twice, and the
    # exact norm then exceeds sqrt(kh*kw) times the matrix norm (by up to 1.5x).
    conv = nn.Conv2d(
        3, 4, kernel, padding=padding, dilation=dilation, padding_mode="circular"
    )
    with pytest.raises(ValueError, match="circular"):
        lipschitz_bound(conv)


@pytest.mark.parametrize("padding", [0, 1, 3])
def test_zero_padding_of_any_width_is_bounded(padding):
    torch.manual_seed(1)
    conv = nn.Conv2d(3, 4, 3, padding=padding)
    with torch.no_grad():
        conv.weight.abs_()
    assert _exact_conv_norm(conv, (1, 3, 6, 6)) <= lipschitz_bound(conv) * (1 + 1e-12)


@pytest.mark.parametrize("shape", [(3000, 2), (2, 3000), (40, 7)])
def test_a_layer_is_bounded_by_its_spectral_norm_on_either_side(shape):
    # sigma_max(W) = sigma_max(Wᵀ): a 2-input layer needs a 2 x 2 Gram matrix.
    torch.manual_seed(0)
    layer = nn.Linear(shape[1], shape[0])
    exact = torch.linalg.matrix_norm(layer.weight.detach().double(), ord=2)
    assert lipschitz_bound(layer) == pytest.approx(float(exact), rel=1e-12)


def test_the_gram_matrix_of_a_wide_output_layer_is_small(monkeypatch):
    import eigenkit.orthogonal.torch.robust as robust

    sides = []
    original = robust.sigma_max
    monkeypatch.setattr(
        robust, "sigma_max", lambda w: sides.append(tuple(w.shape)) or original(w)
    )
    lipschitz_bound(nn.Linear(2, 3000))
    assert sides == [(2, 3000)]


def test_head_distances_stay_exact_above_25_classes():
    # Above 25 rows the default cdist switches to a matrix product, which shrinks
    # a small distance by about 10% here and so overstates the certified radius.
    g = torch.Generator().manual_seed(0)
    w = torch.randn(30, 588, dtype=torch.float64, generator=g)
    u = torch.randn(588, dtype=torch.float64, generator=g)
    w[1] = w[0] + 1e-6 * u / u.norm()
    assert head_distances(w)[0, 1].item() == pytest.approx(1e-6, rel=1e-6)
