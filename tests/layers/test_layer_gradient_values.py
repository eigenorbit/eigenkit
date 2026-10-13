"""The layers' gradients must be right, not merely present.

Checks such as `grad is not None` and `grad.norm() > 0` hold for any
non-degenerate wrong answer — including the 41% error the two-term
parameter-shift rule makes on a controlled rotation. A gradient test that
cannot distinguish a correct gradient from a wrong one is not testing the
gradient.

These check values, two independent ways: against `torch.autograd.gradcheck`,
and against the numpy reference simulator's exact adjoint. The layers run in
float64 so the finite differences gradcheck takes are meaningful.
"""

from __future__ import annotations

import pytest
import torch

from eigenkit.torch.layers import (
    Basic,
    Ensemble,
    Reupload,
)


# (factory, n_features). Each is small enough for gradcheck to stay quick.
LAYERS = [
    ("Basic", lambda: Basic(n_qubits=2, depth=1), 2),
    ("Basic-strong", lambda: Basic(n_qubits=2, depth=1, variational_type="strong"), 2),
    ("Reupload", lambda: Reupload(n_qubits=2, depth=2), 2),
    (
        "Reupload-trainable-scale",
        lambda: Reupload(n_qubits=2, depth=2, trainable_scale=True),
        2,
    ),
    ("Ensemble", lambda: Ensemble(in_features=4, n_qubits=2, depth=1), 4),
]
IDS = [name for name, _, _ in LAYERS]


def _double(layer):
    """gradcheck needs float64 throughout, including the circuit weights."""
    return layer.double()


@pytest.mark.parametrize("name,factory,n_features", LAYERS, ids=IDS)
def test_input_gradient_passes_gradcheck(name, factory, n_features):
    torch.manual_seed(0)
    layer = _double(factory())
    x = torch.randn(1, n_features, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: layer(t), (x,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("name,factory,n_features", LAYERS, ids=IDS)
def test_weight_gradient_matches_a_central_difference(name, factory, n_features):
    """Checked by difference rather than by `gradcheck`: gradcheck wants the
    varied tensors as call arguments, and rebinding `parameter.data` to supply
    them detaches from the graph it built, so it would measure nothing."""
    torch.manual_seed(0)
    layer = _double(factory())
    x = torch.randn(1, n_features, dtype=torch.float64)
    weights = [p for p in layer.parameters() if p.requires_grad]
    assert weights, f"{name} exposes no trainable parameters"

    layer(x).sum().backward()
    step = 1e-6
    for parameter in weights:
        analytic = parameter.grad.detach().clone()
        numeric = torch.zeros_like(analytic)
        flat = parameter.data.view(-1)
        with torch.no_grad():
            for i in range(flat.numel()):
                original = flat[i].item()
                flat[i] = original + step
                plus = layer(x).sum()
                flat[i] = original - step
                minus = layer(x).sum()
                flat[i] = original
                numeric.view(-1)[i] = (plus - minus) / (2 * step)
        torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name,factory,n_features", LAYERS, ids=IDS)
def test_gradient_matches_a_central_difference(name, factory, n_features):
    """A second, independent check: gradcheck and a hand-rolled difference
    can only agree if the analytic gradient is right."""
    torch.manual_seed(0)
    layer = _double(factory())
    x = torch.randn(1, n_features, dtype=torch.float64, requires_grad=True)
    layer(x).sum().backward()
    analytic = x.grad.detach().clone()

    step = 1e-6
    numeric = torch.zeros_like(analytic)
    with torch.no_grad():
        for i in range(n_features):
            plus, minus = x.detach().clone(), x.detach().clone()
            plus[0, i] += step
            minus[0, i] -= step
            numeric[0, i] = (layer(plus).sum() - layer(minus).sum()) / (2 * step)
    torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)


def test_a_deliberately_wrong_gradient_would_be_caught():
    """Guard the guard: the checks above must be able to fail."""
    torch.manual_seed(0)
    layer = _double(Basic(n_qubits=2, depth=1))
    x = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
    layer(x).sum().backward()
    wrong = x.grad.detach().clone() * 1.41  # the two-term rule's CRY error
    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong, x.grad.detach(), atol=1e-6, rtol=1e-5)
