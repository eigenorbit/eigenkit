"""A layer built with nothing but its required arguments must be trainable.

A layer with `RX` encoding, `RZ` variational and `Z` measurement is inert in
its weights. `RZ` and the CNOT ladder both commute with the `Z` observable —
`RZ` because it is diagonal, the ladder because it maps Z-strings to Z-strings
— so the variational weights cancel out of `<Z>` exactly and every gradient is
identically zero. Such a layer runs, produces sensible-looking numbers and
trains to a good score on the surrounding classical layers while its own
weights never move.

The defaults are what the docstring examples, the getting-started page and the
`GenericClassifier` examples build, so they must not land on a configuration
like that.

These tests pin the property that matters — the weights reach the output —
rather than the specific rotation that provides it.
"""

from __future__ import annotations

import pytest
import torch

from eigenkit.torch.layers import Basic, Ensemble, Reupload


def _max_grad(layer: torch.nn.Module, x: torch.Tensor) -> float:
    layer(x).sum().backward()
    return max(p.grad.abs().max().item() for p in layer.parameters())


DEFAULT_LAYERS = [
    pytest.param(lambda: Basic(n_qubits=4, depth=2), 4, id="Basic"),
    pytest.param(lambda: Reupload(n_qubits=4, depth=2), 4, id="Reupload"),
    pytest.param(lambda: Ensemble(in_features=4, n_qubits=2, depth=2), 4, id="Ensemble"),
]


@pytest.mark.parametrize("make,in_features", DEFAULT_LAYERS)
def test_weights_reach_the_output_on_default_arguments(make, in_features):
    torch.manual_seed(0)
    x = torch.linspace(0.2, 1.1, in_features).reshape(1, in_features)
    assert _max_grad(make(), x) > 1e-6


@pytest.mark.parametrize("make,in_features", DEFAULT_LAYERS)
def test_data_reaches_the_output_on_default_arguments(make, in_features):
    torch.manual_seed(0)
    layer = make()
    with torch.no_grad():
        zeros = layer(torch.zeros(1, in_features))
        other = layer(torch.linspace(0.2, 1.1, in_features).reshape(1, in_features))
    assert (zeros - other).abs().max().item() > 1e-6


def test_a_default_layer_inside_a_network_actually_trains():
    # The shape docs/user-guide/estimators.md documents: a quantum layer
    # sandwiched between classical ones. With an inert circuit such a network
    # still reaches a good score — the two `nn.Linear` layers do all the work
    # while the circuit is decoration — so the check is that the quantum weights
    # move. Trained here with a plain torch loop rather
    # than an estimator, because the estimators deep-copy the model they are
    # given and the object handed in would show no movement either way.
    from torch import nn

    torch.manual_seed(0)
    quantum = Basic(n_qubits=4, depth=2)
    net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), quantum, nn.Linear(4, 2))
    before = [p.detach().clone() for p in quantum.parameters()]

    generator = torch.Generator().manual_seed(0)
    X = torch.randn(60, 2, generator=generator)
    y = (X[:, 0] + X[:, 1] > 0).long()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(10):
        optimizer.zero_grad()
        nn.functional.cross_entropy(net(X), y).backward()
        optimizer.step()

    moved = max(
        (a - b).abs().max().item()
        for a, b in zip(quantum.parameters(), before, strict=True)
    )
    assert moved > 1e-6, "the quantum layer trained no weights"


def test_rz_with_a_z_basis_is_reachable_and_degenerate():
    # The defaults avoid this combination; the physics does not. RZ variational
    # with Z measurement stays constructible on request, warns, and has
    # identically zero gradients.
    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="is degenerate"):
        layer = Basic(n_qubits=4, depth=2, var_rotation="RZ", basis="Z")
    x = torch.linspace(0.2, 1.1, 4).reshape(1, 4)
    assert _max_grad(layer, x) < 1e-12
