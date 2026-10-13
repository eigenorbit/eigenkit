"""An estimator builds the layer the layer documentation describes.

`HQNC(layer_type="Basic", ...)` builds a `Basic` layer and forwards four
arguments to it verbatim: `enc_rotation`, `var_rotation`, `variational_type`
and `basis`. On the estimator all four default to None, meaning "the default
of the layer you named". There is one place each value is decided, and it is
the layer: an estimator-level default would silently override the layer's own,
and rebuilding an estimator's architecture by hand from the layer page would
then produce a different model.
"""

from __future__ import annotations

import inspect
import warnings

import pytest
import torch

from eigenkit.torch.estimators._nn import _make_q_layer
from eigenkit.torch.layers import Basic, Ensemble, Reupload


ANSATZ_ARGS = ("enc_rotation", "var_rotation", "variational_type", "basis")

ESTIMATORS = ["HQNC", "HQNR", "RecurrentHQNC", "RecurrentHQNR"]

LAYER_TYPES = [
    pytest.param("Basic", lambda: Basic(n_qubits=4, depth=2), id="Basic"),
    pytest.param("Reupload", lambda: Reupload(n_qubits=4, depth=2), id="Reupload"),
    pytest.param(
        "Ensemble",
        lambda: Ensemble(in_features=4, n_qubits=2, depth=2),
        id="Ensemble",
    ),
]


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("argument", ANSATZ_ARGS)
def test_the_estimator_holds_no_opinion_of_its_own(name, argument):
    import eigenkit

    signature = inspect.signature(getattr(eigenkit, name).__init__)
    assert signature.parameters[argument].default is None


@pytest.mark.parametrize("layer_type,build_by_hand", LAYER_TYPES)
def test_the_estimator_builds_what_the_layer_page_describes(layer_type, build_by_hand):
    estimator_built = _make_q_layer(4, 2, 2, layer_type, None, None, None, None, "numpy")
    by_hand = build_by_hand()

    assert [tuple(p.shape) for p in estimator_built.parameters()] == [
        tuple(p.shape) for p in by_hand.parameters()
    ]

    # Same weights in, same numbers out — the circuits are identical, not just
    # the same size.
    with torch.no_grad():
        for target, source in zip(
            estimator_built.parameters(), by_hand.parameters(), strict=True
        ):
            target.copy_(source)
        x = torch.linspace(0.2, 1.1, by_hand.in_features).reshape(1, -1)
        assert torch.allclose(estimator_built(x), by_hand(x))


EXPLICIT = dict(enc_rotation="RY", var_rotation="RZ", variational_type="basic", basis="X")

EXPLICIT_BY_HAND = [
    pytest.param("Basic", lambda: Basic(n_qubits=4, depth=2, **EXPLICIT), id="Basic"),
    pytest.param(
        "Reupload", lambda: Reupload(n_qubits=4, depth=2, **EXPLICIT), id="Reupload"
    ),
    pytest.param(
        "Ensemble",
        lambda: Ensemble(in_features=4, n_qubits=2, depth=2, **EXPLICIT),
        id="Ensemble",
    ),
]


@pytest.mark.parametrize("layer_type,build_by_hand", EXPLICIT_BY_HAND)
def test_an_explicit_argument_is_forwarded_verbatim(layer_type, build_by_hand):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overridden = _make_q_layer(4, 2, 2, layer_type, "RY", "RZ", "basic", "X", "numpy")
        by_hand = build_by_hand()

    with torch.no_grad():
        for target, source in zip(
            overridden.parameters(), by_hand.parameters(), strict=True
        ):
            target.copy_(source)
        x = torch.linspace(0.2, 1.1, by_hand.in_features).reshape(1, -1)
        assert torch.allclose(overridden(x), by_hand(x))

    # And it is genuinely an override: this configuration is not any layer's
    # own default, so the result differs from the no-arguments build.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = _make_q_layer(4, 2, 2, layer_type, None, None, None, None, "numpy")
    with torch.no_grad():
        differs = default(x).shape != overridden(x).shape or not torch.allclose(
            default(x), overridden(x)
        )
    assert differs


def test_a_bad_value_is_still_rejected():
    for bad in (
        dict(enc_rotation="RQ"),
        dict(var_rotation="nope"),
        dict(basis="W"),
        dict(variational_type="stronger"),
    ):
        kwargs = dict(
            enc_rotation=None, var_rotation=None, variational_type=None, basis=None
        )
        kwargs.update(bad)
        with pytest.raises(ValueError):
            _make_q_layer(
                4,
                2,
                2,
                "Basic",
                kwargs["enc_rotation"],
                kwargs["var_rotation"],
                kwargs["variational_type"],
                kwargs["basis"],
                "numpy",
            )


def test_the_default_estimator_layer_is_trainable():
    # The point of the whole exercise: the layer an estimator builds with no
    # ansatz arguments has weights that reach its output.
    layer = _make_q_layer(4, 2, 2, "Basic", None, None, None, None, "numpy")
    layer(torch.linspace(0.2, 1.1, 4).reshape(1, 4)).sum().backward()
    assert max(p.grad.abs().max().item() for p in layer.parameters()) > 1e-6
