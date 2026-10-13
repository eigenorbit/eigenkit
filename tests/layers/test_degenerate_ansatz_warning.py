"""The degeneracy warning fires exactly when the circuit is degenerate.

`_expressivity.degenerate_reason` encodes four mechanisms that make a layer
inert — see that module for what each one is. A warning that cries wolf is
worse than no warning, and one that stays quiet on the configuration a user
actually built is no warning at all, so this file does not test the rules
against themselves: it measures every configuration and asserts the predicate
agrees with the measurement.

Degenerate means either the variational weights cannot reach the output (every
gradient exactly zero) or the input cannot (the output is constant in x). The
sweep below covers three layers, three qubit counts, three depths, three chunk
counts for `Ensemble`, both variational types and every rotation/basis
combination — 1944 configurations, of which 733 are degenerate.
"""

from __future__ import annotations

import itertools
import warnings

import pytest
import torch

from eigenkit.torch.layers import Basic, Ensemble, Reupload
from eigenkit.torch.layers._expressivity import degenerate_reason


TOL = 1e-9
ROTATIONS = ("RX", "RY", "RZ")
BASES = ("X", "Y", "Z")
TYPES = ("basic", "strong")


def _shapes():
    """(label, n_qubits, n_encodings, factory) for every layer shape swept."""
    for n_qubits, depth in itertools.product((1, 2, 4), (1, 2, 3)):
        yield (
            f"Basic(n_qubits={n_qubits}, depth={depth})",
            n_qubits,
            1,
            lambda nq=n_qubits, d=depth, **kw: Basic(n_qubits=nq, depth=d, **kw),
        )
        yield (
            f"Reupload(n_qubits={n_qubits}, depth={depth})",
            n_qubits,
            depth,
            lambda nq=n_qubits, d=depth, **kw: Reupload(n_qubits=nq, depth=d, **kw),
        )
    for n_qubits, depth, chunks in itertools.product((1, 2), (1, 2, 3), (1, 2, 3)):
        in_features = n_qubits * chunks
        yield (
            f"Ensemble(in_features={in_features}, n_qubits={n_qubits}, depth={depth})",
            n_qubits,
            1,
            lambda f=in_features, nq=n_qubits, d=depth, **kw: Ensemble(
                in_features=f, n_qubits=nq, depth=d, **kw
            ),
        )


def _build(factory, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return factory(**kwargs)


def _is_degenerate(layer, in_features) -> bool:
    x = torch.linspace(0.2, 1.3, in_features).reshape(1, in_features)
    layer(x).sum().backward()
    weights_dead = max(p.grad.abs().max().item() for p in layer.parameters()) < TOL
    with torch.no_grad():
        data_dead = (layer(torch.zeros(1, in_features)) - layer(x)).abs().max() < TOL
    return weights_dead or bool(data_dead)


@pytest.mark.parametrize("label,n_qubits,n_encodings,factory", list(_shapes()))
def test_the_predicate_agrees_with_the_measurement(label, n_qubits, n_encodings, factory):
    mismatches = []
    for enc, var, vtype, basis in itertools.product(ROTATIONS, ROTATIONS, TYPES, BASES):
        kwargs = dict(
            enc_rotation=enc,
            var_rotation=var,
            variational_type=vtype,
            basis=basis,
        )
        torch.manual_seed(0)
        layer = _build(factory, **kwargs)
        measured = _is_degenerate(layer, layer.in_features)
        predicted = (
            degenerate_reason(n_qubits=n_qubits, n_encodings=n_encodings, **kwargs)
            is not None
        )
        if measured != predicted:
            verdict = "warned but trains" if predicted else "degenerate but silent"
            mismatches.append(f"{label} {enc}/{var}/{vtype}/{basis}: {verdict}")
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize("label,n_qubits,n_encodings,factory", list(_shapes()))
def test_the_warning_fires_iff_the_reason_is_there(label, n_qubits, n_encodings, factory):
    # The predicate is verified above against measurement; this checks the
    # layers are actually wired to it, with the right n_encodings for each.
    for enc, var, vtype, basis in itertools.product(ROTATIONS, ROTATIONS, TYPES, BASES):
        kwargs = dict(
            enc_rotation=enc,
            var_rotation=var,
            variational_type=vtype,
            basis=basis,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            factory(**kwargs)
        expected = degenerate_reason(n_qubits=n_qubits, n_encodings=n_encodings, **kwargs)
        fired = [w for w in caught if "is degenerate" in str(w.message)]
        assert bool(fired) == (expected is not None), (
            f"{label} {enc}/{var}/{vtype}/{basis}"
        )
        if expected is not None:
            assert expected in str(fired[0].message)


def test_the_layer_is_still_built():
    # A degenerate circuit is a legitimate thing to construct on purpose — it is
    # the baseline for "does the quantum part contribute anything at all".
    with pytest.warns(UserWarning, match="is degenerate"):
        layer = Basic(n_qubits=4, depth=2, var_rotation="RZ", basis="Z")
    assert layer(torch.zeros(1, 4)).shape == (1, 4)


def test_the_default_layers_are_all_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Basic(n_qubits=4, depth=2)
        Reupload(n_qubits=4, depth=2)
        Ensemble(in_features=4, n_qubits=2, depth=2)


@pytest.mark.parametrize(
    "kwargs,phrase",
    [
        (dict(var_rotation="RZ", basis="Z"), "commutes with the <Z> observable"),
        (dict(var_rotation="RX", basis="X"), "commutes with the <X> observable"),
        (dict(enc_rotation="RZ"), "starts in |0...0>"),
        (dict(enc_rotation="RZ", var_rotation="RZ"), "every gate is diagonal"),
        (
            dict(enc_rotation="RY", var_rotation="RY", basis="Y"),
            "real matrices",
        ),
    ],
)
def test_each_mechanism_names_itself(kwargs, phrase):
    # The four mechanisms are distinct and the message must say which one it is:
    # the remedy differs. "Keep the axes distinct" would be wrong advice for the
    # RZ-encoding case, where no choice of the other two axes helps.
    with pytest.warns(UserWarning, match="is degenerate") as caught:
        Basic(n_qubits=4, depth=2, **kwargs)
    assert phrase in str(caught[0].message)


def test_an_unknown_rotation_gets_the_blocks_error_not_a_key_error():
    """The check runs before the blocks are built, so it must not judge a
    rotation it does not recognise: looking it up would raise `KeyError` from a
    private module for a plain typo, ahead of the `ValueError` the block
    raises. The predicate abstains and the block speaks."""
    for layer, kwargs in [
        (Basic, {"n_qubits": 2, "depth": 1}),
        (Reupload, {"n_qubits": 2, "depth": 1}),
        (Ensemble, {"n_qubits": 2, "depth": 1, "in_features": 2}),
    ]:
        with pytest.raises(ValueError, match="rotation must be RX/RY/RZ"):
            layer(enc_rotation="X", **kwargs)


def test_the_predicate_abstains_on_an_unknown_rotation():
    assert (
        degenerate_reason(
            enc_rotation="X",
            var_rotation="RZ",
            variational_type="basic",
            basis="Z",
            n_qubits=2,
            n_encodings=1,
        )
        is None
    )
