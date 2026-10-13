"""A layer can be asked which backend it runs on.

`provider=` is the one knob that decides where a circuit executes, so it must
be readable back, not write-only. The circuit-building layers hand it to
`TorchCircuit`, and `TorchCircuit` wraps a `core.Circuit` that exposes
`.provider`; both forward it, so the whole torch ring can answer the question
`Circuit.provider` answers.

Which layers have the property follows the same line as which layers take the
argument: the three that own a circuit. `MinMaxNorm` has no circuit, so it
neither takes `provider=` nor answers for one.
"""

from __future__ import annotations

import pytest

from eigenkit import Circuit, Encoding, Variational
from eigenkit.configs import NumpyConfig, PennyLaneConfig
from eigenkit.core.provider import Provider
from eigenkit.torch.circuit import TorchCircuit
from eigenkit.torch.layers import Basic, Ensemble, MinMaxNorm, Reupload


LAYERS = [
    pytest.param(lambda p: Basic(n_qubits=2, depth=1, provider=p), id="Basic"),
    pytest.param(lambda p: Reupload(n_qubits=2, depth=1, provider=p), id="Reupload"),
    pytest.param(
        lambda p: Ensemble(in_features=4, n_qubits=2, depth=1, provider=p),
        id="Ensemble",
    ),
]


@pytest.mark.parametrize("build", LAYERS)
def test_the_name_comes_back(build):
    assert build("numpy").provider.name == "numpy"


@pytest.mark.parametrize("build", LAYERS)
def test_it_is_the_resolved_provider_object(build):
    provider = build("numpy").provider
    assert isinstance(provider, Provider)
    # Same object the circuit compiled against, not a copy or a bare string.
    layer = build("numpy")
    inner = layer.circuit if hasattr(layer, "circuit") else layer.circuits[0]
    assert layer.provider is inner.provider


@pytest.mark.parametrize("build", LAYERS)
@pytest.mark.parametrize(
    "spec,expected",
    [("numpy", "numpy"), (NumpyConfig(), "numpy")],
    ids=["name", "config"],
)
def test_every_spec_form_resolves_to_the_same_name(build, spec, expected):
    # `provider=` takes a registered name, a config, or a backend object. All
    # three resolve to one Provider, so the property answers the same way
    # whichever form was used — that is the point of reading it back.
    assert build(spec).provider.name == expected


def test_a_config_form_resolves_too():
    pytest.importorskip("pennylane")
    assert Basic(n_qubits=2, depth=1, provider=PennyLaneConfig()).provider.name == (
        "pennylane"
    )


def test_torch_circuit_forwards_what_core_circuit_exposes():
    blocks = [Encoding(in_features=2), Variational(depth=1)]
    assert TorchCircuit(blocks=blocks, provider="numpy").provider.name == (
        Circuit(blocks, provider="numpy").provider.name
    )


def test_with_provider_is_visible_through_the_property():
    layer = Basic(n_qubits=2, depth=1, provider="numpy")
    assert layer.provider.name == "numpy"
    layer.circuit = layer.circuit.with_provider("numpy")
    assert layer.provider.name == "numpy"


def test_minmaxnorm_takes_no_provider_and_exposes_none():
    # Deliberate, and the same boundary as the constructor argument: it owns no
    # circuit, so it has no backend of its own to report.
    assert not hasattr(MinMaxNorm(feature_dim=4), "provider")
