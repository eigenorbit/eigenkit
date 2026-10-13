"""The network an estimator trained is reachable after `fit()`.

`fit()` trains `copy.deepcopy(self.model)`, so it never mutates the object the
estimator was configured with. That is sklearn semantics and it is right, and
it is why the copy needs a public handle: `model_` is the network that trained.

Without it the obvious check misleads. Hand a quantum layer to
`GenericClassifier(model=...)`, train, then inspect the weights of the layer
you passed — they have not moved, because they never could. Read as evidence
about the circuit, that is simply wrong, and it is an easy reading to make.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError
from torch import nn

from eigenkit import (
    HQNC,
    HQNR,
    CustomQuantumClassifier,
    CustomQuantumRegressor,
    GenericClassifier,
    GenericRegressor,
    RecurrentHQNC,
    RecurrentHQNR,
)


def _classification_data(n_features=4):
    rng = np.random.RandomState(0)
    X = rng.randn(24, n_features).astype("float32")
    return X, (X[:, 0] > 0).astype(int)


def _regression_data(n_features=4):
    rng = np.random.RandomState(0)
    X = rng.randn(24, n_features).astype("float32")
    return X, X[:, 0].astype("float32")


def _sequence_data(n_features=4):
    rng = np.random.RandomState(0)
    X = rng.randn(16, 5, n_features).astype("float32")
    return X, (X[:, -1, 0] > 0).astype(int)


COMMON = dict(epochs=1, show_progress=False, random_state=0)


def _net(out_features):
    return nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, out_features))


def _blocks():
    from eigenkit import Encoding, Variational

    return [Encoding(in_features=4), Variational(depth=1)]


ESTIMATORS = [
    pytest.param(
        lambda: HQNC(in_features=4, depth=1, **COMMON), _classification_data, id="HQNC"
    ),
    pytest.param(
        lambda: HQNR(in_features=4, depth=1, **COMMON), _regression_data, id="HQNR"
    ),
    pytest.param(
        lambda: RecurrentHQNC(seq_feature_dim=4, rnn_hidden_dim=4, depth=1, **COMMON),
        _sequence_data,
        id="RecurrentHQNC",
    ),
    pytest.param(
        lambda: RecurrentHQNR(seq_feature_dim=4, rnn_hidden_dim=4, depth=1, **COMMON),
        lambda: (_sequence_data()[0], _sequence_data()[1].astype("float32")),
        id="RecurrentHQNR",
    ),
    pytest.param(
        lambda: GenericClassifier(model=_net(2), in_features=4, **COMMON),
        _classification_data,
        id="GenericClassifier",
    ),
    pytest.param(
        lambda: GenericRegressor(model=_net(1), in_features=4, **COMMON),
        _regression_data,
        id="GenericRegressor",
    ),
    pytest.param(
        lambda: CustomQuantumClassifier(blocks=_blocks(), **COMMON),
        _classification_data,
        id="CustomQuantumClassifier",
    ),
    pytest.param(
        lambda: CustomQuantumRegressor(blocks=_blocks(), **COMMON),
        _regression_data,
        id="CustomQuantumRegressor",
    ),
]


@pytest.mark.parametrize("build,data", ESTIMATORS)
def test_model_is_unreachable_before_fit(build, data):
    estimator = build()
    with pytest.raises(NotFittedError):
        _ = estimator.model_


@pytest.mark.parametrize("build,data", ESTIMATORS)
def test_model_is_the_trained_network_after_fit(build, data):
    estimator = build()
    X, y = data()
    estimator.fit(X, y)

    assert isinstance(estimator.model_, nn.Module)
    assert list(estimator.model_.parameters()), "the trained model has no parameters"
    # It is the estimator's own object, not a fresh copy handed out per access.
    assert estimator.model_ is estimator.model_


def test_the_trap_this_closes():
    # The misreading `model_` guards against: "I passed a layer, I trained,
    # its weights did not move, so the layer does not train." The
    # weights of the object passed in never move — fit works on a copy — and
    # `model_` is what actually has to be inspected.
    torch.manual_seed(0)
    passed_in = _net(2)
    before = [p.detach().clone() for p in passed_in.parameters()]

    estimator = GenericClassifier(
        model=passed_in,
        in_features=4,
        epochs=8,
        lr=0.05,
        show_progress=False,
        random_state=0,
    )
    X, y = _classification_data()
    estimator.fit(X, y)

    handed_in_moved = max(
        (a - b).abs().max().item()
        for a, b in zip(passed_in.parameters(), before, strict=True)
    )
    trained_moved = max(
        (a - b).abs().max().item()
        for a, b in zip(estimator.model_.parameters(), before, strict=True)
    )
    assert handed_in_moved == 0.0, "fit() must not mutate the module it was given"
    assert trained_moved > 1e-6, "model_ is not the network that trained"
    assert estimator.model_ is not passed_in


def test_model_is_not_mistaken_for_a_fitted_attribute():
    # `model_` is a class-level property, so sklearn's `check_is_fitted` must
    # not see it and conclude the estimator is fitted. If it did, every
    # unfitted estimator would start passing its own guard.
    from sklearn.utils.validation import check_is_fitted

    with pytest.raises(NotFittedError):
        check_is_fitted(GenericClassifier(model=_net(2), in_features=4))


def test_mutating_the_model_reaches_the_estimator():
    # Documented as read access, but it hands back the live object rather than
    # a copy, and callers will fine-tune through it. Pin that it is the same
    # module the estimator predicts with.
    estimator = GenericClassifier(model=_net(2), in_features=4, **COMMON)
    X, y = _classification_data()
    estimator.fit(X, y)

    with torch.no_grad():
        for parameter in estimator.model_.parameters():
            parameter.zero_()
    outputs = estimator.predict_proba(X)
    assert np.allclose(outputs, outputs[0]), "zeroed weights still gave varied output"
