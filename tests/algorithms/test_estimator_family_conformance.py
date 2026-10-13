"""One contract, checked against every torch estimator at once.

The torch estimators share one contract: `set_params` reaches the architecture
`fit()` builds, a saved checkpoint loads back, a classifier exposes `classes_`.
A property that holds in one estimator can be missing from a sibling without
any test noticing, as long as each estimator is tested only on its own.

This file tests them as a family. A new estimator joins by being registered;
a sibling that drifts fails here rather than in a user's script.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import eigenkit.torch.estimators  # noqa: F401 — populates the registry
from eigenkit.qml._base import BaseQuantumClassifier
from eigenkit.qml._registry import _REGISTRY


# The neural estimators: those built on the torch ring. The kernel and
# reservoir estimators have a different lifecycle and are covered elsewhere.
assert _REGISTRY, "estimator registry is empty; the import above stopped working"
NEURAL = {
    name: cls
    for name, cls in sorted(_REGISTRY.items())
    if cls.__module__.startswith("eigenkit.torch.estimators")
}
# Wrappers that take another estimator rather than being built directly.
WRAPPERS = {"QuantumTimeSeriesClassifier", "QuantumTimeSeriesRegressor"}
DIRECT = {k: v for k, v in NEURAL.items() if k not in WRAPPERS}


def _build(name, cls):
    """Construct with the smallest workable configuration for each shape."""
    common = dict(epochs=1, n_qubits=2, depth=1)
    if name.startswith("Custom"):
        from eigenkit.configs import Encoding, Variational

        return cls(
            blocks=[Encoding(in_features=2), Variational(depth=1)],
            in_features=2,
            epochs=1,
        )
    if name.startswith("Generic"):
        from torch import nn

        out = 2 if name.endswith("Classifier") else 1
        return cls(model=nn.Sequential(nn.Linear(2, out)), epochs=1)
    if name.startswith("Recurrent"):
        return cls(seq_feature_dim=2, **common)
    return cls(in_features=2, **common)


def _fit(name, estimator):
    rng = np.random.default_rng(0)
    if name.startswith("Recurrent"):
        X = rng.normal(size=(16, 3, 2))
    else:
        X = rng.normal(size=(16, 2))
    y = (rng.normal(size=16) > 0).astype(int)
    if not issubclass(type(estimator), BaseQuantumClassifier):
        y = rng.normal(size=16)
    return estimator.fit(X, y), X, y


IDS = sorted(DIRECT)


@pytest.mark.parametrize("name", IDS)
def test_fit_leaves_the_model_ready_to_predict(name):
    """`model_` is the documented handle on the fitted network. Left in
    training mode it disagrees with `predict()` for any model with dropout or
    normalisation."""
    fitted, _, _ = _fit(name, _build(name, DIRECT[name]))
    assert not fitted.model_.training


@pytest.mark.parametrize("name", IDS)
def test_n_features_in_is_recorded(name):
    fitted, X, _ = _fit(name, _build(name, DIRECT[name]))
    assert hasattr(fitted, "n_features_in_"), f"{name} does not record n_features_in_"


@pytest.mark.parametrize("name", [n for n in IDS if n.endswith(("Classifier", "HQNC"))])
def test_classifiers_expose_classes(name):
    fitted, _, y = _fit(name, _build(name, DIRECT[name]))
    assert hasattr(fitted, "classes_"), f"{name} does not expose classes_"
    np.testing.assert_array_equal(np.asarray(fitted.classes_), np.unique(y))


@pytest.mark.parametrize("name", IDS)
def test_save_load_round_trips(name, tmp_path):
    estimator = _build(name, DIRECT[name])
    fitted, X, _ = _fit(name, estimator)
    path = tmp_path / f"{name}.pt"
    fitted.save(str(path))
    # Generic* holds a user-supplied nn.Module it cannot reconstruct, and says
    # so: its `load` takes the architecture back. That is documented, not drift.
    if name.startswith("Generic"):
        restored = type(fitted).load(str(path), model=_build(name, DIRECT[name]).model)
    else:
        restored = type(fitted).load(str(path))
    np.testing.assert_allclose(restored.predict(X), fitted.predict(X), atol=1e-5)


@pytest.mark.parametrize("name", IDS)
def test_set_params_reaches_the_built_model(name):
    """`set_params` is how GridSearchCV applies a grid point. An architecture
    cached at construction would make every point train the same model."""
    estimator = _build(name, DIRECT[name])
    if "n_qubits" not in estimator.get_params():
        pytest.skip(f"{name} has no n_qubits to vary")
    estimator.set_params(n_qubits=3)
    fitted, _, _ = _fit(name, estimator)
    used = {m.n_qubits for m in fitted.model_.modules() if hasattr(m, "n_qubits")}
    assert used == {3}, f"{name} built {used} after set_params(n_qubits=3)"


# The kernel and reservoir estimators live in the same registry but have a
# different lifecycle (no `model_`, no torch RNG), so they get the part of the
# contract that does apply to them.
# PortfolioRiskEstimator is registered too, but fits a return series rather
# than a feature matrix; tests/algorithms/test_portfolio_risk.py covers it. It
# registers only once imported, so it is excluded by name rather than left to
# depend on which test module happened to import it first.
NON_NEURAL = {
    name: cls
    for name, cls in sorted(_REGISTRY.items())
    if not cls.__module__.startswith("eigenkit.torch.estimators")
    and name != "PortfolioRiskEstimator"
}


def _build_non_neural(name, cls):
    if name.startswith("QSV"):
        return cls(feature_map="Angle", reps=1)
    from eigenkit.core.blocks import AngleEncoding, BasicVariational

    return cls(
        blocks=[AngleEncoding(in_features=2), BasicVariational(depth=1)],
        random_state=0,
    )


def _fit_non_neural(name, estimator):
    rng = np.random.default_rng(0)
    if name.startswith("QSV"):
        X = rng.normal(size=(12, 2))
        y = (X[:, 0] > 0).astype(int) if name == "QSVC" else X[:, 0]
    else:
        X = rng.normal(size=(12, 4, 2))
        y = (
            (rng.normal(size=(12, 4)) > 0).astype(int)
            if name.endswith("Classifier")
            else rng.normal(size=(12, 4))
        )
    return estimator.fit(X, y), X, y


@pytest.mark.parametrize("name", sorted(NON_NEURAL))
def test_non_neural_estimators_record_their_feature_count(name):
    fitted, _, _ = _fit_non_neural(name, _build_non_neural(name, NON_NEURAL[name]))
    assert hasattr(fitted, "n_features_in_"), f"{name} does not record it"


@pytest.mark.parametrize("name", sorted(NON_NEURAL))
def test_non_neural_estimators_reject_the_wrong_feature_count(name):
    """`n_features_in_` is only worth recording if something checks it. An
    estimator that recorded 2 and accepted 5 columns at predict time would
    silently read the first 2 of each row."""
    fitted, X, _ = _fit_non_neural(name, _build_non_neural(name, NON_NEURAL[name]))
    wrong = np.concatenate([X, X[..., :1]], axis=-1)
    with pytest.raises(ValueError):
        fitted.predict(wrong)


@pytest.mark.parametrize("name", sorted(NON_NEURAL))
def test_predicting_before_fit_raises_not_fitted(name):
    from sklearn.exceptions import NotFittedError

    estimator = _build_non_neural(name, NON_NEURAL[name])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 2)) if name.startswith("QSV") else rng.normal(size=(4, 3, 2))
    with pytest.raises(NotFittedError):
        estimator.predict(X)


@pytest.mark.parametrize("name", IDS)
def test_the_same_random_state_reproduces_the_fit(name):
    def weights():
        torch.manual_seed(0)
        fitted, X, _ = _fit(name, _build(name, DIRECT[name]))
        return torch.cat([p.detach().flatten() for p in fitted.model_.parameters()])

    torch.testing.assert_close(weights(), weights())
