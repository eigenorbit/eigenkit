"""`set_params` must change the model that `fit()` builds.

`set_params` is how scikit-learn's search estimators apply a grid point, so the
architecture is derived from the public attributes when the model is built, not
cached in `__init__`. With a cached architecture `set_params(n_qubits=...)`
would update the public attribute and nothing else: a `GridSearchCV` over
`n_qubits`, `in_quantum` or `layer_type` would train one architecture for the
whole grid and report a comparison between identical models.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit import HQNC, HQNR


def _classification(n_features=4):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, n_features))
    return X, (X[:, 0] > 0).astype(int)


def _regression(n_features=4):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, n_features))
    return X, X[:, 0]


def _qubits_in(model) -> list[int]:
    return [m.n_qubits for m in model.modules() if hasattr(m, "n_qubits")]


ESTIMATORS = [
    (HQNC, _classification),
    (HQNR, _regression),
]


@pytest.mark.parametrize("cls,data", ESTIMATORS, ids=lambda v: getattr(v, "__name__", ""))
def test_set_params_changes_the_built_architecture(cls, data):
    X, y = data()
    estimator = cls(in_features=4, n_qubits=2, depth=1, epochs=1)
    estimator.set_params(n_qubits=3)
    estimator.fit(X, y)
    assert estimator.n_qubits == 3
    assert _qubits_in(estimator.model_) == [3] * len(_qubits_in(estimator.model_))
    assert 2 not in _qubits_in(estimator.model_)


@pytest.mark.parametrize("cls,data", ESTIMATORS, ids=lambda v: getattr(v, "__name__", ""))
def test_two_grid_points_give_two_architectures(cls, data):
    """The failure this guards is silent: both fits succeed and return
    plausible scores, they are just the same model twice."""
    X, y = data()
    built = []
    for n_qubits in (2, 3):
        estimator = cls(in_features=4, n_qubits=2, depth=1, epochs=1)
        estimator.set_params(n_qubits=n_qubits)
        estimator.fit(X, y)
        built.append(_qubits_in(estimator.model_))
    assert built[0] != built[1], f"both grid points built the same model: {built}"


@pytest.mark.parametrize("cls,data", ESTIMATORS, ids=lambda v: getattr(v, "__name__", ""))
def test_get_params_round_trips_through_set_params(cls, data):
    estimator = cls(in_features=4, n_qubits=2, depth=1, epochs=1)
    params = estimator.get_params()
    clone = cls(**{k: v for k, v in params.items()})
    assert clone.get_params() == params


@pytest.mark.parametrize("cls,data", ESTIMATORS, ids=lambda v: getattr(v, "__name__", ""))
def test_the_derived_values_follow_the_public_attribute(cls, data):
    estimator = cls(in_features=4, n_qubits=2, depth=1, epochs=1)
    assert estimator._resolved_n_qubits == 2
    estimator.set_params(n_qubits=5)
    assert estimator._resolved_n_qubits == 5


@pytest.mark.parametrize("cls,data", ESTIMATORS, ids=lambda v: getattr(v, "__name__", ""))
def test_the_adapter_warning_fires_once_per_fit(cls, data, recwarn):
    """Deriving on read is right, but `_resolve_quantum_dims` warns, so
    `_build_model` must resolve once per fit. Reading the derived properties
    repeatedly would repeat the same warning and redo the resolution each time."""
    X, y = data(4)
    # in_quantum != n_qubits for layer_type="Basic", which is what warns.
    estimator = cls(in_features=4, n_qubits=2, depth=1, epochs=1, show_progress=False)
    estimator.fit(X, y)
    adapter_warnings = [w for w in recwarn if "input adapter" in str(w.message)]
    assert len(adapter_warnings) == 1, (
        f"expected one adapter warning per fit, got {len(adapter_warnings)}"
    )
