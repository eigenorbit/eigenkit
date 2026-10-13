import numpy as np
import pytest
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from torch import nn

from eigenkit.configs import Encoding, StrongVariational
from eigenkit.core.observables import all_z
from eigenkit.torch.estimators import (
    CustomQuantumClassifier,
    CustomQuantumRegressor,
)


def _default_blocks(in_features: int = 4):
    return [
        Encoding(in_features=in_features, rotation="RY"),
        StrongVariational(depth=1),
    ]


def _default_observables(in_features: int = 4):
    return all_z(in_features)


def test_classifier_fit_predict(clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    model = CustomQuantumClassifier(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
        n_classes=2,
        epochs=2,
        batch_size=8,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert set(preds).issubset({0, 1})


def test_regressor_fit_predict(reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    model = CustomQuantumRegressor(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
        epochs=2,
        batch_size=8,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert preds.dtype == np.float32


def test_in_features_inferred_from_encoding_block(clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    model = CustomQuantumClassifier(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
        epochs=2,
        batch_size=8,
    )
    model.fit(X_tr, y_tr)
    assert model.predict(X_te).shape == (len(X_te),)


def test_input_adapter_inserted_when_mismatched():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model = CustomQuantumClassifier(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
        in_features=6,
        epochs=1,
        batch_size=8,
    )
    model.fit(X, y)
    assert isinstance(model._model[0], nn.Linear)
    assert model._model[0].in_features == 6
    assert model._model[0].out_features == 4


def test_no_input_adapter_when_matched(clf_data):
    X_tr, y_tr, _, _ = clf_data
    model = CustomQuantumClassifier(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
        in_features=4,
        epochs=1,
        batch_size=8,
    )
    model.fit(X_tr, y_tr)
    assert not isinstance(model._model[0], nn.Linear)


def test_provider_string_is_stored():
    """The provider string is stored verbatim, not wrapped in a device object."""
    model = CustomQuantumClassifier(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
    )
    assert model.provider == "numpy"


def test_is_classifier_and_regressor():
    assert isinstance(
        CustomQuantumClassifier(
            blocks=_default_blocks(4),
            observables=_default_observables(4),
        ),
        ClassifierMixin,
    )
    assert isinstance(
        CustomQuantumRegressor(
            blocks=_default_blocks(4),
            observables=_default_observables(4),
        ),
        RegressorMixin,
    )


def test_sklearn_clone_classifier():
    blocks = _default_blocks(4)
    obs = _default_observables(4)
    model = CustomQuantumClassifier(
        blocks=blocks,
        observables=obs,
        provider="numpy",
        n_classes=3,
        epochs=5,
    )
    cloned = clone(model)
    assert cloned.n_classes == 3
    assert cloned.epochs == 5
    assert cloned.blocks == blocks


def test_sklearn_clone_regressor():
    blocks = _default_blocks(4)
    obs = _default_observables(4)
    cloned = clone(
        CustomQuantumRegressor(blocks=blocks, observables=obs, provider="numpy", epochs=7)
    )
    assert cloned.epochs == 7
    assert cloned.blocks == blocks


def test_classifier_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    model = CustomQuantumClassifier(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
    )
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((3, 4), dtype=np.float32))


def test_regressor_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    model = CustomQuantumRegressor(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
    )
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((3, 4), dtype=np.float32))


def test_train_losses_recorded(clf_data):
    X_tr, y_tr, _, _ = clf_data
    model = CustomQuantumClassifier(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
        epochs=3,
        batch_size=8,
    )
    model.fit(X_tr, y_tr)
    assert len(model.train_losses_) == 3


def test_classifier_validation_losses_recorded(clf_data):
    X_tr, y_tr, X_te, y_te = clf_data
    model = CustomQuantumClassifier(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
        epochs=2,
        batch_size=8,
    )
    model.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    assert len(model.train_losses_) == 2
    assert len(model.validation_losses_) == 2


def test_regressor_validation_losses_persisted(tmp_path, reg_data):
    X_tr, y_tr, X_te, y_te = reg_data
    model = CustomQuantumRegressor(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
        epochs=2,
        batch_size=8,
    )
    model.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    losses_before = list(model.validation_losses_)

    model.save(tmp_path / "custom_reg")
    loaded = CustomQuantumRegressor.load(tmp_path / "custom_reg")
    assert loaded.validation_losses_ == losses_before


def test_classifier_predict_proba(clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    model = CustomQuantumClassifier(
        blocks=_default_blocks(4),
        observables=_default_observables(4),
        provider="numpy",
        epochs=2,
        batch_size=8,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), len(np.unique(y_tr)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    preds = model.predict(X_te)
    assert np.array_equal(preds, model._label_encoder.classes_[np.argmax(proba, axis=1)])
