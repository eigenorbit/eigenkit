import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone


def test_hqnc_fit_predict(clf_data):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, X_te, _ = clf_data
    model = HQNC(in_features=4, depth=1, n_classes=2, epochs=2, provider="numpy")
    model.fit(X_tr, y_tr)
    assert model.predict(X_te).shape == (len(X_te),)


def test_hqnc_predict_classes(clf_data):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, X_te, _ = clf_data
    model = HQNC(in_features=4, depth=1, epochs=2, provider="numpy")
    model.fit(X_tr, y_tr)
    assert set(model.predict(X_te)).issubset({0, 1})


def test_hqnc_is_classifier():
    from eigenkit.torch.estimators.hqnc import HQNC

    assert isinstance(HQNC(in_features=4), ClassifierMixin)


def test_hqnc_score(clf_data):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, X_te, y_te = clf_data
    model = HQNC(in_features=4, depth=1, epochs=2, provider="numpy")
    model.fit(X_tr, y_tr)
    assert 0.0 <= model.score(X_te, y_te) <= 1.0


def test_hqnc_clone():
    from eigenkit.torch.estimators.hqnc import HQNC

    cloned = clone(HQNC(in_features=4, depth=2, n_classes=3))
    assert cloned.in_features == 4 and cloned.n_classes == 3


def test_hqnc_eq_layer(clf_data):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, X_te, _ = clf_data
    model = HQNC(
        in_features=4,
        n_qubits=2,
        depth=1,
        layer_type="Ensemble",
        epochs=2,
        provider="numpy",
    )
    model.fit(X_tr, y_tr)
    assert model.predict(X_te).shape == (len(X_te),)


def test_hqnc_q_layer_rejects_n_qubits_mismatch():
    from eigenkit.torch.estimators.hqnc import HQNC

    rng = np.random.default_rng(123)
    X = rng.normal(size=(16, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    with pytest.warns(UserWarning, match="For layer_type"):
        HQNC(
            in_features=8, n_qubits=4, depth=1, layer_type="Basic", provider="numpy"
        ).fit(X, y)


def test_hqnc_q_layer_mismatch_roundtrip_fit_predict():
    from eigenkit.torch.estimators.hqnc import HQNC

    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(24, 8)).astype(np.float32)
    y_train = (X_train[:, 0] + 0.2 * X_train[:, 1] > 0.0).astype(np.int64)
    X_test = rng.normal(size=(10, 8)).astype(np.float32)

    with pytest.warns(UserWarning, match="For layer_type"):
        model = HQNC(
            in_features=8,
            n_qubits=4,
            depth=1,
            layer_type="Basic",
            epochs=2,
            provider="numpy",
        )
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(preds).issubset(set(np.unique(y_train)))


def test_hqnc_train_losses(clf_data):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, _, _ = clf_data
    model = HQNC(in_features=4, depth=1, epochs=3, provider="numpy")
    model.fit(X_tr, y_tr)
    assert len(model.train_losses_) == 3


def test_hqnc_validation_losses(clf_data):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, X_te, y_te = clf_data
    model = HQNC(in_features=4, depth=1, epochs=2, provider="numpy")
    model.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    assert len(model.validation_losses_) == 2


def test_hqnc_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    from eigenkit.torch.estimators.hqnc import HQNC

    model = HQNC(in_features=4)
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((5, 4), dtype=np.float32))


def test_hqnc_predict_proba(clf_data):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, X_te, _ = clf_data
    model = HQNC(in_features=4, depth=1, epochs=2, provider="numpy")
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), len(np.unique(y_tr)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    preds = model.predict(X_te)
    assert np.array_equal(preds, model._label_encoder.classes_[np.argmax(proba, axis=1)])


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw", "rmsprop", "sgd"])
def test_hqnc_supported_optimizers_train(clf_data, optimizer_name):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, _, _ = clf_data
    model = HQNC(
        in_features=4,
        depth=1,
        epochs=2,
        provider="numpy",
        optimizer=optimizer_name,
    )
    model.fit(X_tr, y_tr)
    assert len(model.train_losses_) == 2


def test_hqnc_unknown_optimizer_raises(clf_data):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, _, _ = clf_data
    with pytest.raises(ValueError, match="Unknown optimizer"):
        HQNC(
            in_features=4,
            depth=1,
            epochs=2,
            provider="numpy",
            optimizer="unknown",
        ).fit(X_tr, y_tr)
