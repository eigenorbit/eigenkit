import numpy as np
import pytest
from sklearn.base import RegressorMixin, clone


def test_hqnr_predict_shape(reg_data):
    from eigenkit.torch.estimators.hqnr import HQNR

    X_tr, y_tr, X_te, _ = reg_data
    model = HQNR(in_features=4, depth=1, epochs=2, provider="numpy")
    model.fit(X_tr, y_tr)
    assert model.predict(X_te).shape == (len(X_te),)


def test_hqnr_predict_dtype(reg_data):
    from eigenkit.torch.estimators.hqnr import HQNR

    X_tr, y_tr, X_te, _ = reg_data
    model = HQNR(in_features=4, depth=1, epochs=2, provider="numpy")
    model.fit(X_tr, y_tr)
    assert model.predict(X_te).dtype == np.float32


def test_hqnr_is_regressor():
    from eigenkit.torch.estimators.hqnr import HQNR

    assert isinstance(HQNR(in_features=4), RegressorMixin)


def test_hqnr_score(reg_data):
    from eigenkit.torch.estimators.hqnr import HQNR

    X_tr, y_tr, X_te, y_te = reg_data
    model = HQNR(in_features=4, depth=1, epochs=2, provider="numpy")
    model.fit(X_tr, y_tr)
    assert isinstance(model.score(X_te, y_te), float)


def test_hqnr_clone():
    from eigenkit.torch.estimators.hqnr import HQNR

    assert clone(HQNR(in_features=4, depth=2)).in_features == 4


def test_hqnr_train_losses(reg_data):
    from eigenkit.torch.estimators.hqnr import HQNR

    X_tr, y_tr, _, _ = reg_data
    model = HQNR(in_features=4, depth=1, epochs=3, provider="numpy")
    model.fit(X_tr, y_tr)
    assert len(model.train_losses_) == 3


def test_hqnr_validation_losses(reg_data):
    from eigenkit.torch.estimators.hqnr import HQNR

    X_tr, y_tr, X_te, y_te = reg_data
    model = HQNR(in_features=4, depth=1, epochs=2, provider="numpy")
    model.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    assert len(model.validation_losses_) == 2


def test_hqnr_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    from eigenkit.torch.estimators.hqnr import HQNR

    model = HQNR(in_features=4)
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((5, 4), dtype=np.float32))


@pytest.mark.parametrize("loss_name", ["mse", "mae", "huber"])
def test_hqnr_supported_losses_train(reg_data, loss_name):
    from eigenkit.torch.estimators.hqnr import HQNR

    X_tr, y_tr, _, _ = reg_data
    model = HQNR(in_features=4, depth=1, epochs=2, provider="numpy", loss_func=loss_name)
    model.fit(X_tr, y_tr)
    assert len(model.train_losses_) == 2


def test_hqnr_unknown_loss_raises(reg_data):
    from eigenkit.torch.estimators.hqnr import HQNR

    X_tr, y_tr, _, _ = reg_data
    with pytest.raises(ValueError, match="Unknown loss_func"):
        HQNR(in_features=4, depth=1, epochs=2, provider="numpy", loss_func="unknown")


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw", "rmsprop", "sgd"])
def test_hqnr_supported_optimizers_train(reg_data, optimizer_name):
    from eigenkit.torch.estimators.hqnr import HQNR

    X_tr, y_tr, _, _ = reg_data
    model = HQNR(
        in_features=4,
        depth=1,
        epochs=2,
        provider="numpy",
        optimizer=optimizer_name,
    )
    model.fit(X_tr, y_tr)
    assert len(model.train_losses_) == 2


def test_hqnr_unknown_optimizer_raises(reg_data):
    from eigenkit.torch.estimators.hqnr import HQNR

    X_tr, y_tr, _, _ = reg_data
    with pytest.raises(ValueError, match="Unknown optimizer"):
        HQNR(
            in_features=4,
            depth=1,
            epochs=2,
            provider="numpy",
            optimizer="unknown",
        ).fit(X_tr, y_tr)


def test_hqnr_q_layer_mismatch_roundtrip_fit_predict():
    from eigenkit.torch.estimators.hqnr import HQNR

    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(24, 8)).astype(np.float32)
    y_train = (0.5 * X_train[:, 0] - 0.3 * X_train[:, 1]).astype(np.float32)
    X_test = rng.normal(size=(12, 8)).astype(np.float32)

    with pytest.warns(UserWarning, match="For layer_type"):
        model = HQNR(
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
    assert preds.dtype == np.float32
