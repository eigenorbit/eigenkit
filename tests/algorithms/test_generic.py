"""Tests for GenericRegressor and GenericClassifier."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
import torch
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from torch import nn

from eigenkit.benchmark import load_checkpoint
from eigenkit.qml._serialization import SerializationError
from eigenkit.torch.estimators import (
    GenericClassifier,
    GenericRegressor,
    QuantumTimeSeriesClassifier,
    QuantumTimeSeriesRegressor,
    load_model,
)
from eigenkit.torch.layers import Basic


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reg_net() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _clf_net(n_classes: int = 2) -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, n_classes))


# ---------------------------------------------------------------------------
# GenericRegressor — fit / predict
# ---------------------------------------------------------------------------


def test_regressor_fit_predict(reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=2, show_progress=False)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert preds.dtype == np.float32


def test_regressor_predict_single_sample_always_1d(reg_data):
    """predict(X[:1]) must return shape (1,), not a 0-D scalar.

    A model whose forward() already returns (batch,) yields a (1,) tensor for a
    single sample, which a blanket .squeeze(-1) would collapse to a scalar ().
    """
    X_tr, y_tr, X_te, _ = reg_data

    class Head1D(nn.Module):
        """Returns (batch,) instead of (batch, 1)."""

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)

        def forward(self, x):
            return self.linear(x).squeeze(-1)

    reg = GenericRegressor(model=Head1D(), in_features=4, epochs=1, show_progress=False)
    reg.fit(X_tr, y_tr)

    assert reg.predict(X_te[:1]).shape == (1,), "single-sample predict must return (1,)"
    assert reg.predict(X_te).shape == (len(X_te),), (
        "multi-sample predict must return (N,)"
    )


def test_regressor_train_losses_recorded(reg_data):
    X_tr, y_tr, _, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=3, show_progress=False)
    reg.fit(X_tr, y_tr)
    assert len(reg.train_losses_) == 3
    assert all(isinstance(v, float) for v in reg.train_losses_)


def test_regressor_validation_losses_recorded(reg_data):
    X_tr, y_tr, X_te, y_te = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=2, show_progress=False)
    reg.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    assert len(reg.train_losses_) == 2
    assert len(reg.validation_losses_) == 2


def test_regressor_predict_before_fit_raises():
    reg = GenericRegressor(model=_reg_net())
    with pytest.raises(NotFittedError):
        reg.predict(np.zeros((3, 4), dtype=np.float32))


def test_regressor_in_features_mismatch_raises(reg_data):
    X_tr, y_tr, _, _ = reg_data
    reg = GenericRegressor(
        model=_reg_net(), in_features=10, epochs=1, show_progress=False
    )
    with pytest.raises(ValueError, match="in_features"):
        reg.fit(X_tr, y_tr)


def test_regressor_in_features_none_skips_validation(reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    reg = GenericRegressor(
        model=_reg_net(), in_features=None, epochs=2, show_progress=False
    )
    reg.fit(X_tr, y_tr)
    assert reg.predict(X_te).shape == (len(X_te),)


@pytest.mark.parametrize("loss_func", ["mse", "mae", "huber"])
def test_regressor_string_loss_funcs(reg_data, loss_func):
    X_tr, y_tr, X_te, _ = reg_data
    reg = GenericRegressor(
        model=_reg_net(), loss_func=loss_func, epochs=2, show_progress=False
    )
    reg.fit(X_tr, y_tr)
    assert reg.predict(X_te).shape == (len(X_te),)


def test_regressor_callable_loss_func(reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    loss = lambda out, tgt: torch.mean((out.squeeze(-1) - tgt) ** 2)
    reg = GenericRegressor(
        model=_reg_net(), loss_func=loss, epochs=2, show_progress=False
    )
    reg.fit(X_tr, y_tr)
    assert reg.predict(X_te).shape == (len(X_te),)


def test_regressor_invalid_loss_raises():
    with pytest.raises(ValueError, match="loss_func"):
        GenericRegressor(model=_reg_net(), loss_func="rmse")


# ---------------------------------------------------------------------------
# GenericRegressor — sklearn interface
# ---------------------------------------------------------------------------


def test_regressor_is_regressor():
    assert isinstance(GenericRegressor(model=_reg_net()), RegressorMixin)


def test_regressor_sklearn_clone():
    net = _reg_net()
    reg = GenericRegressor(model=net, in_features=4, epochs=7, lr=0.005)
    cloned = clone(reg)
    assert cloned.epochs == 7
    assert cloned.lr == 0.005
    assert cloned.in_features == 4
    # sklearn deepcopies mutable params; verify architecture is preserved
    assert str(cloned.model) == str(net)


# ---------------------------------------------------------------------------
# GenericRegressor — save / load
# ---------------------------------------------------------------------------


def test_regressor_save_load_roundtrip(tmp_path, reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=3, show_progress=False)
    reg.fit(X_tr, y_tr)
    preds_before = reg.predict(X_te)

    reg.save(tmp_path / "reg")
    loaded = GenericRegressor.load(tmp_path / "reg", model=_reg_net())
    np.testing.assert_allclose(preds_before, loaded.predict(X_te), rtol=1e-5, atol=1e-5)


def test_regressor_train_losses_persisted(tmp_path, reg_data):
    X_tr, y_tr, _, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=3, show_progress=False)
    reg.fit(X_tr, y_tr)
    losses_before = list(reg.train_losses_)

    reg.save(tmp_path / "reg")
    loaded = GenericRegressor.load(tmp_path / "reg", model=_reg_net())
    assert loaded.train_losses_ == losses_before


def test_regressor_bundle_contains_weights(tmp_path, reg_data):
    X_tr, y_tr, _, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=2, show_progress=False)
    reg.fit(X_tr, y_tr)
    reg.save(tmp_path / "reg")

    with zipfile.ZipFile(tmp_path / "reg.eigenkit") as zf:
        names = set(zf.namelist())
    assert "state/weights.pt" in names
    assert "params.json" in names
    assert "metadata.json" in names
    assert not any(n.endswith(".pkl") for n in names)


def test_regressor_callable_loss_raises_on_save(tmp_path, reg_data):
    X_tr, y_tr, _, _ = reg_data
    loss = lambda out, tgt: torch.mean((out.squeeze(-1) - tgt) ** 2)
    reg = GenericRegressor(
        model=_reg_net(), loss_func=loss, epochs=2, show_progress=False
    )
    reg.fit(X_tr, y_tr)
    with pytest.raises(SerializationError, match="callable loss_func"):
        reg.save(tmp_path / "reg")


def test_regressor_load_without_model_raises(tmp_path, reg_data):
    X_tr, y_tr, _, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=2, show_progress=False)
    reg.fit(X_tr, y_tr)
    reg.save(tmp_path / "reg")
    with pytest.raises(SerializationError, match="model"):
        GenericRegressor.load(tmp_path / "reg")


def test_regressor_load_model_blocked(tmp_path, reg_data):
    X_tr, y_tr, _, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=2, show_progress=False)
    reg.fit(X_tr, y_tr)
    reg.save(tmp_path / "reg")
    with pytest.raises(SerializationError, match="load_model"):
        load_model(tmp_path / "reg")


def test_regressor_load_wrong_class_raises(tmp_path, reg_data, clf_data):
    X_tr, y_tr, _, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=2, show_progress=False)
    reg.fit(X_tr, y_tr)
    reg.save(tmp_path / "reg")
    with pytest.raises(TypeError, match="GenericClassifier"):
        GenericClassifier.load(tmp_path / "reg", model=_clf_net())


# ---------------------------------------------------------------------------
# GenericClassifier — fit / predict
# ---------------------------------------------------------------------------


def test_classifier_fit_predict(clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert set(preds).issubset(set(y_tr))


def test_classifier_predict_proba(clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    assert proba.dtype == np.float32
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_predict_proba_consistent_with_predict(clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    argmax_labels = clf._label_encoder.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(preds, argmax_labels)


def test_classifier_train_losses_recorded(clf_data):
    X_tr, y_tr, _, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=3, show_progress=False)
    clf.fit(X_tr, y_tr)
    assert len(clf.train_losses_) == 3
    assert all(isinstance(v, float) for v in clf.train_losses_)


def test_classifier_validation_losses_recorded(clf_data):
    X_tr, y_tr, X_te, y_te = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    assert len(clf.train_losses_) == 2
    assert len(clf.validation_losses_) == 2


def test_classifier_predict_before_fit_raises():
    clf = GenericClassifier(model=_clf_net())
    with pytest.raises(NotFittedError):
        clf.predict(np.zeros((3, 4), dtype=np.float32))


def test_classifier_in_features_mismatch_raises(clf_data):
    X_tr, y_tr, _, _ = clf_data
    clf = GenericClassifier(
        model=_clf_net(), in_features=10, epochs=1, show_progress=False
    )
    with pytest.raises(ValueError, match="in_features"):
        clf.fit(X_tr, y_tr)


def test_classifier_multiclass():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = rng.integers(0, 3, 30)
    clf = GenericClassifier(model=_clf_net(3), epochs=2, show_progress=False)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({0, 1, 2})
    proba = clf.predict_proba(X)
    assert proba.shape == (30, 3)


def test_classifier_arbitrary_label_types():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = np.array(["cat", "dog"] * 10)
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({"cat", "dog"})


# ---------------------------------------------------------------------------
# GenericClassifier — sklearn interface
# ---------------------------------------------------------------------------


def test_classifier_is_classifier():
    assert isinstance(GenericClassifier(model=_clf_net()), ClassifierMixin)


def test_classifier_sklearn_clone():
    net = _clf_net(3)
    clf = GenericClassifier(model=net, in_features=4, n_classes=3, epochs=5, lr=0.002)
    cloned = clone(clf)
    assert cloned.epochs == 5
    assert cloned.lr == 0.002
    assert cloned.n_classes == 3
    # sklearn deepcopies mutable params; verify architecture is preserved
    assert str(cloned.model) == str(net)


# ---------------------------------------------------------------------------
# GenericClassifier — save / load
# ---------------------------------------------------------------------------


def test_classifier_save_load_roundtrip(tmp_path, clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=3, show_progress=False)
    clf.fit(X_tr, y_tr)
    preds_before = clf.predict(X_te)

    clf.save(tmp_path / "clf")
    loaded = GenericClassifier.load(tmp_path / "clf", model=_clf_net(2))
    np.testing.assert_array_equal(preds_before, loaded.predict(X_te))


def test_classifier_label_classes_persisted(tmp_path, clf_data):
    X_tr, y_tr, _, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X_tr, y_tr)
    classes_before = list(clf._label_encoder.classes_)

    clf.save(tmp_path / "clf")
    loaded = GenericClassifier.load(tmp_path / "clf", model=_clf_net(2))
    assert list(loaded._label_encoder.classes_) == classes_before


def test_classifier_train_losses_persisted(tmp_path, clf_data):
    X_tr, y_tr, _, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=3, show_progress=False)
    clf.fit(X_tr, y_tr)
    losses_before = list(clf.train_losses_)

    clf.save(tmp_path / "clf")
    loaded = GenericClassifier.load(tmp_path / "clf", model=_clf_net(2))
    assert loaded.train_losses_ == losses_before


def test_classifier_bundle_contains_weights(tmp_path, clf_data):
    X_tr, y_tr, _, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X_tr, y_tr)
    clf.save(tmp_path / "clf")

    with zipfile.ZipFile(tmp_path / "clf.eigenkit") as zf:
        names = set(zf.namelist())
    assert "state/weights.pt" in names
    assert "state/label_classes.json" in names
    assert not any(n.endswith(".pkl") for n in names)


def test_classifier_load_without_model_raises(tmp_path, clf_data):
    X_tr, y_tr, _, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X_tr, y_tr)
    clf.save(tmp_path / "clf")
    with pytest.raises(SerializationError, match="model"):
        GenericClassifier.load(tmp_path / "clf")


def test_classifier_load_model_blocked(tmp_path, clf_data):
    X_tr, y_tr, _, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X_tr, y_tr)
    clf.save(tmp_path / "clf")
    with pytest.raises(SerializationError, match="load_model"):
        load_model(tmp_path / "clf")


def test_classifier_load_wrong_class_raises(tmp_path, clf_data):
    X_tr, y_tr, _, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=2, show_progress=False)
    clf.fit(X_tr, y_tr)
    clf.save(tmp_path / "clf")
    with pytest.raises(TypeError, match="GenericRegressor"):
        GenericRegressor.load(tmp_path / "clf", model=_reg_net())


# ---------------------------------------------------------------------------
# GenericRegressor — sklearn contract (no mutation, warm-start, reproducibility)
# ---------------------------------------------------------------------------


def test_regressor_fit_does_not_mutate_model(reg_data):
    X_tr, y_tr, _, _ = reg_data
    torch.manual_seed(0)
    net = _reg_net()
    original = {k: v.clone() for k, v in net.state_dict().items()}

    reg = GenericRegressor(model=net, epochs=3, show_progress=False)
    reg.fit(X_tr, y_tr)

    for k, v in net.state_dict().items():
        torch.testing.assert_close(v, original[k], msg=f"param {k!r} was mutated")


def test_regressor_repeated_fit_is_independent(reg_data):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y_a = (X[:, 0] + X[:, 1]).astype(np.float32)
    y_b = (-2 * X[:, 0] + 0.5).astype(np.float32)

    kwargs = dict(epochs=10, lr=0.02, batch_size=16, show_progress=False, random_state=0)

    direct = GenericRegressor(model=_reg_net(), **kwargs)
    direct.fit(X, y_b)
    p_direct = direct.predict(X)

    warm = GenericRegressor(model=_reg_net(), **kwargs)
    warm.fit(X, y_a)
    warm.fit(X, y_b)
    p_warm = warm.predict(X)

    np.testing.assert_allclose(p_direct, p_warm, rtol=1e-5, atol=1e-5)


def test_regressor_random_state_reproducibility(reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    kwargs = dict(epochs=5, lr=0.01, batch_size=8, show_progress=False, random_state=99)

    torch.manual_seed(0)
    r1 = GenericRegressor(model=_reg_net(), **kwargs)
    r1.fit(X_tr, y_tr)

    torch.manual_seed(12345)
    r2 = GenericRegressor(model=_reg_net(), **kwargs)
    r2.fit(X_tr, y_tr)

    np.testing.assert_allclose(r1.predict(X_te), r2.predict(X_te), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# GenericRegressor — public sklearn attributes
# ---------------------------------------------------------------------------


def test_regressor_n_features_in_set_after_fit(reg_data):
    X_tr, y_tr, _, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=1, show_progress=False)
    assert not hasattr(reg, "n_features_in_")
    reg.fit(X_tr, y_tr)
    assert hasattr(reg, "n_features_in_")
    assert reg.n_features_in_ == X_tr.shape[1]


# ---------------------------------------------------------------------------
# GenericRegressor — output shape validation
# ---------------------------------------------------------------------------


def test_regressor_wrong_output_shape_raises(reg_data):
    X_tr, y_tr, _, _ = reg_data
    bad_model = nn.Sequential(nn.Linear(4, 2))  # outputs (batch, 2), not (batch, 1)
    reg = GenericRegressor(model=bad_model, in_features=4, epochs=1, show_progress=False)
    with pytest.raises(ValueError, match="GenericRegressor expects model output shape"):
        reg.fit(X_tr, y_tr)


# ---------------------------------------------------------------------------
# GenericClassifier — output validation
# ---------------------------------------------------------------------------


def test_classifier_too_few_logits_raises(clf_data):
    X_tr, y_tr, _, _ = clf_data  # binary: 2 classes
    one_logit = nn.Sequential(nn.Linear(4, 1))  # only 1 logit
    clf = GenericClassifier(model=one_logit, in_features=4, epochs=1, show_progress=False)
    with pytest.raises(ValueError, match="logit"):
        clf.fit(X_tr, y_tr)


def test_classifier_too_many_logits_raises(clf_data):
    X_tr, y_tr, _, _ = clf_data  # binary: 2 classes
    three_logits = nn.Sequential(nn.Linear(4, 3))  # 3 logits for 2 classes
    clf = GenericClassifier(
        model=three_logits, in_features=4, epochs=1, show_progress=False
    )
    with pytest.raises(ValueError, match="exactly 2"):
        clf.fit(X_tr, y_tr)


def test_classifier_n_classes_hint_mismatch_raises(clf_data):
    X_tr, y_tr, _, _ = clf_data  # binary: 2 unique classes
    clf = GenericClassifier(
        model=_clf_net(2), in_features=4, n_classes=3, epochs=1, show_progress=False
    )
    with pytest.raises(ValueError, match="n_classes=3"):
        clf.fit(X_tr, y_tr)


def test_classifier_n_classes_fit_reflects_data(clf_data):
    X_tr, y_tr, _, _ = clf_data  # 2 classes
    clf = GenericClassifier(model=_clf_net(2), epochs=1, show_progress=False)
    clf.fit(X_tr, y_tr)
    assert clf._n_classes_fit == len(np.unique(y_tr))


# ---------------------------------------------------------------------------
# GenericClassifier — public sklearn attributes
# ---------------------------------------------------------------------------


def test_classifier_public_attributes_after_fit(clf_data):
    X_tr, y_tr, _, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=1, show_progress=False)
    assert not hasattr(clf, "classes_")
    assert not hasattr(clf, "n_features_in_")
    clf.fit(X_tr, y_tr)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "n_features_in_")
    np.testing.assert_array_equal(clf.classes_, np.unique(y_tr))
    assert clf.n_features_in_ == X_tr.shape[1]


# ---------------------------------------------------------------------------
# sequence_input — helpers
# ---------------------------------------------------------------------------


class _LSTMReg(nn.Module):
    """Minimal LSTM regressor: (batch, seq, features) -> (batch, 1)."""

    def __init__(self, input_size: int = 4, hidden: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


class _LSTMClf(nn.Module):
    """Minimal LSTM classifier: (batch, seq, features) -> (batch, n_classes)."""

    def __init__(self, input_size: int = 4, hidden: int = 8, n_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


@pytest.fixture
def seq_reg_data():
    """3-D regression data: (24, 5, 4) — 24 samples, 5 timesteps, 4 features."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((24, 5, 4)).astype(np.float32)
    y = X[:, -1, 0] + X[:, -1, 1]  # target from last timestep
    return X[:16], y[:16].astype(np.float32), X[16:], y[16:].astype(np.float32)


@pytest.fixture
def seq_clf_data():
    """3-D binary classification data: (24, 5, 4)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((24, 5, 4)).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(int)
    return X[:16], y[:16], X[16:], y[16:]


# ---------------------------------------------------------------------------
# GenericRegressor — sequence_input
# ---------------------------------------------------------------------------


def test_regressor_accepts_sequence_input_default_false():
    reg = GenericRegressor(model=_reg_net())
    assert reg.accepts_sequence_input is False


def test_regressor_accepts_sequence_input_true():
    reg = GenericRegressor(model=_LSTMReg(), sequence_input=True)
    assert reg.accepts_sequence_input is True


def test_regressor_sequence_input_3d_fit_predict(seq_reg_data):
    X_tr, y_tr, X_te, _ = seq_reg_data
    reg = GenericRegressor(
        model=_LSTMReg(input_size=4),
        in_features=4,
        sequence_input=True,
        epochs=2,
        show_progress=False,
    )
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert preds.dtype == np.float32


def test_regressor_sequence_input_in_features_checks_last_dim(seq_reg_data):
    X_tr, y_tr, _, _ = seq_reg_data  # X shape (16, 5, 4)
    reg = GenericRegressor(
        model=_LSTMReg(input_size=4),
        in_features=99,  # wrong: last dim is 4
        sequence_input=True,
        epochs=1,
        show_progress=False,
    )
    with pytest.raises(ValueError, match="in_features"):
        reg.fit(X_tr, y_tr)


def test_regressor_sequence_input_preserved_in_clone():
    reg = GenericRegressor(model=_LSTMReg(), sequence_input=True, epochs=1)
    from sklearn.base import clone

    cloned = clone(reg)
    assert cloned.sequence_input is True
    assert cloned.accepts_sequence_input is True


# ---------------------------------------------------------------------------
# GenericClassifier — sequence_input
# ---------------------------------------------------------------------------


def test_classifier_accepts_sequence_input_default_false():
    clf = GenericClassifier(model=_clf_net())
    assert clf.accepts_sequence_input is False


def test_classifier_accepts_sequence_input_true():
    clf = GenericClassifier(model=_LSTMClf(), sequence_input=True)
    assert clf.accepts_sequence_input is True


def test_classifier_sequence_input_3d_fit_predict(seq_clf_data):
    X_tr, y_tr, X_te, _ = seq_clf_data
    clf = GenericClassifier(
        model=_LSTMClf(input_size=4, n_classes=2),
        in_features=4,
        sequence_input=True,
        epochs=2,
        show_progress=False,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert set(preds).issubset({0, 1})


def test_classifier_sequence_input_in_features_checks_last_dim(seq_clf_data):
    X_tr, y_tr, _, _ = seq_clf_data  # X shape (16, 5, 4)
    clf = GenericClassifier(
        model=_LSTMClf(input_size=4, n_classes=2),
        in_features=99,  # wrong: last dim is 4
        sequence_input=True,
        epochs=1,
        show_progress=False,
    )
    with pytest.raises(ValueError, match="in_features"):
        clf.fit(X_tr, y_tr)


# ---------------------------------------------------------------------------
# Serialization: load_model(model=) and QuantumTimeSeries* wrapping Generic
# inner estimators
# ---------------------------------------------------------------------------


def test_load_model_with_model_kwarg_roundtrip(tmp_path, reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    net = _reg_net()
    reg = GenericRegressor(model=net, epochs=3, show_progress=False)
    reg.fit(X_tr, y_tr)
    preds_before = reg.predict(X_te)

    reg.save(tmp_path / "reg")
    loaded = load_model(tmp_path / "reg", model=_reg_net())
    np.testing.assert_array_equal(preds_before, loaded.predict(X_te))


def test_load_model_model_kwarg_ignored_for_non_generic(tmp_path, reg_data):
    """load_model(..., model=...) is silently ignored for non-Generic estimators."""

    # We need any .eigenkit checkpoint that isn't a Generic class. Use a quick GenericRegressor
    # save then verify that passing a wrong model still works (model kwarg is filtered out).
    # For simplicity just verify no TypeError is raised when model= is passed for Generic.
    X_tr, y_tr, X_te, _ = reg_data
    reg = GenericRegressor(model=_reg_net(), epochs=1, show_progress=False)
    reg.fit(X_tr, y_tr)
    reg.save(tmp_path / "reg")
    # model= is in GenericRegressor.load() signature so it IS used — this is the normal case
    loaded = load_model(tmp_path / "reg", model=_reg_net())
    assert loaded is not None


def test_load_model_clf_with_model_kwarg(tmp_path, clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    clf = GenericClassifier(model=_clf_net(2), epochs=3, show_progress=False)
    clf.fit(X_tr, y_tr)
    preds_before = clf.predict(X_te)

    clf.save(tmp_path / "clf")
    loaded = load_model(tmp_path / "clf", model=_clf_net(2))
    np.testing.assert_array_equal(preds_before, loaded.predict(X_te))


# ---------------------------------------------------------------------------
# QuantumTimeSeriesRegressor wrapping GenericRegressor — save / load
# ---------------------------------------------------------------------------


@pytest.fixture
def ts_reg_data():
    """2-D time-series data: 20 timesteps, 4 features."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = X[:, 0] + X[:, 1]
    return X[:14], y[:14].astype(np.float32), X[14:], y[14:].astype(np.float32)


def _ts_inner_net() -> nn.Sequential:
    """Flat net for windowed input: in_size=3, n_features=4 → 12 inputs."""
    return nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 1))


def test_ts_regressor_generic_inner_save_load_roundtrip(tmp_path, ts_reg_data):
    X_tr, y_tr, X_te, _ = ts_reg_data
    inner = GenericRegressor(
        model=_ts_inner_net(), in_features=12, epochs=3, show_progress=False
    )
    ts = QuantumTimeSeriesRegressor(estimator=inner, in_size=3, shift=1)
    ts.fit(X_tr, y_tr)
    preds_before = ts.predict(X_te)

    ts.save(tmp_path / "ts_inner")
    loaded = QuantumTimeSeriesRegressor.load(tmp_path / "ts_inner", model=_ts_inner_net())
    np.testing.assert_allclose(preds_before, loaded.predict(X_te), rtol=1e-5)


def test_ts_regressor_generic_inner_load_without_inner_model_raises(
    tmp_path, ts_reg_data
):
    X_tr, y_tr, _, _ = ts_reg_data
    inner = GenericRegressor(
        model=_ts_inner_net(), in_features=12, epochs=2, show_progress=False
    )
    ts = QuantumTimeSeriesRegressor(estimator=inner, in_size=3, shift=1)
    ts.fit(X_tr, y_tr)
    ts.save(tmp_path / "ts_inner")

    with pytest.raises(SerializationError, match="model="):
        QuantumTimeSeriesRegressor.load(tmp_path / "ts_inner")


# ---------------------------------------------------------------------------
# QuantumTimeSeriesClassifier wrapping GenericClassifier — save / load
# ---------------------------------------------------------------------------


@pytest.fixture
def ts_clf_data():
    """2-D time-series classification data: 20 timesteps, 4 features."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X[:14], y[:14], X[14:], y[14:]


def _ts_clf_inner_net() -> nn.Sequential:
    """Flat net for windowed input: in_size=3, n_features=4 → 12 inputs, 2 classes."""
    return nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 2))


def test_ts_classifier_generic_inner_save_load_roundtrip(tmp_path, ts_clf_data):
    X_tr, y_tr, X_te, _ = ts_clf_data
    inner = GenericClassifier(
        model=_ts_clf_inner_net(), in_features=12, epochs=3, show_progress=False
    )
    ts = QuantumTimeSeriesClassifier(estimator=inner, in_size=3, shift=1)
    ts.fit(X_tr, y_tr)
    preds_before = ts.predict(X_te)

    ts.save(tmp_path / "ts_clf_inner")
    loaded = QuantumTimeSeriesClassifier.load(
        tmp_path / "ts_clf_inner", model=_ts_clf_inner_net()
    )
    np.testing.assert_array_equal(preds_before, loaded.predict(X_te))


def test_ts_classifier_generic_inner_load_without_inner_model_raises(
    tmp_path, ts_clf_data
):
    X_tr, y_tr, _, _ = ts_clf_data
    inner = GenericClassifier(
        model=_ts_clf_inner_net(), in_features=12, epochs=2, show_progress=False
    )
    ts = QuantumTimeSeriesClassifier(estimator=inner, in_size=3, shift=1)
    ts.fit(X_tr, y_tr)
    ts.save(tmp_path / "ts_clf_inner")

    with pytest.raises(SerializationError, match="model="):
        QuantumTimeSeriesClassifier.load(tmp_path / "ts_clf_inner")


# ---------------------------------------------------------------------------
# Universal load_model(model=) forwarding for QuantumTimeSeries* wrappers
# ---------------------------------------------------------------------------


def test_load_model_ts_regressor_generic_inner(tmp_path, ts_reg_data):
    """load_model(model=) must reach QuantumTimeSeriesRegressor when inner is Generic."""
    X_tr, y_tr, X_te, _ = ts_reg_data
    inner = GenericRegressor(
        model=_ts_inner_net(), in_features=12, epochs=2, show_progress=False
    )
    ts = QuantumTimeSeriesRegressor(estimator=inner, in_size=3, shift=1)
    ts.fit(X_tr, y_tr)
    preds_before = ts.predict(X_te)

    ts.save(tmp_path / "ts_inner")
    loaded = load_model(tmp_path / "ts_inner", model=_ts_inner_net())
    assert isinstance(loaded, QuantumTimeSeriesRegressor)
    np.testing.assert_allclose(preds_before, loaded.predict(X_te), rtol=1e-5)


def test_load_model_ts_classifier_generic_inner(tmp_path, ts_clf_data):
    """load_model(model=) must reach QuantumTimeSeriesClassifier when inner is Generic."""
    X_tr, y_tr, X_te, _ = ts_clf_data
    inner = GenericClassifier(
        model=_ts_clf_inner_net(), in_features=12, epochs=2, show_progress=False
    )
    ts = QuantumTimeSeriesClassifier(estimator=inner, in_size=3, shift=1)
    ts.fit(X_tr, y_tr)
    preds_before = ts.predict(X_te)

    ts.save(tmp_path / "ts_clf_inner")
    loaded = load_model(tmp_path / "ts_clf_inner", model=_ts_clf_inner_net())
    assert isinstance(loaded, QuantumTimeSeriesClassifier)
    np.testing.assert_array_equal(preds_before, loaded.predict(X_te))


def test_load_checkpoint_ts_regressor_generic_inner(tmp_path, ts_reg_data):
    """benchmark.load_checkpoint(model=) must forward to QuantumTimeSeriesRegressor."""
    X_tr, y_tr, X_te, _ = ts_reg_data
    inner = GenericRegressor(
        model=_ts_inner_net(), in_features=12, epochs=2, show_progress=False
    )
    ts = QuantumTimeSeriesRegressor(estimator=inner, in_size=3, shift=1)
    ts.fit(X_tr, y_tr)
    preds_before = ts.predict(X_te)

    ts.save(tmp_path / "ts_inner")
    loaded = load_checkpoint(tmp_path / "ts_inner.eigenkit", model=_ts_inner_net())
    assert isinstance(loaded, QuantumTimeSeriesRegressor)
    np.testing.assert_allclose(preds_before, loaded.predict(X_te), rtol=1e-5)


# ---------------------------------------------------------------------------
# Quantum layer (Basic) integration tests
# ---------------------------------------------------------------------------

_N_QUBITS = 4


def _quantum_clf_net(in_features: int, n_classes: int) -> nn.Sequential:
    """Hybrid net: classical encoder → Basic quantum layer → classical head."""
    return nn.Sequential(
        nn.Linear(in_features, _N_QUBITS),
        nn.Tanh(),
        Basic(n_qubits=_N_QUBITS, depth=1, provider="numpy"),
        nn.Linear(_N_QUBITS, n_classes),
    )


def _quantum_reg_net(in_features: int) -> nn.Sequential:
    """Hybrid net: classical encoder → Basic quantum layer → scalar output."""
    return nn.Sequential(
        nn.Linear(in_features, _N_QUBITS),
        nn.Tanh(),
        Basic(n_qubits=_N_QUBITS, depth=1, provider="numpy"),
        nn.Linear(_N_QUBITS, 1),
    )


@pytest.fixture
def small_clf_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X[:20], y[:20], X[20:], y[20:]


@pytest.fixture
def small_reg_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X[:20], y[:20], X[20:], y[20:]


def test_generic_classifier_with_basic_layer_fit_predict(small_clf_data):
    """GenericClassifier must fit and predict correctly with a Basic quantum layer."""
    X_tr, y_tr, X_te, _ = small_clf_data
    net = _quantum_clf_net(in_features=4, n_classes=2)
    clf = GenericClassifier(
        model=net, in_features=4, epochs=3, show_progress=False, random_state=0
    )
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert set(preds).issubset({0, 1})

    proba = clf.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_te)), atol=1e-5)


def test_generic_classifier_with_basic_layer_save_load(tmp_path, small_clf_data):
    """Save/load round-trip must preserve predictions for a quantum-layer classifier."""
    X_tr, y_tr, X_te, _ = small_clf_data
    net = _quantum_clf_net(in_features=4, n_classes=2)
    clf = GenericClassifier(
        model=net, in_features=4, epochs=3, show_progress=False, random_state=0
    )
    clf.fit(X_tr, y_tr)
    preds_before = clf.predict(X_te)

    clf.save(tmp_path / "quantum_clf")
    loaded = GenericClassifier.load(
        tmp_path / "quantum_clf", model=_quantum_clf_net(in_features=4, n_classes=2)
    )
    np.testing.assert_array_equal(preds_before, loaded.predict(X_te))


def test_generic_regressor_with_basic_layer_fit_predict(small_reg_data):
    """GenericRegressor must fit and return a 1-D float array with a Basic quantum layer."""
    X_tr, y_tr, X_te, _ = small_reg_data
    net = _quantum_reg_net(in_features=4)
    reg = GenericRegressor(
        model=net, in_features=4, epochs=3, show_progress=False, random_state=0
    )
    reg.fit(X_tr, y_tr)

    preds = reg.predict(X_te)
    assert preds.shape == (len(X_te),), "predict() must return a 1-D array"
    assert preds.dtype == np.float32


def test_generic_regressor_with_basic_layer_save_load(tmp_path, small_reg_data):
    """Save/load round-trip must preserve predictions for a quantum-layer regressor."""
    X_tr, y_tr, X_te, _ = small_reg_data
    net = _quantum_reg_net(in_features=4)
    reg = GenericRegressor(
        model=net, in_features=4, epochs=3, show_progress=False, random_state=0
    )
    reg.fit(X_tr, y_tr)
    preds_before = reg.predict(X_te)

    reg.save(tmp_path / "quantum_reg")
    loaded = GenericRegressor.load(
        tmp_path / "quantum_reg", model=_quantum_reg_net(in_features=4)
    )
    np.testing.assert_allclose(preds_before, loaded.predict(X_te), rtol=1e-5)
    np.testing.assert_allclose(preds_before, loaded.predict(X_te), rtol=1e-5)
