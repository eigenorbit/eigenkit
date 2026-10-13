from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression


class _Flatten2DWrapper(BaseEstimator):
    """Adapts estimators that expect 2-D tabular X to 3-D (batch, time, feat)."""

    def __init__(self, estimator: Any = None):
        self.estimator = LinearRegression() if estimator is None else estimator

    def fit(self, X, y):
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.estimator.fit(X, y)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.estimator.predict(X)


class _Flatten2DClassifier(BaseEstimator, ClassifierMixin):
    """Adapts classifiers that expect 2-D tabular X to 3-D windows."""

    def __init__(self, estimator: Any = None):
        self.estimator = (
            LogisticRegression(max_iter=200) if estimator is None else estimator
        )

    def fit(self, X, y):
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.estimator.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.estimator.predict_proba(X)


def test_ts_is_regressor():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    assert isinstance(
        QuantumTimeSeriesRegressor(estimator=_Flatten2DWrapper(LinearRegression())),
        RegressorMixin,
    )


def test_ts_classifier_is_classifier():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    assert isinstance(
        QuantumTimeSeriesClassifier(estimator=_Flatten2DClassifier()),
        ClassifierMixin,
    )


def test_score_metrics_dict():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = (X[:, 0] * 0.5 + 1).astype(np.float32)
    model = QuantumTimeSeriesRegressor(
        estimator=_Flatten2DWrapper(LinearRegression()),
        in_size=2,
        shift=2,
    )
    model.fit(X, y)
    r2_only = model.score(X, y)
    assert isinstance(r2_only, float)
    d = model.score(X, y, return_dict=True)
    assert set(d.keys()) == {"r2", "mse", "mae", "rmse", "mape"}
    assert d["r2"] == r2_only
    assert d["mse"] >= 0.0
    assert d["mae"] >= 0.0
    assert d["rmse"] >= 0.0


def test_classifier_score_metrics_dict():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = (X[:, 0] > 30).astype(int)
    model = QuantumTimeSeriesClassifier(
        estimator=_Flatten2DClassifier(),
        in_size=2,
        shift=2,
    )
    model.fit(X, y)
    accuracy_only = model.score(X, y)
    assert isinstance(accuracy_only, float)
    d = model.score(X, y, return_dict=True)
    assert set(d.keys()) == {"accuracy", "f1", "precision", "recall"}
    assert d["accuracy"] == accuracy_only
    assert 0.0 <= d["f1"] <= 1.0
    assert 0.0 <= d["precision"] <= 1.0
    assert 0.0 <= d["recall"] <= 1.0


def test_ts_clone():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    model = QuantumTimeSeriesRegressor(
        estimator=_Flatten2DWrapper(LinearRegression()), shift=5
    )
    cloned = clone(model)
    assert cloned.shift == 5


def test_ts_classifier_clone():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    model = QuantumTimeSeriesClassifier(estimator=_Flatten2DClassifier(), shift=5)
    cloned = clone(model)
    assert cloned.shift == 5


def test_supervised_shift_output_shape_hqnr_flatten():
    from eigenkit.torch.estimators.hqnr import HQNR
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = (X[:, 0] * 0.5 + 1).astype(np.float32)

    model = QuantumTimeSeriesRegressor(
        estimator=HQNR(in_features=6, depth=1, epochs=2, provider="numpy"),
        in_size=2,
        shift=2,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X) - 2 - 2 + 1,)
    assert model.output == 1
    assert model.train_losses_ is not None
    assert len(model.train_losses_) == 2
    assert model.train_losses_ is model.estimator.train_losses_


def test_timeseries_wraps_recurrent_hqnr_without_flatten():
    from eigenkit.torch.estimators.recurrent_hqnr import RecurrentHQNR
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = X[:, 0].astype(np.float32)

    inner = RecurrentHQNR(
        seq_feature_dim=4,
        rnn_type="gru",
        rnn_hidden_dim=8,
        rnn_num_layers=1,
        depth=1,
        epochs=2,
        provider="numpy",
    )
    model = QuantumTimeSeriesRegressor(estimator=inner, in_size=3, shift=1)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X) - 3 - 1 + 1,)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert model.train_losses_ is not None
    assert len(model.train_losses_) == 2
    assert model.train_losses_ is inner.train_losses_


def test_timeseries_wraps_recurrent_hqnc_without_flatten():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = np.where(X[:, 0] > 0, "up", "down")

    inner = RecurrentHQNC(
        seq_feature_dim=4,
        rnn_type="gru",
        rnn_hidden_dim=8,
        rnn_num_layers=1,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model = QuantumTimeSeriesClassifier(estimator=inner, in_size=3, shift=1)
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert preds.shape == (len(X) - 3 - 1 + 1,)
    assert proba.shape == (len(preds), 2)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert model.train_losses_ is not None
    assert len(model.train_losses_) == 2
    assert model.train_losses_ is inner.train_losses_


def test_ts_train_losses_empty_when_inner_has_no_history():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = (X[:, 0] * 0.5 + 1).astype(np.float32)
    model = QuantumTimeSeriesRegressor(
        estimator=_Flatten2DWrapper(LinearRegression()),
        in_size=2,
        shift=2,
    )
    model.fit(X, y)
    assert model.train_losses_ == []


def test_ts_classifier_train_losses_none_when_inner_has_no_history():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = (X[:, 0] > 30).astype(int)
    model = QuantumTimeSeriesClassifier(
        estimator=_Flatten2DClassifier(),
        in_size=2,
        shift=2,
    )
    model.fit(X, y)
    assert model.train_losses_ == []


def test_autoregressive_requires_history_on_predict():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    X = np.arange(36, dtype=np.float32).reshape(12, 3)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)

    model = QuantumTimeSeriesRegressor(
        estimator=_Flatten2DWrapper(LinearRegression()),
        shift=1,
        autoregressive=True,
    )
    model.fit(X, y)

    with np.testing.assert_raises(ValueError):
        model.predict(X)

    with np.testing.assert_raises(ValueError):
        model.predict(X, y_history=np.array([1.0, 2.0], dtype=np.float32))

    preds_ok = model.predict(X, y_history=y)
    assert preds_ok.shape == (len(X) - 1,)


def test_classifier_autoregressive_requires_history_on_predict():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    X = np.arange(36, dtype=np.float32).reshape(12, 3)
    y = np.where(X[:, 0] > 15, "high", "low")

    model = QuantumTimeSeriesClassifier(
        estimator=_Flatten2DClassifier(),
        shift=1,
        autoregressive=True,
    )
    model.fit(X, y)

    with np.testing.assert_raises(ValueError):
        model.predict(X)

    with np.testing.assert_raises(ValueError):
        model.predict(X, y_history=np.array(["low", "high"], dtype=object))

    preds_ok = model.predict(X, y_history=y)
    proba_ok = model.predict_proba(X, y_history=y)
    assert preds_ok.shape == (len(X) - 1,)
    assert proba_ok.shape == (len(X) - 1, 2)


def test_supervised_invalid_shapes():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    model = QuantumTimeSeriesRegressor(
        estimator=_Flatten2DWrapper(LinearRegression()), shift=1
    )
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(9, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        model.fit(X, y)


def test_invalid_params_raise_on_init():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    # shift must be a non-negative int; -1 is invalid.
    with np.testing.assert_raises(ValueError):
        QuantumTimeSeriesRegressor(
            estimator=_Flatten2DWrapper(LinearRegression()), shift=-1
        )
    with np.testing.assert_raises(ValueError):
        QuantumTimeSeriesRegressor(
            estimator=_Flatten2DWrapper(LinearRegression()), in_size=0
        )


def test_shift_zero_predicts_current_target_alignment_regressor():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    class _CaptureY:
        accepts_sequence_input = True

        def fit(self, X, y):
            self.y_fit_ = np.asarray(y).copy()
            return self

        def predict(self, X):
            return np.zeros(np.asarray(X).shape[0], dtype=np.float32)

    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.arange(10, dtype=np.float32)
    m = QuantumTimeSeriesRegressor(estimator=_CaptureY(), in_size=4, shift=0)
    m.fit(X, y)
    # For shift=0 and in_size=4 -> targets are y[3], y[4], ..., y[9].
    np.testing.assert_array_equal(m.estimator.y_fit_, y[3:])


def test_shift_zero_with_autoregressive_warns_regressor():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    with pytest.warns(UserWarning, match="Potential target leakage"):
        QuantumTimeSeriesRegressor(
            estimator=_Flatten2DWrapper(LinearRegression()),
            in_size=3,
            shift=0,
            autoregressive=True,
        )


def test_windowed_X_is_three_dimensional():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    class _ShapeRegressor:
        accepts_sequence_input = True

        def fit(self, X, y):
            X = np.asarray(X)
            assert X.ndim == 3
            assert X.shape == (20 - 2 - 1 + 1, 2, 3)
            return self

        def predict(self, X):
            X = np.asarray(X)
            assert X.ndim == 3
            return np.zeros(X.shape[0], dtype=np.float64)

    X = np.random.default_rng(0).random((20, 3)).astype(np.float32)
    y = X[:, 0]
    m = QuantumTimeSeriesRegressor(estimator=_ShapeRegressor(), in_size=2, shift=1)
    m.fit(X, y)
    m.predict(X)


def test_classifier_windowed_X_is_three_dimensional():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    class _ShapeClassifier:
        accepts_sequence_input = True

        def fit(self, X, y):
            X = np.asarray(X)
            assert X.ndim == 3
            assert X.shape == (20 - 2 - 1 + 1, 2, 3)
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            X = np.asarray(X)
            assert X.ndim == 3
            return np.full(X.shape[0], self.classes_[0])

        def predict_proba(self, X):
            return np.full((np.asarray(X).shape[0], len(self.classes_)), 0.5)

    X = np.random.default_rng(0).random((20, 3)).astype(np.float32)
    y = np.where(X[:, 0] > 0.5, "yes", "no")
    m = QuantumTimeSeriesClassifier(estimator=_ShapeClassifier(), in_size=2, shift=1)
    m.fit(X, y)
    m.predict(X)
    assert m.predict_proba(X).shape == (18, 2)


def test_autoregressive_extra_channel_shape():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesRegressor

    class _ShapeAR:
        accepts_sequence_input = True

        def fit(self, X, y):
            X = np.asarray(X)
            assert X.shape[2] == 4  # 3 features + 1 AR channel
            return self

        def predict(self, X):
            return np.zeros(np.asarray(X).shape[0])

    X = np.random.default_rng(1).random((15, 3)).astype(np.float32)
    y = X[:, 0]
    m = QuantumTimeSeriesRegressor(
        estimator=_ShapeAR(), in_size=2, shift=1, autoregressive=True
    )
    m.fit(X, y)


def test_classifier_autoregressive_extra_channel_shape_with_string_labels():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    class _ShapeARClassifier:
        accepts_sequence_input = True

        def fit(self, X, y):
            X = np.asarray(X)
            assert X.shape[2] == 4  # 3 features + 1 encoded label channel
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(np.asarray(X).shape[0], self.classes_[0])

        def predict_proba(self, X):
            return np.full((np.asarray(X).shape[0], len(self.classes_)), 0.5)

    X = np.random.default_rng(1).random((15, 3)).astype(np.float32)
    y = np.where(X[:, 0] > 0.5, "high", "low")
    m = QuantumTimeSeriesClassifier(
        estimator=_ShapeARClassifier(), in_size=2, shift=1, autoregressive=True
    )
    m.fit(X, y)
    assert m.predict(X, y_history=y).shape == (13,)


def test_ts_classifier_predict_proba_not_implemented():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    class _NoProba:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(len(X), self.classes_[0])

    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.where(X[:, 0] > 15, "high", "low")
    model = QuantumTimeSeriesClassifier(estimator=_NoProba(), in_size=2, shift=1)
    model.fit(X, y)

    with np.testing.assert_raises(NotImplementedError):
        model.predict_proba(X)


def test_ts_classifier_shift_zero_autoregressive_warns():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    with pytest.warns(UserWarning, match="Potential target leakage"):
        QuantumTimeSeriesClassifier(
            estimator=_Flatten2DClassifier(),
            in_size=3,
            shift=0,
            autoregressive=True,
        )


def test_ts_classifier_invalid_params_raise_on_init():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    with np.testing.assert_raises(ValueError):
        QuantumTimeSeriesClassifier(estimator=_Flatten2DClassifier(), shift=-1)
    with np.testing.assert_raises(ValueError):
        QuantumTimeSeriesClassifier(estimator=_Flatten2DClassifier(), in_size=0)


def test_ts_classifier_rejects_non_2d_X():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    model = QuantumTimeSeriesClassifier(estimator=_Flatten2DClassifier(), shift=1)
    X_3d = np.zeros((5, 3, 2), dtype=np.float32)
    y = np.array(["a", "b", "a", "b", "a"])
    with np.testing.assert_raises(ValueError):
        model.fit(X_3d, y)


def test_ts_classifier_output_property_after_fit():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = np.where(X[:, 0] > 30, "high", "low")
    model = QuantumTimeSeriesClassifier(
        estimator=_Flatten2DClassifier(), in_size=2, shift=2
    )
    model.fit(X, y)
    assert model.output == 2


def test_ts_classifier_x_y_length_mismatch_raises():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    model = QuantumTimeSeriesClassifier(estimator=_Flatten2DClassifier(), shift=1)
    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.arange(9, dtype=np.float32)  # 9 ≠ 10

    with np.testing.assert_raises(ValueError):
        model.fit(X, y)


def test_ts_classifier_not_enough_samples_raises():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    model = QuantumTimeSeriesClassifier(
        estimator=_Flatten2DClassifier(), in_size=5, shift=1
    )
    X = np.arange(12, dtype=np.float32).reshape(4, 3)  # 4 < in_size(5) + shift(1)
    y = np.array(["a", "b", "a", "b"])

    with np.testing.assert_raises(ValueError):
        model.fit(X, y)


def test_ts_classifier_y_column_vector_accepted():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y_1d = (X[:, 0] > 30).astype(int)
    y_col = y_1d.reshape(-1, 1)  # shape (20, 1)

    model = QuantumTimeSeriesClassifier(
        estimator=_Flatten2DClassifier(), in_size=2, shift=2
    )
    model.fit(X, y_col)
    preds = model.predict(X)
    assert preds.shape == (len(X) - 2 - 2 + 1,)
