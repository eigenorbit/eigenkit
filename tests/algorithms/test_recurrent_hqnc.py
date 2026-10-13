import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone  # noqa: E402


def test_recurrent_hqnc_public_exports():
    from eigenkit import QuantumTimeSeriesClassifier, RecurrentHQNC
    from eigenkit import (
        QuantumTimeSeriesClassifier as AlgorithmsQTSC,
    )
    from eigenkit import RecurrentHQNC as AlgorithmsRecurrentHQNC

    assert RecurrentHQNC is AlgorithmsRecurrentHQNC
    assert QuantumTimeSeriesClassifier is AlgorithmsQTSC


def test_recurrent_hqnc_is_classifier():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    assert isinstance(
        RecurrentHQNC(seq_feature_dim=3, rnn_hidden_dim=4, depth=1, epochs=1),
        ClassifierMixin,
    )


def test_recurrent_hqnc_fit_predict_proba_3d():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 2, 2)).astype(np.float32)
    y = np.where(X[:, -1, 0] > 0, "up", "down")

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_type="lstm",
        rnn_hidden_dim=6,
        rnn_num_layers=1,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X)

    assert pred.shape == (8,)
    assert set(pred) <= {"up", "down"}
    assert proba.shape == (8, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(8), rtol=1e-6)


def test_recurrent_hqnc_rejects_2d():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    model = RecurrentHQNC(seq_feature_dim=2, rnn_hidden_dim=4, depth=1, epochs=1)
    X = np.random.default_rng(1).random((10, 2)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 2, size=10)
    with pytest.raises(ValueError, match="expects X"):
        model.fit(X, y)


def test_recurrent_hqnc_clone():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    m = RecurrentHQNC(seq_feature_dim=4, rnn_hidden_dim=8, depth=2, n_classes=3)
    c = clone(m)
    assert c.seq_feature_dim == 4
    assert c.rnn_hidden_dim == 8
    assert c.n_classes == 3


def test_recurrent_hqnc_set_params_recomputes_quantum_dims():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(5)
    X = rng.normal(size=(10, 2, 2)).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(int)

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=2,
        depth=1,
        epochs=1,
        provider="numpy",
        show_progress=False,
    )
    model.set_params(rnn_hidden_dim=5)
    model.fit(X, y)

    assert model.predict(X).shape == (10,)


def test_recurrent_hqnc_rejects_mismatched_explicit_class_count():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(6)
    X = rng.normal(size=(10, 2, 2)).astype(np.float32)
    y = np.array(["up", "down"] * 5)

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        depth=1,
        n_classes=3,
        epochs=1,
        provider="numpy",
        show_progress=False,
    )

    with pytest.raises(ValueError, match="n_classes must match"):
        model.fit(X, y)


def test_recurrent_hqnc_invalid_rnn_type():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    with pytest.raises(ValueError):
        RecurrentHQNC(seq_feature_dim=2, rnn_type="bogus")  # type: ignore[arg-type]


def test_recurrent_hqnc_full_sequence_requires_sequence_length():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    with pytest.raises(ValueError, match="sequence_length"):
        RecurrentHQNC(
            seq_feature_dim=2,
            rnn_return_last_timestep=False,
            sequence_length=None,
        )


def test_recurrent_hqnc_full_sequence_flatten():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    time_steps, n_features = 2, 2
    rng = np.random.default_rng(3)
    X = rng.normal(size=(8, time_steps, n_features)).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(int)

    model = RecurrentHQNC(
        seq_feature_dim=n_features,
        rnn_hidden_dim=4,
        rnn_return_last_timestep=False,
        sequence_length=time_steps,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X, y)
    assert model.predict(X).shape == (8,)


def test_recurrent_hqnc_train_losses_and_classes_after_fit():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(7)
    X = rng.normal(size=(8, 2, 2)).astype(np.float32)
    y = np.where(X[:, -1, 0] > 0, "up", "down")

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        depth=1,
        epochs=3,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X, y)
    assert len(model.train_losses_) == 3
    assert set(model.classes_) == {"up", "down"}


def test_recurrent_hqnc_validation_losses():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(12)
    X = rng.normal(size=(12, 2, 2)).astype(np.float32)
    y = np.array(["up", "down"] * 6)

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X[:8], y[:8], X_val=X[8:], y_val=y[8:])
    assert len(model.train_losses_) == 2
    assert len(model.validation_losses_) == 2


def test_recurrent_hqnc_sequence_length_must_be_none_when_rnn_return_last_timestep():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    with pytest.raises(ValueError, match="sequence_length must be None"):
        RecurrentHQNC(
            seq_feature_dim=2,
            rnn_return_last_timestep=True,
            sequence_length=5,
        )


def test_recurrent_hqnc_bidirectional():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(8)
    X = rng.normal(size=(8, 2, 2)).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(int)

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        rnn_bidirectional=True,
        depth=1,
        epochs=1,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X, y)
    assert model.predict(X).shape == (8,)


def test_recurrent_hqnc_rnn_type_rnn():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(9)
    X = rng.normal(size=(6, 2, 2)).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(int)

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_type="rnn",
        rnn_hidden_dim=4,
        depth=1,
        epochs=1,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X, y)
    assert model.predict(X).shape == (6,)


def test_recurrent_hqnc_seq_feature_dim_mismatch_raises():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(10)
    X_train = rng.normal(size=(6, 2, 2)).astype(np.float32)
    y = (X_train[:, -1, 0] > 0).astype(int)

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        depth=1,
        epochs=1,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_train, y)

    X_wrong = rng.normal(size=(4, 2, 5)).astype(np.float32)  # wrong last dim
    with pytest.raises(ValueError, match="seq_feature_dim"):
        model.predict(X_wrong)


def test_recurrent_hqnc_sequence_length_mismatch_full_seq_raises():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(6, 3, 2)).astype(np.float32)
    y = (X_train[:, -1, 0] > 0).astype(int)

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        rnn_return_last_timestep=False,
        sequence_length=3,
        depth=1,
        epochs=1,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_train, y)

    X_wrong = rng.normal(size=(4, 5, 2)).astype(np.float32)  # wrong seq len
    with pytest.raises(ValueError, match="sequence_length"):
        model.predict(X_wrong)


def test_recurrent_hqnc_invalid_seq_feature_dim():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    with pytest.raises(ValueError):
        RecurrentHQNC(seq_feature_dim=0)
    with pytest.raises(ValueError):
        RecurrentHQNC(seq_feature_dim=-1)


def test_recurrent_hqnc_y_not_1d_raises():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(12)
    X = rng.normal(size=(6, 2, 2)).astype(np.float32)
    y_2d = np.zeros((6, 2), dtype=np.float32)

    model = RecurrentHQNC(seq_feature_dim=2, rnn_hidden_dim=4, depth=1, epochs=1)
    with pytest.raises(ValueError, match="y must be 1-D"):
        model.fit(X, y_2d)


def test_recurrent_hqnc_x_y_length_mismatch_raises():
    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    rng = np.random.default_rng(13)
    X = rng.normal(size=(6, 2, 2)).astype(np.float32)
    y = np.zeros(5, dtype=np.float32)  # 5 ≠ 6

    model = RecurrentHQNC(seq_feature_dim=2, rnn_hidden_dim=4, depth=1, epochs=1)
    with pytest.raises(ValueError, match="same number of samples"):
        model.fit(X, y)


def test_recurrent_hqnc_not_fitted_raises():
    from sklearn.exceptions import NotFittedError

    from eigenkit.torch.estimators.recurrent_hqnc import RecurrentHQNC

    model = RecurrentHQNC(seq_feature_dim=2, rnn_hidden_dim=4, depth=1, epochs=1)
    X = np.random.default_rng(14).normal(size=(4, 2, 2)).astype(np.float32)
    with pytest.raises(NotFittedError):
        model.predict(X)
    with pytest.raises(NotFittedError):
        model.predict_proba(X)
