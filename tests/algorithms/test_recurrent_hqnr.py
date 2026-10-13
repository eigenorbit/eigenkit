import numpy as np
import pytest
from sklearn.base import RegressorMixin, clone  # noqa: E402


def test_recurrent_hqnr_is_regressor():
    from eigenkit.torch.estimators.recurrent_hqnr import RecurrentHQNR

    assert isinstance(
        RecurrentHQNR(seq_feature_dim=3, rnn_hidden_dim=4, depth=1, epochs=1),
        RegressorMixin,
    )


def test_recurrent_hqnr_fit_predict_3d():
    from eigenkit.torch.estimators.recurrent_hqnr import RecurrentHQNR

    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 2, 2)).astype(np.float32)
    y = X[:, -1, 0].astype(np.float32)

    model = RecurrentHQNR(
        seq_feature_dim=2,
        rnn_type="lstm",
        rnn_hidden_dim=6,
        rnn_num_layers=1,
        depth=1,
        epochs=2,
        provider="numpy",
    )
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (3,)
    assert pred.dtype == np.float32


def test_recurrent_hqnr_validation_losses():
    from eigenkit.torch.estimators.recurrent_hqnr import RecurrentHQNR

    rng = np.random.default_rng(12)
    X = rng.normal(size=(8, 2, 2)).astype(np.float32)
    y = X[:, -1, 0].astype(np.float32)

    model = RecurrentHQNR(
        seq_feature_dim=2,
        rnn_type="lstm",
        rnn_hidden_dim=6,
        rnn_num_layers=1,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X[:5], y[:5], X_val=X[5:], y_val=y[5:])
    assert len(model.train_losses_) == 2
    assert len(model.validation_losses_) == 2


def test_recurrent_hqnr_rejects_2d():
    from eigenkit.torch.estimators.recurrent_hqnr import RecurrentHQNR

    model = RecurrentHQNR(seq_feature_dim=2, rnn_hidden_dim=4, depth=1, epochs=1)
    X = np.random.default_rng(1).random((10, 2)).astype(np.float32)
    y = np.random.default_rng(1).random(10).astype(np.float32)
    with pytest.raises(ValueError, match="expects X"):
        model.fit(X, y)


def test_recurrent_hqnr_clone():
    from eigenkit.torch.estimators.recurrent_hqnr import RecurrentHQNR

    m = RecurrentHQNR(seq_feature_dim=4, rnn_hidden_dim=8, depth=2)
    c = clone(m)
    assert c.seq_feature_dim == 4
    assert c.rnn_hidden_dim == 8


def test_recurrent_hqnr_invalid_rnn_type():
    from eigenkit.torch.estimators.recurrent_hqnr import RecurrentHQNR

    with pytest.raises(ValueError):
        RecurrentHQNR(seq_feature_dim=2, rnn_type="bogus")  # type: ignore[arg-type]


def test_recurrent_hqnr_full_sequence_requires_sequence_length():
    from eigenkit.torch.estimators.recurrent_hqnr import RecurrentHQNR

    with pytest.raises(ValueError, match="sequence_length"):
        RecurrentHQNR(
            seq_feature_dim=2,
            rnn_return_last_timestep=False,
            sequence_length=None,
        )


def test_recurrent_hqnr_full_sequence_flatten():
    from eigenkit.torch.estimators.recurrent_hqnr import RecurrentHQNR

    T, F = 2, 2
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3, T, F)).astype(np.float32)
    y = X[:, -1, 0].astype(np.float32)

    model = RecurrentHQNR(
        seq_feature_dim=F,
        rnn_hidden_dim=4,
        rnn_return_last_timestep=False,
        sequence_length=T,
        depth=1,
        epochs=2,
        provider="numpy",
    )
    model.fit(X, y)
    assert model.predict(X).shape == (3,)
