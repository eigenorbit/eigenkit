import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def test_make_dataloader_classification():
    from eigenkit.torch.estimators._training import make_dataloader

    X = np.random.randn(20, 4).astype(np.float32)
    y = np.array([0, 1] * 10)
    dl = make_dataloader(X, y, batch_size=8)
    assert isinstance(dl, DataLoader)
    X_b, y_b = next(iter(dl))
    assert X_b.dtype == torch.float32
    assert y_b.dtype == torch.long


def test_make_dataloader_regression():
    from eigenkit.torch.estimators._training import make_dataloader

    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randn(20).astype(np.float32)
    dl = make_dataloader(X, y, batch_size=8, target_dtype=torch.float32)
    _, y_b = next(iter(dl))
    assert y_b.dtype == torch.float32


def test_train_loop_returns_losses():
    from eigenkit.torch.estimators._training import make_dataloader, train_loop

    model = nn.Linear(4, 2)
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.array([0, 1] * 10)
    dl = make_dataloader(X, y, batch_size=8)
    train_losses, validation_losses = train_loop(
        model,
        dl,
        torch.optim.SGD(model.parameters(), lr=0.01),
        nn.CrossEntropyLoss(),
        epochs=3,
    )
    assert len(train_losses) == 3
    assert validation_losses == []
    assert all(isinstance(loss, float) for loss in train_losses)


def test_train_loop_appends_validation_losses():
    from eigenkit.torch.estimators._training import make_dataloader, train_loop

    model = nn.Linear(4, 2)
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.array([0, 1] * 10)
    train_dl = make_dataloader(X, y, batch_size=8)
    val_dl = make_dataloader(X[:6], y[:6], batch_size=3, shuffle=False)

    train_losses, validation_losses = train_loop(
        model,
        train_dl,
        torch.optim.SGD(model.parameters(), lr=0.01),
        nn.CrossEntropyLoss(),
        epochs=3,
        show_progress=False,
        validation_dataloader=val_dl,
    )

    assert len(train_losses) == 3
    assert len(validation_losses) == 3
    assert all(isinstance(loss, float) for loss in validation_losses)


def test_train_loop_progress_prints_validation_loss(monkeypatch, capsys):
    import eigenkit.torch.estimators._training as training
    from eigenkit.torch.estimators._training import make_dataloader, train_loop

    monkeypatch.setattr(training, "tqdm", None)
    model = nn.Linear(4, 2)
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.array([0, 1] * 10)
    train_dl = make_dataloader(X, y, batch_size=8)
    val_dl = make_dataloader(X[:6], y[:6], batch_size=3, shuffle=False)

    train_loop(
        model,
        train_dl,
        torch.optim.SGD(model.parameters(), lr=0.01),
        nn.CrossEntropyLoss(),
        epochs=1,
        show_progress=True,
        validation_dataloader=val_dl,
    )

    assert "val_loss=" in capsys.readouterr().out


def test_predict_loop_shape():
    from eigenkit.torch.estimators._training import predict_loop

    model = nn.Linear(4, 2)
    X_t = torch.randn(15, 4)
    out = predict_loop(model, X_t, batch_size=8)
    assert out.shape == (15, 2)
