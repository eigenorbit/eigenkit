import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


try:
    from tqdm.auto import tqdm  # type: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - fallback path when tqdm is absent
    tqdm = None


def validate_optimizer_name(optimizer: str) -> str:
    """Check the name of the optimiser and return it in lower case.

    Args:
        optimizer: The name of the optimiser. It is one of `"adam"`, `"adamw"`,
            `"rmsprop"` and `"sgd"`, in any case.

    Returns:
        The name in lower case.

    Raises:
        TypeError: If `optimizer` is not a string.
        ValueError: If the name is not one of the supported names.
    """
    if not isinstance(optimizer, str):
        raise TypeError("optimizer must be a string.")

    normalized = optimizer.lower()
    supported = {"adam", "adamw", "rmsprop", "sgd"}
    if normalized not in supported:
        names = ", ".join(sorted(supported))
        raise ValueError(f"Unknown optimizer: {optimizer!r}. Supported values: {names}.")
    return normalized


def validate_regression_loss_config(
    loss_func: str,
    huber_delta: float,
    huber_reduction: str,
) -> str:
    """Check the configuration of the regression loss and return its name in lower case.

    Args:
        loss_func: The name of the loss. It is one of `"mse"`, `"mae"` and
            `"huber"`, in any case.
        huber_delta: The threshold of the Huber loss. It is not checked.
        huber_reduction: The reduction of the Huber loss, `"mean"` or `"sum"`. It
            is checked only for the Huber loss.

    Returns:
        The name of the loss in lower case.

    Raises:
        TypeError: If `loss_func` is not a string.
        ValueError: If the name is not supported, or if the Huber loss has a
            reduction other than `"mean"` or `"sum"`.
    """
    if not isinstance(loss_func, str):
        raise TypeError("loss_func must be a string.")

    normalized = loss_func.lower()
    if normalized not in {"mse", "mae", "huber"}:
        raise ValueError("Unknown loss_func. Supported values: 'mse', 'mae', 'huber'.")

    if normalized == "huber" and huber_reduction not in ("mean", "sum"):
        raise ValueError("huber_reduction must be 'mean' or 'sum'.")

    return normalized


def make_optimizer(
    parameters,
    optimizer: str = "adam",
    lr: float = 0.01,
    optimizer_kwargs: dict | None = None,
) -> torch.optim.Optimizer:
    """Build a PyTorch optimiser from its name.

    The supported names are `"adam"`, `"adamw"`, `"rmsprop"` and `"sgd"`.

    Args:
        parameters: The parameters to optimise.
        optimizer: The name of the optimiser. Default: `"adam"`.
        lr: The learning rate. Default: `0.01`.
        optimizer_kwargs: Extra keyword arguments passed to the optimiser.
            Default: `None`.

    Returns:
        The optimiser.
    """
    kwargs = optimizer_kwargs or {}
    normalized = validate_optimizer_name(optimizer)
    registry: dict[str, type[torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "sgd": torch.optim.SGD,
    }
    return registry[normalized](parameters, lr=lr, **kwargs)


def make_dataloader(
    X,
    y,
    batch_size: int = 32,
    shuffle: bool = True,
    target_dtype=torch.long,
) -> DataLoader:
    """Wrap NumPy arrays in a PyTorch `DataLoader`.

    Use `target_dtype=torch.long` for classification and `torch.float32` for
    regression.

    Args:
        X: The samples. They are converted to `torch.float32`.
        y: The targets. They are converted to `target_dtype`.
        batch_size: The size of each mini-batch. Default: `32`.
        shuffle: If True, the samples are shuffled in each epoch.
        target_dtype: The data type of the targets. Default: `torch.long`.

    Returns:
        The `DataLoader`.
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=target_dtype)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_loop(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epochs: int,
    verbose: bool = False,
    show_progress: bool = True,
    progress_desc: str = "Training",
    show_epoch_loss_history: bool = True,
    device: torch.device | None = None,
    validation_dataloader: DataLoader | None = None,
) -> tuple[list[float], list[float]]:
    """Run a standard PyTorch training loop and return the average loss of each epoch.

    Args:
        model: The model to train.
        dataloader: The training batches, as returned by `make_dataloader`.
        optimizer: Any `torch.optim` optimiser.
        loss_fn: A callable `(output, target) -> scalar`.
        epochs: The number of full passes over the data.
        verbose: If True, the loss of each epoch is printed.
        show_progress: If True, a tqdm progress bar is shown (or printed lines,
            when tqdm is not installed).
        progress_desc: The text at the start of the progress output.
        show_epoch_loss_history: If True, one line is printed for each epoch,
            with the loss, the elapsed time and the estimated time remaining, in
            addition to the progress bar.
        device: The device to move each batch to before the forward pass. Pass
            `None` (the default) to leave the batches on their original device.
        validation_dataloader: Optional validation batches, evaluated after each
            epoch.

    Returns:
        epoch_losses: The average training loss of each epoch.
        validation_losses: The average validation loss of each epoch. It is an
            empty list when `validation_dataloader` is `None`.

    Raises:
        ValueError: If an epoch processes no batches.
    """
    epoch_losses = []
    validation_losses = []
    model.train()
    use_tqdm = show_progress and tqdm is not None
    epoch_iterator = range(epochs)
    if use_tqdm:
        epoch_iterator = tqdm(
            epoch_iterator,
            total=epochs,
            desc=progress_desc,
            unit="epoch",
            leave=True,
        )

    start_time = time.perf_counter()
    for epoch in epoch_iterator:
        batch_losses = []
        for X_batch, y_batch in dataloader:
            if device is not None:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        if not batch_losses:
            raise ValueError("No batches were processed in train_loop().")
        avg = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg)
        val_avg = None
        if validation_dataloader is not None:
            val_batch_losses = []
            model.eval()
            with torch.no_grad():
                for X_val, y_val in validation_dataloader:
                    if device is not None:
                        X_val, y_val = X_val.to(device), y_val.to(device)
                    val_loss = loss_fn(model(X_val), y_val)
                    val_batch_losses.append(val_loss.item())
            model.train()
            if not val_batch_losses:
                raise ValueError("No batches were processed in validation loop.")
            val_avg = sum(val_batch_losses) / len(val_batch_losses)
            validation_losses.append(val_avg)
        done = epoch + 1
        elapsed = time.perf_counter() - start_time
        eta = (elapsed / done) * (epochs - done) if done else 0.0
        val_text = f" | val_loss={val_avg:.6f}" if val_avg is not None else ""
        if use_tqdm:
            postfix = {"loss": f"{avg:.4f}"}
            if val_avg is not None:
                postfix["val_loss"] = f"{val_avg:.4f}"
            epoch_iterator.set_postfix(postfix)
            if show_epoch_loss_history:
                tqdm.write(
                    f"{progress_desc} | Epoch {done}/{epochs} | "
                    f"loss={avg:.6f}{val_text} | "
                    f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
                )
        elif show_progress:
            val_status = f" - val_loss={val_avg:.4f}" if val_avg is not None else ""
            print(
                f"\r{progress_desc}: {done}/{epochs} epochs - "
                f"loss {avg:.4f}{val_status} - ETA {eta:.1f}s",
                end="",
                flush=True,
            )
            if show_epoch_loss_history:
                print(
                    f"\n{progress_desc} | Epoch {done}/{epochs} | "
                    f"loss={avg:.6f}{val_text} | "
                    f"elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                    flush=True,
                )
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg:.4f}")
    if show_progress and not use_tqdm:
        print()
    # Leave the model ready to predict with. `train_loop` puts it in training
    # mode; left there, the fitted `model_` a user reaches for would disagree
    # with `predict()` on the same input whenever the network holds dropout or
    # a normalisation layer -- silently, and looking like a numerical bug
    # rather than a mode difference.
    model.eval()
    return epoch_losses, validation_losses


@torch.no_grad()
def predict_loop(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int = 64,
) -> torch.Tensor:
    """Run the model on `X` in batches and return the raw outputs, concatenated.

    The function puts `model` in evaluation mode.

    Args:
        model: The model to run.
        X: The inputs.
        batch_size: The size of each batch. Default: `64`.

    Returns:
        The outputs of all the batches, concatenated along the first axis.
    """
    model.eval()
    results = []
    for i in range(0, len(X), batch_size):
        results.append(model(X[i : i + batch_size]))
    return torch.cat(results, dim=0)


def make_regression_loss_fn(loss_func, huber_delta: float, huber_reduction: str):
    """Return the loss function named by `loss_func`.

    Every quantum regressor takes the same three options and builds the loss in
    the same way, so they all share this function. The output of the model
    is squeezed on the last axis, so that an output of shape `(batch, 1)` is
    compared with a target of shape `(batch,)`.

    Args:
        loss_func (str): The name of the loss: `"mse"`, `"mae"` or `"huber"`. A
            callable is rejected with a `TypeError`.
        huber_delta (float): The transition point of the Huber loss. It is ignored
            for the other losses.
        huber_reduction (str): The reduction of the Huber loss. It is ignored for
            the other losses.

    Returns:
        A callable that takes `(output, target)`.
    """
    loss_name = validate_regression_loss_config(loss_func, huber_delta, huber_reduction)
    if callable(loss_name):
        return loss_name
    if loss_name == "mse":
        return lambda output, target: nn.functional.mse_loss(output.squeeze(-1), target)
    if loss_name == "mae":
        return lambda output, target: nn.functional.l1_loss(output.squeeze(-1), target)
    if loss_name == "huber":
        return lambda output, target: nn.functional.huber_loss(
            output.squeeze(-1),
            target,
            reduction=huber_reduction,
            delta=huber_delta,
        )
    raise ValueError(f"unknown regression loss {loss_name!r}")
