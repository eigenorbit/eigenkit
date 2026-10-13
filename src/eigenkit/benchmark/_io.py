"""Checkpoint loading helpers for benchmark-saved models."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_checkpoint(path: str | Path, *, model: Any = None) -> Any:
    """Load a model checkpoint written by `compare(..., save_dir=...)`.

    Args:
        path: The path to the checkpoint file (`*.eigenkit` or `*.joblib`).
        model: The `nn.Module` architecture for
            [`GenericRegressor`][eigenkit.torch.estimators.GenericRegressor] or
            [`GenericClassifier`][eigenkit.torch.estimators.GenericClassifier]
            checkpoints. It is ignored for other types of estimator.

    Returns:
        The loaded estimator.

    Raises:
        TypeError: If a `.joblib` file does not contain a scikit-learn estimator.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    if path.suffix == ".eigenkit":
        from ..torch.estimators import load_model

        kwargs: dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        return load_model(path, **kwargs)
    if path.suffix in {".joblib", ".pkl"}:
        import joblib
        from sklearn.base import BaseEstimator

        obj = joblib.load(path)
        if not isinstance(obj, BaseEstimator):
            raise TypeError(
                f"Expected a scikit-learn estimator, got {type(obj).__name__!r}."
                " Only checkpoints saved by 'compare(..., save_dir=...)' are supported."
            )
        return obj
    raise ValueError("Unsupported benchmark model checkpoint suffix.")
