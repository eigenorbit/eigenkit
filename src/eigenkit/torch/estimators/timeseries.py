from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from eigenkit.core.circuit import ProviderSpec
from eigenkit.qml import BaseQuantumClassifier, BaseQuantumRegressor
from eigenkit.qml._registry import get_estimator_class, register_estimator
from eigenkit.qml._serialization import (
    SerializationError,
    check_format_version,
    deserialize_circuit_spec,
    deserialize_provider_param,
    load_bundle,
)


def _windowed_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute regression metrics on aligned 1-D vectors."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = float(mean_squared_error(yt, yp))
    return {
        "r2": float(r2_score(yt, yp)),
        "mse": mse,
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mse)),
        "mape": float(mean_absolute_percentage_error(yt, yp)),
    }


def _windowed_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute classification metrics on aligned 1-D label vectors."""
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "precision": float(precision_score(yt, yp, average="weighted", zero_division=0)),
        "recall": float(recall_score(yt, yp, average="weighted", zero_division=0)),
    }


def _validate_bundle_class(metadata: dict, expected_cls: type) -> None:
    actual = metadata.get("class_name")
    if actual != expected_cls.__name__:
        raise TypeError(
            f"{expected_cls.__name__}.load() expected a "
            f"{expected_cls.__name__!r} checkpoint, but the file contains "
            f"{actual!r}. Use load_model() to infer the class from metadata."
        )


@register_estimator
class QuantumTimeSeriesRegressor(BaseQuantumRegressor):
    """Supervised wrapper with a single output, for time-series regression.

    It builds sliding windows over the rows of `X` (one row for each time step).
    Each window has shape `(in_size, n_features)`. Stacked along the batch axis,
    they form an array of shape `(n_windows, in_size, n_features)`, ready for
    sequence models such as `torch.nn.LSTM`. When `autoregressive=True`, an extra
    last channel holds the target value of each timestep, so each step has
    `n_features + 1` values. For a window that starts at row `i`, the regression
    target is `y[i + in_size - 1 + shift]`.

    Args:
        estimator: The inner estimator. It must have `fit(X, y)` and
            `predict(X)`.
        in_size: The length of the window, in time steps. If None, it is
            treated as `1`.
        shift: How many steps after the last timestep of the window the target
            is. It must be zero or more. `shift=0` predicts the target at the last
            timestep.
        autoregressive: If True, a channel with `y[i + k]` for
            `k = 0, ..., in_size - 1` is added to each window. With `shift=0`,
            this leaks the target into the inputs, and a `UserWarning` is raised.

    Attributes:
        train_losses_: The training loss of each epoch, copied from the inner
            estimator if it has `train_losses_` (`HQNR`, `RecurrentHQNR`), and
            an empty list otherwise. It is set by `fit()`.

    Example:
        ```python
        import numpy as np
        from eigenkit.torch.estimators import QuantumTimeSeriesRegressor, RecurrentHQNR

        rng = np.random.default_rng(0)
        X_train, X_test = rng.normal(size=(40, 3)), rng.normal(size=(15, 3))
        y_train = np.sin(np.arange(40) / 4)

        # The inner estimator should accept X of shape (batch, in_size, n_features).
        inner = RecurrentHQNR(seq_feature_dim=3, rnn_hidden_dim=4, epochs=5)
        model = QuantumTimeSeriesRegressor(estimator=inner, in_size=10, shift=1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        ```
    """

    def __init__(
        self,
        estimator: Any,
        in_size: int | None = None,
        shift: int = 1,
        autoregressive: bool = False,
    ):
        self.estimator = estimator
        self.in_size = in_size
        self.shift = shift
        self.autoregressive = autoregressive
        self._validate_init_params()

    @property
    def output(self) -> int:
        """The number of outputs, which is always `1`."""
        return 1

    def _validate_init_params(self) -> None:
        if not isinstance(self.shift, int) or self.shift < 0:
            raise ValueError("shift must be a non-negative integer.")
        if self.in_size is not None and (
            not isinstance(self.in_size, int) or self.in_size < 1
        ):
            raise ValueError("in_size must be a positive integer or None.")
        if not isinstance(self.autoregressive, bool):
            raise TypeError("autoregressive must be a boolean.")
        if self.shift == 0 and self.autoregressive:
            warnings.warn(
                "Potential target leakage: shift=0 with autoregressive=True "
                "injects the current target into input windows.",
                UserWarning,
                stacklevel=2,
            )
        self._validate_estimator_interface()

    def _validate_estimator_interface(self) -> None:
        if not hasattr(self.estimator, "fit") or not hasattr(self.estimator, "predict"):
            raise TypeError("Estimator must implement fit(X, y) and predict(X).")

    def _estimator_accepts_sequence_input(self) -> bool:
        return getattr(self.estimator, "accepts_sequence_input", False)

    def _prepare_estimator_features(self, X_pairs: np.ndarray) -> np.ndarray:
        """Flatten window tensors for tabular estimators; keep 3-D for
        sequence models.

        Reshaping is done on NumPy arrays before calling `fit` / `predict`
        on the inner estimator, so autograd remains intact end-to-end
        (HQNR trains as one unit; RecurrentHQNR trains RNN + quantum
        jointly inside its own `fit`).
        """
        if self._estimator_accepts_sequence_input():
            return X_pairs
        if X_pairs.ndim == 3:
            return X_pairs.reshape(X_pairs.shape[0], -1)
        return X_pairs

    def _validate_xy_shapes(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.reshape(-1)
        if y_arr.ndim != 1:
            raise ValueError("y must be 1-D or shape (n_samples, 1).")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must contain the same number of samples.")
        win = self.in_size if self.in_size is not None else 1
        if len(X_arr) < win + self.shift:
            raise ValueError(
                "Not enough samples: need at least in_size + shift rows "
                f"(got n_samples={len(X_arr)}, in_size={win}, shift={self.shift})."
            )
        return X_arr, y_arr

    def _input_window_length(self) -> int:
        return self.in_size if self.in_size is not None else 1

    def _n_window_samples(self, n_timesteps: int) -> int:
        win = self._input_window_length()
        return n_timesteps - win - self.shift + 1

    def _make_supervised_windows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_history: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr, y_arr = self._validate_xy_shapes(X, y)
        win = self._input_window_length()
        n_feat = X_arr.shape[1]
        n_win = self._n_window_samples(len(X_arr))

        # Rows i .. i+win-1 -> (win, n_feat); batch dim = n_win -> (n_win, win, n_feat)
        X_flat = X_arr.reshape(len(X_arr), n_feat)
        X_stacked = np.stack([X_flat[i : i + win] for i in range(n_win)])

        y_pairs = np.array(
            [y_arr[i + win - 1 + self.shift] for i in range(n_win)],
            dtype=y_arr.dtype,
        )

        if self.autoregressive:
            if y_history is None:
                y_src = y_arr
            else:
                y_src = np.asarray(y_history)
                if y_src.ndim == 2 and y_src.shape[1] == 1:
                    y_src = y_src.reshape(-1)
                if y_src.ndim != 1:
                    raise ValueError("y_history must be 1-D or shape (n_samples, 1).")
                min_len = (n_win - 1) + (win - 1) + 1
                if len(y_src) < min_len:
                    raise ValueError(
                        "y_history length is too short for autoregressive mode."
                    )
            ar_chan = np.stack([y_src[i : i + win] for i in range(n_win)])
            ar_chan = ar_chan[..., np.newaxis].astype(X_stacked.dtype, copy=False)
            X_stacked = np.concatenate([X_stacked, ar_chan], axis=2)

        return X_stacked, y_pairs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "QuantumTimeSeriesRegressor":
        """Fit the inner estimator on the windows and their targets.

        Args:
            X: The features in time order, of shape `(n_timesteps, n_features)`.
            y: The targets, of shape `(n_timesteps,)` or `(n_timesteps, 1)`.

        Returns:
            This estimator, so that calls can be chained.
        """
        self.train_losses_ = []
        X_pairs, y_pairs = self._make_supervised_windows(np.asarray(X), np.asarray(y))
        self.estimator.fit(self._prepare_estimator_features(X_pairs), y_pairs)
        self.train_losses_ = getattr(self.estimator, "train_losses_", [])
        return self

    def predict(
        self,
        X: np.ndarray,
        y_history: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict the regression targets of the windows of `X`.

        Args:
            X: The features in time order, of shape `(n_timesteps, n_features)`.
            y_history: The past targets, aligned with the rows of `X`. They fill
                the autoregressive channel of each window. It is required when
                `autoregressive=True`.

        Returns:
            The predictions, of shape `(n_timesteps - in_size - shift + 1,)`.
        """
        X_arr = np.asarray(X)
        if self.autoregressive and y_history is None:
            raise ValueError("y_history is required when autoregressive=True.")
        dummy_y = np.zeros(X_arr.shape[0], dtype=np.float32)
        X_pairs, _ = self._make_supervised_windows(X_arr, dummy_y, y_history=y_history)
        preds = self.estimator.predict(self._prepare_estimator_features(X_pairs))
        return np.asarray(preds).reshape(-1)

    @overload
    def score(
        self, X: np.ndarray, y: np.ndarray, *, return_dict: Literal[False] = False
    ) -> float: ...

    @overload
    def score(
        self, X: np.ndarray, y: np.ndarray, *, return_dict: Literal[True]
    ) -> dict[str, float]: ...

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        return_dict: bool = False,
    ) -> float | dict[str, float]:
        """Score the estimator on the window targets, aligned as in `predict`.

        By default, it returns `R²`, for compatibility with scikit-learn
        (`cross_val_score`, `GridSearchCV`). Pass `return_dict=True` to get a
        dict with `r2`, `mse`, `mae`, `rmse` and `mape`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`. The
                windows are built inside.
            y (np.ndarray): One target for each sample, of shape `(n_samples,)`.
            return_dict (bool): If True, every metric is returned in a dict,
                instead of the single main score. Default: `False`.

        Returns:
            The `R²` score, or a dict of metrics when `return_dict` is set.
        """
        X_pairs, y_pairs = self._make_supervised_windows(np.asarray(X), np.asarray(y))
        preds = np.asarray(
            self.estimator.predict(self._prepare_estimator_features(X_pairs))
        ).reshape(-1)
        metrics = _windowed_regression_metrics(y_pairs, preds)
        if return_dict:
            return metrics
        return metrics["r2"]

    def _extra_metadata(self) -> dict:
        return {"estimator_class": type(self.estimator).__name__}

    def _serialize_params(self) -> tuple[dict, dict | None]:
        inner = self.estimator
        if not hasattr(inner, "_collect_state") or not hasattr(
            inner, "_serialize_params"
        ):
            raise SerializationError(
                f"Inner estimator {type(inner).__name__!r} does not support save/load. "
                "Wrap a eigenkit BaseQuantumEstimator."
            )
        inner_params, inner_circuit = inner._serialize_params()
        params = {
            "in_size": self.in_size,
            "shift": self.shift,
            "autoregressive": self.autoregressive,
            "estimator_class": type(inner).__name__,
            "estimator_params": inner_params,
        }
        return params, inner_circuit

    def _collect_state(self) -> dict:
        state = {f"estimator_{k}": v for k, v in self.estimator._collect_state().items()}
        if hasattr(self, "train_losses_"):
            state["train_losses_"] = self.train_losses_
        return state

    def _restore_state(self, state: dict) -> None:
        inner_state = {
            key.removeprefix("estimator_"): value
            for key, value in state.items()
            if key.startswith("estimator_")
        }
        self.estimator._restore_state(inner_state)
        self.train_losses_ = state.get("train_losses_", [])

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        provider: ProviderSpec | None = None,
        model: Any = None,
    ) -> "QuantumTimeSeriesRegressor":
        """Load a fitted time-series wrapper from a `.eigenkit` bundle.

        Args:
            path: The path of the `.eigenkit` bundle.
            provider: An optional provider that replaces the saved one. It is
                passed to the inner estimator.
            model: The `nn.Module` architecture, required when the inner
                estimator is
                [`GenericRegressor`][eigenkit.torch.estimators.GenericRegressor].
                Pass the same architecture as the one used in training.

        Returns:
            The loaded wrapper.

        Raises:
            TypeError: If the bundle holds a different class.
            SerializationError: If the inner estimator needs `model` and it is
                not given.
        """
        metadata, params, circuit_spec, state = load_bundle(path)
        check_format_version(metadata)
        _validate_bundle_class(metadata, cls)

        estimator_class = params.pop("estimator_class")
        estimator_params = params.pop("estimator_params")

        if provider is not None:
            estimator_params["provider"] = provider
        elif "provider" in estimator_params:
            estimator_params["provider"] = deserialize_provider_param(
                estimator_params["provider"]
            )

        if circuit_spec is not None:
            blocks, observables = deserialize_circuit_spec(circuit_spec)
            estimator_params["blocks"] = blocks
            if observables is not None:
                estimator_params["observables"] = observables

        inner_cls = get_estimator_class(estimator_class)
        if "model" in inspect.signature(inner_cls.__init__).parameters:
            if model is None:
                raise SerializationError(
                    f"Inner estimator {estimator_class!r} requires an nn.Module "
                    "architecture at load time. Provide it via model=..."
                )
            estimator_params["model"] = model
        inner = inner_cls(**estimator_params)
        wrapper = cls(estimator=inner, **params)
        wrapper._restore_state(state)
        return wrapper


@register_estimator
class QuantumTimeSeriesClassifier(BaseQuantumClassifier):
    """Supervised wrapper for time-series classification.

    It builds sliding windows over the rows of `X` (one row for each time step).
    Each window has shape `(in_size, n_features)`. Stacked along the batch axis,
    they form an array of shape `(n_windows, in_size, n_features)`, ready for
    sequence models such as `RecurrentHQNC`. When `autoregressive=True`, an extra
    last channel holds the target value of each timestep, so each step has
    `n_features + 1` values. For a window that starts at row `i`, the
    classification target is `y[i + in_size - 1 + shift]`.

    Args:
        estimator: The inner estimator. It must have `fit(X, y)` and
            `predict(X)`.
        in_size: The length of the window, in time steps. If None, it is
            treated as `1`.
        shift: How many steps after the last timestep of the window the target
            is. It must be zero or more. `shift=0` predicts the target at the last
            timestep.
        autoregressive: If True, a channel with `y[i + k]` for
            `k = 0, ..., in_size - 1` is added to each window. With `shift=0`,
            this leaks the target into the inputs, and a `UserWarning` is raised.
            When `autoregressive=True`, `predict()` and `predict_proba()` require
            a `y_history` argument.

    Attributes:
        classes_: The distinct class labels seen by `fit()`.
        train_losses_: The training loss of each epoch, copied from the inner
            estimator if it has `train_losses_` (`HQNC`, `RecurrentHQNC`), and an
            empty list otherwise. It is set by `fit()`.

    Example:
        ```python
        import numpy as np
        from eigenkit.torch.estimators import (
            QuantumTimeSeriesClassifier, RecurrentHQNC,
        )

        rng = np.random.default_rng(0)
        X_train, X_test = rng.normal(size=(40, 4)), rng.normal(size=(15, 4))
        y_train = np.arange(40) % 2

        inner = RecurrentHQNC(seq_feature_dim=4, rnn_hidden_dim=4, depth=2)
        model = QuantumTimeSeriesClassifier(estimator=inner, in_size=10, shift=1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        ```
    """

    def __init__(
        self,
        estimator: Any,
        in_size: int | None = None,
        shift: int = 1,
        autoregressive: bool = False,
    ):
        self.estimator = estimator
        self.in_size = in_size
        self.shift = shift
        self.autoregressive = autoregressive
        self._validate_init_params()

    @property
    def output(self) -> int | None:
        """The number of classes.

        After `fit()`, it is the number of classes seen. Before, it is the
        `n_classes` of the inner estimator, or None if it has none.
        """
        if hasattr(self, "classes_"):
            return len(self.classes_)
        return getattr(self.estimator, "n_classes", None)

    def _validate_init_params(self) -> None:
        if not isinstance(self.shift, int) or self.shift < 0:
            raise ValueError("shift must be a non-negative integer.")
        if self.in_size is not None and (
            not isinstance(self.in_size, int) or self.in_size < 1
        ):
            raise ValueError("in_size must be a positive integer or None.")
        if not isinstance(self.autoregressive, bool):
            raise TypeError("autoregressive must be a boolean.")
        if self.shift == 0 and self.autoregressive:
            warnings.warn(
                "Potential target leakage: shift=0 with autoregressive=True "
                "injects the current target into input windows.",
                UserWarning,
                stacklevel=2,
            )
        self._validate_estimator_interface()

    def _validate_estimator_interface(self) -> None:
        if not hasattr(self.estimator, "fit") or not hasattr(self.estimator, "predict"):
            raise TypeError("Estimator must implement fit(X, y) and predict(X).")

    def _estimator_accepts_sequence_input(self) -> bool:
        return getattr(self.estimator, "accepts_sequence_input", False)

    def _prepare_estimator_features(self, X_pairs: np.ndarray) -> np.ndarray:
        if self._estimator_accepts_sequence_input():
            return X_pairs
        if X_pairs.ndim == 3:
            return X_pairs.reshape(X_pairs.shape[0], -1)
        return X_pairs

    def _validate_xy_shapes(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.reshape(-1)
        if y_arr.ndim != 1:
            raise ValueError("y must be 1-D or shape (n_samples, 1).")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must contain the same number of samples.")
        win = self.in_size if self.in_size is not None else 1
        if len(X_arr) < win + self.shift:
            raise ValueError(
                "Not enough samples: need at least in_size + shift rows "
                f"(got n_samples={len(X_arr)}, in_size={win}, shift={self.shift})."
            )
        return X_arr, y_arr

    def _input_window_length(self) -> int:
        return self.in_size if self.in_size is not None else 1

    def _n_window_samples(self, n_timesteps: int) -> int:
        win = self._input_window_length()
        return n_timesteps - win - self.shift + 1

    def _history_values(self, y_src: np.ndarray, *, fit: bool) -> np.ndarray:
        y_arr = np.asarray(y_src)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.reshape(-1)
        if y_arr.ndim != 1:
            raise ValueError("y_history must be 1-D or shape (n_samples, 1).")
        if fit:
            self._history_encoder = LabelEncoder()
            encoded = self._history_encoder.fit_transform(y_arr)
            return encoded.astype(np.float32)
        if not hasattr(self, "_history_encoder"):
            raise ValueError("Model must be fitted before using y_history.")
        return self._history_encoder.transform(y_arr).astype(np.float32)

    def _make_supervised_windows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_history: np.ndarray | None = None,
        *,
        fit_history: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr, y_arr = self._validate_xy_shapes(X, y)
        win = self._input_window_length()
        n_feat = X_arr.shape[1]
        n_win = self._n_window_samples(len(X_arr))

        X_flat = X_arr.reshape(len(X_arr), n_feat)
        X_stacked = np.stack([X_flat[i : i + win] for i in range(n_win)])

        y_pairs = np.array(
            [y_arr[i + win - 1 + self.shift] for i in range(n_win)],
            dtype=y_arr.dtype,
        )

        if self.autoregressive:
            y_src = y_arr if y_history is None else np.asarray(y_history)
            y_encoded = self._history_values(y_src, fit=fit_history)
            min_len = (n_win - 1) + (win - 1) + 1
            if len(y_encoded) < min_len:
                raise ValueError("y_history length is too short for autoregressive mode.")
            ar_chan = np.stack([y_encoded[i : i + win] for i in range(n_win)])
            ar_chan = ar_chan[..., np.newaxis].astype(X_stacked.dtype, copy=False)
            X_stacked = np.concatenate([X_stacked, ar_chan], axis=2)

        return X_stacked, y_pairs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "QuantumTimeSeriesClassifier":
        """Fit the inner classifier on the windows and their labels.

        Args:
            X: The features in time order, of shape `(n_timesteps, n_features)`.
            y: The class labels, of shape `(n_timesteps,)` or `(n_timesteps, 1)`.

        Returns:
            This estimator, so that calls can be chained.
        """
        self.train_losses_ = []
        X_pairs, y_pairs = self._make_supervised_windows(
            np.asarray(X), np.asarray(y), fit_history=self.autoregressive
        )
        self.estimator.fit(self._prepare_estimator_features(X_pairs), y_pairs)
        self.train_losses_ = getattr(self.estimator, "train_losses_", [])
        # Every classifier in the library sets `classes_`. A wrapped estimator
        # from outside the library may omit it, so the labels seen are the
        # fallback.
        if hasattr(self.estimator, "classes_"):
            self.classes_ = np.asarray(self.estimator.classes_)
        else:
            self.classes_ = np.unique(y_pairs)
        return self

    def predict(
        self,
        X: np.ndarray,
        y_history: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict the class labels of the windows of `X`.

        Args:
            X: The features in time order, of shape `(n_timesteps, n_features)`.
            y_history: The past class labels, aligned with the rows of `X`. They
                fill the autoregressive channel of each window. It is required
                when `autoregressive=True`, and it must have at least
                `n_timesteps - shift` elements (that is,
                `n_timesteps - in_size - shift + 1 + in_size - 1`).

        Returns:
            The predicted labels, of shape `(n_timesteps - in_size - shift + 1,)`.
        """
        X_arr = np.asarray(X)
        if self.autoregressive and y_history is None:
            raise ValueError("y_history is required when autoregressive=True.")
        dummy_y = np.zeros(X_arr.shape[0], dtype=np.int64)
        X_pairs, _ = self._make_supervised_windows(
            X_arr, dummy_y, y_history=y_history, fit_history=False
        )
        preds = self.estimator.predict(self._prepare_estimator_features(X_pairs))
        return np.asarray(preds).reshape(-1)

    def predict_proba(
        self,
        X: np.ndarray,
        y_history: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the estimated class probabilities of the windows of `X`.

        Args:
            X: The features in time order, of shape `(n_timesteps, n_features)`.
            y_history: The past class labels. It is required when
                `autoregressive=True`. See `predict()`.

        Returns:
            A float32 array of shape `(n_windows, n_classes)`.

        Raises:
            NotImplementedError: If the inner estimator has no `predict_proba`.
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError(
                "Inner estimator does not implement predict_proba()."
            )
        X_arr = np.asarray(X)
        if self.autoregressive and y_history is None:
            raise ValueError("y_history is required when autoregressive=True.")
        dummy_y = np.zeros(X_arr.shape[0], dtype=np.int64)
        X_pairs, _ = self._make_supervised_windows(
            X_arr, dummy_y, y_history=y_history, fit_history=False
        )
        proba = self.estimator.predict_proba(self._prepare_estimator_features(X_pairs))
        return np.asarray(proba, dtype=np.float32)

    @overload
    def score(
        self, X: np.ndarray, y: np.ndarray, *, return_dict: Literal[False] = False
    ) -> float: ...

    @overload
    def score(
        self, X: np.ndarray, y: np.ndarray, *, return_dict: Literal[True]
    ) -> dict[str, float]: ...

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        return_dict: bool = False,
    ) -> float | dict[str, float]:
        """Score the estimator on the window targets, aligned as in `predict`.

        By default, it returns the accuracy, for compatibility with scikit-learn
        (`cross_val_score`, `GridSearchCV`). Pass `return_dict=True` to get a
        dict with `accuracy`, `f1` (weighted), `precision` (weighted) and
        `recall` (weighted).

        Note:
            When `autoregressive=True`, the true labels `y` are used as the
            autoregressive history channel (teacher forcing). This is the correct
            behaviour for held-out evaluation, but it means that `score` cannot be
            called without the true labels of all the timesteps in `X`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`. The
                windows are built inside.
            y (np.ndarray): One target for each sample, of shape `(n_samples,)`.
            return_dict (bool): If True, every metric is returned in a dict,
                instead of the single main score. Default: `False`.

        Returns:
            The accuracy, or a dict of metrics when `return_dict` is set.
        """
        X_pairs, y_pairs = self._make_supervised_windows(
            np.asarray(X), np.asarray(y), fit_history=False
        )
        preds = self.estimator.predict(self._prepare_estimator_features(X_pairs))
        metrics = _windowed_classification_metrics(y_pairs, preds)
        if return_dict:
            return metrics
        return metrics["accuracy"]

    def _extra_metadata(self) -> dict:
        return {"estimator_class": type(self.estimator).__name__}

    def _serialize_params(self) -> tuple[dict, dict | None]:
        inner = self.estimator
        if not hasattr(inner, "_collect_state") or not hasattr(
            inner, "_serialize_params"
        ):
            raise SerializationError(
                f"Inner estimator {type(inner).__name__!r} does not support save/load. "
                "Wrap a eigenkit BaseQuantumEstimator."
            )
        inner_params, inner_circuit = inner._serialize_params()
        params = {
            "in_size": self.in_size,
            "shift": self.shift,
            "autoregressive": self.autoregressive,
            "estimator_class": type(inner).__name__,
            "estimator_params": inner_params,
        }
        return params, inner_circuit

    def _collect_state(self) -> dict:
        state = {f"estimator_{k}": v for k, v in self.estimator._collect_state().items()}
        if hasattr(self, "train_losses_") and self.train_losses_ is not None:
            state["train_losses_"] = self.train_losses_
        if hasattr(self, "classes_"):
            state["classes"] = list(self.classes_)
        if self.autoregressive and hasattr(self, "_history_encoder"):
            state["history_label_classes"] = list(self._history_encoder.classes_)
        return state

    def _restore_state(self, state: dict) -> None:
        inner_state = {
            key.removeprefix("estimator_"): value
            for key, value in state.items()
            if key.startswith("estimator_")
        }
        self.estimator._restore_state(inner_state)
        self.train_losses_ = state.get("train_losses_")
        if "classes" in state:
            self.classes_ = np.asarray(state["classes"])
        if "history_label_classes" in state:
            le = LabelEncoder()
            le.classes_ = np.asarray(state["history_label_classes"])
            self._history_encoder = le

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        provider: ProviderSpec | None = None,
        model: Any = None,
    ) -> "QuantumTimeSeriesClassifier":
        """Load a fitted time-series classifier wrapper from a `.eigenkit` bundle.

        Args:
            path: The path of the `.eigenkit` bundle.
            provider: An optional provider that replaces the saved one. It is
                passed to the inner estimator.
            model: The `nn.Module` architecture, required when the inner
                estimator is
                [`GenericClassifier`][eigenkit.torch.estimators.GenericClassifier].
                Pass the same architecture as the one used in training.

        Returns:
            The loaded wrapper.

        Raises:
            TypeError: If the bundle holds a different class.
            SerializationError: If the inner estimator needs `model` and it is
                not given.
        """
        metadata, params, circuit_spec, state = load_bundle(path)
        check_format_version(metadata)
        _validate_bundle_class(metadata, cls)

        estimator_class = params.pop("estimator_class")
        estimator_params = params.pop("estimator_params")

        if provider is not None:
            estimator_params["provider"] = provider
        elif "provider" in estimator_params:
            estimator_params["provider"] = deserialize_provider_param(
                estimator_params["provider"]
            )

        if circuit_spec is not None:
            blocks, observables = deserialize_circuit_spec(circuit_spec)
            estimator_params["blocks"] = blocks
            if observables is not None:
                estimator_params["observables"] = observables

        inner_cls = get_estimator_class(estimator_class)
        if "model" in inspect.signature(inner_cls.__init__).parameters:
            if model is None:
                raise SerializationError(
                    f"Inner estimator {estimator_class!r} requires an nn.Module "
                    "architecture at load time. Provide it via model=..."
                )
            estimator_params["model"] = model
        inner = inner_cls(**estimator_params)
        wrapper = cls(estimator=inner, **params)
        wrapper._restore_state(state)
        return wrapper
