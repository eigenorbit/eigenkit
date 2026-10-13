"""Estimators that wrap any `nn.Module`.

They wrap any `torch.nn.Module` that you supply in a `fit` and `predict`
interface compatible with scikit-learn, with the standard training and inference
loops of the library. You must build the complete model, including any output
head.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch import nn

from eigenkit.core.circuit import ProviderSpec
from eigenkit.qml import BaseQuantumClassifier, BaseQuantumRegressor
from eigenkit.qml._registry import register_estimator
from eigenkit.qml._serialization import (
    SerializationError,
    check_format_version,
    load_bundle,
    to_json_safe,
)

from ._nn import ModelAccessMixin
from ._seeding import isolated_rng
from ._training import (
    make_dataloader,
    make_optimizer,
    predict_loop,
    train_loop,
    validate_optimizer_name,
    validate_regression_loss_config,
)


def _model_device(model: nn.Module) -> torch.device:
    """Return the device of the first parameter; fall back to CPU."""
    return next((p.device for p in model.parameters()), torch.device("cpu"))


@register_estimator
class GenericRegressor(ModelAccessMixin, BaseQuantumRegressor):
    """Regressor compatible with scikit-learn that wraps any `torch.nn.Module`.

    You supply a complete model whose `forward(X)` accepts a float32 tensor of
    shape `(batch, in_features)` and returns a tensor of shape `(batch, 1)` or
    `(batch,)`. For sequence models (for example a long short-term memory
    network), set `sequence_input=True`. The wrapper then passes a 3-D tensor of
    shape `(batch, seq_len, in_features)` as it is, instead of flattening it.

    The wrapper trains the model with the standard `train_loop()` of the library,
    and returns the predictions as a float32 NumPy array.

    Args:
        model: Any `nn.Module`. It must include its own output head.
        in_features: The expected number of features **at each time step** (the
            last dimension of the input). When it is given, `fit` raises an error
            if `X.shape[-1]` does not match. Pass `None` to skip the check.
        sequence_input: If `True`, the estimator tells
            [`QuantumTimeSeriesRegressor`][eigenkit.torch.estimators.QuantumTimeSeriesRegressor]
            that it expects a 3-D tensor of shape `(batch, seq_len, in_features)`,
            so the windowed data must not be flattened. Default: `False`.
        loss_func: The loss: `"mse"`, `"mae"`, `"huber"`, or any callable
            `(output, target) -> scalar_tensor`.
        huber_delta: The threshold of the Huber loss. It is ignored for the other
            losses.
        huber_reduction: The reduction of the Huber loss: `"mean"` or `"sum"`.
        lr: The learning rate of the optimiser.
        optimizer: The name of the optimiser: `"adam"`, `"adamw"`, `"rmsprop"` or
            `"sgd"`.
        optimizer_kwargs: Extra keyword arguments passed to the constructor of the
            optimiser.
        epochs: The number of full passes over the training data.
        batch_size: The size of each mini-batch.
        show_progress: If True, a tqdm progress bar is shown during training.
        random_state: The seed, applied before the first training step.

    Attributes:
        train_losses_: The list of the average loss of each epoch, set by
            `fit()`.
        validation_losses_: The list of the validation loss of each epoch, set by
            `fit()` when validation data is given.

    Example:
        ```python
        import numpy as np
        from torch import nn
        from eigenkit.torch.estimators import GenericRegressor
        from eigenkit.torch.layers import Basic

        rng = np.random.default_rng(0)
        X_train, X_test = rng.normal(size=(30, 8)), rng.normal(size=(5, 8))
        y_train = X_train.sum(axis=1)

        # A plain PyTorch network.
        net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        reg = GenericRegressor(model=net, in_features=8, epochs=30)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)

        # A quantum layer with a classical head.
        q = Basic(n_qubits=4, depth=2)
        hybrid = nn.Sequential(q, nn.Linear(4, 1))
        reg2 = GenericRegressor(model=hybrid, in_features=4)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        in_features: int | None = None,
        sequence_input: bool = False,
        loss_func: str | Callable = "mse",
        huber_delta: float = 1.0,
        huber_reduction: str = "mean",
        lr: float = 0.01,
        optimizer: str = "adam",
        optimizer_kwargs: dict | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        show_progress: bool = True,
        random_state: int | None = 42,
    ):
        self.model = model
        self.in_features = in_features
        self.sequence_input = sequence_input
        self.loss_func = loss_func
        self.huber_delta = huber_delta
        self.huber_reduction = huber_reduction
        self.lr = lr
        validate_optimizer_name(optimizer)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.random_state = random_state
        if not callable(loss_func):
            validate_regression_loss_config(loss_func, huber_delta, huber_reduction)

    @property
    def accepts_sequence_input(self) -> bool:
        """Whether the model expects a 3-D input `(batch, seq_len, features)`."""
        return self.sequence_input

    def _make_loss_fn(self) -> Callable:
        if callable(self.loss_func):
            return self.loss_func
        loss_name = validate_regression_loss_config(
            self.loss_func, self.huber_delta, self.huber_reduction
        )
        if loss_name == "mse":
            return lambda out, tgt: nn.functional.mse_loss(out.squeeze(-1), tgt)
        if loss_name == "mae":
            return lambda out, tgt: nn.functional.l1_loss(out.squeeze(-1), tgt)
        delta, reduction = self.huber_delta, self.huber_reduction
        return lambda out, tgt: nn.functional.huber_loss(
            out.squeeze(-1), tgt, reduction=reduction, delta=delta
        )

    @isolated_rng
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "GenericRegressor":
        """Fit the model on training data.

        Args:
            X: The samples, of shape `(n_samples, in_features)` for flat models, or
                `(n_samples, seq_len, in_features)` when `sequence_input=True`.
            y: The real-valued targets, of shape `(n_samples,)`.
            X_val: Optional validation samples, used to compute the validation
                loss of each epoch.
            y_val: Optional validation targets, used to compute the validation
                loss of each epoch.

        Returns:
            This estimator, so that calls can be chained.

        Raises:
            ValueError: If `X.shape[-1]` or `X_val.shape[-1]` differs from
                `in_features`, or if the output of the model does not have shape
                `(batch,)` or `(batch, 1)`.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if self.in_features is not None and X_arr.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected in_features={self.in_features}, "
                f"got X.shape[-1]={X_arr.shape[-1]}."
            )
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self._model = copy.deepcopy(self.model)
        self._model.apply(
            lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
        )
        device = _model_device(self._model)
        # Probe in eval mode. A freshly built model is in training mode, and a
        # one-sample forward through any batch-statistics layer is either an
        # error (`BatchNorm1d`: "Expected more than 1 value per channel") or a
        # corruption (`MinMaxNorm`: its running statistics get seeded from this
        # single probe row). Neither belongs in a shape check.
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                probe_out = self._model(
                    torch.as_tensor(X_arr[:1], dtype=torch.float32).to(device)
                )
        finally:
            self._model.train(was_training)
        if not (probe_out.ndim == 1 or (probe_out.ndim == 2 and probe_out.shape[1] == 1)):
            raise ValueError(
                f"GenericRegressor expects model output shape (batch,) or (batch, 1), "
                f"got {tuple(probe_out.shape)}. "
                "Ensure the model's forward() returns a scalar or single-column output."
            )
        optimizer = make_optimizer(
            self._model.parameters(),
            optimizer=self.optimizer,
            lr=self.lr,
            optimizer_kwargs=self.optimizer_kwargs,
        )
        dl = make_dataloader(
            X_arr, y, batch_size=self.batch_size, target_dtype=torch.float32
        )
        val_dl = None
        if X_val is not None and y_val is not None:
            X_val_arr = np.asarray(X_val, dtype=np.float32)
            if self.in_features is not None and X_val_arr.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected in_features={self.in_features}, "
                    f"got X_val.shape[-1]={X_val_arr.shape[-1]}."
                )
            val_dl = make_dataloader(
                X_val_arr,
                y_val,
                batch_size=self.batch_size,
                target_dtype=torch.float32,
                shuffle=False,
            )
        self.train_losses_, self.validation_losses_ = train_loop(
            self._model,
            dl,
            optimizer,
            self._make_loss_fn(),
            self.epochs,
            show_progress=self.show_progress,
            progress_desc="GenericRegressor training",
            device=device,
            validation_dataloader=val_dl,
        )
        self.n_features_in_ = X_arr.shape[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the continuous targets of `X`. It returns a float32 array.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            One prediction for each sample.
        """
        check_is_fitted(self, "_model")
        X_t = torch.as_tensor(X, dtype=torch.float32).to(_model_device(self._model))
        out = predict_loop(self._model, X_t).cpu()
        if out.ndim == 2:
            out = out.squeeze(-1)  # (N, 1) → (N,)
        # ndim == 1: already (N,) or (1,) — no squeeze to avoid collapsing to scalar
        return out.numpy().astype(np.float32)

    def _collect_state(self) -> dict:
        return {
            "weights": self._model.state_dict(),
            "train_losses": self.train_losses_,
            "validation_losses": self.validation_losses_,
            "n_features_in": self.n_features_in_,
        }

    def _restore_state(self, state: dict) -> None:
        self._model = copy.deepcopy(self.model)
        self._model.load_state_dict(state["weights"])
        self._model.eval()
        self.train_losses_ = state.get("train_losses", [])
        self.validation_losses_ = state.get("validation_losses", [])
        self.n_features_in_ = state["n_features_in"]

    def _serialize_params(self) -> tuple[dict, dict | None]:
        import json

        if callable(self.loss_func):
            raise SerializationError(
                "GenericRegressor with a callable loss_func cannot be saved. "
                "Use a string loss_func ('mse', 'mae', 'huber') to enable save/load."
            )
        all_params = dict(self.get_params(deep=False))
        all_params.pop("model")  # architecture is not serializable; passed at load time
        params = {}
        for k, v in all_params.items():
            safe_v = to_json_safe(v)
            try:
                json.dumps(safe_v)
                params[k] = safe_v
            except (TypeError, ValueError) as exc:
                raise SerializationError(
                    f"Parameter {k!r} of GenericRegressor is not JSON-serializable."
                ) from exc
        return params, None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model: nn.Module | None = None,
        provider: ProviderSpec | None = None,
    ) -> "GenericRegressor":
        """Load a fitted `GenericRegressor` from a `.eigenkit` bundle.

        The architecture of the model cannot be rebuilt automatically, so you must
        give it. The saved weights are loaded into a copy of it.

        Args:
            path: The path of the `.eigenkit` bundle.
            model: An `nn.Module` with the **same architecture** as the one used
                in training. Its weights are not changed: the saved weights go into
                a deep copy, which is `model_` of the loaded estimator.
            provider: It is ignored. It is kept so that the signature matches the
                other estimators.

        Returns:
            The loaded estimator.

        Raises:
            SerializationError: If `model` is not given.
            TypeError: If the bundle holds a different class.

        Example:
            ```python
            from torch import nn
            from eigenkit.torch.estimators import GenericRegressor

            net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
            reg = GenericRegressor.load("checkpoint.eigenkit", model=net)
            ```
        """
        if model is None:
            raise SerializationError(
                "GenericRegressor cannot be loaded via load_model(). "
                "The nn.Module architecture must be provided manually: "
                "GenericRegressor.load(path, model=my_net)"
            )
        metadata, params, _, state = load_bundle(path)
        check_format_version(metadata)
        if metadata.get("class_name") != cls.__name__:
            raise TypeError(
                f"{cls.__name__}.load() expected a {cls.__name__!r} checkpoint, "
                f"but the file contains {metadata['class_name']!r}."
            )
        instance = cls(model=model, **params)
        instance._restore_state(state)
        return instance


@register_estimator
class GenericClassifier(ModelAccessMixin, BaseQuantumClassifier):
    """Classifier compatible with scikit-learn that wraps any `torch.nn.Module`.

    You supply a complete model whose `forward(X)` accepts a float32 tensor of
    shape `(batch, in_features)` and returns logits of shape
    `(batch, n_classes)`. For sequence models (for example a long short-term
    memory network), set `sequence_input=True`. The wrapper then passes a 3-D
    tensor of shape `(batch, seq_len, in_features)` as it is, instead of
    flattening it.

    The wrapper trains with `CrossEntropyLoss` and encodes the labels
    automatically.

    Args:
        model: Any `nn.Module`. It must include its own classification head, which
            gives `n_classes` logits.
        in_features: The expected number of features **at each time step** (the
            last dimension of the input). When it is given, `fit` raises an error
            if `X.shape[-1]` does not match. Pass `None` to skip the check.
        sequence_input: If `True`, the estimator tells
            [`QuantumTimeSeriesClassifier`][eigenkit.torch.estimators.QuantumTimeSeriesClassifier]
            that it expects a 3-D tensor of shape `(batch, seq_len, in_features)`,
            so the windowed data must not be flattened. Default: `False`.
        n_classes: The number of output classes. If `None`, it is inferred from
            the training labels. If it is given, it must equal the number of
            distinct labels in `y`.
        lr: The learning rate of the optimiser.
        optimizer: The name of the optimiser: `"adam"`, `"adamw"`, `"rmsprop"` or
            `"sgd"`.
        optimizer_kwargs: Extra keyword arguments passed to the constructor of the
            optimiser.
        epochs: The number of full passes over the training data.
        batch_size: The size of each mini-batch.
        show_progress: If True, a tqdm progress bar is shown during training.
        random_state: The seed, applied before the first training step.

    Attributes:
        train_losses_: The average cross-entropy loss of each epoch, set by
            `fit()`.
        validation_losses_: The validation cross-entropy loss of each epoch, set
            by `fit()` when validation data is given.

    Example:
        ```python
        import numpy as np
        from torch import nn
        from eigenkit.torch.estimators import GenericClassifier
        from eigenkit.torch.layers import Basic

        rng = np.random.default_rng(0)
        X_train, X_test = rng.normal(size=(30, 4)), rng.normal(size=(5, 4))
        y_train = np.arange(30) % 3

        # A plain PyTorch network, with a head for 3 classes.
        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        clf = GenericClassifier(model=net, in_features=4, epochs=20)
        clf.fit(X_train, y_train)
        labels = clf.predict(X_test)
        probs  = clf.predict_proba(X_test)

        # A quantum layer with a classification head.
        q = Basic(n_qubits=4, depth=2)
        hybrid = nn.Sequential(q, nn.Linear(4, 3))
        clf2 = GenericClassifier(model=hybrid, in_features=4)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        in_features: int | None = None,
        sequence_input: bool = False,
        n_classes: int | None = None,
        lr: float = 0.01,
        optimizer: str = "adam",
        optimizer_kwargs: dict | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        show_progress: bool = True,
        random_state: int | None = 42,
    ):
        self.model = model
        self.in_features = in_features
        self.sequence_input = sequence_input
        self.n_classes = n_classes
        self.lr = lr
        validate_optimizer_name(optimizer)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.random_state = random_state

    @property
    def accepts_sequence_input(self) -> bool:
        """Whether the model expects a 3-D input `(batch, seq_len, features)`."""
        return self.sequence_input

    @isolated_rng
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "GenericClassifier":
        """Fit the model on training data.

        Args:
            X: The samples, of shape `(n_samples, in_features)` for flat models, or
                `(n_samples, seq_len, in_features)` when `sequence_input=True`.
            y: The class labels, of shape `(n_samples,)`.
            X_val: Optional validation samples, used to compute the validation
                loss of each epoch.
            y_val: Optional validation labels, used to compute the validation
                loss of each epoch.

        Returns:
            This estimator, so that calls can be chained.

        Raises:
            ValueError: If `X.shape[-1]` or `X_val.shape[-1]` differs from
                `in_features`, if `n_classes` differs from the number of distinct
                labels, or if the model does not give one logit for each class.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if self.in_features is not None and X_arr.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected in_features={self.in_features}, "
                f"got X.shape[-1]={X_arr.shape[-1]}."
            )
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self._model = copy.deepcopy(self.model)
        self._model.apply(
            lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
        )
        device = _model_device(self._model)
        n_classes_data = len(self._label_encoder.classes_)
        # Probe in eval mode. A freshly built model is in training mode, and a
        # one-sample forward through any batch-statistics layer is either an
        # error (`BatchNorm1d`: "Expected more than 1 value per channel") or a
        # corruption (`MinMaxNorm`: its running statistics get seeded from this
        # single probe row). Neither belongs in a shape check.
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                probe_out = self._model(
                    torch.as_tensor(X_arr[:1], dtype=torch.float32).to(device)
                )
        finally:
            self._model.train(was_training)
        n_logits = probe_out.shape[-1] if probe_out.ndim >= 2 else 1
        if self.n_classes is not None and self.n_classes != n_classes_data:
            raise ValueError(
                f"n_classes={self.n_classes} does not match the number of unique "
                f"labels in y ({n_classes_data})."
            )
        if n_logits != n_classes_data:
            raise ValueError(
                f"Model outputs {n_logits} logit(s) but {n_classes_data} classes "
                f"found in y. Ensure the final layer has exactly {n_classes_data} "
                "output units."
            )
        self._n_classes_fit = n_classes_data
        optimizer = make_optimizer(
            self._model.parameters(),
            optimizer=self.optimizer,
            lr=self.lr,
            optimizer_kwargs=self.optimizer_kwargs,
        )
        dl = make_dataloader(
            X_arr, y_enc, batch_size=self.batch_size, target_dtype=torch.long
        )
        val_dl = None
        if X_val is not None and y_val is not None:
            X_val_arr = np.asarray(X_val, dtype=np.float32)
            if self.in_features is not None and X_val_arr.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected in_features={self.in_features}, "
                    f"got X_val.shape[-1]={X_val_arr.shape[-1]}."
                )
            y_val_enc = self._label_encoder.transform(y_val)
            val_dl = make_dataloader(
                X_val_arr,
                y_val_enc,
                batch_size=self.batch_size,
                target_dtype=torch.long,
                shuffle=False,
            )
        self.train_losses_, self.validation_losses_ = train_loop(
            self._model,
            dl,
            optimizer,
            nn.CrossEntropyLoss(),
            self.epochs,
            show_progress=self.show_progress,
            progress_desc="GenericClassifier training",
            device=device,
            validation_dataloader=val_dl,
        )
        self.classes_ = self._label_encoder.classes_
        self.n_features_in_ = X_arr.shape[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels of `X`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            One prediction for each sample.
        """
        check_is_fitted(self, "_model")
        X_t = torch.as_tensor(X, dtype=torch.float32).to(_model_device(self._model))
        logits = predict_loop(self._model, X_t).cpu()
        return self._label_encoder.inverse_transform(logits.argmax(dim=1).numpy())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the probability of each class, with a softmax over the logits.

        It returns a float32 array of shape `(n_samples, n_classes)`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            The probabilities, of shape `(n_samples, n_classes)`, with the columns
                in the order of `classes_`.
        """
        check_is_fitted(self, "_model")
        X_t = torch.as_tensor(X, dtype=torch.float32).to(_model_device(self._model))
        logits = predict_loop(self._model, X_t).cpu()
        return torch.softmax(logits, dim=1).numpy().astype(np.float32)

    def _collect_state(self) -> dict:
        return {
            "weights": self._model.state_dict(),
            "label_classes": list(self._label_encoder.classes_),
            "n_classes": self._n_classes_fit,
            "train_losses": self.train_losses_,
            "validation_losses": self.validation_losses_,
            "n_features_in": self.n_features_in_,
        }

    def _restore_state(self, state: dict) -> None:
        le = LabelEncoder()
        le.classes_ = np.array(state["label_classes"])
        self._label_encoder = le
        self._n_classes_fit = state["n_classes"]
        self._model = copy.deepcopy(self.model)
        self._model.load_state_dict(state["weights"])
        self._model.eval()
        self.train_losses_ = state.get("train_losses", [])
        self.validation_losses_ = state.get("validation_losses", [])
        # classes_ is derived from the already-stored encoder — no extra bundle data needed
        self.classes_ = self._label_encoder.classes_
        self.n_features_in_ = state["n_features_in"]

    def _serialize_params(self) -> tuple[dict, dict | None]:
        import json

        all_params = dict(self.get_params(deep=False))
        all_params.pop("model")  # architecture is not serializable; passed at load time
        params = {}
        for k, v in all_params.items():
            safe_v = to_json_safe(v)
            try:
                json.dumps(safe_v)
                params[k] = safe_v
            except (TypeError, ValueError) as exc:
                raise SerializationError(
                    f"Parameter {k!r} of GenericClassifier is not JSON-serializable."
                ) from exc
        return params, None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model: nn.Module | None = None,
        provider: ProviderSpec | None = None,
    ) -> "GenericClassifier":
        """Load a fitted `GenericClassifier` from a `.eigenkit` bundle.

        The architecture of the model cannot be rebuilt automatically, so you must
        give it. The saved weights and the label encoder are restored
        automatically.

        Args:
            path: The path of the `.eigenkit` bundle.
            model: An `nn.Module` with the **same architecture** as the one used
                in training. Its weights are not changed: the saved weights go into
                a deep copy, which is `model_` of the loaded estimator.
            provider: It is ignored. It is kept so that the signature matches the
                other estimators.

        Returns:
            The loaded estimator.

        Raises:
            SerializationError: If `model` is not given.
            TypeError: If the bundle holds a different class.

        Example:
            ```python
            from torch import nn
            from eigenkit.torch.estimators import GenericClassifier

            net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
            clf = GenericClassifier.load("checkpoint.eigenkit", model=net)
            ```
        """
        if model is None:
            raise SerializationError(
                "GenericClassifier cannot be loaded via load_model(). "
                "The nn.Module architecture must be provided manually: "
                "GenericClassifier.load(path, model=my_net)"
            )
        metadata, params, _, state = load_bundle(path)
        check_format_version(metadata)
        if metadata.get("class_name") != cls.__name__:
            raise TypeError(
                f"{cls.__name__}.load() expected a {cls.__name__!r} checkpoint, "
                f"but the file contains {metadata['class_name']!r}."
            )
        instance = cls(model=model, **params)
        instance._restore_state(state)
        return instance
