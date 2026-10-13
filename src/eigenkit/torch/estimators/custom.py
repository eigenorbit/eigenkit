"""Estimators built from blocks and observables that you supply.

They wrap a list of `Block` objects and an `observables` specification into a
`Circuit`, and then add a small classical head (an `nn.Linear` layer that gives
the classification logits or a single regression output). When the number of
features of the data differs from the number of inputs that the circuit expects,
an `nn.Linear` adapter is added in front.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch import nn

from eigenkit.configs import Encoding, Reuploading
from eigenkit.core.blocks import Block
from eigenkit.core.circuit import ObservableLike, ProviderSpec
from eigenkit.qml import BaseQuantumClassifier, BaseQuantumRegressor
from eigenkit.qml._registry import register_estimator

from ..circuit import TorchCircuit
from ._nn import ModelAccessMixin, _ToFloat32
from ._seeding import isolated_rng
from ._training import (
    make_dataloader,
    make_optimizer,
    make_regression_loss_fn,
    predict_loop,
    train_loop,
    validate_optimizer_name,
    validate_regression_loss_config,
)


def _infer_q_in_features(blocks: list[Block]) -> int:
    """Derive the data input dim from the first input-consuming block."""
    for block in blocks:
        if isinstance(block, Encoding):
            return block.in_features
        if isinstance(block, Reuploading):
            return block.in_features
    raise ValueError(
        "Cannot infer in_features: blocks must contain at least one "
        "Encoding or Reuploading block."
    )


def _build_custom_model(
    blocks: list[Block],
    observables: ObservableLike,
    provider: ProviderSpec,
    data_in_features: int,
    out_dim: int,
    random_state: int | None,
) -> nn.Sequential:
    if random_state is not None:
        torch.manual_seed(random_state)
    q_in = _infer_q_in_features(blocks)
    q_layer = TorchCircuit(blocks=blocks, observables=observables, provider=provider)
    q_out = q_layer.out_features

    modules: list[nn.Module] = []
    if data_in_features != q_in:
        modules.append(nn.Linear(data_in_features, q_in))
    modules.extend([q_layer, _ToFloat32(), nn.Linear(q_out, out_dim)])
    return nn.Sequential(*modules)


@register_estimator
class CustomQuantumClassifier(ModelAccessMixin, BaseQuantumClassifier):
    """Classifier compatible with scikit-learn, built from your own list of blocks.

    Give any sequence of blocks (`Encoding`, `Variational`, `StrongVariational`,
    `Reuploading`, or a custom block from `CircuitBuilder.to_block()`) and an
    `observables` specification, as for `Circuit`. The estimator connects them
    into a quantum layer followed by a classical `nn.Linear` head, and trains the
    whole model with the cross-entropy loss. When the number of features of the
    data differs from the number of inputs that the circuit expects, an
    `nn.Linear` adapter is added in front automatically.

    Args:
        blocks: The list of blocks, in order.
        observables: The observables. It takes the same options as
            `Circuit(observables=...)`. Default: `None` (`Z` on every qubit).
        provider: The backend. Default: `"numpy"`.
        in_features: The number of features of the data. If None, it is inferred
            from the first `Encoding` or `Reuploading` block.
        n_classes: The number of output classes. If None, it is inferred from the
            training labels.
        lr: The learning rate of the optimiser.
        optimizer: The name of the optimiser. The supported names are `"adam"`,
            `"adamw"`, `"rmsprop"` and `"sgd"`.
        optimizer_kwargs: Extra keyword arguments passed to the optimiser.
        epochs: The number of training epochs.
        batch_size: The size of each mini-batch.
        show_progress: If True, a tqdm progress bar is shown.
        random_state: The seed.

    Attributes:
        train_losses_: The cross-entropy loss of each epoch, set by `fit()`.
        validation_losses_: The validation cross-entropy loss of each epoch, set
            by `fit()` when validation data is given.

    Example:
        ```python
        import numpy as np
        from eigenkit import (
            CustomQuantumClassifier, Encoding, Variational, all_z,
        )

        rng = np.random.default_rng(0)
        X_train, X_test = rng.uniform(0, np.pi, (20, 4)), rng.uniform(0, np.pi, (5, 4))
        y_train = np.array([0, 1] * 10)

        blocks = [
            Encoding(in_features=4, rotation="RY"),
            Variational(depth=2, rotation="RY"),
        ]
        model = CustomQuantumClassifier(
            blocks=blocks, observables=all_z(4), epochs=10,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        ```
    """

    def __init__(
        self,
        blocks: list[Block],
        observables: ObservableLike = None,
        provider: ProviderSpec = "numpy",
        in_features: int | None = None,
        n_classes: int | None = None,
        lr: float = 0.01,
        optimizer: str = "adam",
        optimizer_kwargs: dict | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        show_progress: bool = True,
        random_state: int | None = 42,
    ):
        super().__init__(provider=provider, random_state=random_state)
        self.blocks = blocks
        self.observables = observables
        self.in_features = in_features
        self.n_classes = n_classes
        self.lr = lr
        validate_optimizer_name(optimizer)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.show_progress = show_progress

    def _resolve_in_features(self) -> int:
        if self.in_features is not None:
            return self.in_features
        return _infer_q_in_features(self.blocks)

    def _build_model(self, n_classes: int) -> nn.Sequential:
        return _build_custom_model(
            blocks=self.blocks,
            observables=self.observables,
            provider=self.provider,
            data_in_features=self._resolve_in_features(),
            out_dim=n_classes,
            random_state=self.random_state,
        )

    @isolated_rng
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "CustomQuantumClassifier":
        """Fit the classifier on training data.

        Args:
            X: The samples, of shape `(n_samples, in_features)`.
            y: The class labels, of shape `(n_samples,)`. They can be of any type
                that scikit-learn's `LabelEncoder` accepts.
            X_val: Optional validation samples, used to compute the validation
                loss of each epoch.
            y_val: Optional validation labels, used to compute the validation
                loss of each epoch.

        Returns:
            This estimator, so that calls can be chained.

        Raises:
            ValueError: If `in_features` is set and `X` does not have
                `in_features` columns.
        """
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        # The fitted attributes scikit-learn defines, set by every classifier in
        # the library: the time-series wrapper reads `classes_`, and
        # `n_features_in_` lets a wrong column count be reported by name rather
        # than as a raw torch shape error.
        self.classes_ = self._label_encoder.classes_
        self.n_features_in_ = int(np.asarray(X).shape[-1])
        if self.in_features is not None and self.n_features_in_ != self.in_features:
            raise ValueError(
                f"{type(self).__name__} was built with in_features="
                f"{self.in_features}, but X has {self.n_features_in_} features."
            )
        n_classes = (
            self.n_classes
            if self.n_classes is not None
            else len(self._label_encoder.classes_)
        )

        # See HQNC.fit: the declared `n_classes` may exceed the labels present.
        self._n_outputs = int(n_classes)
        self._model = self._build_model(n_classes)
        optimizer = make_optimizer(
            self._model.parameters(),
            optimizer=self.optimizer,
            lr=self.lr,
            optimizer_kwargs=self.optimizer_kwargs,
        )
        dl = make_dataloader(
            X, y_enc, batch_size=self.batch_size, target_dtype=torch.long
        )
        val_dl = None
        if X_val is not None and y_val is not None:
            y_val_enc = self._label_encoder.transform(y_val)
            val_dl = make_dataloader(
                X_val,
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
            progress_desc="Custom classifier training",
            validation_dataloader=val_dl,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels of `X`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            One prediction for each sample.
        """
        check_is_fitted(self, "_model")
        X = self._check_feature_count(X)
        X_t = torch.as_tensor(X, dtype=torch.float32)
        logits = predict_loop(self._model, X_t)
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
        X = self._check_feature_count(X)
        X_t = torch.as_tensor(X, dtype=torch.float32)
        logits = predict_loop(self._model, X_t)
        return torch.softmax(logits, dim=1).numpy().astype(np.float32)

    def _collect_state(self) -> dict:
        return {
            "weights": self._model.state_dict(),
            "label_classes": list(self._label_encoder.classes_),
            "n_classes": int(
                getattr(self, "_n_outputs", len(self._label_encoder.classes_))
            ),
            "train_losses": self.train_losses_,
            "validation_losses": self.validation_losses_,
        }

    @isolated_rng
    def _restore_state(self, state: dict) -> None:
        le = LabelEncoder()
        le.classes_ = np.array(state["label_classes"])
        self._label_encoder = le
        self._n_outputs = int(state["n_classes"])
        self._model = self._build_model(self._n_outputs)
        self._model.load_state_dict(state["weights"])
        self._model.eval()
        self.train_losses_ = state.get("train_losses", [])
        self.validation_losses_ = state.get("validation_losses", [])


@register_estimator
class CustomQuantumRegressor(ModelAccessMixin, BaseQuantumRegressor):
    """Regressor compatible with scikit-learn, built from your own list of blocks.

    It is the regression counterpart of
    [`CustomQuantumClassifier`][eigenkit.torch.estimators.custom.CustomQuantumClassifier].
    It wraps your quantum `Circuit` with an `nn.Linear` head of one neuron, trained
    with the chosen regression loss. An `nn.Linear` input adapter is added when the
    number of features of the data differs from the number of inputs of the
    circuit.

    Args:
        blocks: The list of blocks, in order.
        observables: The observables. It takes the same options as
            `Circuit(observables=...)`.
        provider: The backend. Default: `"numpy"`.
        in_features: The number of features of the data. If None, it is inferred
            from the first `Encoding` or `Reuploading` block.
        lr: The learning rate of the optimiser.
        optimizer: The name of the optimiser: `"adam"`, `"adamw"`, `"rmsprop"` or
            `"sgd"`.
        optimizer_kwargs: Extra keyword arguments passed to the optimiser.
        epochs: The number of training epochs.
        batch_size: The size of each mini-batch.
        loss_func: The regression loss: `"mse"`, `"mae"` or `"huber"`.
        huber_delta: The threshold of the Huber loss.
        huber_reduction: The reduction of the Huber loss: `"mean"` or `"sum"`.
        show_progress: If True, a tqdm progress bar is shown.
        random_state: The seed.

    Attributes:
        train_losses_: The training loss of each epoch, set by `fit()`.
        validation_losses_: The validation loss of each epoch, set by `fit()` when
            validation data is given.
    """

    def __init__(
        self,
        blocks: list[Block],
        observables: ObservableLike = None,
        provider: ProviderSpec = "numpy",
        in_features: int | None = None,
        lr: float = 0.01,
        optimizer: str = "adam",
        optimizer_kwargs: dict | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        show_progress: bool = True,
        loss_func: str = "mse",
        huber_delta: float = 1.0,
        huber_reduction: str = "mean",
        random_state: int | None = 42,
    ):
        super().__init__(provider=provider, random_state=random_state)
        self.blocks = blocks
        self.observables = observables
        self.in_features = in_features
        self.lr = lr
        validate_optimizer_name(optimizer)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.show_progress = show_progress
        validate_regression_loss_config(loss_func, huber_delta, huber_reduction)
        self.loss_func = loss_func
        self.huber_delta = huber_delta
        self.huber_reduction = huber_reduction

    def _resolve_in_features(self) -> int:
        if self.in_features is not None:
            return self.in_features
        return _infer_q_in_features(self.blocks)

    def _build_model(self) -> nn.Sequential:
        return _build_custom_model(
            blocks=self.blocks,
            observables=self.observables,
            provider=self.provider,
            data_in_features=self._resolve_in_features(),
            out_dim=1,
            random_state=self.random_state,
        )

    def _make_loss_fn(self):
        return make_regression_loss_fn(
            self.loss_func, self.huber_delta, self.huber_reduction
        )

    @isolated_rng
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "CustomQuantumRegressor":
        """Fit the regressor on training data.

        Args:
            X: The samples, of shape `(n_samples, in_features)`.
            y: The real-valued targets, of shape `(n_samples,)`.
            X_val: Optional validation samples, used to compute the validation
                loss of each epoch.
            y_val: Optional validation targets, used to compute the validation
                loss of each epoch.

        Returns:
            This estimator, so that calls can be chained.
        """
        # The fitted attribute scikit-learn defines, recorded by every sibling.
        self.n_features_in_ = int(np.asarray(X).shape[-1])
        self._model = self._build_model()
        optimizer = make_optimizer(
            self._model.parameters(),
            optimizer=self.optimizer,
            lr=self.lr,
            optimizer_kwargs=self.optimizer_kwargs,
        )
        dl = make_dataloader(
            X,
            y,
            batch_size=self.batch_size,
            target_dtype=torch.float32,
        )
        val_dl = None
        if X_val is not None and y_val is not None:
            val_dl = make_dataloader(
                X_val,
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
            progress_desc="Custom regressor training",
            validation_dataloader=val_dl,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the continuous targets of `X`. It returns a float32 array.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            One prediction for each sample.
        """
        check_is_fitted(self, "_model")
        X = self._check_feature_count(X)
        X_t = torch.as_tensor(X, dtype=torch.float32)
        return predict_loop(self._model, X_t).squeeze(-1).numpy().astype(np.float32)

    def _collect_state(self) -> dict:
        return {
            "weights": self._model.state_dict(),
            "train_losses": self.train_losses_,
            "validation_losses": self.validation_losses_,
        }

    @isolated_rng
    def _restore_state(self, state: dict) -> None:
        self._model = self._build_model()
        self._model.load_state_dict(state["weights"])
        self._model.eval()
        self.train_losses_ = state.get("train_losses", [])
        self.validation_losses_ = state.get("validation_losses", [])
