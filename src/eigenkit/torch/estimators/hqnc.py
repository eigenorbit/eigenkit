import math
from typing import Literal

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch import nn

from eigenkit.core.circuit import ProviderSpec
from eigenkit.qml import BaseQuantumClassifier
from eigenkit.qml._registry import register_estimator
from eigenkit.torch.layers import MinMaxNorm

from ._nn import ModelAccessMixin, _make_q_layer, _resolve_quantum_dims, _ToFloat32
from ._seeding import isolated_rng
from ._training import (
    make_dataloader,
    make_optimizer,
    predict_loop,
    train_loop,
    validate_optimizer_name,
)


@register_estimator
class HQNC(ModelAccessMixin, BaseQuantumClassifier):
    """Hybrid quantum neural network classifier (HQNC).

    The model is `MinMaxNorm -> quantum_layer -> nn.Linear(q_out, n_classes)`. If
    the number of features of the data differs from the resolved input size of the
    quantum layer (derived from `in_quantum`, `n_qubits` and `layer_type`), an
    `nn.Linear` adapter is added in front automatically.

    Args:
        in_features: The number of input features.
        in_quantum: The number of inputs that the quantum layer expects. It
            defaults to `in_features`.
        n_qubits: The number of qubits in each circuit. It defaults to
            `in_quantum`.
        depth: The number of repetitions of the variational layer.
        n_classes: The number of output classes. If None, it is inferred from the
            training labels.
        layer_type: The quantum layer. It is one of `"Basic"`, `"Reupload"` and
            `"Ensemble"`.
        enc_rotation: The encoding rotation: `"RX"`, `"RY"`, `"RZ"` or None.
        var_rotation: The variational rotation of the `"basic"` ansatz: `"RX"`,
            `"RY"`, `"RZ"` or None.
        variational_type: The type of variational block: `"basic"`, `"strong"` or
            None.
        basis: The measurement basis: `"X"`, `"Y"`, `"Z"` or None. For these four
            arguments, None means "use the default of the layer named by
            `layer_type`", which is where each of these values is decided.
        provider: The backend. It takes the same options as
            `Circuit(provider=...)`.
        lr: The learning rate of the optimiser.
        optimizer: The name of the optimiser. The supported names are `"adam"`,
            `"adamw"`, `"rmsprop"` and `"sgd"`.
        optimizer_kwargs: Extra keyword arguments passed to the optimiser.
        epochs: The number of training epochs.
        batch_size: The size of each mini-batch.
        show_progress: If True, a tqdm progress bar is shown.
        random_state: The seed for the initialisation of the weights.

    Attributes:
        train_losses_: The average cross-entropy loss of each epoch, set by
            `fit()`.
        validation_losses_: The validation cross-entropy loss of each epoch, set
            by `fit()` when validation data is given.

    Example:
        ```python
        import numpy as np
        from eigenkit.torch.estimators import HQNC

        rng = np.random.default_rng(0)
        X_train, X_test = rng.normal(size=(30, 4)), rng.normal(size=(5, 4))
        y_train = np.arange(30) % 3

        model = HQNC(in_features=4, depth=2, n_classes=3)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        ```
    """

    def __init__(
        self,
        in_features: int,
        in_quantum: int | None = None,
        n_qubits: int | None = None,
        depth: int = 2,
        n_classes: int | None = None,
        layer_type: str = "Basic",
        enc_rotation: Literal["RX", "RY", "RZ"] | None = None,
        var_rotation: Literal["RX", "RY", "RZ"] | None = None,
        variational_type: Literal["basic", "strong"] | None = None,
        basis: Literal["X", "Y", "Z"] | None = None,
        provider: ProviderSpec = "numpy",
        lr: float = 0.01,
        optimizer: str = "adam",
        optimizer_kwargs: dict | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        show_progress: bool = True,
        random_state: int | None = 42,
    ):
        """Initialise a hybrid quantum neural network classifier (HQNC).

        Args:
            in_features: The number of input features that `fit()` expects.
            in_quantum: The number of inputs that the quantum layer expects. It
                defaults to `in_features`.
            n_qubits: The number of qubits in each circuit. It defaults to
                `in_quantum`.
            depth: The number of repetitions of the variational layer.
            n_classes: The number of output classes. If `None`, it is inferred
                from the labels passed to `fit()`.
            layer_type: The family of the quantum layer. It is one of `"Basic"`,
                `"Reupload"` and `"Ensemble"`.
            enc_rotation: The encoding rotation: `"RX"`, `"RY"`, `"RZ"` or None.
            var_rotation: The variational rotation of the basic ansatz: `"RX"`,
                `"RY"`, `"RZ"` or None.
            variational_type: The type of variational block: `"basic"`,
                `"strong"` or None.
            basis: The measurement basis: `"X"`, `"Y"`, `"Z"` or None. For these
                four arguments, None means "use the default of the layer named by
                `layer_type`", which is where each of these values is decided.
            provider: The backend. It takes the same options as
                `Circuit(provider=...)`.
            lr: The learning rate of the optimiser.
            optimizer: The name of the optimiser. The supported names are
                `"adam"`, `"adamw"`, `"rmsprop"` and `"sgd"`.
            optimizer_kwargs: Extra keyword arguments passed to the optimiser.
            epochs: The number of training epochs.
            batch_size: The size of each mini-batch.
            show_progress: If True, a tqdm progress bar is shown during training.
            random_state: The seed for the initialisation of the weights.
        """
        super().__init__(provider=provider, random_state=random_state)
        self.in_features = in_features
        self.in_quantum = in_quantum if in_quantum is not None else in_features
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_classes = n_classes
        self.layer_type = layer_type
        self.enc_rotation = enc_rotation
        self.var_rotation = var_rotation
        self.variational_type = variational_type
        self.basis = basis
        self.lr = lr
        validate_optimizer_name(optimizer)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.show_progress = show_progress

    # Architecture derived from the constructor arguments. Properties, not
    # fields assigned in `__init__`: `set_params` writes the public attribute
    # and nothing else, so a value cached at construction would go stale and
    # the model would be built from the old one, making every `GridSearchCV`
    # over `n_qubits`, `in_quantum` or `layer_type` train one architecture for
    # the whole grid. Deriving on read also keeps `__init__` to plain assignment,
    # which is what scikit-learn's clone/get_params contract expects.
    @property
    def _quantum_dims(self) -> tuple[int, int]:
        return _resolve_quantum_dims(
            in_features=self.in_features,
            in_quantum=self.in_quantum,
            n_qubits=self.n_qubits,
            layer_type=self.layer_type,
        )

    @property
    def _resolved_n_qubits(self) -> int:
        return self._quantum_dims[0]

    @property
    def _quantum_input_dim(self) -> int:
        return self._quantum_dims[1]

    def _build_model(self, n_classes: int) -> nn.Sequential:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        # Once per build, as in `RecurrentHQNC`: the properties re-derive on
        # every read and `_resolve_quantum_dims` warns, so reading them through
        # the build would repeat that warning five times per `fit()`.
        resolved_n_qubits, quantum_input_dim = self._quantum_dims
        q_layer = _make_q_layer(
            quantum_input_dim,
            resolved_n_qubits,
            self.depth,
            self.layer_type,
            self.enc_rotation,
            self.var_rotation,
            self.variational_type,
            self.basis,
            self.provider,
        )

        modules: list[nn.Module] = []
        if self.in_features != quantum_input_dim:
            modules.append(nn.Linear(self.in_features, quantum_input_dim))
        modules.extend(
            [
                MinMaxNorm(quantum_input_dim, min_val=0.0, max_val=math.pi),
                q_layer,
                _ToFloat32(),
                nn.Linear(q_layer.out_features, n_classes),
            ]
        )
        return nn.Sequential(*modules)

    @isolated_rng
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "HQNC":
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
            ValueError: If `X` does not have `in_features` columns.
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

        # Remember the width actually built. `n_classes` may exceed the number
        # of labels present in y -- that is what declaring `n_classes` is for --
        # and saving the observed count instead would make the checkpoint
        # unloadable.
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
            progress_desc="HQNC training",
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

        It returns a float32 array of shape `(n_samples, n_classes)`. Each row
        sums to `1`.

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
