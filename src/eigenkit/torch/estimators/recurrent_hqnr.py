import math
from typing import Literal

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted
from torch import nn

from eigenkit.core.circuit import ProviderSpec
from eigenkit.qml import BaseQuantumRegressor
from eigenkit.qml._registry import register_estimator
from eigenkit.torch.layers import MinMaxNorm

from ._nn import ModelAccessMixin, _make_q_layer, _resolve_quantum_dims, _ToFloat32
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


class _SequenceEncoder(nn.Module):
    """RNN over `(batch, time, feat)`.

    With `return_last_timestep=True` (default) returns only the last
    timestep, shape `(batch, hidden_dim)`. Otherwise returns the full
    output sequence, shape `(batch, time, hidden_dim)`.
    """

    def __init__(
        self,
        *,
        rnn_type: Literal["rnn", "gru", "lstm"],
        input_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        return_last_timestep: bool = True,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.return_last_timestep = return_last_timestep
        cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        drop = dropout if num_layers > 1 else 0.0
        self.rnn = cls(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop,
            bidirectional=bidirectional,
        )
        self._out_dim = hidden_dim * (2 if bidirectional else 1)

    @property
    def out_features(self) -> int:
        """Per-timestep feature size (last dimension of the RNN output)."""
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        if self.return_last_timestep:
            return out[:, -1, :]
        return out


@register_estimator
class RecurrentHQNR(ModelAccessMixin, BaseQuantumRegressor):
    """Hybrid regressor: a recurrent encoder, a quantum layer and a linear head.

    The recurrent encoder is a recurrent neural network (RNN). The estimator
    expects 3-D inputs of shape `(n_samples, seq_len, seq_feature_dim)`. By
    default, only the last timestep of the RNN output goes to the quantum layer.
    Set `rnn_return_last_timestep=False`, together with `sequence_length`, to use
    the full RNN output (flattened) instead. The quantum part is the same as in
    [`HQNR`][eigenkit.torch.estimators.HQNR]: `MinMaxNorm`, the quantum layer and
    an `nn.Linear` head.

    Args:
        seq_feature_dim: The number of features at each time step (the last axis
            of `X`).
        rnn_type: The type of recurrent layer: `"rnn"`, `"gru"` (gated recurrent
            unit) or `"lstm"` (long short-term memory).
        rnn_hidden_dim: The hidden size in each direction.
        rnn_num_layers: The number of stacked RNN layers.
        rnn_dropout: The dropout between the RNN layers. It applies only if
            `rnn_num_layers > 1`.
        rnn_bidirectional: If True, the RNN is bidirectional, and its output size
            doubles.
        rnn_return_last_timestep: If True, only the last output step is kept.
            Otherwise, the full sequence is flattened, which requires
            `sequence_length`.
        sequence_length: The number of time steps in each sample. It is required
            when `rnn_return_last_timestep=False`, and it must equal `X.shape[1]`
            in `fit`.
        in_quantum: The number of inputs of the quantum layer. It defaults to the
            size of the RNN output that reaches the quantum layer: `rnn_hidden_dim`,
            doubled when the RNN is bidirectional, and multiplied by
            `sequence_length` when `rnn_return_last_timestep=False`.
        n_qubits: The number of qubits in each circuit. It defaults to
            `in_quantum`.
        depth: The number of repetitions of the variational layer.
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
        optimizer: The name of the optimiser.
        optimizer_kwargs: Extra keyword arguments passed to the optimiser.
        epochs: The number of training epochs.
        batch_size: The size of each mini-batch.
        show_progress: If True, a tqdm progress bar is shown.
        loss_func: The regression loss: `"mse"`, `"mae"` or `"huber"`.
        huber_delta: The threshold of the Huber loss.
        huber_reduction: The reduction of the Huber loss: `"mean"` or `"sum"`.
        random_state: The seed.

    Attributes:
        train_losses_: The training loss of each epoch, set by `fit()`.
        validation_losses_: The validation loss of each epoch, set by `fit()` when
            validation data is given.
    """

    accepts_sequence_input: bool = True

    def __init__(
        self,
        seq_feature_dim: int,
        rnn_type: Literal["rnn", "gru", "lstm"] = "gru",
        rnn_hidden_dim: int = 32,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
        rnn_bidirectional: bool = False,
        rnn_return_last_timestep: bool = True,
        sequence_length: int | None = None,
        in_quantum: int | None = None,
        n_qubits: int | None = None,
        depth: int = 2,
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
        loss_func: str = "mse",
        huber_delta: float = 1.0,
        huber_reduction: str = "mean",
        random_state: int | None = 42,
    ):
        super().__init__(provider=provider, random_state=random_state)
        if not isinstance(seq_feature_dim, int) or seq_feature_dim < 1:
            raise ValueError("seq_feature_dim must be a positive integer.")
        self.seq_feature_dim = seq_feature_dim
        self.rnn_type = rnn_type
        if rnn_type not in ("rnn", "gru", "lstm"):
            raise ValueError("rnn_type must be 'rnn', 'gru', or 'lstm'.")
        if not isinstance(rnn_hidden_dim, int) or rnn_hidden_dim < 1:
            raise ValueError("rnn_hidden_dim must be a positive integer.")
        if not isinstance(rnn_num_layers, int) or rnn_num_layers < 1:
            raise ValueError("rnn_num_layers must be a positive integer.")
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout = float(rnn_dropout)
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_return_last_timestep = rnn_return_last_timestep
        self.sequence_length = sequence_length
        if not rnn_return_last_timestep:
            if sequence_length is None or sequence_length < 1:
                raise ValueError(
                    "sequence_length must be a positive integer when "
                    "rnn_return_last_timestep=False."
                )
        elif sequence_length is not None:
            raise ValueError(
                "sequence_length must be None when rnn_return_last_timestep=True."
            )
        self.n_qubits = n_qubits
        self.depth = depth
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

        validate_regression_loss_config(loss_func, huber_delta, huber_reduction)
        self.loss_func = loss_func
        self.huber_delta = huber_delta
        self.huber_reduction = huber_reduction

        # Stored as given, not as resolved. Writing the derived width into the
        # public parameter would make `get_params()` report an int the caller
        # never passed, so `clone()` would not reproduce the constructor
        # arguments -- the contract the properties below rely on.
        # `RecurrentHQNC` stores it the same way.
        self.in_quantum = in_quantum

    @property
    def _rnn_effective_out_dim(self) -> int:
        """Width the RNN hands the quantum block, derived on read.

        Not cached in `__init__`, for the same reason as the quantum dims:
        `set_params(rnn_hidden_dim=...)` writes the public attribute and nothing
        else, so with a cached width a search over the RNN width would build the
        encoder from the new value and the input adapter from the old.
        `sequence_length` is re-checked for the same reason -- `__init__`
        validates it, but `set_params` can invalidate it afterwards. Mirrors
        `RecurrentHQNC._resolve_model_dims`.
        """
        rnn_out_dim = self.rnn_hidden_dim * (2 if self.rnn_bidirectional else 1)
        if self.rnn_return_last_timestep:
            return rnn_out_dim
        if self.sequence_length is None or self.sequence_length < 1:
            raise ValueError(
                "sequence_length must be a positive integer when "
                "rnn_return_last_timestep=False."
            )
        return self.sequence_length * rnn_out_dim

    # Architecture derived from the constructor arguments. Properties, not
    # fields assigned in `__init__`: `set_params` writes the public attribute
    # and nothing else, so a value cached at construction would go stale and
    # the model would be built from the old one, making every `GridSearchCV`
    # over `n_qubits`, `in_quantum` or `layer_type` train one architecture for
    # the whole grid. Deriving on read also keeps `__init__` to plain assignment,
    # which is what scikit-learn's clone/get_params contract expects.
    @property
    def _quantum_dims(self) -> tuple[int, int]:
        effective_out_dim = self._rnn_effective_out_dim
        return _resolve_quantum_dims(
            in_features=effective_out_dim,
            in_quantum=(
                self.in_quantum if self.in_quantum is not None else effective_out_dim
            ),
            n_qubits=self.n_qubits,
            layer_type=self.layer_type,
        )

    @property
    def _resolved_n_qubits(self) -> int:
        return self._quantum_dims[0]

    @property
    def _quantum_input_dim(self) -> int:
        return self._quantum_dims[1]

    def _make_loss_fn(self):
        return make_regression_loss_fn(
            self.loss_func, self.huber_delta, self.huber_reduction
        )

    def _build_model(self) -> nn.Module:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        # Once per build, as in `RecurrentHQNC`: the properties re-derive on
        # every read and `_resolve_quantum_dims` warns, so reading them through
        # the build would repeat that warning five times per `fit()`.
        resolved_n_qubits, quantum_input_dim = self._quantum_dims
        encoder = _SequenceEncoder(
            rnn_type=self.rnn_type,
            input_size=self.seq_feature_dim,
            hidden_dim=self.rnn_hidden_dim,
            num_layers=self.rnn_num_layers,
            dropout=self.rnn_dropout,
            bidirectional=self.rnn_bidirectional,
            return_last_timestep=self.rnn_return_last_timestep,
        )
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

        modules: list[nn.Module] = [encoder]
        if not self.rnn_return_last_timestep:
            modules.append(nn.Flatten(start_dim=1, end_dim=-1))
        if self._rnn_effective_out_dim != quantum_input_dim:
            modules.append(nn.Linear(self._rnn_effective_out_dim, quantum_input_dim))
        modules.extend(
            [
                MinMaxNorm(quantum_input_dim, min_val=0.0, max_val=math.pi),
                q_layer,
                _ToFloat32(),
                nn.Linear(q_layer.out_features, 1),
            ]
        )
        return nn.Sequential(*modules)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim != 3:
            raise ValueError(
                "RecurrentHQNR expects X with shape "
                "(n_samples, seq_len, seq_feature_dim)."
            )
        if X_arr.shape[2] != self.seq_feature_dim:
            raise ValueError(
                f"Last dimension of X must equal seq_feature_dim={self.seq_feature_dim}, "
                f"got {X_arr.shape[2]}."
            )
        if not self.rnn_return_last_timestep:
            if X_arr.shape[1] != self.sequence_length:
                raise ValueError(
                    f"X.shape[1] must equal sequence_length={self.sequence_length}, "
                    f"got {X_arr.shape[1]}."
                )
        return X_arr

    @isolated_rng
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RecurrentHQNR":
        """Fit the estimator on `X` and `y`.

        Args:
            X (np.ndarray): The sequences, of shape
                `(n_samples, seq_len, seq_feature_dim)`.
            y (np.ndarray): One target for each sequence, of shape `(n_samples,)`
                or `(n_samples, 1)`.
            X_val (np.ndarray | None): Optional validation sequences, of the same
                shape as `X`. Default: `None`.
            y_val (np.ndarray | None): The targets of `X_val`. Default: `None`.

        Returns:
            This estimator, so that calls can be chained.
        """
        X_arr = self._validate_X(X)
        y_arr = np.asarray(y)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.reshape(-1)
        if y_arr.ndim != 1:
            raise ValueError("y must be 1-D or shape (n_samples, 1).")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must have the same number of samples.")

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
            X_arr, y_arr, batch_size=self.batch_size, target_dtype=torch.float32
        )
        loss_fn = self._make_loss_fn()
        val_dl = None
        if X_val is not None and y_val is not None:
            X_val_arr = self._validate_X(X_val)
            y_val_arr = np.asarray(y_val)
            if y_val_arr.ndim == 2 and y_val_arr.shape[1] == 1:
                y_val_arr = y_val_arr.reshape(-1)
            if y_val_arr.ndim != 1:
                raise ValueError("y_val must be 1-D or shape (n_samples, 1).")
            if len(X_val_arr) != len(y_val_arr):
                raise ValueError("X_val and y_val must have the same number of samples.")
            val_dl = make_dataloader(
                X_val_arr,
                y_val_arr,
                batch_size=self.batch_size,
                target_dtype=torch.float32,
                shuffle=False,
            )
        self.train_losses_, self.validation_losses_ = train_loop(
            self._model,
            dl,
            optimizer,
            loss_fn,
            self.epochs,
            show_progress=self.show_progress,
            progress_desc="RecurrentHQNR training",
            validation_dataloader=val_dl,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a value for every sample in `X`.

        Args:
            X (np.ndarray): The sequences, of shape
                `(n_samples, seq_len, seq_feature_dim)`.

        Returns:
            One prediction for each sequence, as a float32 array of shape
                `(n_samples,)`.
        """
        check_is_fitted(self, "_model")
        X_arr = self._validate_X(X)
        X_t = torch.as_tensor(X_arr, dtype=torch.float32)
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
