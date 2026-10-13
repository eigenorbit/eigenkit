"""Neural network helpers for the hybrid estimators, which need PyTorch.

The factory of quantum layers maps each `layer_type` string to the matching layer
class:

    "Basic"    -> Basic
    "Reupload" -> Reupload
    "Ensemble" -> Ensemble

These helpers need PyTorch, because they
build `nn.Module` objects. `HQNC`, `HQNR`, `RecurrentHQNC` and `RecurrentHQNR`
share them. The estimator base classes compatible with scikit-learn, and the
functions that check the rotations (`RX`, `RY`, `RZ`) and the bases (`X`, `Y`,
`Z`), are in a separate package, `eigenkit.qml`, which depends on no framework.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted
from torch import nn

from eigenkit.core.circuit import ProviderSpec
from eigenkit.qml._base import _validate_basis, _validate_rotation

from ..layers import Basic, Ensemble, Reupload


class ModelAccessMixin:
    """Read access to the network that an estimator built and trained.

    `fit()` trains a deep copy of what the estimator was configured with, so
    fitting never changes the object that it was given, as in scikit-learn. This
    means that the weights of a layer passed to `GenericClassifier(model=...)` do
    not change during `fit()`, however well the estimator trained. `model_` is the
    copy that was trained.

    Attributes:
        model_: The trained `torch.nn.Module`. Reading it before `fit()` raises a
            `NotFittedError`. Changing it changes the fitted estimator.
    """

    def _check_feature_count(self, X):
        """Reject an X whose column count is not the one this was fitted on.

        The qml estimators make the same check. Without it, a torch estimator
        would let X reach the first `nn.Linear`, where it surfaces as
        `mat1 and mat2 shapes cannot be multiplied (4x7 and 3x2)`, which names
        neither the estimator nor what it wanted.

        Args:
            X: Samples to score.

        Returns:
            `X` unchanged.

        Raises:
            ValueError: If the column count differs from `n_features_in_`.
        """
        expected = getattr(self, "n_features_in_", None)
        if expected is not None:
            given = int(np.asarray(X).shape[-1])
            if given != expected:
                raise ValueError(
                    f"{type(self).__name__} was fitted on {expected} features; "
                    f"got {given}."
                )
        return X

    @property
    def model_(self) -> nn.Module:
        """The `torch.nn.Module` that this estimator trained.

        Returns:
            model_ (torch.nn.Module): The network that `fit()` trained. It is the
                copy that belongs to the estimator, not the object passed to the
                constructor.

        Raises:
            NotFittedError: If `fit()` has not been called.
        """
        check_is_fitted(self, "_model")
        return self._model


class _ToFloat32(nn.Module):
    # Quantum layers may output float64; cast to float32 before nn.Linear.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


# ---- quantum-layer factory shared across HQNC, HQNR, RecurrentHQNR ----


_SUPPORTED_LAYER_TYPES = {"Basic", "Reupload", "Ensemble"}


def _validate_layer_type(layer_type: str) -> None:
    if layer_type not in _SUPPORTED_LAYER_TYPES:
        raise ValueError(
            f"Unknown layer_type: {layer_type!r}. "
            f"Supported: {sorted(_SUPPORTED_LAYER_TYPES)}"
        )


def _resolve_quantum_dims(
    *,
    in_features: int,
    in_quantum: int,
    n_qubits: int | None,
    layer_type: str,
) -> tuple[int, int]:
    _validate_layer_type(layer_type)
    resolved_n_qubits = n_qubits if n_qubits is not None else in_quantum

    if layer_type == "Ensemble" and in_quantum % resolved_n_qubits != 0:
        raise ValueError(
            f"For layer_type='{layer_type}', "
            f"in_quantum ({in_quantum}) must be divisible by "
            f"n_qubits ({resolved_n_qubits})."
        )

    # Basic and Reupload require in_quantum == n_qubits; insert a Linear
    # adapter and warn if they differ.
    if in_quantum != resolved_n_qubits and layer_type in {"Basic", "Reupload"}:
        warnings.warn(
            f"For layer_type='{layer_type}', in_quantum should equal n_qubits. "
            "Inserting an input adapter Linear(in_features -> n_qubits).",
            UserWarning,
            stacklevel=2,
        )
        return resolved_n_qubits, resolved_n_qubits

    return resolved_n_qubits, in_quantum


def _ansatz_kwargs(
    enc_rotation: str | None,
    var_rotation: str | None,
    variational_type: str | None,
    basis: str | None,
) -> dict[str, str]:
    """Validate the four ansatz arguments, dropping the ones left as None.

    A dropped argument is not passed to the layer at all, so the layer's own
    default applies. Validation still runs on everything that was supplied, so
    a typo is caught here rather than deep inside the block constructors.
    """
    kwargs: dict[str, str] = {}
    if enc_rotation is not None:
        kwargs["enc_rotation"] = _validate_rotation(enc_rotation)
    if var_rotation is not None:
        kwargs["var_rotation"] = _validate_rotation(var_rotation)
    if basis is not None:
        kwargs["basis"] = _validate_basis(basis)
    if variational_type is not None:
        if variational_type not in {"basic", "strong"}:
            raise ValueError(
                f"variational_type must be 'basic' or 'strong', got {variational_type!r}"
            )
        kwargs["variational_type"] = variational_type
    return kwargs


def _make_q_layer(
    in_quantum: int,
    n_qubits: int,
    depth: int,
    layer_type: str,
    enc_rotation: Literal["RX", "RY", "RZ"] | None,
    var_rotation: Literal["RX", "RY", "RZ"] | None,
    variational_type: Literal["basic", "strong"] | None,
    basis: Literal["X", "Y", "Z"] | None,
    provider: ProviderSpec,
) -> nn.Module:
    """Quantum-layer factory used by HQNC, HQNR, and RecurrentHQNR.

    Returns the layer matching `layer_type`:

        "Basic"    -> Basic
        "Reupload" -> Reupload
        "Ensemble" -> Ensemble

    `enc_rotation`, `var_rotation`, `variational_type` and `basis` accept None,
    meaning "whatever this layer's own default is". The estimators pass None by
    default so that there is one place each of these values is decided — the
    layer — instead of an estimator default silently overriding it with a
    different one.
    """
    _validate_layer_type(layer_type)
    ansatz = _ansatz_kwargs(enc_rotation, var_rotation, variational_type, basis)

    if layer_type == "Basic":
        return Basic(
            n_qubits=in_quantum,
            depth=depth,
            provider=provider,
            **ansatz,
        )

    if layer_type == "Reupload":
        return Reupload(
            n_qubits=in_quantum,
            depth=depth,
            provider=provider,
            **ansatz,
        )

    return Ensemble(
        in_features=in_quantum,
        n_qubits=n_qubits,
        depth=depth,
        provider=provider,
        **ansatz,
    )
