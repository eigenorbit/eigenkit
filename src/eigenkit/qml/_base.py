"""The base estimator classes of quantum machine learning, for scikit-learn.

`BaseQuantumEstimator` and its classifier and regressor subclasses depend on no
framework: they need only NumPy and scikit-learn. The `provider` argument is the
same as in `Circuit`: a registered name, a configuration instance, or a backend.
The layer factory that the hybrid estimators use needs PyTorch, so it is in a
separate package, `eigenkit.torch.estimators._nn`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from eigenkit.core.circuit import ProviderSpec

from ._serialization import (
    SerializationError,
    check_format_version,
    deserialize_circuit_spec,
    deserialize_provider_param,
    load_bundle,
    resolve_bundle_path,
    save_bundle,
    serialize_circuit_spec,
    serialize_provider_param,
    to_json_safe,
)


# ---- shared helpers ----


_VALID_BASES = {"X", "Y", "Z"}


def _validate_basis(basis: str) -> str:
    if basis not in _VALID_BASES:
        raise ValueError(f"basis must be one of 'X', 'Y', 'Z', got {basis!r}")
    return basis


def _validate_rotation(rotation: str) -> str:
    if rotation not in {"RX", "RY", "RZ"}:
        raise ValueError(f"rotation must be one of 'RX', 'RY', 'RZ', got {rotation!r}")
    return rotation


# ---- base estimator ----


class BaseQuantumEstimator(BaseEstimator, ABC):
    """Base class, compatible with scikit-learn, for all quantum algorithms.

    Args:
        provider: The backend. It can be a registered name (`"pennylane"`,
            `"qiskit"`, `"braket"`, or any name passed to `register()`), a
            configuration instance (for example `PennyLaneConfig(device_name=...)`),
            or a backend instance.
        random_state: The seed for reproducibility, applied inside `fit()` and
            `load()`. What it seeds depends on the estimator. The PyTorch
            estimators isolate the random number generator of PyTorch, so seeding
            one of them does not reseed the random stream of the caller. The kernel
            estimators are deterministic, and they take it only so that all the
            estimators have the same interface.
    """

    def __init__(
        self,
        provider: ProviderSpec = "numpy",
        random_state: int | None = 42,
    ):
        self.provider = provider
        self.random_state = random_state

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseQuantumEstimator":
        """Fit the estimator on `X` and `y`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.
            y (np.ndarray): One target for each sample, of shape `(n_samples,)`.

        Returns:
            This estimator, so that calls can be chained.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a value for every sample in `X`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            One prediction for each sample.
        """
        ...

    def save(self, path: str | Path) -> None:
        """Save a fitted model to a `.eigenkit` bundle.

        The bundles of models that hold scikit-learn objects (`QSVC`, `QSVR`, the
        reservoir estimators and the time-series estimators) contain joblib
        pickles, so loading one can run code. Share them only with people who
        trust you.

        Args:
            path: The path of the bundle. The suffix `.eigenkit` is added when it
                is missing.
        """
        import importlib.metadata
        from datetime import datetime, timezone

        path = resolve_bundle_path(path)

        provider_val = getattr(self, "provider", None)
        provider_str = (
            provider_val if isinstance(provider_val, str) else repr(provider_val)
        )
        metadata = {
            "class_name": type(self).__name__,
            "library_version": importlib.metadata.version("eigenkit"),
            "format_version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider_str,
            **self._extra_metadata(),
        }

        params, circuit_spec = self._serialize_params()
        state = self._collect_state()

        save_bundle(
            path,
            metadata=metadata,
            params=params,
            circuit_spec=circuit_spec,
            state=state,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        provider: ProviderSpec | None = None,
    ) -> "BaseQuantumEstimator":
        """Load a fitted model from a `.eigenkit` bundle.

        Load only bundles that you trust. A bundle can contain joblib pickles, and
        unpickling runs code, as with `torch.load` or `joblib.load`.

        Args:
            path: The path of the bundle.
            provider: A provider that replaces the saved one.

        Returns:
            The loaded estimator.

        Raises:
            TypeError: If the method is called on a subclass and the bundle holds
                a different class. `BaseQuantumEstimator.load()` accepts any class.
        """
        from ._registry import get_estimator_class

        metadata, params, circuit_spec, state = load_bundle(path)
        check_format_version(metadata)

        if provider is not None:
            params["provider"] = provider
        elif "provider" in params:
            params["provider"] = deserialize_provider_param(params["provider"])

        for key, value in list(params.items()):
            # A QuantumKernel parameter (`QSVC(kernel=...)`) is stored as its
            # own JSON description; rebuild it, on the override provider if any.
            if isinstance(value, dict) and "__quantum_kernel__" in value:
                from ._kernels import QuantumKernel

                params[key] = QuantumKernel.from_json_dict(value, provider=provider)

        if circuit_spec is not None:
            blocks, observables = deserialize_circuit_spec(circuit_spec)
            params["blocks"] = blocks
            if observables is not None:
                params["observables"] = observables

        actual_cls = get_estimator_class(metadata["class_name"])
        if cls is not BaseQuantumEstimator and actual_cls is not cls:
            raise TypeError(
                f"{cls.__name__}.load() expected a {cls.__name__!r} checkpoint, "
                f"but the file contains {actual_cls.__name__!r}. "
                f"Use {actual_cls.__name__}.load() or BaseQuantumEstimator.load() instead."
            )
        model = actual_cls(**params)
        model._restore_state(state)
        return model

    def _serialize_params(self) -> tuple[dict, dict | None]:
        """Return ``(params_dict, circuit_spec_or_None)`` for the bundle."""
        all_params = dict(self.get_params(deep=False))
        circuit_spec = None
        params: dict = {}

        blocks = all_params.pop("blocks", None)
        observables = all_params.pop("observables", None)

        if blocks is not None:
            circuit_spec = serialize_circuit_spec(blocks, observables)

        for k, v in all_params.items():
            if k == "provider":
                params[k] = serialize_provider_param(v)
                continue
            if callable(getattr(v, "to_json_dict", None)):
                params[k] = v.to_json_dict()
                continue
            safe_v = to_json_safe(v)
            try:
                json.dumps(safe_v)
                params[k] = safe_v
            except (TypeError, ValueError) as exc:
                raise SerializationError(
                    f"Parameter {k!r} of {type(self).__name__} has type "
                    f"{type(v).__name__} which is not JSON-serializable. "
                    "Override _serialize_params() to handle it."
                ) from exc

        return params, circuit_spec

    def _extra_metadata(self) -> dict:
        """Extra fields to merge into bundle metadata (override in subclasses)."""
        return {}

    def _collect_state(self) -> dict:
        """Return trainable state for ``save()`` (override in subclasses)."""
        return {}

    def _restore_state(self, state: dict) -> None:
        """Restore trainable state after ``load()`` (override in subclasses)."""
        pass


# The mixin comes first, ahead of the base. scikit-learn reads an estimator's
# kind from `__sklearn_tags__`, and with the mixin last in the MRO
# `BaseEstimator`'s implementation would shadow `ClassifierMixin`'s -- so
# `is_classifier()` would be False for every estimator in this library.
# `cross_val_score` would then choose KFold over StratifiedKFold (scoring
# 0.7/1.0/0.4 where a stratified split gives 0.9/1.0/0.8 on the same data), and
# `VotingClassifier` would refuse them outright with "should be a classifier".
class BaseQuantumClassifier(ClassifierMixin, BaseQuantumEstimator):
    """Base class for quantum classifiers. Its `score()` returns the accuracy."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the class probabilities.

        Subclasses should return the probabilities in the order of `classes_`,
        with shape `(n_samples, n_classes)`. The default implementation raises a
        `NotImplementedError`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            The probabilities, of shape `(n_samples, n_classes)`, with the columns
                in the order of `classes_`.
        """
        raise NotImplementedError("This classifier does not implement predict_proba().")


class BaseQuantumRegressor(RegressorMixin, BaseQuantumEstimator):
    """Base class for quantum regressors. Its `score()` returns `R²`."""
