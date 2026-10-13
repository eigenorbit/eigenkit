"""Model evaluation helpers for compare()."""

from __future__ import annotations

import copy
import inspect
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from ._metrics import Metric, compute_metrics, timer


def _base_quantum_estimator():
    """Lazily fetch ``BaseQuantumEstimator`` from the torch-free qml tier.

    Taking it from the torch ring's re-export would make ``compare()`` import
    torch for any quantum model — a QSVC benchmark included. The class is the
    same object either way."""
    from ..qml import BaseQuantumEstimator

    return BaseQuantumEstimator


def _is_sklearn_builtin_estimator(model: Any) -> bool:
    """True for estimators from the sklearn package (not kit quantum bases)."""
    if not isinstance(model, BaseEstimator) or isinstance(
        model, _base_quantum_estimator()
    ):
        return False
    module = type(model).__module__
    return module == "sklearn" or module.startswith("sklearn.")


def _collect_non_sklearn_components(model: Any) -> list[str]:
    """Return param keys whose values are non-sklearn BaseEstimator sub-components."""
    if not hasattr(model, "get_params"):
        return []
    return [
        f"{key} ({type(val).__name__})"
        for key, val in model.get_params(deep=True).items()
        if isinstance(val, BaseEstimator) and not _is_sklearn_builtin_estimator(val)
    ]


def _validate_benchmark_models(
    models: list[Any], model_names: list[str] | None = None
) -> None:
    """Require kit quantum bases or built-in sklearn estimators."""
    for i, model in enumerate(models):
        name = type(model).__name__
        locator = f"models[{i}]" if model_names is None else f"models[{model_names[i]!r}]"
        is_kit = isinstance(model, _base_quantum_estimator())
        is_sklearn = _is_sklearn_builtin_estimator(model)

        if not is_kit and not is_sklearn:
            raise TypeError(
                f"{locator} ({name!r}) is not supported. "
                "Each model must subclass BaseQuantumEstimator, "
                "BaseQuantumClassifier, or BaseQuantumRegressor, "
                "or be a built-in sklearn estimator (module sklearn.*)."
            )
        if not callable(getattr(model, "fit", None)):
            raise TypeError(f"{locator} ({name!r}) has no callable fit() method.")
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"{locator} ({name!r}) has no callable predict() method.")

        if is_sklearn:
            non_sklearn = _collect_non_sklearn_components(model)
            if non_sklearn:
                warnings.warn(
                    f"{locator} ({name!r}) contains non-sklearn components: "
                    f"{', '.join(non_sklearn)}. "
                    "The model will be saved as .joblib, but deserialization may fail "
                    "in a different Python environment where these classes are not importable.",
                    UserWarning,
                    stacklevel=3,
                )


def _get_model_name(model: Any) -> str:
    return type(model).__name__


def _task_type(model: Any) -> str:
    return "regression" if isinstance(model, RegressorMixin) else "classification"


def _clone_model(model: Any, seed: int) -> Any:
    """Clone a model for a fresh fit and best-effort seed it."""
    try:
        cloned = clone(model)
    except TypeError:
        try:
            return copy.deepcopy(model)
        except Exception:
            return model

    if hasattr(cloned, "get_params") and hasattr(cloned, "set_params"):
        # Every `random_state` the estimator exposes, nested ones included.
        # Setting the top-level name alone raises on a `Pipeline` -- which has
        # no `random_state` of its own -- so no estimator inside a pipeline or
        # any meta-estimator would be seeded, and three identical `compare()`
        # calls would return three different scores.
        try:
            names = [
                name
                for name in cloned.get_params(deep=True)
                if name == "random_state" or name.endswith("__random_state")
            ]
        except (ValueError, TypeError):
            names = []
        if names:
            try:
                cloned.set_params(**dict.fromkeys(names, seed))
            except (ValueError, TypeError):
                pass
    return cloned


def _maybe_predict_proba(model: Any, X: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(X)
    except Exception:
        return None


def _fit_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
) -> None:
    if X_val is not None and y_val is not None:
        params = inspect.signature(model.fit).parameters
        if "X_val" in params and "y_val" in params:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            return
    model.fit(X_train, y_train)


def _loss_curve_for(model: Any) -> dict[str, list[float]] | None:
    train = getattr(model, "train_losses_", None)
    validation = getattr(model, "validation_losses_", None)
    if train is None and validation is None:
        return None
    return {
        "train": [] if train is None else list(train),
        "validation": [] if validation is None else list(validation),
    }


def _try_save_model(
    model: Any, save_dir: Path, seed: int, save_name: str | None = None
) -> Path | None:
    # `_append_suffix` lives in the qml tier; import lazily so loading the
    # benchmark module stays cheap and torch-free until a save actually runs.
    from ..qml._serialization import _append_suffix

    save_dir.mkdir(parents=True, exist_ok=True)
    name = save_name if save_name is not None else type(model).__name__
    path_base = save_dir / f"{name}_seed{seed}"

    if isinstance(model, _base_quantum_estimator()):
        path = _append_suffix(path_base, ".eigenkit")
        model.save(path)
        return path

    if _is_sklearn_builtin_estimator(model):
        import io

        import joblib

        buf = io.BytesIO()
        try:
            joblib.dump(model, buf)
        except Exception as exc:
            warnings.warn(
                f"{name} (seed {seed}) could not be serialized: {exc}. Skipping save.",
                UserWarning,
                stacklevel=2,
            )
            return None

        path = _append_suffix(path_base, ".joblib")
        path.write_bytes(buf.getvalue())
        return path

    warnings.warn(
        f"{name} could not be saved for seed {seed} "
        "(not a kit quantum estimator or built-in sklearn model).",
        UserWarning,
        stacklevel=2,
    )
    return None


def _evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: list[Metric],
    seed: int,
    save_dir: Path | None = None,
    save_name: str | None = None,
    store_predictions: bool = False,
) -> dict[str, Any]:
    """Fit, predict, and compute metrics for one model / seed.

    Returns:
        Dict containing metric rows, the fitted clone, optional predictions,
        and optional saved artifact path.
    """
    model_name = save_name if save_name is not None else _get_model_name(model)
    task = _task_type(model)
    cloned = _clone_model(model, seed)
    artifact_path: Path | None = None

    with timer() as t_fit:
        _fit_model(cloned, X_train, y_train, X_val, y_val)

    if save_dir is not None:
        artifact_path = _try_save_model(cloned, save_dir, seed, save_name=save_name)

    with timer() as t_predict:
        y_pred = cloned.predict(X_test)

    y_prob = None
    needs_proba = any(isinstance(m, str) and m == "roc_auc" for m in metrics)
    if task == "classification" and (needs_proba or store_predictions):
        y_prob = _maybe_predict_proba(cloned, X_test)

    values = compute_metrics(y_test, y_pred, y_prob, metrics, task=task)
    if "fit_time" in metrics:
        values["fit_time"] = t_fit["elapsed"]
    if "predict_time" in metrics:
        values["predict_time"] = t_predict["elapsed"]

    rows = [
        {"model": model_name, "seed": seed, "metric": m, "value": v}
        for m, v in values.items()
    ]

    predictions = None
    if store_predictions:
        predictions = {"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob}

    return {
        "rows": rows,
        "fitted_model": cloned,
        "predictions": predictions,
        "loss_curve": _loss_curve_for(cloned),
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
    }
