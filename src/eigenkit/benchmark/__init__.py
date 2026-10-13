"""Comparison of quantum and classical estimators.

The [`compare()`][eigenkit.benchmark.compare] function trains models that follow
the scikit-learn interface on a dataset with several random seeds. It returns a
[`BenchmarkResult`][eigenkit.benchmark.BenchmarkResult] with the aggregated
metrics.

Example:
    ```python
    from sklearn.svm import SVC
    from eigenkit.qml import QSVC
    from eigenkit.benchmark import compare

    results = compare(
        models={
            "Quantum SVC": QSVC(
                feature_map="Angle", reps=1, probability=True
            ),
            "Regularized SVC": SVC(C=0.3),
            "Flexible SVC": SVC(C=3.0),
        },
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=1,
    )
    print(list(results.fitted_models))
    # ['Quantum SVC', 'Regularized SVC', 'Flexible SVC']
    ```
"""

from __future__ import annotations

import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ._baselines import quick_baselines
from ._io import load_checkpoint
from ._metrics import (
    DEFAULT_CLASSIFICATION_METRICS,
    DEFAULT_REGRESSION_METRICS,
    Metric,
    collect_metric_directions,
    custom_metric,
    timer,
)
from ._results import BenchmarkResult
from ._runner import (
    _evaluate_model,
    _task_type,
    _validate_benchmark_models,
)


__all__ = [
    "compare",
    "BenchmarkResult",
    "quick_baselines",
    "load_checkpoint",
    "custom_metric",
    "timer",
]


_UNSAFE_MODEL_NAME_CHARS = frozenset('<>:"/\\|?*')
_MAX_MODEL_NAME_FILENAME_BYTES = 200


def _validate_model_name(name: object, *, for_filename: bool = False) -> str:
    if not isinstance(name, str):
        raise TypeError("model names (dict keys) must be strings.")
    if not name.strip():
        raise ValueError("model names must be non-empty strings.")
    if any(ord(ch) < 32 for ch in name):
        raise ValueError("model names must not contain control characters.")
    if name != name.strip():
        raise ValueError("model names must not have surrounding whitespace.")
    if not for_filename:
        return name

    if any(ch in _UNSAFE_MODEL_NAME_CHARS for ch in name):
        raise ValueError(
            "model names used with save_dir must not contain reserved filename "
            "characters."
        )
    if name.endswith("."):
        raise ValueError("model names used with save_dir must not end with a period.")
    if len(name.encode("utf-8")) > _MAX_MODEL_NAME_FILENAME_BYTES:
        raise ValueError(
            "model names used with save_dir must be "
            f"{_MAX_MODEL_NAME_FILENAME_BYTES} UTF-8 bytes or fewer."
        )
    return name


def _default_metrics(models: list[Any]) -> list[str]:
    if not models:
        return list(DEFAULT_CLASSIFICATION_METRICS)
    return (
        list(DEFAULT_REGRESSION_METRICS)
        if _task_type(models[0]) == "regression"
        else list(DEFAULT_CLASSIFICATION_METRICS)
    )


def _load_split(dataset, seed: int, test_size: float, validation_size: float) -> tuple:
    if isinstance(dataset, str):
        from ..datasets import load_dataset

        data = load_dataset(
            dataset,
            # `seed` here is the benchmark's run index; the dataset API calls
            # the same thing `random_state`.
            random_state=seed,
            test_size=test_size,
            validation_size=validation_size,
        )
        X_train = data["X_train"]
        # Keyed on the shape, not on `metadata.task`. Some time-series
        # datasets do hand back the time axis as a 1-D `X` -- windowing into a
        # feature matrix is the algorithm layer's job, and the raw series comes
        # back under "sequence" -- and passing that straight to an estimator
        # produces sklearn's "Expected 2D array, got 1D array instead", which
        # says nothing about what the caller should do. But the label does not
        # imply the shape: `beijing_pm25` is registered `task="timeseries"`
        # and its loader returns a (41757, 13) feature matrix, so keying on
        # the label would refuse a dataset that compares fine, with a message
        # that is untrue about it.
        if np.ndim(X_train) == 1:
            raise ValueError(
                f"{dataset!r} returns a 1-D X (the raw series / time axis), "
                f"not a feature matrix, so it cannot be compared directly. "
                f"Window it into (n_samples, n_features) first, or pass a "
                f"dataset dict with your own X_train/X_test."
            )
        return (
            X_train,
            data["y_train"],
            data.get("X_val"),
            data.get("y_val"),
            data["X_test"],
            data["y_test"],
        )

    if not isinstance(dataset, dict):
        raise TypeError(
            "dataset must be a string (registry name) or a dict with "
            "X_train/y_train/X_test/y_test keys."
        )

    missing = {"X_train", "y_train", "X_test", "y_test"} - set(dataset)
    if missing:
        raise ValueError(f"dataset dict missing keys: {sorted(missing)}")
    return (
        dataset["X_train"],
        dataset["y_train"],
        dataset.get("X_val"),
        dataset.get("y_val"),
        dataset["X_test"],
        dataset["y_test"],
    )


def compare(
    models: list | dict[str, Any],
    dataset: str | dict = "breast_cancer_wisconsin",
    metrics: list[Metric] | None = None,
    n_seeds: int = 5,
    test_size: float = 0.15,
    show_progress: bool = False,
    save_dir: str | Path | None = None,
    store_predictions: bool = False,
    validation_size: float = 0.15,
) -> BenchmarkResult:
    """Compare eigenkit models and built-in scikit-learn models on a dataset.

    Each model must be a built-in scikit-learn estimator (from the `sklearn`
    package), or a subclass of
    [`BaseQuantumEstimator`][eigenkit.torch.estimators.BaseQuantumEstimator],
    [`BaseQuantumClassifier`][eigenkit.torch.estimators.BaseQuantumClassifier] or
    [`BaseQuantumRegressor`][eigenkit.torch.estimators.BaseQuantumRegressor]. For
    every seed, each model is copied with `sklearn.base.clone()`.

    Args:
        models (list | dict[str, Any]): The estimator instances to compare. If a
            list is passed, the display names come from `type(model).__name__`, and
            repeated names are numbered (for example, `SVC_0` and `SVC_1`). If a
            dict is passed, its keys are used as display names. Dict keys must be
            non-empty strings, with no whitespace at the start or the end and no
            control characters. Use a different key for each model. If the same key
            appears twice in a Python dict literal, Python keeps only the last
            model before `compare()` receives the dict. When `save_dir` is set, the
            keys must also be valid prefixes for file names.
        dataset (str | dict): A dataset name, or a dict that is already split, with
            `X_train`, `y_train`, `X_test` and `y_test`. Such a dict can also
            include `X_val` and `y_val`.
        metrics (list | None): The metrics to compute. For classification, the
            options are `accuracy`, `f1`, `precision`, `recall`, `roc_auc`,
            `fit_time` and `predict_time`. For regression, they are `r2`, `mae`,
            `rmse`, `mse`, `fit_time` and `predict_time`. A custom function must be
            wrapped with `custom_metric(...)`, so that its direction (higher or
            lower is better) is explicit. The default depends on the type of task
            of the first model.
        n_seeds (int): The number of seeds to average over. The default is `5`.
        test_size (float): The fraction of the data kept for the final test set.
        show_progress (bool): Whether to print the progress of the benchmark, with
            the current model and an estimate of the remaining time. The default is
            `False`, which runs silently.
        save_dir (str | Path | None): If set, each fitted model is saved in this
            directory, as `{ModelName}_seed{N}.eigenkit` with `save()`, or as
            `{ModelName}_seed{N}.joblib` with joblib for scikit-learn estimators.
        store_predictions (bool): If `True`, the result keeps `y_true`, `y_pred`
            and `y_prob` for the last seed of each model.
        validation_size (float): The fraction of the data kept for validation.

    Returns:
        An object with the results, and methods such as `table()`, `plot()` and
            `to_dataframe()`.

    Raises:
        ValueError: If `models` is empty, if a model name is not valid, or if
            `dataset` is not well formed.
        TypeError: If a model name is not a string, or if a model is not a
            supported type of estimator.
    """
    names_from_dict = isinstance(models, dict)
    if isinstance(models, dict):
        save_names = [
            _validate_model_name(name, for_filename=save_dir is not None)
            for name in models
        ]
        if len(save_names) != len(set(save_names)):
            raise ValueError("model names (dict keys) must be unique.")
        models = list(models.values())
    else:
        save_names = []

    if not models:
        raise ValueError("models must be a non-empty list or dict.")
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1.")
    if isinstance(dataset, str):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not 0 <= validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if test_size + validation_size >= 1:
            raise ValueError("test_size + validation_size must be < 1.")

    if not names_from_dict:
        # Assign unique save names: append _0, _1 only when class names clash.
        _class_counts: Counter = Counter(type(m).__name__ for m in models)
        _class_seen: dict[str, int] = {}
        for m in models:
            cls_name = type(m).__name__
            if _class_counts[cls_name] > 1:
                idx = _class_seen.get(cls_name, 0)
                _class_seen[cls_name] = idx + 1
                save_names.append(f"{cls_name}_{idx}")
            else:
                save_names.append(cls_name)

    _validate_benchmark_models(models, save_names if names_from_dict else None)

    if metrics is None:
        metrics = _default_metrics(models)
    else:
        metrics = list(metrics)
    metric_directions = collect_metric_directions(metrics)

    if "roc_auc" in metrics:
        no_proba = [
            name
            for m, name in zip(models, save_names)
            if getattr(m, "probability", None) is False
            or not callable(getattr(m, "predict_proba", None))
        ]
        if no_proba:
            warnings.warn(
                f"roc_auc requires predict_proba, but the following models do not "
                f"support it and will produce no roc_auc results: {no_proba}. "
                "Set probability=True or choose a different metric.",
                UserWarning,
                stacklevel=2,
            )

    total_steps = n_seeds * len(models)
    steps_done = 0
    start_ts = time.perf_counter()
    rows: list[dict] = []
    save_path = Path(save_dir) if save_dir is not None else None
    fitted_models: dict[str, list[Any]] = {name: [] for name in save_names}
    predictions: dict[str, dict[str, Any]] = {}
    loss_curves: dict[str, list[dict[str, list[float]] | None]] = {
        name: [] for name in save_names
    }
    model_artifacts: dict[str, list[str | None]] = {name: [] for name in save_names}

    for seed in range(n_seeds):
        X_train, y_train, X_val, y_val, X_test, y_test = _load_split(
            dataset, seed, test_size, validation_size
        )
        for model, save_name in zip(models, save_names):
            steps_done += 1
            if show_progress:
                elapsed = time.perf_counter() - start_ts
                avg_step = elapsed / steps_done if steps_done else 0.0
                remaining_steps = total_steps - steps_done
                eta_seconds = remaining_steps * avg_step
                print(
                    f"[benchmark] {steps_done}/{total_steps} | "
                    f"seed={seed + 1}/{n_seeds} | "
                    f"model={save_name} | "
                    f"remaining={remaining_steps} | "
                    f"eta~{eta_seconds:.1f}s"
                )
            try:
                result = _evaluate_model(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    metrics,
                    seed,
                    save_dir=save_path,
                    save_name=save_name,
                    store_predictions=store_predictions,
                )
            except Exception as exc:  # noqa: BLE001 — one model must not sink the run
                # A comparison is a set of independent runs. Letting one
                # model's failure propagate would discard every result already
                # computed, including the other models' -- so a long sweep would
                # lose everything to a single bad fit. The failure is reported and
                # that model simply has no row for this seed; `_summarize`
                # already handles a model with fewer seeds than the others,
                # because it reads `raw`, which carries an explicit "seed" key.
                warnings.warn(
                    f"{save_name} failed on seed {seed} "
                    f"({type(exc).__name__}: {exc}); it is skipped for this "
                    f"seed and the remaining models continue.",
                    UserWarning,
                    stacklevel=2,
                )
                # These three are positional, though: `best_model(seed=n)` and
                # `inference(seed=n)` document `n` as the n-th seed, and index
                # straight into them. Skipping the appends would slide every
                # later seed down one, so a model that failed on seed 0 would
                # answer `seed=0` with seed 1's estimator -- in silence, and
                # mislabel its artifact and loss curve the same way. Hold the slot.
                fitted_models[save_name].append(None)
                model_artifacts[save_name].append(None)
                loss_curves[save_name].append(None)
                continue
            rows.extend(result["rows"])
            fitted_models[save_name].append(result["fitted_model"])
            model_artifacts[save_name].append(result["artifact_path"])
            loss_curves[save_name].append(result["loss_curve"])
            if store_predictions and result["predictions"] is not None:
                predictions[save_name] = result["predictions"]

    return BenchmarkResult(
        raw=rows,
        fitted_models=fitted_models,
        predictions=predictions,
        model_artifacts=model_artifacts,
        loss_curves=loss_curves,
        metric_directions=metric_directions,
    )
