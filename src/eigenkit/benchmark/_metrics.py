"""Metric computation and timing utilities for the benchmark module."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


try:
    from sklearn.metrics import root_mean_squared_error as _rmse
except ImportError:  # sklearn < 1.4

    def _rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


def _roc_auc(y_true, y_pred, y_prob):
    if y_prob is None:
        warnings.warn(
            "roc_auc requested but model has no predict_proba; skipping for this model.",
            UserWarning,
            stacklevel=2,
        )
        return None
    classes = np.unique(y_true)
    if len(classes) == 2 and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    return roc_auc_score(y_true, y_prob, multi_class="ovr")


CLASSIFICATION_METRICS = {
    "accuracy": lambda y_true, y_pred, y_prob: accuracy_score(y_true, y_pred),
    "f1": lambda y_true, y_pred, y_prob: f1_score(y_true, y_pred, average="weighted"),
    "precision": lambda y_true, y_pred, y_prob: precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "recall": lambda y_true, y_pred, y_prob: recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "roc_auc": _roc_auc,
}

REGRESSION_METRICS = {
    "r2": lambda y_true, y_pred, y_prob: r2_score(y_true, y_pred),
    "mae": lambda y_true, y_pred, y_prob: mean_absolute_error(y_true, y_pred),
    "rmse": lambda y_true, y_pred, y_prob: _rmse(y_true, y_pred),
    "mse": lambda y_true, y_pred, y_prob: mean_squared_error(y_true, y_pred),
}

DEFAULT_CLASSIFICATION_METRICS = ["accuracy", "f1", "fit_time", "predict_time"]
DEFAULT_REGRESSION_METRICS = ["r2", "mae", "mse", "fit_time", "predict_time"]


MetricDirection = Literal["higher", "lower"]
_VALID_DIRECTIONS: frozenset[str] = frozenset({"higher", "lower"})

BUILTIN_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "accuracy": "higher",
    "f1": "higher",
    "precision": "higher",
    "recall": "higher",
    "roc_auc": "higher",
    "r2": "higher",
    "mae": "lower",
    "rmse": "lower",
    "mse": "lower",
    "fit_time": "lower",
    "predict_time": "lower",
    "loss": "lower",
    "mape": "lower",
}
KNOWN_METRICS: frozenset[str] = frozenset(
    set(CLASSIFICATION_METRICS) | set(REGRESSION_METRICS) | {"fit_time", "predict_time"}
)


@dataclass(frozen=True)
class CustomMetric:
    """Callable metric plus explicit optimization direction."""

    fn: Callable[[np.ndarray, np.ndarray], float]
    direction: MetricDirection
    name: str

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError("custom metric must be callable.")
        _validate_metric_name(self.name)
        _validate_metric_direction(self.direction)


Metric = str | CustomMetric


def custom_metric(
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    direction: MetricDirection,
    name: str | None = None,
) -> CustomMetric:
    """Define a custom metric and the direction in which it improves.

    Args:
        fn: The metric function. It receives `(y_true, y_pred)` and returns a
            number.
        direction: `"higher"` if higher values are better, or `"lower"` if lower
            values are better.
        name: The column label of the metric. The default is the name of `fn`.

    Returns:
        A metric specification to pass to `compare(metrics=[...])`.
    """
    metric_name_value = (
        name if name is not None else getattr(fn, "__name__", type(fn).__name__)
    )
    return CustomMetric(fn=fn, direction=direction, name=metric_name_value)


def metric_name(metric: Metric) -> str:
    """Return the display/storage name for a metric spec."""
    if isinstance(metric, CustomMetric):
        return metric.name
    return metric


def _validate_metric_name(name: object) -> None:
    if not isinstance(name, str):
        raise TypeError("custom metric name must be a string.")
    if not name.strip():
        raise ValueError("custom metric name must be non-empty.")
    if name != name.strip():
        raise ValueError("custom metric name must not have surrounding whitespace.")


def _validate_metric_direction(direction: object) -> MetricDirection:
    if direction not in _VALID_DIRECTIONS:
        raise ValueError("direction must be 'higher' or 'lower'.")
    return cast(MetricDirection, direction)


def validate_metric_directions(
    metric_directions: dict[str, str],
) -> dict[str, MetricDirection]:
    """Validate a persisted metric direction map."""
    validated: dict[str, MetricDirection] = {}
    for name, direction in metric_directions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric direction names must be non-empty strings.")
        validated[name] = _validate_metric_direction(direction)
    return validated


def validate_metric_specs(metric_names: list[Metric]) -> None:
    """Validate benchmark metric specs before a compare() run."""
    seen: set[str] = set()
    for metric in metric_names:
        if callable(metric) and not isinstance(metric, CustomMetric):
            raise ValueError(
                "Custom metric callables must be wrapped with "
                "custom_metric(fn, direction='higher'|'lower')."
            )
        if not isinstance(metric, (str, CustomMetric)):
            raise TypeError("metrics must be strings or custom_metric(...) results.")

        name = metric_name(metric)
        if name in seen:
            raise ValueError(f"Duplicate metric name {name!r}.")
        seen.add(name)

        if isinstance(metric, str) and metric not in KNOWN_METRICS:
            if metric == "runtime":
                raise ValueError(
                    '"runtime" is not a valid metric; it was replaced by '
                    '"fit_time" and "predict_time".'
                )
            raise ValueError(
                f"{metric!r} is not a recognised metric. "
                f"Valid options: {sorted(KNOWN_METRICS)}."
            )


def collect_metric_directions(
    metric_names: list[Metric],
) -> dict[str, MetricDirection]:
    """Return per-metric best-value directions keyed by display name."""
    validate_metric_specs(metric_names)
    directions: dict[str, MetricDirection] = {}
    for metric in metric_names:
        name = metric_name(metric)
        if isinstance(metric, CustomMetric):
            directions[name] = metric.direction
        else:
            directions[name] = BUILTIN_METRIC_DIRECTIONS[name]
    return directions


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    metric_names: list[Metric],
    *,
    task: str = "classification",
) -> dict[str, float]:
    """Compute the requested metrics.

    Args:
        y_true: Ground truth labels / targets.
        y_pred: Predicted labels / values.
        y_prob: Predicted probabilities (classification only). May be ``None``.
        metric_names: Metrics to compute. Strings are resolved from the built-in
            registries; custom metrics are called as ``fn(y_true, y_pred)``.
            Time metrics (``"fit_time"``, ``"predict_time"``) are ignored here
            because timing is populated by the runner.
        task: ``"classification"`` or ``"regression"``.

    Returns:
        Mapping of metric name to computed value. Metrics that cannot be
        computed (e.g. ``roc_auc`` without probabilities) are omitted.
    """
    if task == "classification":
        registry = CLASSIFICATION_METRICS
    elif task == "regression":
        registry = REGRESSION_METRICS
    else:
        raise ValueError(f"Unknown task: {task!r}")

    validate_metric_specs(metric_names)

    values: dict[str, float] = {}
    for metric in metric_names:
        name = metric_name(metric)

        if isinstance(metric, CustomMetric):
            values[name] = float(metric.fn(y_true, y_pred))
            continue

        name = metric
        if name in ("fit_time", "predict_time"):
            continue
        fn = registry.get(name)
        if fn is None:
            continue
        result = fn(y_true, y_pred, y_prob)
        if result is not None:
            values[name] = float(result)
    return values


@contextmanager
def timer() -> Iterator[dict]:
    """Measure the wall-clock time of a block of code.

    Yields:
        A dict. When the `with` block ends, its `"elapsed"` key holds the elapsed
            time in seconds.

    Example:
        ```python
        with timer() as t:
            do_work()
        print(t["elapsed"])
        ```
    """
    holder: dict = {}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder["elapsed"] = time.perf_counter() - start
