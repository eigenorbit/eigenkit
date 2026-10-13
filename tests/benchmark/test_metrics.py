import time

import numpy as np
import pytest

from eigenkit.benchmark._metrics import (
    collect_metric_directions,
    compute_metrics,
    custom_metric,
    timer,
)


def test_timer_measures_elapsed():
    with timer() as t:
        time.sleep(0.01)
    assert t["elapsed"] >= 0.01


def test_compute_classification_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    values = compute_metrics(
        y_true, y_pred, None, ["accuracy", "f1"], task="classification"
    )
    assert set(values) == {"accuracy", "f1"}
    assert 0.0 <= values["accuracy"] <= 1.0


def test_compute_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    values = compute_metrics(
        y_true, y_pred, None, ["r2", "mae", "rmse", "mse"], task="regression"
    )
    assert set(values) == {"r2", "mae", "rmse", "mse"}
    assert values["mae"] > 0


def test_roc_auc_without_proba_warns_and_skips():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    with pytest.warns(UserWarning, match="predict_proba"):
        values = compute_metrics(
            y_true, y_pred, None, ["accuracy", "roc_auc"], task="classification"
        )
    assert "roc_auc" not in values
    assert "accuracy" in values


def test_roc_auc_with_proba_binary():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])
    values = compute_metrics(y_true, y_pred, y_prob, ["roc_auc"], task="classification")
    assert "roc_auc" in values
    assert 0.0 <= values["roc_auc"] <= 1.0


def test_roc_auc_with_proba_multiclass():
    rng = np.random.default_rng(42)
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    y_prob = rng.dirichlet(alpha=[1, 1, 1], size=6)
    values = compute_metrics(y_true, y_pred, y_prob, ["roc_auc"], task="classification")
    assert "roc_auc" in values
    assert 0.0 <= values["roc_auc"] <= 1.0


def test_timing_metrics_ignored_by_compute_metrics():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    values = compute_metrics(
        y_true,
        y_pred,
        None,
        ["fit_time", "predict_time", "accuracy"],
        task="classification",
    )
    assert "fit_time" not in values
    assert "predict_time" not in values
    assert "accuracy" in values


def test_compute_metrics_accepts_custom_metric_wrapper():
    def exact_match_rate(y_true, y_pred):
        return np.mean(y_true == y_pred)

    metric = custom_metric(exact_match_rate, direction="higher")

    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 0, 1])
    values = compute_metrics(y_true, y_pred, None, [metric], task="classification")
    assert values == {"exact_match_rate": pytest.approx(2 / 3)}


def test_custom_metric_registers_direction_and_name():
    def penalty(y_true, y_pred):
        return np.mean(y_true != y_pred)

    metric = custom_metric(penalty, direction="lower", name="error_rate")

    assert collect_metric_directions([metric]) == {"error_rate": "lower"}
    values = compute_metrics(
        np.array([0, 1, 1]),
        np.array([0, 0, 1]),
        None,
        [metric],
        task="classification",
    )
    assert values == {"error_rate": pytest.approx(1 / 3)}


def test_bare_callable_metric_requires_direction():
    def loss(y_true, y_pred):
        return np.mean(y_true != y_pred)

    with pytest.raises(ValueError, match="custom_metric"):
        collect_metric_directions([loss])


def test_custom_metric_rejects_invalid_direction():
    def score(y_true, y_pred):
        return 1.0

    with pytest.raises(ValueError, match="direction"):
        custom_metric(score, direction="sideways")


def test_custom_metric_rejects_bad_name():
    def score(y_true, y_pred):
        return 1.0

    with pytest.raises(TypeError, match="name"):
        custom_metric(score, direction="higher", name=123)
    with pytest.raises(ValueError, match="non-empty"):
        custom_metric(score, direction="higher", name="")
    with pytest.raises(ValueError, match="non-empty"):
        custom_metric(score, direction="higher", name=" ")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        custom_metric(score, direction="higher", name=" score")


def test_duplicate_metric_names_raise_value_error():
    def score(y_true, y_pred):
        return 0.1

    def score2(y_true, y_pred):
        return 0.9

    score2.__name__ = "score"

    with pytest.raises(ValueError, match="Duplicate metric name"):
        compute_metrics(
            np.array([0, 1]),
            np.array([0, 1]),
            None,
            [
                custom_metric(score, direction="higher"),
                custom_metric(score2, direction="higher"),
            ],
            task="classification",
        )


def test_metric_specs_reject_invalid_objects_and_unknown_strings():
    with pytest.raises(TypeError, match="metrics must be"):
        collect_metric_directions([object()])

    with pytest.raises(ValueError, match="not a recognised metric"):
        compute_metrics(
            np.array([0, 1]),
            np.array([0, 1]),
            None,
            ["not_a_metric"],
            task="classification",
        )
