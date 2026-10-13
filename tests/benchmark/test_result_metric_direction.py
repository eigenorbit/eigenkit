import pytest

from eigenkit.benchmark import BenchmarkResult


class _Predictor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value for _ in X]


def _raw_for(metric, good_values, bad_values):
    raw = []
    for seed, value in enumerate(good_values):
        raw.append({"model": "good", "seed": seed, "metric": metric, "value": value})
    for seed, value in enumerate(bad_values):
        raw.append({"model": "bad", "seed": seed, "metric": metric, "value": value})
    return raw


@pytest.mark.parametrize("metric", ["mae", "mse", "rmse", "fit_time", "predict_time"])
def test_best_model_minimizes_loss_and_runtime_metrics(metric):
    result = BenchmarkResult(
        raw=_raw_for(metric, [0.1, 0.2, 0.1], [1.0, 1.2, 1.1]),
        fitted_models={
            "good": [_Predictor("good")],
            "bad": [_Predictor("bad")],
        },
    )

    assert result.best_model(metric).value == "good"
    assert result.predict([1, 2], metric=metric) == ["good", "good"]


def test_best_model_still_maximizes_quality_metrics():
    result = BenchmarkResult(
        raw=_raw_for("accuracy", [0.9, 0.8, 0.9], [0.2, 0.1, 0.2]),
        fitted_models={
            "good": [_Predictor("good")],
            "bad": [_Predictor("bad")],
        },
    )

    assert result.best_model("accuracy").value == "good"


def test_best_model_uses_custom_metric_direction():
    result = BenchmarkResult(
        raw=_raw_for("custom_loss", [0.1, 0.2, 0.1], [1.0, 1.2, 1.1])
        + _raw_for("custom_score", [0.1, 0.2, 0.1], [1.0, 1.2, 1.1]),
        fitted_models={
            "good": [_Predictor("good")],
            "bad": [_Predictor("bad")],
        },
        metric_directions={"custom_loss": "lower", "custom_score": "higher"},
    )

    assert result.best_model("custom_loss").value == "good"
    assert result.best_model("custom_score").value == "bad"


def test_best_model_rejects_custom_metric_without_direction():
    result = BenchmarkResult(
        raw=_raw_for("custom_loss", [0.1, 0.2, 0.1], [1.0, 1.2, 1.1]),
        fitted_models={
            "good": [_Predictor("good")],
            "bad": [_Predictor("bad")],
        },
    )

    with pytest.raises(ValueError, match="No direction registered"):
        result.best_model("custom_loss")


def test_result_rejects_unknown_custom_metric_direction_name():
    with pytest.raises(ValueError, match="Unknown metric direction"):
        BenchmarkResult(
            raw=_raw_for("custom_loss", [0.1, 0.2, 0.1], [1.0, 1.2, 1.1]),
            metric_directions={"custom_los": "lower"},
        )


def test_significance_marker_uses_loss_metric_direction():
    result = BenchmarkResult(
        raw=_raw_for("mae", [0.1] * 6, [1.0] * 6),
    )

    table = result.table(show_significance=True)

    assert "0.100 ± 0.000*" in table
    assert "1.000 ± 0.000*" not in table
