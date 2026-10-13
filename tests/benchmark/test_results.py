import json
import warnings

import matplotlib
import numpy as np
import pytest

from eigenkit.benchmark import BenchmarkResult


matplotlib.use("Agg")


def _sample_raw():
    return [
        {"model": "QSVC", "seed": 0, "metric": "accuracy", "value": 0.80},
        {"model": "QSVC", "seed": 1, "metric": "accuracy", "value": 0.90},
        {"model": "SVC", "seed": 0, "metric": "accuracy", "value": 0.85},
        {"model": "SVC", "seed": 1, "metric": "accuracy", "value": 0.95},
    ]


def test_table_contains_model_names():
    result = BenchmarkResult(raw=_sample_raw())
    out = result.table()
    assert "QSVC" in out
    assert "SVC" in out
    assert "Accuracy" in out


def test_table_empty_result():
    assert "no results" in BenchmarkResult().table()


def test_summarize_mean_std():
    result = BenchmarkResult(raw=_sample_raw())
    summary = result._summarize()
    mean, std = summary["QSVC"]["accuracy"]
    assert abs(mean - 0.85) < 1e-9
    assert std > 0


def test_plot_does_not_crash(tmp_path):
    result = BenchmarkResult(raw=_sample_raw())
    result.plot(save_path=str(tmp_path / "out.png"))
    assert (tmp_path / "out.png").exists()


def test_to_dataframe():
    pd = __import__("pandas")
    result = BenchmarkResult(raw=_sample_raw())
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"model", "seed", "metric", "value"}
    assert len(df) == 4


def test_result_save_load_json_roundtrip(tmp_path):
    result = BenchmarkResult(
        raw=_sample_raw()
        + [{"model": "QSVC", "seed": 0, "metric": "custom_loss", "value": 0.1}],
        predictions={
            "QSVC": {
                "y_true": np.array([0, 1, 1]),
                "y_pred": np.array([0, 1, 0]),
                "y_prob": None,
            }
        },
        model_artifacts={"QSVC": ["model.eigenkit"]},
        loss_curves={"QSVC": [{"train": [1.0, 0.5], "validation": [1.2, 0.7]}]},
        metric_directions={"accuracy": "higher", "custom_loss": "lower"},
    )

    path = result.save(tmp_path / "bench_result")
    loaded = BenchmarkResult.load(path)

    assert path.name == "bench_result.json"
    assert loaded.raw == result.raw
    assert loaded.fitted_models == {}
    assert loaded.model_artifacts == {"QSVC": ["model.eigenkit"]}
    assert loaded.loss_curves == {
        "QSVC": [{"train": [1.0, 0.5], "validation": [1.2, 0.7]}]
    }
    assert loaded.metric_directions == {"accuracy": "higher", "custom_loss": "lower"}
    np.testing.assert_array_equal(
        loaded.predictions["QSVC"]["y_pred"], np.array([0, 1, 0])
    )


def test_result_preserves_metric_directions_positional_argument():
    result = BenchmarkResult([], {}, {}, {}, {"custom_loss": "lower"})

    assert result.metric_directions == {"custom_loss": "lower"}
    assert result.loss_curves == {}


def test_result_rejects_invalid_metric_direction():
    with pytest.raises(ValueError, match="direction"):
        BenchmarkResult(metric_directions={"custom_loss": "smaller"})


def test_plot_history_writes_file(tmp_path):
    result = BenchmarkResult(
        loss_curves={"HQNC": [{"train": [1.0, 0.5, 0.25], "validation": [1.1, 0.7, 0.4]}]}
    )

    result.plot_history(save_path=str(tmp_path / "losses.png"))

    assert (tmp_path / "losses.png").exists()


def test_plot_history_warns_about_models_without_history(tmp_path):
    result = BenchmarkResult(
        loss_curves={
            "HQNC": [{"train": [1.0, 0.5], "validation": [1.1, 0.6]}],
            "SVC": [None],
        }
    )

    with pytest.warns(UserWarning, match="SVC"):
        result.plot_history(save_path=str(tmp_path / "history.png"))

    assert (tmp_path / "history.png").exists()


def test_plot_history_accepts_integer_seed(tmp_path):
    result = BenchmarkResult(
        loss_curves={
            "HQNC": [
                {"train": [1.0, 0.5], "validation": [1.1, 0.6]},
                {"train": [0.9, 0.4], "validation": [1.0, 0.5]},
            ]
        }
    )

    result.plot_history(save_path=str(tmp_path / "history_seed0.png"), seed=0)

    assert (tmp_path / "history_seed0.png").exists()


def test_plot_history_rejects_invalid_seed_before_plotting():
    result = BenchmarkResult(
        loss_curves={"HQNC": [{"train": [1.0, 0.5], "validation": [1.1, 0.6]}]}
    )

    with pytest.raises(TypeError, match="seed"):
        result.plot_history(seed="bad")


def test_plot_history_closes_empty_figure_on_error():
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    result = BenchmarkResult(loss_curves={"SVC": [None]})

    with pytest.warns(UserWarning, match="SVC"):
        with pytest.raises(ValueError, match="No train or validation loss curves"):
            result.plot_history()

    assert set(plt.get_fignums()) == before


def test_result_load_rejects_invalid_metric_direction(tmp_path):
    path = tmp_path / "bad_result.json"
    path.write_text(
        json.dumps(
            {
                "raw": [],
                "predictions": {},
                "model_artifacts": {},
                "metric_directions": {"custom_loss": "smaller"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="direction"):
        BenchmarkResult.load(path)


def test_significance_test_returns_pairwise_matrix():
    raw = []
    for seed in range(6):
        raw.append({"model": "A", "seed": seed, "metric": "accuracy", "value": 1.0})
        raw.append({"model": "B", "seed": seed, "metric": "accuracy", "value": 0.0})
    result = BenchmarkResult(raw=raw)

    matrix = result.significance_test("accuracy")

    assert list(matrix.index) == ["A", "B"]
    assert matrix.loc["A", "A"] == 1.0
    assert matrix.loc["A", "B"] == pytest.approx(matrix.loc["B", "A"])
    assert matrix.loc["A", "B"] < 0.05


def test_table_can_show_significance_markers():
    raw = []
    for seed in range(6):
        raw.append({"model": "A", "seed": seed, "metric": "accuracy", "value": 1.0})
        raw.append({"model": "B", "seed": seed, "metric": "accuracy", "value": 0.0})
    result = BenchmarkResult(raw=raw)

    assert "*" in result.table(show_significance=True)


def test_significance_test_requires_two_models():
    result = BenchmarkResult(
        raw=[{"model": "A", "seed": 0, "metric": "accuracy", "value": 1.0}]
    )

    with pytest.raises(ValueError, match="at least two models"):
        result.significance_test("accuracy")


def test_confusion_matrix_requires_stored_predictions():
    result = BenchmarkResult()
    with pytest.raises(ValueError, match="No predictions"):
        result.confusion_matrix("SVC")


def _regression_raw():
    return [
        {"model": "ModelA", "seed": 0, "metric": "mae", "value": 0.5},
        {"model": "ModelA", "seed": 0, "metric": "r2", "value": 0.7},
        {"model": "ModelB", "seed": 0, "metric": "mae", "value": 0.2},
        {"model": "ModelB", "seed": 0, "metric": "r2", "value": 0.6},
    ]


def _make_mock_estimator():
    class _Mock:
        def predict(self, X):
            return X[:, 0]

    return _Mock()


def test_best_model_lower_is_better_uses_min():
    mock_a = _make_mock_estimator()
    mock_b = _make_mock_estimator()
    result = BenchmarkResult(
        raw=_regression_raw(),
        fitted_models={"ModelA": [mock_a], "ModelB": [mock_b]},
    )
    # MAE: lower is better — ModelB (0.2) should win over ModelA (0.5)
    best = result.best_model(metric="mae")
    assert best is mock_b


def test_best_model_higher_is_better_uses_max():
    mock_a = _make_mock_estimator()
    mock_b = _make_mock_estimator()
    result = BenchmarkResult(
        raw=_regression_raw(),
        fitted_models={"ModelA": [mock_a], "ModelB": [mock_b]},
    )
    # R²: higher is better — ModelA (0.7) should win over ModelB (0.6)
    best = result.best_model(metric="r2")
    assert best is mock_a


def test_best_model_auto_selects_primary_metric():
    mock_a = _make_mock_estimator()
    mock_b = _make_mock_estimator()
    result = BenchmarkResult(
        raw=_regression_raw(),
        fitted_models={"ModelA": [mock_a], "ModelB": [mock_b]},
    )
    # Primary metric (first non-timing): mae — lower is better, ModelB wins
    best = result.best_model()
    assert best is mock_b


# ── inference() predict_proba warnings ───────────────────────────────────────


def _make_result_with(estimator, name="Model"):
    """BenchmarkResult with a single fitted estimator, no raw scores needed."""
    return BenchmarkResult(
        raw=[{"model": name, "seed": 0, "metric": "accuracy", "value": 1.0}],
        fitted_models={name: [estimator]},
    )


def test_inference_warns_on_not_implemented_error():
    class _Estimator:
        def predict(self, X):
            return X[:, 0]

        def predict_proba(self, X):
            raise NotImplementedError("not supported")

    result = _make_result_with(_Estimator())
    X = np.zeros((3, 2))
    with pytest.warns(UserWarning, match="predict_proba failed"):
        out = result.inference(X)
    assert out["Model"]["probabilities"] is None


def test_inference_warns_on_attribute_error():
    class _Estimator:
        def predict(self, X):
            return X[:, 0]

        def predict_proba(self, X):
            raise AttributeError("broken internals")

    result = _make_result_with(_Estimator())
    X = np.zeros((3, 2))
    with pytest.warns(UserWarning, match="predict_proba failed"):
        out = result.inference(X)
    assert out["Model"]["probabilities"] is None


def test_inference_warns_on_value_error():
    class _Estimator:
        def predict(self, X):
            return X[:, 0]

        def predict_proba(self, X):
            raise ValueError("incompatible input shape")

    result = _make_result_with(_Estimator())
    X = np.zeros((3, 2))
    with pytest.warns(UserWarning, match="predict_proba failed"):
        out = result.inference(X)
    assert out["Model"]["probabilities"] is None


def test_inference_warning_contains_model_name_and_exception():
    class _Estimator:
        def predict(self, X):
            return X[:, 0]

        def predict_proba(self, X):
            raise ValueError("bad shape")

    result = _make_result_with(_Estimator(), name="MyModel")
    X = np.zeros((3, 2))
    with pytest.warns(UserWarning, match="MyModel") as record:
        result.inference(X)
    assert "ValueError" in str(record[0].message)
    assert "bad shape" in str(record[0].message)


def test_inference_no_warning_when_predict_proba_succeeds():
    class _Estimator:
        def predict(self, X):
            return X[:, 0]

        def predict_proba(self, X):
            return np.ones((len(X), 2)) * 0.5

    result = _make_result_with(_Estimator())
    X = np.zeros((3, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = result.inference(X)
    assert out["Model"]["probabilities"].shape == (3, 2)


def test_inference_no_warning_without_predict_proba():
    result = _make_result_with(_make_mock_estimator())
    X = np.zeros((3, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = result.inference(X)
    assert out["Model"]["probabilities"] is None


class TestSignificancePowerIsPerComparison:
    """The floor is real -- an exact two-sided Wilcoxon over n pairs cannot go
    below `2 ** (1 - n)`, so 5 pairs bottom out at 0.0625 -- but it applies to
    each comparison, not to the run.

    Taking `max` over every model's seed count would ask the wrong question:
    seed counts differ whenever a model failed on a seed, so a run could hold
    one 6-seed model and one 3-seed model, warn about nothing, and still leave
    the 3-seed pair unmarkable -- read by the reader as "no significant
    difference".
    """

    @staticmethod
    def _result(**per_model):
        raw = [
            {"model": name, "metric": "accuracy", "value": value, "seed": seed}
            for name, values in per_model.items()
            for seed, value in enumerate(values)
        ]
        return BenchmarkResult(raw=raw)

    def test_a_lopsided_run_is_not_excused_by_its_longest_model(self):
        result = self._result(A=[0.9] * 6, B=[0.60, 0.61, 0.62])
        with pytest.warns(UserWarning, match="best-supported comparison"):
            assert result._significance_annotations("accuracy") == {}

    def test_an_underpowered_pair_is_named_when_others_are_powered(self):
        result = self._result(
            A=[0.90 + 0.001 * i for i in range(8)],
            B=[0.60 + 0.001 * i for i in range(8)],
            C=[0.55, 0.56, 0.57],
        )
        with pytest.warns(UserWarning, match=r"C \(3\)"):
            annotations = result._significance_annotations("accuracy")
        # The powered comparison still earns its marker.
        assert annotations == {"A": "**"}

    def test_a_fully_powered_run_says_nothing(self):
        result = self._result(
            A=[0.90 + 0.001 * i for i in range(6)],
            B=[0.60 + 0.001 * i for i in range(6)],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Six pairs bottom out at exactly 0.03125: significant at 0.05,
            # so "*", and not reachable at 0.01, so never "**".
            assert result._significance_annotations("accuracy") == {"A": "*"}

    def test_the_pair_count_is_the_intersection_of_seeds(self):
        """Seeds 0-5 for A, seeds 3-8 for B: six each, three in common."""
        raw = [
            {"model": "A", "metric": "accuracy", "value": 0.9, "seed": s}
            for s in range(6)
        ] + [
            {"model": "B", "metric": "accuracy", "value": 0.6, "seed": s}
            for s in range(3, 9)
        ]
        result = BenchmarkResult(raw=raw)
        assert result._significance_pair_counts("accuracy", "A") == {"B": 3}
