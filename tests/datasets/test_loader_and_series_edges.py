"""Edge cases in the dataset layer that must not fail quietly.

* The CSV loader drops the first line only when it is a header; skipping it
  unconditionally would cost a headerless file its first sample -- no error,
  one row short.
* A single-column CSV is refused with a message, not a bare `IndexError` from
  inside numpy.
* `load_dataset("sine_wave")` returns the 1-D time axis as `X`, so
  `compare(dataset="sine_wave")` must say what to do rather than die with
  scikit-learn's "Expected 2D array".
* A series shorter than the inferred period is not called seasonal: a
  `seasonality_strength` saturated at 1.0 would produce a false "seasonality
  detected" recommendation on a monotone ramp and leak numpy
  mean-of-empty-slice warnings.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from eigenkit.datasets import load_dataset
from eigenkit.datasets._loaders import load_url_dataset
from eigenkit.datasets.eda import DatasetReport


class TestCsvLoading:
    @staticmethod
    def _write(tmp_path, name, text):
        path = tmp_path / name
        path.write_text(text)
        return str(path)

    def test_a_headerless_csv_keeps_every_row(self, tmp_path):
        path = self._write(tmp_path, "n.csv", "1,2,0\n3,4,1\n5,6,0\n7,8,1\n")
        result = load_url_dataset(path, test_size=0.25, random_state=0, split=False)
        assert len(result["y"]) == 4

    def test_a_csv_with_a_header_still_drops_it(self, tmp_path):
        path = self._write(tmp_path, "h.csv", "a,b,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n")
        result = load_url_dataset(path, test_size=0.25, random_state=0, split=False)
        assert len(result["y"]) == 4
        assert result["X"].shape == (4, 2)

    def test_a_single_column_csv_says_what_is_wrong(self, tmp_path):
        path = self._write(tmp_path, "o.csv", "a\n1\n2\n")
        with pytest.raises(ValueError, match="at least one feature column"):
            load_url_dataset(path, test_size=0.25, random_state=0, split=False)


class TestTimeSeriesDatasetsExplainThemselves:
    """`X` is the 1-D time axis for some of these, by design -- windowing
    belongs to the algorithm layer, and the raw series comes back under
    "sequence" too. Handing that to an estimator would give sklearn's
    "Expected 2D array", which tells the caller nothing about what to do.

    The refusal is keyed on the shape, not on `metadata.task`: the label does
    not imply a 1-D `X`, and keying on it would refuse `beijing_pm25` -- a
    13-column feature matrix -- with a message that is untrue about it.
    """

    def test_x_is_the_one_dimensional_time_axis(self):
        assert np.asarray(load_dataset("sine_wave")["X_train"]).ndim == 1

    def test_compare_says_what_is_wrong_and_what_to_do(self):
        from sklearn.linear_model import Ridge

        from eigenkit.benchmark import compare

        with pytest.raises(ValueError, match="returns a 1-D X"):
            compare(models={"r": Ridge()}, dataset="sine_wave", n_seeds=1)

    def test_the_label_does_not_imply_the_shape(self):
        """`beijing_pm25` is registered `task="timeseries"` and its loader
        returns `arr[:, :-1]` -- thirteen named feature columns."""
        from eigenkit.datasets._registry import REGISTRY

        info = REGISTRY["beijing_pm25"]
        assert info.task == "timeseries"
        assert info.n_features == 13
        assert len(info.feature_names) == 13

    def test_a_two_dimensional_x_is_compared_even_when_labelled_timeseries(
        self, monkeypatch
    ):
        """The `beijing_pm25` shape, without the download."""
        from sklearn.linear_model import Ridge

        import eigenkit.datasets as datasets_module
        from eigenkit.benchmark import compare
        from eigenkit.datasets._registry import REGISTRY

        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 13))
        y = X[:, 0] + 0.1 * rng.normal(size=60)

        def fake_load_dataset(name, **kwargs):
            assert name == "beijing_pm25"
            return {
                "X_train": X[:45],
                "y_train": y[:45],
                "X_test": X[45:],
                "y_test": y[45:],
                "metadata": REGISTRY["beijing_pm25"],
            }

        monkeypatch.setattr(datasets_module, "load_dataset", fake_load_dataset)
        result = compare(models={"r": Ridge()}, dataset="beijing_pm25", n_seeds=1)
        assert result.raw

    def test_a_windowed_dataset_dict_still_works(self):
        from sklearn.linear_model import Ridge

        from eigenkit.benchmark import compare

        data = load_dataset("sine_wave")
        windowed = {
            "X_train": np.asarray(data["X_train"]).reshape(-1, 1),
            "y_train": data["y_train"],
            "X_test": np.asarray(data["X_test"]).reshape(-1, 1),
            "y_test": data["y_test"],
            "task": "regression",
        }
        assert compare(models={"r": Ridge()}, dataset=windowed, n_seeds=1).raw


class TestShortSeriesSeasonality:
    def test_a_monotone_ramp_is_not_called_seasonal(self):
        y = np.arange(8, dtype=float)
        report = DatasetReport.from_data(X=np.zeros((8, 1)), y=y, task="timeseries")
        summary = report.summarize()
        assert not any("easonal" in r for r in summary["recommendations"])

    def test_it_does_not_leak_numpy_warnings(self):
        y = np.arange(8, dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DatasetReport.from_data(
                X=np.zeros((8, 1)), y=y, task="timeseries"
            ).summarize()
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert runtime == [], f"leaked {len(runtime)} numpy warnings"

    def test_a_genuinely_seasonal_series_is_still_detected(self):
        """The guard must suppress the false positive without costing the
        true one: 240 points against an inferred period of 12."""
        t = np.arange(240, dtype=float)
        y = np.sin(2 * np.pi * t / 12) + 0.01 * t
        summary = DatasetReport.from_data(
            X=t.reshape(-1, 1), y=y, task="timeseries"
        ).summarize()
        assert any("easonal" in r for r in summary["recommendations"])
