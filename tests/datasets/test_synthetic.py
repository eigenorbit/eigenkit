"""Tests for load_dataset (synthetic sources) and _synthetic generators."""

import numpy as np
import pytest

from eigenkit.datasets import load_dataset
from eigenkit.datasets._synthetic import (
    GENERATORS,
    generate_checkerboard,
    generate_circles,
    generate_moons,
    generate_regression,
    generate_timeseries,
)


@pytest.mark.parametrize("name", ["moons", "circles", "checkerboard"])
def test_synthetic_shapes(name):
    data = load_dataset(name, random_state=42)
    assert {"X_train", "y_train", "X_test", "y_test", "metadata"} <= data.keys()
    assert data["X_train"].shape[1] == 2
    assert set(data["y_train"]).issubset({0, 1})
    assert set(data["y_test"]).issubset({0, 1})


@pytest.mark.parametrize("name", ["moons", "circles", "checkerboard"])
def test_synthetic_no_split(name):
    data = load_dataset(name, split=False, random_state=0)
    assert "X" in data and "y" in data
    assert data["X"].shape[1] == 2


def test_reproducibility():
    d1 = load_dataset("moons", random_state=42)
    d2 = load_dataset("moons", random_state=42)
    np.testing.assert_array_equal(d1["X_train"], d2["X_train"])
    np.testing.assert_array_equal(d1["y_train"], d2["y_train"])


def test_different_seeds_differ():
    d1 = load_dataset("circles", random_state=0)
    d2 = load_dataset("circles", random_state=99)
    assert not np.array_equal(d1["X_train"], d2["X_train"])


def test_generators_dict_keys():
    assert set(GENERATORS.keys()) == {
        "make_moons",
        "make_circles",
        "make_checkerboard",
        "generate_regression",
        "generate_sine_wave",
        "generate_ar_process",
    }


def test_generate_moons_shape():
    X, y = generate_moons(n_samples=100, random_state=0)
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert set(y) == {0, 1}


def test_generate_circles_shape():
    X, y = generate_circles(n_samples=80, random_state=7)
    assert X.shape == (80, 2)
    assert y.shape == (80,)


def test_generate_checkerboard_labels():
    X, y = generate_checkerboard(n_samples=400, grid_size=2, random_state=1)
    assert X.shape == (400, 2)
    assert set(y).issubset({0, 1})
    # Should have roughly equal class balance for a fine grid
    assert 0.3 < y.mean() < 0.7


def test_generate_regression_shape():
    X, y = generate_regression(n_samples=100, n_features=4, random_state=0)
    assert X.shape == (100, 4)
    assert y.shape == (100,)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_generate_regression_reproducible():
    X1, y1 = generate_regression(random_state=7)
    X2, y2 = generate_regression(random_state=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_timeseries_sine_shape():
    X, y = generate_timeseries(n_samples=300, kind="sine", random_state=0)
    assert X.shape == (300,)
    assert y.shape == (300,)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_generate_timeseries_ar_shape():
    X, y = generate_timeseries(n_samples=200, kind="ar", random_state=42)
    assert X.shape == (200,)
    assert y.shape == (200,)


def test_generate_timeseries_sine_range():
    X, y = generate_timeseries(n_samples=1000, kind="sine", noise=0.0, random_state=1)
    assert X[0] == pytest.approx(0.0, abs=1e-5)
    assert X[-1] == pytest.approx(1.0, abs=1e-5)
    assert y.max() < 1.1
    assert y.min() > -1.1


def test_generate_timeseries_unknown_kind():
    with pytest.raises(ValueError, match="Unknown kind"):
        generate_timeseries(kind="unknown")


def test_generators_dict_contains_the_regression_and_timeseries_generators():
    assert "generate_regression" in GENERATORS
    assert "generate_sine_wave" in GENERATORS
    assert "generate_ar_process" in GENERATORS


@pytest.mark.parametrize("name", ["sine_wave", "ar_process"])
def test_timeseries_dataset_split_shapes(name):
    data = load_dataset(name, random_state=42)
    assert {
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "sequence",
        "metadata",
    } <= data.keys()
    assert data["X_train"].shape == data["y_train"].shape
    assert data["X_test"].shape == data["y_test"].shape
    n_train = len(data["X_train"])
    n_test = len(data["X_test"])
    assert n_train > n_test  # default 80/20 split


@pytest.mark.parametrize("name", ["sine_wave", "ar_process"])
def test_timeseries_no_shuffle(name):
    data = load_dataset(name, random_state=42)
    assert data["X_train"].max() <= data["X_test"].min()


@pytest.mark.parametrize("name", ["sine_wave", "ar_process"])
def test_timeseries_validation_split_is_chronological(name):
    data = load_dataset(name, test_size=0.15, validation_size=0.15, random_state=42)
    assert {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"} <= data.keys()
    assert data["X_train"].max() <= data["X_val"].min()
    assert data["X_val"].max() <= data["X_test"].min()


@pytest.mark.parametrize("name", ["sine_wave", "ar_process"])
def test_timeseries_sequence_key(name):
    data = load_dataset(name, random_state=42)
    assert "sequence" in data
    assert len(data["sequence"]) == len(data["X_train"]) + len(data["X_test"])


@pytest.mark.parametrize("name", ["sine_wave", "ar_process"])
def test_timeseries_metadata(name):
    data = load_dataset(name, random_state=42)
    assert data["metadata"].task == "timeseries"
    assert data["metadata"].window_size == 10


@pytest.mark.parametrize("name", ["sine_wave", "ar_process"])
def test_timeseries_no_split(name):
    data = load_dataset(name, split=False, random_state=42)
    assert {"X", "y", "sequence", "metadata"} <= data.keys()


class TestAdHoc:
    """Havlicek et al.'s dataset: labels a ZZ-feature-map kernel separates."""

    def test_shape_balance_and_range(self):
        from eigenkit.datasets import generate_ad_hoc

        X, y = generate_ad_hoc(n_samples=60, n_qubits=3, random_state=0)
        assert X.shape == (60, 3) and y.shape == (60,)
        assert np.bincount(y).tolist() == [30, 30]
        assert X.min() >= 0 and X.max() < 2 * np.pi

    def test_reproducible(self):
        from eigenkit.datasets import generate_ad_hoc

        a = generate_ad_hoc(n_samples=40, random_state=3)
        b = generate_ad_hoc(n_samples=40, random_state=3)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_the_generating_kernel_separates_it_and_a_classical_one_struggles(self):
        from sklearn.svm import SVC

        from eigenkit.datasets import generate_ad_hoc
        from eigenkit.datasets._synthetic import _zz_feature_map_spec
        from eigenkit.qml import QSVC, QuantumKernel

        X, y = generate_ad_hoc(n_samples=160, n_qubits=3, random_state=1)
        kernel = QuantumKernel(n_features=3, feature_map=_zz_feature_map_spec(3))
        quantum = QSVC(kernel=kernel).fit(X[:100], y[:100]).score(X[100:], y[100:])
        rbf = SVC().fit(X[:100], y[:100]).score(X[100:], y[100:])
        assert quantum > 0.9
        assert quantum - rbf > 0.2

    def test_gap_is_validated(self):
        import pytest

        from eigenkit.datasets import generate_ad_hoc

        with pytest.raises(ValueError, match="gap"):
            generate_ad_hoc(gap=1.0)
