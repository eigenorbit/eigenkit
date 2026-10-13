"""Tests for load_dataset (sklearn sources) and list_datasets."""

import io
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eigenkit.datasets import list_datasets, load_dataset


@pytest.mark.parametrize("name", ["breast_cancer_wisconsin", "iris", "wine"])
def test_sklearn_dataset_shapes(name):
    data = load_dataset(name, random_state=42)
    assert {"X_train", "y_train", "X_test", "y_test", "metadata"} <= data.keys()
    assert data["X_train"].shape[0] == data["y_train"].shape[0]
    assert data["X_test"].shape[0] == data["y_test"].shape[0]
    assert data["X_train"].shape[1] == data["metadata"].n_features


@pytest.mark.parametrize("name", ["breast_cancer_wisconsin", "iris", "wine"])
def test_sklearn_dataset_metadata_attached(name):
    data = load_dataset(name, random_state=42)
    meta = data["metadata"]
    assert meta.name == name
    assert meta.source == "sklearn"


def test_load_dataset_no_split():
    data = load_dataset("iris", split=False)
    assert set(data.keys()) == {"X", "y", "metadata"}
    assert data["X"].shape == (150, 4)
    assert data["y"].shape == (150,)


def test_load_dataset_test_size():
    data = load_dataset("iris", test_size=0.3, random_state=0)
    n_total = data["X_train"].shape[0] + data["X_test"].shape[0]
    assert n_total == 150
    assert data["X_test"].shape[0] == pytest.approx(45, abs=2)


def test_load_dataset_validation_split():
    data = load_dataset("iris", test_size=0.15, validation_size=0.15, random_state=0)
    assert {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"} <= data.keys()
    n_total = data["X_train"].shape[0] + data["X_val"].shape[0] + data["X_test"].shape[0]
    assert n_total == 150
    assert data["X_train"].shape[0] == 105
    assert data["X_val"].shape[0] == pytest.approx(22, abs=1)
    assert data["X_test"].shape[0] == pytest.approx(23, abs=1)


def test_load_dataset_validation_size_zero_omits_validation_split():
    data = load_dataset("iris", test_size=0.15, validation_size=0.0, random_state=0)
    assert {"X_train", "y_train", "X_test", "y_test", "metadata"} <= data.keys()
    assert "X_val" not in data
    assert "y_val" not in data


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="not found"):
        load_dataset("nonexistent_dataset")


def test_list_datasets():
    datasets = list_datasets()
    assert len(datasets) >= 6
    assert all(hasattr(d, "name") for d in datasets)
    names = [d.name for d in datasets]
    assert "iris" in names
    assert "moons" in names


@pytest.mark.parametrize("name", ["california_housing", "diabetes"])
def test_regression_dataset_shapes(name):
    data = load_dataset(name, random_state=42)
    assert {"X_train", "y_train", "X_test", "y_test", "metadata"} <= data.keys()
    assert data["X_train"].shape[0] == data["y_train"].shape[0]
    assert data["X_test"].shape[0] == data["y_test"].shape[0]
    assert data["X_train"].shape[1] == data["metadata"].n_features


@pytest.mark.parametrize("name", ["california_housing", "diabetes"])
def test_regression_y_is_float(name):
    data = load_dataset(name, random_state=42)
    assert data["y_train"].dtype == np.float32
    assert data["y_test"].dtype == np.float32


@pytest.mark.parametrize("name", ["california_housing", "diabetes"])
def test_regression_metadata_task(name):
    data = load_dataset(name, random_state=42)
    assert data["metadata"].task == "regression"


@pytest.fixture(autouse=True)
def _unpin_url_datasets(monkeypatch):
    # The tests below serve synthetic CSVs, which can never match the pinned
    # SHA-256 of the real files.
    from eigenkit.datasets._registry import REGISTRY

    for info in REGISTRY.values():
        if info.source == "url":
            monkeypatch.setattr(info, "sha256", None)


def _make_concrete_csv(n_rows: int = 50) -> bytes:
    rng = np.random.default_rng(0)
    header = "cement,slag,ash,water,superplastic,coarseagg,fineagg,age,strength\n"
    data = rng.random((n_rows, 9)) * 100
    rows = "\n".join(",".join(f"{v:.4f}" for v in row) for row in data)
    return (header + rows).encode()


def test_concrete_dataset_shapes(tmp_path):
    csv_bytes = _make_concrete_csv(50)

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(csv_bytes)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            data = load_dataset("concrete", random_state=42)

    assert {"X_train", "y_train", "X_test", "y_test", "metadata"} <= data.keys()
    assert data["X_train"].shape[1] == 8
    assert data["X_train"].shape[0] + data["X_test"].shape[0] == 50
    assert data["metadata"].task == "regression"


def test_concrete_dataset_y_is_float(tmp_path):
    csv_bytes = _make_concrete_csv(50)

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(csv_bytes)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            data = load_dataset("concrete", random_state=42)

    assert data["y_train"].dtype == np.float32
    assert data["y_test"].dtype == np.float32


def _make_beijing_csv(n_rows: int = 100) -> bytes:
    rng = np.random.default_rng(0)
    header = (
        "year,month,day,hour,DEWP,TEMP,PRES,Iws,Is,Ir,"
        "wnd_dir_NW,wnd_dir_SE,wnd_dir_cv,pollution\n"
    )
    data = rng.random((n_rows, 14)) * 100
    rows = "\n".join(",".join(f"{v:.4f}" for v in row) for row in data)
    return (header + rows).encode()


def test_beijing_pm25_dataset_shapes(tmp_path):
    csv_bytes = _make_beijing_csv(100)

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(csv_bytes)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            data = load_dataset("beijing_pm25", random_state=42)

    assert {"X_train", "y_train", "X_test", "y_test", "metadata"} <= data.keys()
    assert data["X_train"].shape[1] == 13
    assert data["X_train"].shape[0] + data["X_test"].shape[0] == 100
    assert data["metadata"].task == "timeseries"


def test_beijing_pm25_chronological_split(tmp_path):
    csv_bytes = _make_beijing_csv(100)

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(csv_bytes)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            data = load_dataset("beijing_pm25", test_size=0.2, random_state=42)

    raw = np.loadtxt(io.BytesIO(csv_bytes), delimiter=",", skiprows=1)
    np.testing.assert_array_equal(data["X_train"][0], raw[0, :-1])
    np.testing.assert_array_equal(data["X_test"][-1], raw[-1, :-1])
    assert data["X_train"].shape[0] == 80
    assert data["X_test"].shape[0] == 20
    assert data["y_train"].dtype == np.float32
    assert data["y_test"].dtype == np.float32


def test_beijing_pm25_no_split(tmp_path):
    csv_bytes = _make_beijing_csv(100)

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(csv_bytes)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            data = load_dataset("beijing_pm25", split=False)

    assert {"X", "y", "metadata"} <= data.keys()
    assert data["X"].shape == (100, 13)
    assert data["y"].dtype == np.float32
