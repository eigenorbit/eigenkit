import pytest
from pydantic import ValidationError

from eigenkit.datasets._registry import REGISTRY, DatasetInfo, get_dataset_info


class TestDatasetInfo:
    def test_fields_present(self):
        info = DatasetInfo(
            name="iris",
            source="sklearn",
            url=None,
            sklearn_loader="load_iris",
            synthetic_generator=None,
            task="classification",
            n_features=4,
            n_samples=150,
            n_classes=3,
            description="Classic multi-class classification dataset.",
        )
        assert info.name == "iris"
        assert info.source == "sklearn"
        assert info.sklearn_loader == "load_iris"
        assert info.synthetic_generator is None
        assert info.url is None
        assert info.task == "classification"
        assert info.n_features == 4
        assert info.n_samples == 150
        assert info.n_classes == 3

    def test_source_literal_validation(self):
        with pytest.raises(ValidationError):
            DatasetInfo(
                name="bad",
                source="unknown_source",
                url=None,
                sklearn_loader=None,
                synthetic_generator=None,
                task="classification",
                n_features=2,
                n_samples=100,
                n_classes=2,
                description="bad source",
            )

    def test_task_literal_validation(self):
        with pytest.raises(ValidationError):
            DatasetInfo(
                name="bad",
                source="sklearn",
                url=None,
                sklearn_loader="load_iris",
                synthetic_generator=None,
                task="unsupported_task",
                n_features=2,
                n_samples=100,
                n_classes=2,
                description="bad task",
            )

    def test_n_features_must_be_positive(self):
        with pytest.raises(ValidationError):
            DatasetInfo(
                name="bad",
                source="sklearn",
                url=None,
                sklearn_loader="load_iris",
                synthetic_generator=None,
                task="classification",
                n_features=0,
                n_samples=100,
                n_classes=2,
                description="zero features",
            )

    def test_n_samples_none_for_synthetic(self):
        info = DatasetInfo(
            name="moons",
            source="synthetic",
            url=None,
            sklearn_loader=None,
            synthetic_generator="make_moons",
            task="classification",
            n_features=2,
            n_samples=None,
            n_classes=2,
            description="Two interleaving half-circles.",
        )
        assert info.n_samples is None

    def test_window_size_field_optional(self):
        info = DatasetInfo(
            name="ts",
            source="synthetic",
            synthetic_generator="generate_sine_wave",
            task="timeseries",
            n_features=1,
            description="A time series.",
            window_size=10,
        )
        assert info.window_size == 10

    def test_window_size_defaults_to_none(self):
        info = DatasetInfo(
            name="iris",
            source="sklearn",
            sklearn_loader="load_iris",
            task="classification",
            n_features=4,
            description="Iris.",
        )
        assert info.window_size is None

    def test_feature_names_optional(self):
        info = DatasetInfo(
            name="iris",
            source="sklearn",
            sklearn_loader="load_iris",
            task="classification",
            n_features=4,
            description="Iris.",
            feature_names=["a", "b", "c", "d"],
        )
        assert info.feature_names == ["a", "b", "c", "d"]

    def test_timeseries_task_type_valid(self):
        info = DatasetInfo(
            name="ts",
            source="synthetic",
            synthetic_generator="generate_sine_wave",
            task="timeseries",
            n_features=1,
            description="A time series.",
        )
        assert info.task == "timeseries"


class TestRegistry:
    EXPECTED_DATASETS = [
        "breast_cancer_wisconsin",
        "iris",
        "wine",
        "moons",
        "circles",
        "checkerboard",
        "california_housing",
        "diabetes",
        "concrete",
        "beijing_pm25",
        "sine_wave",
        "ar_process",
    ]

    def test_registry_contains_expected_keys(self):
        for name in self.EXPECTED_DATASETS:
            assert name in REGISTRY, f"Missing dataset: {name}"

    def test_registry_values_are_dataset_info(self):
        for name, info in REGISTRY.items():
            assert isinstance(info, DatasetInfo), f"{name} value is not DatasetInfo"

    def test_registry_names_match_keys(self):
        for key, info in REGISTRY.items():
            assert info.name == key, f"Key '{key}' does not match info.name '{info.name}'"

    def test_sklearn_datasets_have_loader(self):
        for name, info in REGISTRY.items():
            if info.source == "sklearn":
                assert info.sklearn_loader is not None, f"{name} missing sklearn_loader"

    def test_synthetic_datasets_have_generator(self):
        for name, info in REGISTRY.items():
            if info.source == "synthetic":
                assert info.synthetic_generator is not None, (
                    f"{name} missing synthetic_generator"
                )

    def test_url_datasets_have_url(self):
        for name, info in REGISTRY.items():
            if info.source == "url":
                assert info.url is not None, f"{name} missing url"

    def test_timeseries_datasets_have_window_size(self):
        for name, info in REGISTRY.items():
            if info.task == "timeseries":
                assert info.window_size is not None, f"{name} missing window_size"
                assert info.window_size > 0


class TestGetDatasetInfo:
    def test_returns_correct_info(self):
        info = get_dataset_info("iris")
        assert isinstance(info, DatasetInfo)
        assert info.name == "iris"

    def test_raises_value_error_for_unknown(self):
        with pytest.raises(ValueError, match="nonexistent_dataset_xyz"):
            get_dataset_info("nonexistent_dataset_xyz")

    def test_error_message_lists_available_datasets(self):
        with pytest.raises(ValueError) as exc_info:
            get_dataset_info("nonexistent_dataset_xyz")
        msg = str(exc_info.value)
        assert "breast_cancer_wisconsin" in msg or "iris" in msg


class TestPublicAPI:
    def test_top_level_import(self):
        import eigenkit

        assert hasattr(eigenkit, "datasets")

    def test_datasets_module_exports(self):
        from eigenkit import datasets

        assert hasattr(datasets, "DatasetInfo")
        assert hasattr(datasets, "REGISTRY")
        assert hasattr(datasets, "get_dataset_info")


def test_url_datasets_pin_their_sha256():
    from eigenkit.datasets._registry import REGISTRY

    for info in REGISTRY.values():
        if info.source == "url":
            assert info.sha256 and len(info.sha256) == 64, info.name
