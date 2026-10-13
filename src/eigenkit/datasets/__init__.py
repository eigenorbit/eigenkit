"""The datasets of eigenkit: the registry of their metadata, and the loaders."""

from __future__ import annotations

import warnings

import numpy as np

from ._registry import REGISTRY, DatasetInfo, get_dataset_info
from ._synthetic import (
    generate_ad_hoc,
    generate_checkerboard,
    generate_circles,
    generate_moons,
    generate_regression,
    generate_timeseries,
)
from .eda import (
    FONTSIZE_AXES,
    FONTSIZE_LEGEND,
    FONTSIZE_TICKS,
    STYLE_COLORS,
    DatasetReport,
    rc_params,
)


__all__ = [
    "DatasetInfo",
    "REGISTRY",
    "get_dataset_info",
    "load_dataset",
    "list_datasets",
    "generate_moons",
    "generate_circles",
    "generate_checkerboard",
    "generate_ad_hoc",
    "generate_regression",
    "generate_timeseries",
    "DatasetReport",
    # theme-aware plotting, shared by the EDA report, the benchmark curves
    # and the worked examples
    "rc_params",
    "STYLE_COLORS",
    "FONTSIZE_AXES",
    "FONTSIZE_LEGEND",
    "FONTSIZE_TICKS",
]


def load_dataset(
    name: str,
    *,
    split: bool = True,
    test_size: float = 0.2,
    validation_size: float = 0.0,
    random_state: int | None = 42,
    cache_dir: str | None = None,
) -> dict:
    """Load a dataset by name.

    Args:
        name: The name of the dataset in `REGISTRY`, for example `"iris"`.
        split: If `True`, the default, the data is split into a training set and a
            test set.
        test_size: The fraction of the samples used for the test set.
        validation_size: The fraction of the samples used for the validation set.
            With `0`, the default, there is no validation set.
        random_state: The random seed of the split, and of the generator for the
            synthetic datasets.
        cache_dir: A cache directory for the downloaded datasets, in place of the
            default one.

    Returns:
        A dictionary with these keys:

            - With `split=True`: `"X_train"`, `"y_train"`, `"X_test"`, `"y_test"`
              and `"metadata"`, and also `"X_val"` and `"y_val"` when
              `validation_size` is above `0`.
            - With `split=False`: `"X"`, `"y"` and `"metadata"`.

            The synthetic time-series datasets also have a `"sequence"` key, which
            holds the whole series. `"metadata"` is the
            [`DatasetInfo`][eigenkit.datasets.DatasetInfo] of the dataset.

    Raises:
        ValueError: If `name` is not in the registry, if `validation_size` is not in
            `[0, 1)`, or if `test_size + validation_size` is `1` or more with
            `split=True`.

    Example:
        ```python
        data = load_dataset("iris")
        X_train, y_train = data["X_train"], data["y_train"]
        ```
    """
    from sklearn.model_selection import train_test_split

    info = get_dataset_info(name)
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if split and test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be < 1.")
    holdout_size = test_size + validation_size if split else test_size

    if info.source == "sklearn":
        from ._loaders import load_sklearn_dataset

        result = load_sklearn_dataset(
            info.sklearn_loader,
            test_size=holdout_size,
            random_state=random_state,
            split=split,
            task=info.task,
        )

    elif info.source == "synthetic":
        from ._synthetic import GENERATORS

        X, y = GENERATORS[info.synthetic_generator](random_state=random_state)

        if info.task == "timeseries":
            n = len(y)
            split_idx = int(n * (1 - holdout_size))
            if split:
                result = {
                    "X_train": X[:split_idx],
                    "y_train": y[:split_idx],
                    "X_test": X[split_idx:],
                    "y_test": y[split_idx:],
                    "sequence": y,
                }
            else:
                result = {"X": X, "y": y, "sequence": y}
        else:
            if split:
                stratify = y if info.task == "classification" else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=holdout_size,
                    random_state=random_state,
                    stratify=stratify,
                )
                result = {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                }
            else:
                result = {"X": X, "y": y}

    elif info.source == "url":
        import os

        from ._cache import cached_download
        from ._loaders import load_url_dataset

        filename = info.url.rsplit("/", 1)[-1]
        if cache_dir is None:
            filepath = cached_download(info.url, filename, expected_hash=info.sha256)
        else:
            # Scoped to this call. Assigning the variable outright would redirect
            # every later download in the process, including ones from other
            # code that never asked for a custom cache.
            previous = os.environ.get("EIGENKIT_DATA_DIR")
            os.environ["EIGENKIT_DATA_DIR"] = cache_dir
            try:
                filepath = cached_download(info.url, filename, expected_hash=info.sha256)
            finally:
                if previous is None:
                    os.environ.pop("EIGENKIT_DATA_DIR", None)
                else:
                    os.environ["EIGENKIT_DATA_DIR"] = previous
        result = load_url_dataset(
            filepath,
            test_size=holdout_size,
            random_state=random_state,
            split=split,
            task=info.task,
        )

    else:
        raise ValueError(f"Unknown dataset source: {info.source!r}")

    if split and validation_size:
        _add_validation_split(result, info, test_size, validation_size, random_state)

    result["metadata"] = info
    return result


def _add_validation_split(
    result: dict,
    info: DatasetInfo,
    test_size: float,
    validation_size: float,
    random_state: int | None,
) -> None:
    from sklearn.model_selection import train_test_split

    X_holdout, y_holdout = result["X_test"], result["y_test"]
    test_ratio = test_size / (test_size + validation_size)

    if info.task == "timeseries":
        split_idx = int(len(y_holdout) * (1 - test_ratio))
        result["X_val"] = X_holdout[:split_idx]
        result["y_val"] = y_holdout[:split_idx]
        result["X_test"] = X_holdout[split_idx:]
        result["y_test"] = y_holdout[split_idx:]
        return

    stratify = y_holdout if info.task == "classification" else None
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout,
            y_holdout,
            test_size=test_ratio,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        if stratify is None:
            raise
        # Falling back to an unstratified split is better than failing, but it
        # can produce a single-class y_val against a multi-class y_test, and
        # silently -- so a validation score that means nothing looks like a
        # real one.
        warnings.warn(
            "Could not stratify the validation/test split (too few samples in "
            "some class), so it was split without stratification. The class "
            "balance of y_val and y_test may differ from the training set.",
            UserWarning,
            stacklevel=3,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout,
            y_holdout,
            test_size=test_ratio,
            random_state=random_state,
        )
    result["X_val"] = X_val
    result["y_val"] = y_val
    result["X_test"] = X_test
    result["y_test"] = y_test


def list_datasets() -> list[DatasetInfo]:
    """Return the metadata of every available dataset.

    Returns:
        One [`DatasetInfo`][eigenkit.datasets.DatasetInfo] for each dataset in the
            registry.

    Example:
        ```python
        datasets = list_datasets()
        print(datasets[0].name)  # 'breast_cancer_wisconsin'
        ```
    """
    return list(REGISTRY.values())
