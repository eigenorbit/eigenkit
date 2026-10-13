"""
Dataset loaders for sklearn-bundled and URL-based datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_sklearn_dataset(
    loader_name: str,
    test_size: float = 0.2,
    random_state: int | None = 42,
    split: bool = True,
    task: str = "classification",
) -> dict:
    """Load a dataset from sklearn.datasets.

    Args:
        loader_name: Name of the sklearn.datasets function, e.g. ``"load_iris"``.
        test_size: Fraction of data held out for the test set (only used when split=True).
        random_state: Random random_state for train/test splitting.
        split: If True, return train/test split; otherwise return the full dataset.
        task: ML task type — ``"classification"`` stratifies the split and returns integer
            targets; any other value skips stratification and returns float32 targets.

    Returns:
        If split=True: ``{"X_train", "y_train", "X_test", "y_test"}``
        If split=False: ``{"X", "y"}``
    """
    import sklearn.datasets
    from sklearn.model_selection import train_test_split

    loader_fn = getattr(sklearn.datasets, loader_name)
    data = loader_fn()
    X, y = data.data, data.target

    if task == "classification":
        y = y.astype(int)
        stratify = y if split else None
    else:
        y = y.astype(np.float32)
        stratify = None

    if split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    return {"X": X, "y": y}


def load_url_dataset(
    filepath: Path,
    test_size: float = 0.2,
    random_state: int | None = 42,
    split: bool = True,
    task: str = "classification",
) -> dict:
    """Load a dataset from a locally cached file (CSV or NPZ).

    Args:
        filepath: Path to the cached file on disk.
        test_size: Fraction of data held out for the test set (only used when split=True).
        random_state: Random random_state for train/test splitting (ignored when ``task='timeseries'``,
            which always uses a chronological no-shuffle split).
        split: If True, return train/test split; otherwise return the full dataset.
        task: ML task type — ``"classification"`` stratifies the split and encodes string
            labels; ``"timeseries"`` uses a chronological (no-shuffle) split and float32
            targets; any other value skips stratification and returns float32 targets.

    Returns:
        If split=True: ``{"X_train", "y_train", "X_test", "y_test"}``
        If split=False: ``{"X", "y"}``
    """
    from sklearn.model_selection import train_test_split

    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".npz":
        loaded = np.load(filepath)
        X, y = loaded["X"], loaded["y"]
    elif suffix == ".csv":
        # Skip the first line only when it is actually a header. An
        # unconditional `skiprows=1` would make a headerless CSV silently lose
        # its first sample -- no error, one row short, and nothing to notice it by.
        with open(filepath, encoding="utf-8") as handle:
            first_line = handle.readline()
        if not first_line.strip():
            # Otherwise `float("")` raises, the file is treated as having a
            # header, and `np.loadtxt` fails from inside numpy -- the same
            # inscrutable failure the header sniffing above exists to remove.
            raise ValueError(
                f"{filepath.name} is empty; a dataset CSV needs at least one "
                f"row with one feature column plus a target column."
            )
        try:
            [float(cell) for cell in first_line.strip().split(",")]
        except ValueError:
            skiprows = 1  # not numeric, so it is a header
        else:
            skiprows = 0
        arr = np.loadtxt(filepath, delimiter=",", skiprows=skiprows, ndmin=2)
        if arr.shape[1] < 2:
            raise ValueError(
                f"{filepath.name} has {arr.shape[1]} column(s); a dataset CSV "
                f"needs at least one feature column plus a target column."
            )
        X, y = arr[:, :-1], arr[:, -1]
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}. Expected .csv or .npz")

    if task == "timeseries":
        y = y.astype(np.float32)
        if split:
            split_idx = int(len(y) * (1 - test_size))
            return {
                "X_train": X[:split_idx],
                "y_train": y[:split_idx],
                "X_test": X[split_idx:],
                "y_test": y[split_idx:],
            }
        return {"X": X, "y": y}

    if task == "classification":
        # Encode string labels to integers
        if y.dtype.kind in ("U", "S", "O"):
            classes = sorted(set(y))
            mapping = {c: i for i, c in enumerate(classes)}
            y = np.array([mapping[v] for v in y], dtype=np.int64)
        stratify = y if split else None
    else:
        y = y.astype(np.float32)
        stratify = None

    if split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    return {"X": X, "y": y}
