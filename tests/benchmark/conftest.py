import numpy as np
import pytest


matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("pandas")
matplotlib.use("Agg")


@pytest.fixture
def tiny_clf_split():
    """Small binary classification dict with pre-split arrays."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return {
        "X_train": X[:20],
        "y_train": y[:20],
        "X_test": X[20:],
        "y_test": y[20:],
    }


@pytest.fixture
def tiny_reg_split():
    """Small regression dict with pre-split arrays."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3)).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)
    return {
        "X_train": X[:20],
        "y_train": y[:20],
        "X_test": X[20:],
        "y_test": y[20:],
    }
