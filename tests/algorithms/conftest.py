import numpy as np
import pytest


@pytest.fixture
def clf_data():
    """4-feature binary classification: 20 train, 10 test."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X[:20], y[:20], X[20:], y[20:]


@pytest.fixture
def reg_data():
    """4-feature regression: 20 train, 10 test."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)
    return X[:20], y[:20], X[20:], y[20:]


@pytest.fixture
def ts_data():
    """1-D sine wave, 50 samples."""
    t = np.linspace(0, 4 * np.pi, 50)
    return np.sin(t).astype(np.float32)
