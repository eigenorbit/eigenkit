"""Tests for the qml-tier quantum reservoir estimators."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import ClassifierMixin, RegressorMixin, clone

from eigenkit.configs import Encoding, Variational
from eigenkit.qml import QuantumReservoirClassifier, QuantumReservoirRegressor


def _reg_blocks():
    # 2 qubits, n_inputs == 2 (F=2 for carry policies).
    return [Encoding(in_features=2, rotation="RY"), Variational(depth=2, rotation="RY")]


@pytest.fixture
def seq_reg():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 5, 2))
    y = 0.5 * X[:, :, 0] + X[:, :, 1]  # (6, 5)
    return X, y


@pytest.fixture
def seq_clf():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 5, 2))
    y = (X[:, :, 0] > 0).astype(int)  # (6, 5)
    return X, y


class TestRegressor:
    def test_predict_shape(self, seq_reg):
        X, y = seq_reg
        model = QuantumReservoirRegressor(_reg_blocks(), state_policy="carry").fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (6, 5)

    def test_is_regressor(self):
        assert isinstance(QuantumReservoirRegressor(_reg_blocks()), RegressorMixin)

    def test_score_is_float(self, seq_reg):
        X, y = seq_reg
        model = QuantumReservoirRegressor(_reg_blocks()).fit(X, y)
        assert isinstance(model.score(X, y), float)

    def test_clone_preserves_params(self):
        cloned = clone(
            QuantumReservoirRegressor(_reg_blocks(), window=3, ridge=0.1, washout=2)
        )
        assert cloned.window == 3
        assert cloned.ridge == 0.1
        assert cloned.washout == 2

    def test_frozen_weights_are_deterministic(self, seq_reg):
        X, y = seq_reg
        a = QuantumReservoirRegressor(_reg_blocks(), random_state=3).fit(X, y)
        b = QuantumReservoirRegressor(_reg_blocks(), random_state=3).fit(X, y)
        np.testing.assert_allclose(a._weights, b._weights)
        np.testing.assert_allclose(a.predict(X), b.predict(X))


class TestClassifier:
    def test_predict_and_proba_shapes(self, seq_clf):
        X, y = seq_clf
        model = QuantumReservoirClassifier(_reg_blocks()).fit(X, y)
        assert model.predict(X).shape == (6, 5)
        assert set(np.unique(model.predict(X))).issubset({0, 1})
        assert model.predict_proba(X).shape == (6, 5, 2)
        np.testing.assert_array_equal(model.classes_, np.array([0, 1]))

    def test_is_classifier(self):
        assert isinstance(QuantumReservoirClassifier(_reg_blocks()), ClassifierMixin)

    def test_score_is_float(self, seq_clf):
        X, y = seq_clf
        model = QuantumReservoirClassifier(_reg_blocks()).fit(X, y)
        assert isinstance(model.score(X, y), float)


class TestPoliciesAndFeatures:
    def test_stateless_windowed(self):
        # Circuit consumes window*F = 2 inputs; sequences are F=1.
        blocks = [
            Encoding(in_features=2, rotation="RY"),
            Variational(depth=1, rotation="RY"),
        ]
        rng = np.random.default_rng(4)
        X = rng.standard_normal((5, 6, 1))
        y = X[:, :, 0]
        model = QuantumReservoirRegressor(blocks, state_policy="stateless", window=2).fit(
            X, y
        )
        assert model.predict(X).shape == (5, 6)

    def test_carry_and_stateless_differ(self, seq_reg):
        X, y = seq_reg
        carry = QuantumReservoirRegressor(_reg_blocks(), state_policy="carry").fit(X, y)
        stateless = QuantumReservoirRegressor(
            _reg_blocks(), state_policy="stateless", window=1
        ).fit(X, y)
        assert not np.allclose(carry.predict(X), stateless.predict(X))

    def test_hamiltonian_features(self, seq_reg):
        X, y = seq_reg
        model = QuantumReservoirRegressor(
            _reg_blocks(), features="0.5 Z0 X1 - 0.3 Y0 + Z1"
        ).fit(X, y)
        assert model.predict(X).shape == (6, 5)

    def test_amplitude_features(self, seq_reg):
        X, y = seq_reg
        model = QuantumReservoirRegressor(_reg_blocks(), features="amplitudes").fit(X, y)
        assert model.predict(X).shape == (6, 5)

    def test_leaky_policy_runs(self, seq_reg):
        X, y = seq_reg
        model = QuantumReservoirRegressor(
            _reg_blocks(), state_policy="leaky", leak=0.7
        ).fit(X, y)
        assert model.predict(X).shape == (6, 5)


class TestValidation:
    def test_bad_state_policy_raises(self, seq_reg):
        X, y = seq_reg
        with pytest.raises(ValueError, match="state_policy"):
            QuantumReservoirRegressor(_reg_blocks(), state_policy="nope").fit(X, y)

    def test_X_must_be_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            QuantumReservoirRegressor(_reg_blocks()).fit(np.zeros((4, 2)), np.zeros(4))

    def test_washout_too_large_raises(self, seq_reg):
        X, y = seq_reg  # T == 5
        with pytest.raises(ValueError, match="washout"):
            QuantumReservoirRegressor(_reg_blocks(), washout=5).fit(X, y)

    def test_washout_trains_on_remaining(self, seq_reg):
        X, y = seq_reg
        model = QuantumReservoirRegressor(_reg_blocks(), washout=2).fit(X, y)
        assert model.predict(X).shape == (6, 5)  # predictions still cover all T


class TestSerialization:
    def test_regressor_roundtrip(self, tmp_path, seq_reg):
        X, y = seq_reg
        model = QuantumReservoirRegressor(_reg_blocks(), state_policy="carry").fit(X, y)
        before = model.predict(X)
        model.save(tmp_path / "qrc")
        loaded = QuantumReservoirRegressor.load(tmp_path / "qrc")
        np.testing.assert_allclose(before, loaded.predict(X), rtol=1e-6, atol=1e-6)

    def test_classifier_roundtrip(self, tmp_path, seq_clf):
        X, y = seq_clf
        model = QuantumReservoirClassifier(_reg_blocks()).fit(X, y)
        before = model.predict(X)
        model.save(tmp_path / "qrc_clf")
        loaded = QuantumReservoirClassifier.load(tmp_path / "qrc_clf")
        np.testing.assert_array_equal(before, loaded.predict(X))

    def test_load_via_load_model(self, tmp_path, seq_reg):
        from eigenkit.qml import load_model

        X, y = seq_reg
        model = QuantumReservoirRegressor(_reg_blocks()).fit(X, y)
        model.save(tmp_path / "qrc")
        loaded = load_model(tmp_path / "qrc")
        assert isinstance(loaded, QuantumReservoirRegressor)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X), rtol=1e-6, atol=1e-6
        )
