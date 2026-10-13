"""The fitted-attribute contract, and a checkpoint that can be read back.

Three properties an estimator shares with its siblings:

  * `save()` records the width of the head actually built, not the number of
    labels observed, so declaring `n_classes` larger than the label count
    still produces a checkpoint that `load()` can read.
  * A fitted classifier sets `classes_` and `n_features_in_`, as
    GenericClassifier does, so wrappers such as the time-series estimators
    can rely on both without a fallback for their absence.
  * `fit()` leaves the model in eval mode; in training mode `model_(x)` would
    disagree with `predict()` for any network holding dropout or normalisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigenkit import HQNC, HQNR


def _data(n_features=4, labels=("a", "b")):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, n_features))
    y = np.array(list(labels) * (20 // len(labels)))
    return X, y


class TestSaveLoadRoundTrip:
    @pytest.mark.parametrize("n_classes", [None, 2, 5])
    def test_a_checkpoint_can_be_read_back(self, tmp_path, n_classes):
        X, y = _data()
        model = HQNC(
            in_features=4, n_qubits=2, depth=1, epochs=1, n_classes=n_classes
        ).fit(X, y)
        path = tmp_path / "m.pt"
        model.save(str(path))
        restored = HQNC.load(str(path))
        np.testing.assert_allclose(
            restored.predict_proba(X), model.predict_proba(X), atol=1e-6
        )

    def test_a_declared_width_wider_than_the_labels_survives(self, tmp_path):
        """A head wider than the label count must round-trip: a checkpoint that
        recorded the label count instead would fail to load with a state-dict
        size mismatch."""
        X, y = _data()
        model = HQNC(in_features=4, n_qubits=2, depth=1, epochs=1, n_classes=5).fit(X, y)
        assert model.predict_proba(X).shape[1] == 5
        path = tmp_path / "m.pt"
        model.save(str(path))
        assert HQNC.load(str(path)).predict_proba(X).shape[1] == 5


class TestFittedAttributes:
    def test_classifier_exposes_classes(self):
        X, y = _data()
        model = HQNC(in_features=4, n_qubits=2, depth=1, epochs=1).fit(X, y)
        np.testing.assert_array_equal(model.classes_, np.array(["a", "b"]))

    @pytest.mark.parametrize("cls,regression", [(HQNC, False), (HQNR, True)])
    def test_n_features_in_is_recorded(self, cls, regression):
        X, y = _data()
        target = X[:, 0] if regression else y
        model = cls(in_features=4, n_qubits=2, depth=1, epochs=1).fit(X, target)
        assert model.n_features_in_ == 4

    def test_a_wrong_feature_count_names_the_estimator(self):
        X, y = _data(n_features=9)
        with pytest.raises(ValueError, match="in_features=4"):
            HQNC(in_features=4, n_qubits=2, depth=1, epochs=1).fit(X, y)


class TestModelIsReadyToPredict:
    @pytest.mark.parametrize("cls,regression", [(HQNC, False), (HQNR, True)])
    def test_fit_leaves_the_model_in_eval_mode(self, cls, regression):
        X, y = _data()
        target = X[:, 0] if regression else y
        model = cls(in_features=4, n_qubits=2, depth=1, epochs=1).fit(X, target)
        assert not model.model_.training

    def test_calling_model_directly_agrees_with_predict(self):
        import torch

        X, y = _data()
        model = HQNC(in_features=4, n_qubits=2, depth=1, epochs=1).fit(X, y)
        direct = model.model_(torch.as_tensor(X, dtype=torch.float32))
        viaapi = model.predict_proba(X)
        assert direct.shape[0] == viaapi.shape[0]
