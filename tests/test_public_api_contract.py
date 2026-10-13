"""What an outside user meets on day one.

Cross-cutting checks on the public surface, rather than any one module: the
scikit-learn contract the README advertises, the errors a new user actually
triggers, and the promise that importing the package stays cheap.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import is_classifier, is_regressor

import eigenkit as ek
import eigenkit.torch.estimators  # noqa: F401 — populates the registry
from eigenkit.qml._registry import _REGISTRY


def _instantiate(name):
    """Smallest workable instance of a registered estimator, by name."""
    cls = _REGISTRY[name]
    if name.startswith("QSV"):
        return cls(feature_map="Angle", reps=1)
    if name.startswith("Custom"):
        from eigenkit.configs import Encoding, Variational

        return cls(
            blocks=[Encoding(in_features=3), Variational(depth=1)],
            in_features=3,
            epochs=1,
        )
    if name.startswith("Generic"):
        from torch import nn

        out = 2 if name.endswith("Classifier") else 1
        return cls(model=nn.Sequential(nn.Linear(3, out)), epochs=1)
    return cls(in_features=3, n_qubits=2, depth=1, epochs=1)


def _classifier_data(n_features=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, n_features))
    return X, (X[:, 0] > 0).astype(int)


class TestScikitLearnRecognisesTheEstimators:
    """`is_classifier` / `is_regressor` decide how scikit-learn treats an
    estimator. With the mixins after the base in the MRO, `BaseEstimator`'s
    `__sklearn_tags__` would shadow theirs and both would answer False for every
    estimator here -- so `cross_val_score` would silently use KFold instead of
    StratifiedKFold, and the ensemble meta-estimators would refuse them.
    """

    @pytest.mark.parametrize(
        "name", ["QSVC", "HQNC", "CustomQuantumClassifier", "GenericClassifier"]
    )
    def test_classifiers_are_classifiers(self, name):
        # An instance, not the class: sklearn reads tags off `self`.
        assert is_classifier(_instantiate(name))

    @pytest.mark.parametrize(
        "name", ["QSVR", "HQNR", "CustomQuantumRegressor", "GenericRegressor"]
    )
    def test_regressors_are_regressors(self, name):
        assert is_regressor(_instantiate(name))

    def test_a_classifier_works_inside_a_voting_ensemble(self):
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression

        from eigenkit.qml import QSVC

        X, y = _classifier_data(2)
        ensemble = VotingClassifier(
            [("q", QSVC(feature_map="Angle", reps=1)), ("l", LogisticRegression())]
        )
        assert ensemble.fit(X, y).predict(X).shape == (20,)

    def test_cross_val_score_stratifies(self):
        """On sorted labels an unstratified KFold produces single-class folds."""
        from sklearn.model_selection import cross_val_score

        from eigenkit.qml import QSVC

        X, y = _classifier_data(2)
        order = np.argsort(y)
        scores = cross_val_score(
            QSVC(feature_map="Angle", reps=1), X[order], y[order], cv=3
        )
        assert min(scores) > 0.5, f"folds look unstratified: {scores}"


class TestTheErrorsANewUserHits:
    def test_a_circuit_says_how_many_parameters_it_takes(self):
        from eigenkit.core import Circuit
        from eigenkit.core.blocks import (
            AngleEncoding,
            BasicVariational,
            assemble,
        )
        from eigenkit.core.observables import all_z

        spec = assemble([AngleEncoding(in_features=3), BasicVariational(depth=1)])
        circuit = Circuit(spec, all_z(3), "numpy", "numpy")
        weights = np.zeros(spec.n_weights)

        with pytest.raises(ValueError, match="takes 3 inputs, got 5"):
            circuit(weights, np.zeros(5))
        with pytest.raises(ValueError, match="takes 3 inputs, got 2"):
            circuit(weights, np.zeros(2))
        with pytest.raises(ValueError, match="takes 3 weights, got 2"):
            circuit(np.zeros(2), np.zeros(3))

    def test_the_torch_interface_says_it_too(self):
        """The torch path -- the one `TorchCircuit` and every layer take -- has
        the same guard as the numpy path; without it, a too-wide input would be
        truncated in silence and a too-narrow one reported as `IndexError`."""
        import torch

        from eigenkit.core import Circuit
        from eigenkit.core.blocks import (
            AngleEncoding,
            BasicVariational,
            assemble,
        )
        from eigenkit.core.observables import all_z

        spec = assemble([AngleEncoding(in_features=3), BasicVariational(depth=1)])
        circuit = Circuit(spec, all_z(3), "numpy", "torch")
        weights = torch.zeros(spec.n_weights, dtype=torch.float64)

        with pytest.raises(ValueError, match="takes 3 inputs, got 5"):
            circuit(weights, torch.zeros(4, 5, dtype=torch.float64))
        with pytest.raises(ValueError, match="takes 3 inputs, got 2"):
            circuit(weights, torch.zeros(4, 2, dtype=torch.float64))
        with pytest.raises(ValueError, match="takes 3 weights, got 2"):
            circuit(
                torch.zeros(2, dtype=torch.float64),
                torch.zeros(3, dtype=torch.float64),
            )

    def test_a_layer_says_how_wide_its_input_must_be(self):
        import torch

        from eigenkit import Basic

        layer = Basic(n_qubits=3, depth=1)
        with pytest.raises(ValueError, match="takes 3 inputs, got 5"):
            layer(torch.randn(4, 5, dtype=torch.float64))
        with pytest.raises(ValueError, match="takes 3 inputs, got 2"):
            layer(torch.randn(4, 2, dtype=torch.float64))
        assert layer(torch.randn(4, 3, dtype=torch.float64)).shape == (4, 3)

    @pytest.mark.parametrize("name", ["HQNC", "HQNR", "QSVC", "QSVR"])
    def test_predicting_with_the_wrong_width_names_the_estimator(self, name):
        X, y = _classifier_data(3)
        target = X[:, 0] if name.endswith("R") else y
        fitted = _instantiate(name).fit(X, target)
        with pytest.raises(ValueError, match=f"{name}.*3 features.*7"):
            fitted.predict(np.zeros((4, 7)))

    def test_an_unknown_provider_lists_the_known_ones(self):
        from eigenkit.core.provider import get

        with pytest.raises(KeyError, match="registered:"):
            get("not-a-backend")

    def test_an_opt_in_provider_says_how_to_register_it(self):
        """Asked of a fresh interpreter, not this one.

        The registry is process-global, so in-process the answer would depend on
        whether another test had registered cudaq -- and the runners where cudaq
        is installed, the ones with something to check, are exactly the runners
        where a fixture does. `tests/conftest.py` restores the registry per test;
        asking a subprocess means no session state can silence the check.
        """
        import subprocess
        import sys
        import textwrap

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    from eigenkit.core.provider import get

                    for name in ("cudaq",):
                        try:
                            get(name)
                        except KeyError as exc:
                            assert "register" in str(exc), (name, str(exc))
                        else:
                            raise AssertionError(
                                f"{name} resolved without a register() call"
                            )
                    """
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestImportStaysCheap:
    def test_importing_pulls_no_heavy_framework(self):
        import subprocess
        import sys
        import textwrap

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import sys
                    import eigenkit  # noqa: F401

                    heavy = sorted(
                        m
                        for m in ("torch", "pennylane", "qiskit", "sklearn", "braket")
                        if m in sys.modules
                    )
                    assert not heavy, f"import pulled: {heavy}"
                    """
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_every_exported_name_resolves(self):
        missing = [n for n in ek.__all__ if not hasattr(ek, n)]
        assert missing == []
