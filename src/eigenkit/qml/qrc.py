"""Quantum reservoir computing (QRC) estimators, compatible with scikit-learn.

A reservoir is a circuit with fixed random weights, which are never trained.
These estimators sample the weights once (with the seed `random_state`), run the
circuit over the input sequences with
[`reservoir.run_reservoir()`][eigenkit.reservoir.run_reservoir], extract classical
features, and train only a linear readout. This split into a fixed reservoir and
a linear readout is the standard form of QRC. So these estimators belong with the
scikit-learn estimators (next to `QSVC` and `QSVR`), and not in the PyTorch
package.

They are thin wrappers around the functions of the `reservoir` package. For full
control (custom processing of the features, custom readouts), call
`run_reservoir` and the feature extractors directly.

Data shapes: `X` has shape `(n_sequences, n_timesteps, n_features)`, and `y` has
shape `(n_sequences, n_timesteps)`, with one target per timestep (sequence to
sequence). `washout` drops the first timesteps from the fit of the readout, so
that the reservoir can forget its initial state. Predictions are still made for
every timestep.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from sklearn.utils.validation import check_is_fitted

from eigenkit.core.blocks import assemble
from eigenkit.core.circuit import Circuit, ProviderSpec
from eigenkit.core.observables import all_z
from eigenkit.reservoir import (
    Carry,
    LeakyCarry,
    LinearReadout,
    Stateless,
    amplitude_features,
    expectation_features,
    pauli_z_features,
    probability_features,
    run_reservoir,
)

from ._base import BaseQuantumClassifier, BaseQuantumRegressor
from ._registry import register_estimator


# A named extractor ("z" / "probabilities" / "amplitudes"), a Pauli/Hamiltonian
# string, or a sequence of observable specs (strings / Observable / PauliExpr).
FeaturesSpec = Union[str, Sequence]

_NAMED_FEATURES = {
    "z": pauli_z_features,
    "probabilities": probability_features,
    "amplitudes": amplitude_features,
}


class _ReservoirMixin:
    """Shared reservoir forward pass for the regressor and classifier.

    Subclasses provide the constructor attributes ``blocks``, ``state_policy``,
    ``window``, ``leak``, ``features``, ``washout``, ``weight_scale`` (plus the
    inherited ``provider`` / ``random_state``).
    """

    def _make_policy(self):
        if self.state_policy == "stateless":
            return Stateless(window=self.window)
        if self.state_policy == "carry":
            return Carry()
        if self.state_policy == "leaky":
            return LeakyCarry(alpha=self.leak)
        raise ValueError(
            f"state_policy must be 'stateless', 'carry', or 'leaky', "
            f"got {self.state_policy!r}"
        )

    def _build_reservoir(self) -> None:
        """Assemble the circuit and sample the frozen random reservoir weights."""
        spec = assemble(self.blocks)
        self._circuit = Circuit(spec, all_z(spec.n_qubits), self.provider, "numpy")
        self._policy = self._make_policy()
        rng = np.random.default_rng(self.random_state)
        self._weights = rng.standard_normal(spec.n_weights) * self.weight_scale

    def _features(self, X: np.ndarray) -> np.ndarray:
        """Drive the reservoir over ``X`` and extract features: ``(B, T, d)``."""
        # `_circuit` is built by `fit`, so without this check reaching here first
        # would raise `AttributeError: no attribute '_circuit'` -- a traceback
        # into a private name rather than the NotFittedError every scikit-learn
        # estimator raises for this.
        check_is_fitted(self, "_circuit")
        traj = run_reservoir(self._circuit, self._policy, X, self._weights)
        spec = self.features
        if isinstance(spec, str) and spec in _NAMED_FEATURES:
            return _NAMED_FEATURES[spec](traj)
        return expectation_features(traj, spec)  # Hamiltonian string(s) / Observable(s)

    @staticmethod
    def _check_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3-D (n_sequences, n_timesteps, n_features), got ndim={X.ndim}"
            )
        return X

    def _drop_washout(self, feats: np.ndarray, y: np.ndarray):
        """Flatten the post-washout (sequence, timestep) grid into rows.

        Both validations live here, next to the slice they protect, so the two
        subclasses share one copy. Only `fit` reads `washout` --
        prediction runs the whole sequence -- so this is every path a bad value
        can reach.
        """
        B, T, d = feats.shape
        if self.washout < 0:
            # A negative value would index from the end, so `washout=-2` would
            # train on only the last two timesteps instead of dropping any.
            raise ValueError(
                f"washout must be >= 0; got {self.washout}. It is how many "
                f"leading timesteps to discard before fitting the readout."
            )
        if self.washout >= T:
            raise ValueError(f"washout={self.washout} >= n_timesteps={T}")
        rows = feats[:, self.washout :, :].reshape(-1, d)
        targets = np.asarray(y)[:, self.washout :]
        return rows, targets


@register_estimator
class QuantumReservoirRegressor(_ReservoirMixin, BaseQuantumRegressor):
    """Quantum reservoir regressor: a fixed random reservoir and a ridge readout.

    Args:
        blocks: The architecture of the reservoir circuit, a list of blocks that
            is assembled once. Its `n_inputs` must match the policy: `F` for the
            policies that carry the state, and `window * F` for
            `state_policy="stateless"`, where `F` is the number of features per
            timestep.
        state_policy: `"carry"` (the unitary of the circuit carries the state
            from one step to the next), `"stateless"` (each step starts from
            `|0…0⟩` and encodes a sliding window of inputs), or `"leaky"` (the
            carried state is mixed with a reset state and renormalised).
        window: The length of the sliding window, for `state_policy="stateless"`.
        leak: The mixing coefficient `alpha`, for `state_policy="leaky"`.
        features: The features read from the reservoir. It is `"z"` (the value
            of `⟨Z⟩` on each qubit, the default), `"probabilities"`,
            `"amplitudes"`, a Pauli or Hamiltonian string (for example
            `"0.5 Z0 X1 - 0.3 Y0"`), or a list of such strings, to read out any
            observables.
        ridge: The strength of the L2 penalty of the linear readout, which is
            solved in closed form.
        washout: The number of timesteps dropped from the fit of the readout.
            They are not dropped from the predictions.
        weight_scale: The standard deviation of the sampled reservoir weights.
        provider: The backend, which must give access to the statevector.
            Default: `"numpy"`.
        random_state: The seed of the fixed reservoir weights.
    """

    def __init__(
        self,
        blocks: list,
        state_policy: str = "carry",
        window: int = 1,
        leak: float = 1.0,
        features: FeaturesSpec = "z",
        ridge: float = 1e-6,
        washout: int = 0,
        weight_scale: float = 1.0,
        provider: ProviderSpec = "numpy",
        random_state: int | None = 42,
    ):
        super().__init__(provider=provider, random_state=random_state)
        self.blocks = blocks
        self.state_policy = state_policy
        self.window = window
        self.leak = leak
        self.features = features
        self.ridge = ridge
        self.washout = washout
        self.weight_scale = weight_scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumReservoirRegressor":
        """Fit the estimator on `X` and `y`.

        Args:
            X (np.ndarray): The sequences, of shape
                `(n_sequences, n_timesteps, n_features)`. A 2-D array is rejected.
            y (np.ndarray): One target for each timestep, of shape
                `(n_sequences, n_timesteps)`, or
                `(n_sequences, n_timesteps, n_outputs)` for several outputs.

        Returns:
            This estimator, so that calls can be chained.
        """
        X = self._check_X(X)
        self._build_reservoir()
        rows, targets = self._drop_washout(
            self._features(X), np.asarray(y, dtype=np.float64)
        )
        targets = (
            targets.reshape(-1)
            if targets.ndim == 2
            else targets.reshape(-1, targets.shape[-1])
        )
        self._readout = LinearReadout(ridge=self.ridge).fit(rows, targets)
        self.n_features_in_ = X.shape[-1]
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Return True when the readout has been fitted."""
        return hasattr(self, "_readout")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a value for every timestep of every sequence in `X`.

        Args:
            X (np.ndarray): The sequences, of shape
                `(n_sequences, n_timesteps, n_features)`. A 2-D array is rejected.

        Returns:
            The predictions, of shape `(n_sequences, n_timesteps)`, or
                `(n_sequences, n_timesteps, n_outputs)` when the targets had
                several outputs.
        """
        X = self._check_X(X)
        feats = self._features(X)
        B, T, d = feats.shape
        preds = self._readout.predict(feats.reshape(B * T, d))
        return preds.reshape(B, T) if preds.ndim == 1 else preds.reshape(B, T, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score the estimator on `X` against `y`.

        Args:
            X (np.ndarray): The sequences, of shape
                `(n_sequences, n_timesteps, n_features)`. A 2-D array is rejected.
            y (np.ndarray): One target for each timestep, of shape
                `(n_sequences, n_timesteps)`.

        Returns:
            The coefficient of determination `R²` of the predictions, over every
                timestep of every sequence.
        """
        from sklearn.metrics import r2_score

        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        if preds.ndim == 3:
            return float(
                r2_score(
                    y.reshape(-1, preds.shape[-1]), preds.reshape(-1, preds.shape[-1])
                )
            )
        return float(r2_score(y.reshape(-1), preds.reshape(-1)))

    def _collect_state(self) -> dict:
        return {
            "weights": self._weights,
            "readout_coef": self._readout.coef_,
            "readout_single_output": bool(self._readout._single_output),
            "n_features_in": int(self.n_features_in_),
        }

    def _restore_state(self, state: dict) -> None:
        self._build_reservoir()
        self._weights = np.asarray(state["weights"], dtype=np.float64)
        readout = LinearReadout(ridge=self.ridge)
        readout.coef_ = np.asarray(state["readout_coef"], dtype=np.float64)
        readout._single_output = bool(state["readout_single_output"])
        self._readout = readout
        self.n_features_in_ = int(state["n_features_in"])


@register_estimator
class QuantumReservoirClassifier(_ReservoirMixin, BaseQuantumClassifier):
    """Quantum reservoir classifier: a fixed random reservoir and a logistic readout.

    It uses the same reservoir as
    [`QuantumReservoirRegressor`][eigenkit.qml.qrc.QuantumReservoirRegressor].
    The readout is a scikit-learn `LogisticRegression` on the features of each
    timestep. `y` holds integer or label targets, of shape
    `(n_sequences, n_timesteps)`.

    Args:
        blocks: The architecture of the reservoir circuit, a list of blocks that
            is assembled once. Its `n_inputs` must match the policy: `F` for the
            policies that carry the state, and `window * F` for
            `state_policy="stateless"`, where `F` is the number of features per
            timestep.
        state_policy: `"carry"` (the unitary of the circuit carries the state
            from one step to the next), `"stateless"` (each step starts from
            `|0…0⟩` and encodes a sliding window of inputs), or `"leaky"` (the
            carried state is mixed with a reset state and renormalised).
        window: The length of the sliding window, for `state_policy="stateless"`.
        leak: The mixing coefficient `alpha`, for `state_policy="leaky"`.
        features: The features read from the reservoir. It is `"z"` (the value
            of `⟨Z⟩` on each qubit, the default), `"probabilities"`,
            `"amplitudes"`, a Pauli or Hamiltonian string (for example
            `"0.5 Z0 X1 - 0.3 Y0"`), or a list of such strings, to read out any
            observables.
        C: The inverse regularisation strength of the logistic readout, passed
            to scikit-learn. Smaller values regularise more. This is the only
            argument that the regressor does not have: there, the readout is a
            ridge regression controlled by `ridge`.
        washout: The number of timesteps dropped from the fit of the readout.
            They are not dropped from the predictions.
        weight_scale: The standard deviation of the sampled reservoir weights.
        provider: The backend, which must give access to the statevector.
            Default: `"numpy"`.
        random_state: The seed of the fixed reservoir weights.
    """

    def __init__(
        self,
        blocks: list,
        state_policy: str = "carry",
        window: int = 1,
        leak: float = 1.0,
        features: FeaturesSpec = "z",
        C: float = 1.0,
        washout: int = 0,
        weight_scale: float = 1.0,
        provider: ProviderSpec = "numpy",
        random_state: int | None = 42,
    ):
        super().__init__(provider=provider, random_state=random_state)
        self.blocks = blocks
        self.state_policy = state_policy
        self.window = window
        self.leak = leak
        self.features = features
        self.C = C
        self.washout = washout
        self.weight_scale = weight_scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumReservoirClassifier":
        """Fit the estimator on `X` and `y`.

        Args:
            X (np.ndarray): The sequences, of shape
                `(n_sequences, n_timesteps, n_features)`. A 2-D array is rejected.
            y (np.ndarray): One target for each timestep, of shape
                `(n_sequences, n_timesteps)`.

        Returns:
            This estimator, so that calls can be chained.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        X = self._check_X(X)
        self._build_reservoir()
        rows, targets = self._drop_washout(self._features(X), np.asarray(y))
        self._label_encoder = LabelEncoder()
        encoded = self._label_encoder.fit_transform(targets.reshape(-1))
        self._clf = LogisticRegression(C=self.C, max_iter=1000).fit(rows, encoded)
        self.classes_ = self._label_encoder.classes_
        self.n_features_in_ = X.shape[-1]
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Return True when the readout has been fitted."""
        return hasattr(self, "_clf")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a label for every timestep of every sequence in `X`.

        Args:
            X (np.ndarray): The sequences, of shape
                `(n_sequences, n_timesteps, n_features)`. A 2-D array is rejected.

        Returns:
            The predicted labels, of shape `(n_sequences, n_timesteps)`.
        """
        X = self._check_X(X)
        feats = self._features(X)
        B, T, d = feats.shape
        encoded = self._clf.predict(feats.reshape(B * T, d))
        return self._label_encoder.inverse_transform(encoded).reshape(B, T)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the class probabilities for every timestep of every sequence in `X`.

        Args:
            X (np.ndarray): The sequences, of shape
                `(n_sequences, n_timesteps, n_features)`. A 2-D array is rejected.

        Returns:
            The probabilities, of shape `(n_sequences, n_timesteps, n_classes)`,
                with the last axis in the order of `classes_`.
        """
        X = self._check_X(X)
        feats = self._features(X)
        B, T, d = feats.shape
        proba = self._clf.predict_proba(feats.reshape(B * T, d))
        return proba.reshape(B, T, proba.shape[-1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score the estimator on `X` against `y`.

        Args:
            X (np.ndarray): The sequences, of shape
                `(n_sequences, n_timesteps, n_features)`. A 2-D array is rejected.
            y (np.ndarray): One target for each timestep, of shape
                `(n_sequences, n_timesteps)`.

        Returns:
            The accuracy of the predictions, over every timestep of every sequence.
        """
        from sklearn.metrics import accuracy_score

        preds = self.predict(X)
        return float(accuracy_score(np.asarray(y).reshape(-1), preds.reshape(-1)))

    def _collect_state(self) -> dict:
        import io

        import joblib

        buf = io.BytesIO()
        joblib.dump(self._clf, buf)
        return {
            "weights": self._weights,
            "clf_joblib": np.frombuffer(buf.getvalue(), dtype=np.uint8).copy(),
            "label_classes": np.asarray(self._label_encoder.classes_),
            "n_features_in": int(self.n_features_in_),
        }

    def _restore_state(self, state: dict) -> None:
        import io

        import joblib
        from sklearn.preprocessing import LabelEncoder

        self._build_reservoir()
        self._weights = np.asarray(state["weights"], dtype=np.float64)
        self._clf = joblib.load(io.BytesIO(state["clf_joblib"].tobytes()))
        le = LabelEncoder()
        le.classes_ = np.asarray(state["label_classes"])
        self._label_encoder = le
        self.classes_ = le.classes_
        self.n_features_in_ = int(state["n_features_in"])
