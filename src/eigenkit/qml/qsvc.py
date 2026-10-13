import numpy as np
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from eigenkit.core.circuit import ProviderSpec

from ._base import BaseQuantumClassifier
from ._kernels import QuantumKernel, _estimator_kernel
from ._registry import register_estimator


@register_estimator
class QSVC(BaseQuantumClassifier):
    """Quantum support vector classifier (QSVC).

    It computes a quantum kernel matrix from a fidelity circuit, and then uses
    `SVC(kernel="precomputed")` from scikit-learn.

    Args:
        feature_map: The feature map. It is `"Angle"` (angle encoding into a
            product state) or `"AngleCNOT"` (angle encoding
            with ladders of controlled-NOT (CNOT) gates). Default: `"Angle"`. It
            is ignored when `kernel` is given.
        reps: The number of repetitions of the feature map. Default: `1`. It is
            ignored when `kernel` is given.
        C: The regularisation parameter of the support vector machine. Default:
            `1.0`.
        probability: If True, the model gives Platt-scaled probabilities. If
            False (the default), `predict_proba` is unavailable. The calibration
            is a sigmoid fitted by cross-validation on the training kernel, as in
            `CalibratedClassifierCV(SVC(), ensemble=False)`. The predictions still
            come from the plain `SVC`.
        provider: The backend. It takes the same options as
            `Circuit(provider=...)`. Any backend with a statevector or a
            `zero_state_probability` method works, including Qiskit Runtime and
            IQM hardware. Default: `"numpy"`. It is ignored when `kernel` is given.
        random_state: The seed.
        kernel: A [`QuantumKernel`][eigenkit.qml.QuantumKernel] to use instead
            of building one from `feature_map`, `reps` and `provider`. It can be a
            custom kernel, a trained kernel or a hardware kernel. Default: `None`.
        scale: The bandwidth. The inputs are multiplied by it before they reach
            the kernel, so it can be tuned with `GridSearchCV`. Default: `1.0`.

    Example:
        ```python
        import numpy as np
        from eigenkit import QSVC, Encoding, QuantumKernel, Variational

        rng = np.random.default_rng(0)
        X_train, X_test = rng.uniform(0, np.pi, (8, 4)), rng.uniform(0, np.pi, (3, 4))
        y_train = np.array([0, 1] * 4)

        model = QSVC(feature_map="AngleCNOT", reps=2)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Any kernel works, including custom and trained ones.
        my_map = [Encoding(in_features=4), Variational(depth=1)]
        model = QSVC(kernel=QuantumKernel(n_features=4, feature_map=my_map))
        ```
    """

    def __init__(
        self,
        feature_map: str = "Angle",
        reps: int = 1,
        C: float = 1.0,
        probability: bool = False,
        provider: ProviderSpec = "numpy",
        random_state: int | None = 42,
        kernel: QuantumKernel | None = None,
        scale: float = 1.0,
    ):
        super().__init__(provider=provider, random_state=random_state)
        self.feature_map = feature_map
        self.reps = reps
        self.C = C
        self.probability = probability
        self.kernel = kernel
        self.scale = scale

    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        """Reject an X whose column count is not the one this was fitted on.

        The kernel is built from `X` against `self._X_train`, and the feature
        map reads only the first `n_features_in_` columns of each row -- so a
        wider X would be silently truncated and give confident predictions
        computed from part of the input.

        Args:
            X (np.ndarray): Samples to score.

        Returns:
            `X` as a 2-D float array.

        Raises:
            ValueError: If the column count differs from `n_features_in_`.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                f"{type(self).__name__} expects a 2-D X of shape "
                f"(n_samples, {self.n_features_in_}); got shape {X.shape}."
            )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{type(self).__name__} was fitted on {self.n_features_in_} "
                f"features; got {X.shape[1]}."
            )
        return X

    def _new_svc(self) -> SVC:
        # `probability` is never passed: scikit-learn 1.9 deprecates it, even
        # as False. Probabilities come from a separate calibration instead.
        return SVC(kernel="precomputed", C=self.C, random_state=self.random_state)

    def _test_kernel(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_features(X)
        return self.kernel_.compute_kernel_matrix(
            X * self.scale, self._X_train * self.scale
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVC":
        """Fit the classifier on training data.

        Args:
            X: The samples, of shape `(n_samples, n_features)`.
            y: The class labels, of shape `(n_samples,)`.

        Returns:
            This estimator, so that calls can be chained.
        """
        X = np.asarray(X, dtype=np.float64)
        self.kernel_ = _estimator_kernel(self, X.shape[1])
        K_train = self.kernel_.compute_kernel_matrix(X * self.scale)
        svc = self._new_svc()
        svc.fit(K_train, y)
        self._svc = svc
        self._calibrated = None
        if self.probability:
            from sklearn.calibration import CalibratedClassifierCV

            # Small QPU datasets can have fewer than 5 samples per class.
            cv = int(min(5, np.unique(y, return_counts=True)[1].min()))
            self._calibrated = CalibratedClassifierCV(
                self._new_svc(), method="sigmoid", cv=cv, ensemble=False
            ).fit(K_train, y)
        self._X_train = X
        self.classes_ = self._svc.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Return True when the wrapped scikit-learn `SVC` has been fitted."""
        return hasattr(self, "_svc")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels of `X`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            One prediction for each sample.
        """
        check_is_fitted(self)
        return self._svc.predict(self._test_kernel(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the signed distance of each sample to the separating hyperplane.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            An array of shape `(n_samples,)` for two classes, and
                `(n_samples, n_classes)` otherwise.
        """
        check_is_fitted(self)
        return self._svc.decision_function(self._test_kernel(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the probability of each class. It requires `probability=True`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            The probabilities, of shape `(n_samples, n_classes)`, with the columns
                in the order of `classes_`.

        Raises:
            ValueError: If the model was created with `probability=False`.
        """
        check_is_fitted(self)
        if not self.probability:
            raise ValueError(
                "predict_proba is unavailable because QSVC was initialized "
                "with probability=False. Recreate the model with "
                "probability=True."
            )
        return self._calibrated.predict_proba(self._test_kernel(X))

    def _collect_state(self) -> dict:
        import io

        import joblib  # sklearn guarantees joblib compat for its own estimators

        buf = io.BytesIO()
        joblib.dump(self._svc, buf)
        state = {
            "svc_joblib": np.frombuffer(buf.getvalue(), dtype=np.uint8).copy(),
            "X_train": self._X_train,
        }
        if self._calibrated is not None:
            buf = io.BytesIO()
            joblib.dump(self._calibrated, buf)
            state["calibrated_joblib"] = np.frombuffer(
                buf.getvalue(), dtype=np.uint8
            ).copy()
        return state

    def _restore_state(self, state: dict) -> None:
        import io

        import joblib  # sklearn guarantees joblib compat for its own estimators

        self._svc = joblib.load(io.BytesIO(state["svc_joblib"].tobytes()))
        self._calibrated = (
            joblib.load(io.BytesIO(state["calibrated_joblib"].tobytes()))
            if "calibrated_joblib" in state
            else None
        )
        self._X_train = state["X_train"]
        self.classes_ = self._svc.classes_
        self.n_features_in_ = self._X_train.shape[1]
        self.kernel_ = _estimator_kernel(self, self._X_train.shape[1], warn=False)
