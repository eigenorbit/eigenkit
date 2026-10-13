import numpy as np
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from eigenkit.core.circuit import ProviderSpec

from ._base import BaseQuantumRegressor
from ._kernels import QuantumKernel, _estimator_kernel
from ._registry import register_estimator


@register_estimator
class QSVR(BaseQuantumRegressor):
    """Quantum support vector regressor (QSVR).

    It is the regression counterpart of [`QSVC`][eigenkit.qml.QSVC]. It uses
    `SVR(kernel="precomputed")` from scikit-learn on the quantum kernel matrix.

    Args:
        feature_map: The feature map. It is `"Angle"` (angle encoding into a
            product state) or `"AngleCNOT"` (angle encoding
            with ladders of controlled-NOT (CNOT) gates). Default: `"Angle"`. It
            is ignored when `kernel` is given.
        reps: The number of repetitions of the feature map. Default: `1`. It is
            ignored when `kernel` is given.
        C: The regularisation parameter of the support vector regressor. Default:
            `1.0`.
        epsilon: The width of the epsilon-insensitive tube. Default: `0.1`.
        provider: The backend. It takes the same options as
            `Circuit(provider=...)`. Default: `"numpy"`. It is ignored when
            `kernel` is given.
        random_state: The seed.
        kernel: A [`QuantumKernel`][eigenkit.qml.QuantumKernel] to use instead
            of building one from `feature_map`, `reps` and `provider`. Default:
            `None`.
        scale: The bandwidth. The inputs are multiplied by it before they reach
            the kernel. Default: `1.0`.

    Example:
        ```python
        import numpy as np
        from eigenkit import QSVR

        rng = np.random.default_rng(0)
        X_train, X_test = rng.uniform(0, np.pi, (8, 2)), rng.uniform(0, np.pi, (3, 2))
        y_train = np.sin(X_train).sum(axis=1)

        model = QSVR(feature_map="Angle", reps=1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        ```
    """

    def __init__(
        self,
        feature_map: str = "Angle",
        reps: int = 1,
        C: float = 1.0,
        epsilon: float = 0.1,
        provider: ProviderSpec = "numpy",
        random_state: int | None = 42,
        kernel: QuantumKernel | None = None,
        scale: float = 1.0,
    ):
        super().__init__(provider=provider, random_state=random_state)
        self.feature_map = feature_map
        self.reps = reps
        self.C = C
        self.epsilon = epsilon
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVR":
        """Fit the regressor on training data.

        Args:
            X: The samples, of shape `(n_samples, n_features)`.
            y: The float targets, of shape `(n_samples,)`.

        Returns:
            This estimator, so that calls can be chained.
        """
        X = np.asarray(X, dtype=np.float64)
        self.kernel_ = _estimator_kernel(self, X.shape[1])
        K_train = self.kernel_.compute_kernel_matrix(X * self.scale)
        svr = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
        svr.fit(K_train, y)
        self._svr = svr
        self._X_train = X
        self.n_features_in_ = X.shape[1]
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Return True when the wrapped scikit-learn `SVR` has been fitted."""
        return hasattr(self, "_svr")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous targets for `X`.

        Args:
            X (np.ndarray): The samples, of shape `(n_samples, n_features)`.

        Returns:
            One prediction for each sample.
        """
        check_is_fitted(self, "_svr")
        X = self._validate_features(X)
        K_test = self.kernel_.compute_kernel_matrix(
            X * self.scale, self._X_train * self.scale
        )
        return self._svr.predict(K_test)

    def _collect_state(self) -> dict:
        import io

        import joblib  # sklearn guarantees joblib compat for its own estimators

        buf = io.BytesIO()
        joblib.dump(self._svr, buf)
        return {
            "svr_joblib": np.frombuffer(buf.getvalue(), dtype=np.uint8).copy(),
            "X_train": self._X_train,
        }

    def _restore_state(self, state: dict) -> None:
        import io

        import joblib  # sklearn guarantees joblib compat for its own estimators

        self._svr = joblib.load(io.BytesIO(state["svr_joblib"].tobytes()))
        self._X_train = state["X_train"]
        self.n_features_in_ = self._X_train.shape[1]
        self.kernel_ = _estimator_kernel(self, self._X_train.shape[1], warn=False)
