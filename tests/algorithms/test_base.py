import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, clone


def test_imports():
    pass


def test_provider_string_stored_numpy():
    from eigenkit.qml import BaseQuantumClassifier

    class Clf(BaseQuantumClassifier):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    assert Clf(provider="numpy").provider == "numpy"


def test_provider_string_stored_pennylane():
    from eigenkit.qml import BaseQuantumClassifier

    class Clf(BaseQuantumClassifier):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    assert Clf(provider="pennylane").provider == "pennylane"


def test_classifier_mixin():
    from eigenkit.qml import BaseQuantumClassifier

    class Clf(BaseQuantumClassifier):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    assert isinstance(Clf(), ClassifierMixin)


def test_regressor_mixin():
    from eigenkit.qml import BaseQuantumRegressor

    class Reg(BaseQuantumRegressor):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    assert isinstance(Reg(), RegressorMixin)


def test_clone_preserves_params():
    from eigenkit.qml import BaseQuantumClassifier

    class Clf(BaseQuantumClassifier):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    cloned = clone(Clf(provider="numpy", random_state=99))
    assert cloned.provider == "numpy"
    assert cloned.random_state == 99
