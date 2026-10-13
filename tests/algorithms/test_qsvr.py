import pytest
from sklearn.base import RegressorMixin, clone


def test_qsvr_predict_shape(reg_data):
    from eigenkit.qml import QSVR

    X_tr, y_tr, X_te, _ = reg_data
    model = QSVR(feature_map="Angle", reps=1)
    model.fit(X_tr, y_tr)
    assert model.predict(X_te).shape == (len(X_te),)


def test_qsvr_is_regressor():
    from eigenkit.qml import QSVR

    assert isinstance(QSVR(), RegressorMixin)


def test_qsvr_score(reg_data):
    from eigenkit.qml import QSVR

    X_tr, y_tr, X_te, y_te = reg_data
    model = QSVR(feature_map="Angle", reps=1)
    model.fit(X_tr, y_tr)
    assert isinstance(model.score(X_te, y_te), float)


def test_qsvr_clone():
    from eigenkit.qml import QSVR

    cloned = clone(QSVR(feature_map="AngleCNOT", C=2.0, epsilon=0.2))
    assert cloned.epsilon == 0.2


@pytest.mark.pennylane
def test_qsvr_grid_search_pipeline(reg_data):
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from eigenkit.qml import QSVR

    X_tr, y_tr, X_te, _ = reg_data
    search = GridSearchCV(
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("qsvr", QSVR(feature_map="Angle", reps=1, provider="pennylane")),
            ]
        ),
        param_grid={"qsvr__C": [0.5, 1.0]},
        cv=2,
        scoring="neg_mean_squared_error",
    )
    search.fit(X_tr, y_tr)
    assert search.predict(X_te).shape == (len(X_te),)
