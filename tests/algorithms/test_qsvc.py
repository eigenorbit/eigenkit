import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone


def test_qsvc_fit_predict(clf_data):
    from eigenkit.qml import QSVC

    X_tr, y_tr, X_te, _ = clf_data
    model = QSVC(feature_map="Angle", reps=1)
    model.fit(X_tr, y_tr)
    assert model.predict(X_te).shape == (len(X_te),)


def test_qsvc_classes(clf_data):
    from eigenkit.qml import QSVC

    X_tr, y_tr, X_te, _ = clf_data
    model = QSVC(feature_map="Angle", reps=1)
    model.fit(X_tr, y_tr)
    assert set(model.predict(X_te)).issubset({0, 1})


def test_qsvc_is_classifier():
    from eigenkit.qml import QSVC

    assert isinstance(QSVC(), ClassifierMixin)


def test_qsvc_score(clf_data):
    from eigenkit.qml import QSVC

    X_tr, y_tr, X_te, y_te = clf_data
    model = QSVC(feature_map="Angle", reps=1)
    model.fit(X_tr, y_tr)
    assert 0.0 <= model.score(X_te, y_te) <= 1.0


def test_qsvc_clone():
    from eigenkit.qml import QSVC

    cloned = clone(QSVC(feature_map="AngleCNOT", reps=2, C=2.0))
    assert cloned.feature_map == "AngleCNOT" and cloned.C == 2.0


@pytest.mark.pennylane
def test_qsvc_grid_search_pipeline(clf_data):
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from eigenkit.qml import QSVC

    X_tr, y_tr, X_te, _ = clf_data
    search = GridSearchCV(
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("qsvc", QSVC(feature_map="Angle", reps=1, provider="pennylane")),
            ]
        ),
        param_grid={"qsvc__C": [0.5, 1.0]},
        cv=2,
        scoring="accuracy",
    )
    search.fit(X_tr, y_tr)
    assert search.predict(X_te).shape == (len(X_te),)


def test_qsvc_predict_proba(clf_data):
    from eigenkit.qml import QSVC

    X_tr, y_tr, X_te, _ = clf_data
    model = QSVC(feature_map="Angle", reps=1, probability=True)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), len(np.unique(y_tr)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_qsvc_predict_proba_with_fewer_than_five_samples_per_class():
    from eigenkit.qml import QSVC

    X = np.random.default_rng(0).uniform(0, np.pi, size=(8, 2))
    y = np.array([0, 1] * 4)
    proba = QSVC(feature_map="Angle", reps=1, probability=True).fit(X, y).predict_proba(X)
    assert proba.shape == (8, 2)


def test_qsvc_predict_proba_requires_probability_flag(clf_data):
    import pytest

    from eigenkit.qml import QSVC

    X_tr, y_tr, X_te, _ = clf_data
    model = QSVC(feature_map="Angle", reps=1, probability=False)
    model.fit(X_tr, y_tr)
    with pytest.raises(ValueError, match="probability=False"):
        model.predict_proba(X_te)


# ---- kernel=, scale=, and the sklearn contract around them ----


def _ry_cz_map(qc, x, w):
    """A small custom, trainable map: RY(w) RX(x) per qubit, then a CZ ring."""
    n = len(x)
    for q in range(n):
        qc.ry(w[0], qubit=q)
        qc.rx(x[q], qubit=q)
    for q in range(n - 1):
        qc.mcz(q, q + 1)


def test_qsvc_with_a_builtin_kernel_matches_feature_map(clf_data):
    from eigenkit.qml import QSVC, QuantumKernel

    X_tr, y_tr, X_te, _ = clf_data
    by_name = QSVC(feature_map="AngleCNOT", reps=2).fit(X_tr, y_tr)
    by_kernel = QSVC(
        kernel=QuantumKernel(n_features=4, feature_map="AngleCNOT", reps=2)
    ).fit(X_tr, y_tr)
    np.testing.assert_array_equal(by_kernel.predict(X_te), by_name.predict(X_te))
    np.testing.assert_allclose(
        by_kernel.decision_function(X_te), by_name.decision_function(X_te), atol=1e-10
    )
    assert by_kernel.kernel_ is by_kernel.kernel


def test_qsvc_custom_kernel_in_grid_search(clf_data):
    from sklearn.model_selection import GridSearchCV

    from eigenkit.qml import QSVC, QuantumKernel

    X_tr, y_tr, X_te, _ = clf_data
    kernel = QuantumKernel(
        n_features=4, feature_map=_ry_cz_map, n_weights=1, weights=[0.2]
    )
    search = GridSearchCV(
        QSVC(kernel=kernel),
        param_grid={"C": [0.5, 1.0], "scale": [0.5, 1.0]},
        cv=2,
    ).fit(X_tr, y_tr)
    assert search.predict(X_te).shape == (len(X_te),)
    np.testing.assert_allclose(search.best_estimator_.kernel_.weights, [0.2])


def test_qsvc_clone_carries_the_kernel():
    from eigenkit.qml import QSVC, QuantumKernel

    kernel = QuantumKernel(
        n_features=4, feature_map=_ry_cz_map, n_weights=1, weights=[0.2]
    )
    cloned = clone(QSVC(kernel=kernel, scale=0.5))
    assert cloned.kernel is not kernel
    assert cloned.scale == 0.5
    np.testing.assert_allclose(cloned.kernel.weights, [0.2])


def test_qsvc_scale_is_the_input_bandwidth(clf_data):
    from eigenkit.qml import QSVC

    X_tr, y_tr, X_te, _ = clf_data
    X_tr, X_te = X_tr.astype(np.float64), X_te.astype(np.float64)
    scaled = QSVC(feature_map="AngleCNOT", reps=2, scale=0.3).fit(X_tr, y_tr)
    prescaled = QSVC(feature_map="AngleCNOT", reps=2).fit(0.3 * X_tr, y_tr)
    np.testing.assert_allclose(
        scaled.decision_function(X_te),
        prescaled.decision_function(0.3 * X_te),
        atol=1e-10,
    )


def test_qsvc_kernel_overrides_warn(clf_data):
    import pytest

    from eigenkit.qml import QSVC, QuantumKernel

    X_tr, y_tr, _, _ = clf_data
    with pytest.warns(UserWarning, match="ignores feature_map"):
        QSVC(kernel=QuantumKernel(n_features=4), feature_map="AngleCNOT").fit(X_tr, y_tr)


def test_qsvc_kernel_feature_count_is_checked(clf_data):
    import pytest

    from eigenkit.qml import QSVC, QuantumKernel

    X_tr, y_tr, _, _ = clf_data
    with pytest.raises(ValueError, match="n_features=3"):
        QSVC(kernel=QuantumKernel(n_features=3)).fit(X_tr, y_tr)


def test_qsvc_custom_kernel_save_load_round_trip(clf_data, tmp_path):
    from eigenkit.qml import QSVC, QuantumKernel

    X_tr, y_tr, X_te, _ = clf_data
    kernel = QuantumKernel(
        n_features=4, feature_map=_ry_cz_map, n_weights=1, weights=[0.2]
    )
    model = QSVC(kernel=kernel, scale=0.7, probability=True).fit(X_tr, y_tr)
    path = tmp_path / "custom.eigenkit"
    model.save(path)
    restored = QSVC.load(path)
    np.testing.assert_array_equal(restored.predict(X_te), model.predict(X_te))
    np.testing.assert_allclose(restored.predict_proba(X_te), model.predict_proba(X_te))
    np.testing.assert_allclose(restored.kernel.weights, [0.2])


def test_qsvc_fit_raises_no_future_warning(clf_data):
    import warnings

    from eigenkit.qml import QSVC

    X_tr, y_tr, X_te, _ = clf_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        QSVC().fit(X_tr, y_tr).predict(X_te)
        QSVC(probability=True).fit(X_tr, y_tr).predict_proba(X_te)


def test_qsvr_with_a_custom_kernel(reg_data):
    from eigenkit.qml import QSVR, QuantumKernel

    X_tr, y_tr, X_te, _ = reg_data
    kernel = QuantumKernel(n_features=4, feature_map=_ry_cz_map, n_weights=1)
    model = QSVR(kernel=kernel, scale=0.5).fit(X_tr, y_tr)
    assert model.predict(X_te).shape == (len(X_te),)
    assert model.kernel_ is kernel
