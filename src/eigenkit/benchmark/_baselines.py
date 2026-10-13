"""Convenience factories for common classical baselines."""

from __future__ import annotations

from typing import Any


def _probabilistic_svc(SVC):
    """An RBF SVC with `predict_proba`, without scikit-learn 1.9's deprecation."""
    import sklearn
    from sklearn.utils.fixes import parse_version

    if parse_version(sklearn.__version__).release < (1, 9):
        return SVC(probability=True)
    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(SVC(), ensemble=False)


def quick_baselines() -> list[Any]:
    """Return a list of common scikit-learn classifiers with sensible defaults.

    The list contains a `SVC` with a radial basis function (RBF) kernel and
    probabilities, so that `roc_auc` can be computed, a `RandomForestClassifier`, a
    `LogisticRegression` and a `KNeighborsClassifier`. `XGBClassifier` and
    `LGBMClassifier` are added when their packages can be imported.

    scikit-learn 1.9 deprecates `SVC(probability=True)`. From that version, the
    support vector classifier is wrapped as
    `CalibratedClassifierCV(SVC(), ensemble=False)`, the replacement that
    scikit-learn documents, so the results table calls it `CalibratedClassifierCV`.

    Returns:
        A list of classifiers that follow the scikit-learn interface, created but
            not fitted.

    Example:
        ```python
        from eigenkit.benchmark import compare, quick_baselines
        results = compare(models=quick_baselines(), dataset="iris", n_seeds=2)
        ```
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    baselines: list[Any] = [
        _probabilistic_svc(SVC),
        RandomForestClassifier(n_estimators=100),
        LogisticRegression(max_iter=1000),
        KNeighborsClassifier(),
    ]

    try:
        from xgboost import XGBClassifier

        baselines.append(XGBClassifier(eval_metric="logloss"))
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        baselines.append(LGBMClassifier(verbose=-1))
    except ImportError:
        pass

    return baselines
