import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from eigenkit.benchmark import (
    BenchmarkResult,
    compare,
    custom_metric,
    load_checkpoint,
)


class RecordingClassifier(BaseEstimator):
    __module__ = "sklearn.recording"

    def fit(self, X, y, X_val=None, y_val=None):
        self.classes_ = np.unique(y)
        self.n_train_seen_ = len(X)
        self.n_validation_seen_ = 0 if X_val is None else len(X_val)
        self.train_losses_ = [0.8, 0.4]
        self.validation_losses_ = [] if X_val is None else [0.9, 0.5]
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


class ArrayLossClassifier(RecordingClassifier):
    __module__ = "sklearn.recording"

    def fit(self, X, y, X_val=None, y_val=None):
        super().fit(X, y, X_val=X_val, y_val=y_val)
        self.train_losses_ = np.array([0.8, 0.4])
        self.validation_losses_ = np.array([0.9, 0.5])
        return self


def test_compare_sklearn_only():
    results = compare(
        models=[SVC(), RandomForestClassifier(n_estimators=5)],
        dataset="iris",
        metrics=["accuracy", "fit_time"],
        n_seeds=1,
    )
    assert isinstance(results, BenchmarkResult)
    models = {r["model"] for r in results.raw}
    assert models == {"SVC", "RandomForestClassifier"}


def test_compare_dict_models_use_custom_names(tiny_clf_split):
    results = compare(
        models={
            "Support Vector": SVC(),
            "Baseline": DummyClassifier(strategy="most_frequent"),
        },
        dataset=tiny_clf_split,
        metrics=["accuracy"],
        n_seeds=1,
    )

    assert set(results.fitted_models) == {"Support Vector", "Baseline"}
    assert {r["model"] for r in results.raw} == {"Support Vector", "Baseline"}
    table = results.table()
    assert "Support Vector" in table
    assert "Baseline" in table


def test_compare_with_custom_data(tiny_clf_split):
    results = compare(
        models=[RandomForestClassifier(n_estimators=5)],
        dataset=tiny_clf_split,
        metrics=["accuracy"],
        n_seeds=1,
    )
    assert len(results.raw) == 1


def test_compare_named_dataset_uses_default_train_validation_test_split():
    results = compare(
        models=[RecordingClassifier()],
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=1,
    )

    fitted = results.fitted_models["RecordingClassifier"][0]
    assert fitted.n_train_seen_ == 105
    assert fitted.n_validation_seen_ == 22
    assert results.loss_curves["RecordingClassifier"] == [
        {"train": [0.8, 0.4], "validation": [0.9, 0.5]}
    ]


def test_compare_named_dataset_validation_size_zero_opt_out():
    results = compare(
        models=[RecordingClassifier()],
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=1,
        validation_size=0.0,
    )

    fitted = results.fitted_models["RecordingClassifier"][0]
    assert fitted.n_validation_seen_ == 0


def test_compare_accepts_numpy_array_loss_histories():
    results = compare(
        models=[ArrayLossClassifier()],
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=1,
    )

    curve = results.loss_curves["ArrayLossClassifier"][0]
    np.testing.assert_allclose(curve["train"], [0.8, 0.4])
    np.testing.assert_allclose(curve["validation"], [0.9, 0.5])


def test_compare_presplit_dict_uses_explicit_validation_data():
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    y = np.array([0, 1] * 20)

    results = compare(
        models=[RecordingClassifier()],
        dataset={
            "X_train": X[:25],
            "y_train": y[:25],
            "X_val": X[25:32],
            "y_val": y[25:32],
            "X_test": X[32:],
            "y_test": y[32:],
        },
        metrics=["accuracy"],
        n_seeds=1,
    )

    fitted = results.fitted_models["RecordingClassifier"][0]
    assert fitted.n_train_seen_ == 25
    assert fitted.n_validation_seen_ == 7


def test_compare_presplit_dict_does_not_create_validation_split():
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    y = np.array([0, 1] * 20)

    results = compare(
        models=[RecordingClassifier()],
        dataset={
            "X_train": X[:30],
            "y_train": y[:30],
            "X_test": X[30:],
            "y_test": y[30:],
        },
        metrics=["accuracy"],
        n_seeds=1,
    )

    fitted = results.fitted_models["RecordingClassifier"][0]
    assert fitted.n_train_seen_ == 30
    assert fitted.n_validation_seen_ == 0


def test_compare_presplit_dict_ignores_split_sizes():
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    y = np.array([0, 1] * 20)

    results = compare(
        models=[RecordingClassifier()],
        dataset={
            "X_train": X[:30],
            "y_train": y[:30],
            "X_test": X[30:],
            "y_test": y[30:],
        },
        metrics=["accuracy"],
        n_seeds=1,
        test_size=0.0,
    )

    assert len(results.raw) == 1


def test_compare_preserves_show_progress_positional_argument(capsys):
    results = compare([RecordingClassifier()], "iris", ["accuracy"], 1, 0.2, True)

    assert len(results.raw) == 1
    assert "[benchmark]" in capsys.readouterr().out


def test_compare_single_model():
    results = compare(models=[SVC()], dataset="iris", n_seeds=1)
    assert len(results.raw) > 0


def test_compare_regression_defaults_to_regression_metrics(tiny_reg_split):
    results = compare(
        models=[LinearRegression()],
        dataset=tiny_reg_split,
        n_seeds=1,
    )
    metrics = {r["metric"] for r in results.raw}
    assert "r2" in metrics
    assert "fit_time" in metrics


def test_compare_integer_presplit_regression_targets_do_not_stratify():
    X = np.arange(300, dtype=np.float32).reshape(100, 3)
    y = np.arange(100)

    presplit = compare(
        models=[LinearRegression()],
        dataset={
            "X_train": X[:70],
            "y_train": y[:70],
            "X_test": X[70:],
            "y_test": y[70:],
        },
        metrics=["r2"],
        n_seeds=1,
    )

    assert len(presplit.raw) == 1


def test_compare_with_quantum_model(tiny_clf_split):
    from eigenkit.qml import QSVC

    results = compare(
        models=[QSVC(feature_map="Angle", reps=1, provider="numpy"), SVC()],
        dataset=tiny_clf_split,
        metrics=["accuracy", "fit_time"],
        n_seeds=1,
    )
    model_names = {r["model"] for r in results.raw}
    assert {"QSVC", "SVC"} <= model_names


def test_compare_empty_models_raises():
    with pytest.raises(ValueError, match="non-empty"):
        compare(models=[], dataset="iris", n_seeds=1)


def test_compare_empty_model_dict_raises():
    with pytest.raises(ValueError, match="non-empty"):
        compare(models={}, dataset="iris", n_seeds=1)


def test_compare_rejects_non_string_model_names():
    with pytest.raises(TypeError, match="model names"):
        compare(models={1: SVC()}, dataset="iris", n_seeds=1)


def test_compare_rejects_duplicate_model_names_from_dict_like_input():
    class DuplicateNameDict(dict):
        def __iter__(self):
            return iter(["Same Name", "Same Name"])

        def values(self):
            return [SVC(), DummyClassifier(strategy="most_frequent")]

    with pytest.raises(ValueError, match="unique"):
        compare(models=DuplicateNameDict({"Same Name": SVC()}), dataset="iris", n_seeds=1)


@pytest.mark.parametrize("name", ["", "   "])
def test_compare_rejects_empty_model_names(name):
    with pytest.raises(ValueError, match="non-empty"):
        compare(models={name: SVC()}, dataset="iris", n_seeds=1)


@pytest.mark.parametrize("name", [" leading", "trailing "])
def test_compare_rejects_model_names_with_surrounding_whitespace(name):
    with pytest.raises(ValueError, match="surrounding whitespace"):
        compare(models={name: SVC()}, dataset="iris", n_seeds=1)


@pytest.mark.parametrize("name", ["bad\x00name", "bad\nname", "\tTabbed"])
def test_compare_rejects_model_names_with_control_characters(name):
    with pytest.raises(ValueError, match="control characters"):
        compare(models={name: SVC()}, dataset="iris", n_seeds=1)


def test_compare_allows_filename_unsafe_display_names_without_save_dir(tiny_clf_split):
    results = compare(
        models={"acc>0.9 SVC": SVC()},
        dataset=tiny_clf_split,
        metrics=["accuracy"],
        n_seeds=1,
    )

    assert set(results.fitted_models) == {"acc>0.9 SVC"}


@pytest.mark.parametrize(
    "name",
    [
        "bad/name",
        r"bad\name",
        "bad:name",
        "bad*name",
        "bad?name",
        'bad"name',
        "bad<name",
        "bad>name",
        "bad|name",
    ],
)
def test_compare_rejects_filename_unsafe_model_names_when_saving(name, tmp_path):
    with pytest.raises(ValueError, match="reserved filename"):
        compare(models={name: SVC()}, dataset="iris", n_seeds=1, save_dir=tmp_path)


def test_compare_rejects_model_names_ending_with_period_when_saving(tmp_path):
    with pytest.raises(ValueError, match="period"):
        compare(models={"SVC.": SVC()}, dataset="iris", n_seeds=1, save_dir=tmp_path)


def test_compare_rejects_too_long_model_names_when_saving(tmp_path):
    with pytest.raises(ValueError, match="200 UTF-8 bytes"):
        compare(models={"a" * 201: SVC()}, dataset="iris", n_seeds=1, save_dir=tmp_path)


def test_compare_rejects_non_sklearn_estimator():
    class NotSklearn:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X[:, 0]

    with pytest.raises(TypeError, match="not supported"):
        compare(models=[NotSklearn()], dataset="iris", n_seeds=1)


def test_compare_dict_validation_error_names_bad_key():
    class NotSklearn:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X[:, 0]

    with pytest.raises(TypeError, match="Bad Estimator"):
        compare(models={"Bad Estimator": NotSklearn()}, dataset="iris", n_seeds=1)


def test_compare_rejects_third_party_base_estimator():
    from sklearn.base import BaseEstimator

    class ThirdPartyEst(BaseEstimator):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(TypeError, match="not supported"):
        compare(models=[ThirdPartyEst()], dataset="iris", n_seeds=1)


def test_compare_accepts_kit_quantum_estimator(tiny_clf_split):
    from eigenkit.qml import QSVC

    results = compare(
        models=[QSVC(feature_map="Angle", reps=1, provider="numpy")],
        dataset=tiny_clf_split,
        metrics=["accuracy"],
        n_seeds=1,
    )
    assert {r["model"] for r in results.raw} == {"QSVC"}


def test_compare_save_dir_writes_sklearn_joblib(tmp_path):
    compare(
        models=[SVC()],
        dataset="iris",
        n_seeds=1,
        save_dir=tmp_path,
    )
    assert (tmp_path / "SVC_seed0.joblib").is_file()
    assert not (tmp_path / "SVC_seed0.eigenkit").exists()


def test_compare_dict_models_use_custom_names_for_save_dir(tmp_path):
    compare(
        models={"NamedSVC": SVC()},
        dataset="iris",
        n_seeds=1,
        save_dir=tmp_path,
    )
    assert (tmp_path / "NamedSVC_seed0.joblib").is_file()


def test_compare_save_dir_preserves_seed_suffix_for_dotted_custom_names(tmp_path):
    results = compare(
        models={"Flexible SVC v1.0": SVC()},
        dataset="iris",
        n_seeds=2,
        save_dir=tmp_path,
    )

    assert (tmp_path / "Flexible SVC v1.0_seed0.joblib").is_file()
    assert (tmp_path / "Flexible SVC v1.0_seed1.joblib").is_file()
    assert not (tmp_path / "Flexible SVC v1.joblib").exists()
    assert results.model_artifacts["Flexible SVC v1.0"] == [
        str(tmp_path / "Flexible SVC v1.0_seed0.joblib"),
        str(tmp_path / "Flexible SVC v1.0_seed1.joblib"),
    ]
    loaded = load_checkpoint(results.model_artifacts["Flexible SVC v1.0"][0])
    assert isinstance(loaded, SVC)


def test_compare_save_dir_writes_bundle_and_joblib(tmp_path):
    from eigenkit.torch.estimators import HQNC

    compare(
        models=[
            HQNC(in_features=4, depth=1, epochs=1, provider="numpy", show_progress=False),
            SVC(),
        ],
        dataset="iris",
        n_seeds=1,
        save_dir=tmp_path,
    )
    assert (tmp_path / "HQNC_seed0.eigenkit").is_file()
    assert (tmp_path / "SVC_seed0.joblib").is_file()


def test_compare_dict_save_dir_writes_quantum_and_sklearn_custom_names(
    tmp_path, tiny_clf_split
):
    from eigenkit import HQNC

    results = compare(
        models={
            "Small HQNC": HQNC(
                in_features=4,
                depth=1,
                epochs=1,
                provider="numpy",
                show_progress=False,
            ),
            "Flexible SVC": SVC(),
        },
        dataset=tiny_clf_split,
        n_seeds=1,
        save_dir=tmp_path,
    )

    assert (tmp_path / "Small HQNC_seed0.eigenkit").is_file()
    assert (tmp_path / "Flexible SVC_seed0.joblib").is_file()
    assert set(results.model_artifacts) == {"Small HQNC", "Flexible SVC"}


def test_compare_save_dir_preserves_seed_suffix_for_dotted_quantum_names(
    tmp_path, tiny_clf_split
):
    from eigenkit import HQNC

    results = compare(
        models={
            "HQNC depth=1.0": HQNC(
                in_features=4,
                depth=1,
                epochs=1,
                provider="numpy",
                show_progress=False,
            )
        },
        dataset=tiny_clf_split,
        n_seeds=1,
        save_dir=tmp_path,
    )

    assert (tmp_path / "HQNC depth=1.0_seed0.eigenkit").is_file()
    assert not (tmp_path / "HQNC depth=1.eigenkit").exists()
    loaded = load_checkpoint(results.model_artifacts["HQNC depth=1.0"][0])
    assert isinstance(loaded, HQNC)
    assert (
        loaded.predict(tiny_clf_split["X_test"]).shape == tiny_clf_split["y_test"].shape
    )


def test_compare_bad_dict_raises():
    with pytest.raises(ValueError, match="missing keys"):
        compare(
            models=[SVC()],
            dataset={"X_train": np.zeros((2, 2)), "y_train": np.zeros(2)},
            n_seeds=1,
        )


def test_compare_seeds_produce_multiple_rows():
    results = compare(
        models=[RandomForestClassifier(n_estimators=5)],
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=3,
    )
    seeds = {r["seed"] for r in results.raw}
    assert seeds == {0, 1, 2}


def test_compare_progress_output_shows_current_model(capsys):
    compare(
        models=[SVC(), RandomForestClassifier(n_estimators=5)],
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=1,
        show_progress=True,
    )
    out = capsys.readouterr().out
    assert "[benchmark] 1/2" in out
    assert "model=SVC" in out
    assert "model=RandomForestClassifier" in out
    assert "remaining=" in out


def test_compare_progress_output_uses_custom_model_name(capsys):
    compare(
        models={"Custom SVC": SVC()},
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=1,
        show_progress=True,
    )
    out = capsys.readouterr().out
    assert "model=Custom SVC" in out


def test_compare_progress_can_be_disabled(capsys):
    compare(
        models=[SVC()],
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=1,
        show_progress=False,
    )
    out = capsys.readouterr().out
    assert out == ""


def test_compare_returns_fitted_models_and_best_model(tiny_clf_split):
    results = compare(
        models=[SVC(), DummyClassifier(strategy="most_frequent")],
        dataset=tiny_clf_split,
        metrics=["accuracy"],
        n_seeds=2,
    )

    assert set(results.fitted_models) == {"SVC", "DummyClassifier"}
    assert len(results.fitted_models["SVC"]) == 2
    best = results.best_model("accuracy")
    assert hasattr(best, "predict")
    pred = results.predict(tiny_clf_split["X_test"], metric="accuracy")
    assert pred.shape == tiny_clf_split["y_test"].shape


def test_compare_custom_names_key_downstream_accessors(tiny_clf_split):
    results = compare(
        models={
            "Recorded Baseline": RecordingClassifier(),
            "Frequent Class": DummyClassifier(strategy="most_frequent"),
        },
        dataset=tiny_clf_split,
        metrics=["accuracy"],
        n_seeds=1,
        store_predictions=True,
    )

    assert set(results.fitted_models) == {"Recorded Baseline", "Frequent Class"}
    assert results.loss_curves["Recorded Baseline"] == [
        {"train": [0.8, 0.4], "validation": []}
    ]
    assert set(results.predictions) == {"Recorded Baseline", "Frequent Class"}
    best = results.best_model("accuracy")
    assert hasattr(best, "predict")
    pred = results.predict(tiny_clf_split["X_test"], metric="accuracy")
    assert pred.shape == tiny_clf_split["y_test"].shape
    out = results.inference(
        tiny_clf_split["X_test"],
        models=["Recorded Baseline"],
        seed=0,
    )
    assert set(out) == {"Recorded Baseline"}
    assert out["Recorded Baseline"]["predictions"].shape == tiny_clf_split["y_test"].shape
    matrix = results.confusion_matrix("Recorded Baseline")
    assert matrix.shape == (2, 2)


def test_compare_dict_keys_are_verbatim_when_they_look_like_class_names(
    tiny_clf_split,
):
    results = compare(
        models={
            "SVC": SVC(C=0.3),
            "SVC_0": SVC(C=3.0),
        },
        dataset=tiny_clf_split,
        metrics=["accuracy"],
        n_seeds=1,
    )

    assert set(results.fitted_models) == {"SVC", "SVC_0"}
    assert {row["model"] for row in results.raw} == {"SVC", "SVC_0"}
    assert "SVC_1" not in results.fitted_models


def test_compare_store_predictions_and_confusion_matrix(tiny_clf_split):
    results = compare(
        models=[SVC()],
        dataset=tiny_clf_split,
        metrics=["accuracy"],
        n_seeds=1,
        store_predictions=True,
    )

    assert "SVC" in results.predictions
    assert set(results.predictions["SVC"]) == {"y_true", "y_pred", "y_prob"}
    matrix = results.confusion_matrix("SVC")
    assert matrix.shape == (2, 2)


def test_confusion_matrix_defaults_to_predictions_when_metrics_skip(tiny_clf_split):
    with pytest.warns(UserWarning, match="predict_proba"):
        results = compare(
            models=[SVC()],
            dataset=tiny_clf_split,
            metrics=["roc_auc"],
            n_seeds=1,
            store_predictions=True,
        )

    assert results.raw == []
    matrix = results.confusion_matrix()
    assert matrix.shape == (2, 2)


def test_compare_roc_auc_warning_uses_custom_model_name(tiny_clf_split):
    with pytest.warns(UserWarning) as warnings_record:
        compare(
            models={"Display Label": SVC()},
            dataset=tiny_clf_split,
            metrics=["roc_auc"],
            n_seeds=1,
        )

    message = next(
        str(record.message)
        for record in warnings_record
        if "following models" in str(record.message)
    )
    assert "Display Label" in message
    assert "SVC" not in message


def test_compare_custom_metric_wrapper(tiny_clf_split):
    def match_rate(y_true, y_pred):
        return np.mean(y_true == y_pred)

    results = compare(
        models=[SVC()],
        dataset=tiny_clf_split,
        metrics=["accuracy", custom_metric(match_rate, direction="higher")],
        n_seeds=1,
    )
    assert {r["metric"] for r in results.raw} == {"accuracy", "match_rate"}


def test_compare_custom_metric_directions_rank_independently(tiny_clf_split):
    def zero_rate(y_true, y_pred):
        return np.mean(y_pred == 0)

    results = compare(
        models=[
            DummyClassifier(strategy="constant", constant=0),
            DummyClassifier(strategy="constant", constant=1),
        ],
        dataset=tiny_clf_split,
        metrics=[
            custom_metric(zero_rate, direction="higher"),
            custom_metric(zero_rate, direction="lower", name="zero_rate_loss"),
        ],
        n_seeds=1,
    )

    assert results.metric_directions == {
        "zero_rate": "higher",
        "zero_rate_loss": "lower",
    }
    assert results.best_model("zero_rate").predict(tiny_clf_split["X_test"])[0] == 0
    assert results.best_model("zero_rate_loss").predict(tiny_clf_split["X_test"])[0] == 1


def test_compare_rejects_bare_custom_metric_callable(tiny_clf_split):
    def loss(y_true, y_pred):
        return np.mean(y_pred == 0)

    with pytest.raises(ValueError, match="custom_metric"):
        compare(
            models=[SVC()],
            dataset=tiny_clf_split,
            metrics=[loss],
            n_seeds=1,
        )


def test_compare_rejects_duplicate_metric_names(tiny_clf_split):
    def match_rate(y_true, y_pred):
        return np.mean(y_true == y_pred)

    with pytest.raises(ValueError, match="Duplicate metric name"):
        compare(
            models=[SVC()],
            dataset=tiny_clf_split,
            metrics=[
                "accuracy",
                custom_metric(match_rate, direction="higher", name="accuracy"),
            ],
            n_seeds=1,
        )


def test_load_checkpoint_loads_joblib_checkpoint(tmp_path):
    compare(
        models=[SVC()],
        dataset="iris",
        metrics=["accuracy"],
        n_seeds=1,
        save_dir=tmp_path,
    )

    path = tmp_path / "SVC_seed0.joblib"
    assert path.is_file()
    loaded = load_checkpoint(path)
    assert isinstance(loaded, SVC)


def test_load_checkpoint_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("not a benchmark checkpoint", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        load_checkpoint(path)


def test_compare_fit_time_and_predict_time_are_positive():
    results = compare(
        models=[SVC()],
        dataset="iris",
        metrics=["accuracy", "fit_time", "predict_time"],
        n_seeds=1,
    )
    fit_times = [r["value"] for r in results.raw if r["metric"] == "fit_time"]
    predict_times = [r["value"] for r in results.raw if r["metric"] == "predict_time"]
    assert len(fit_times) == 1
    assert len(predict_times) == 1
    assert fit_times[0] >= 0.0
    assert predict_times[0] >= 0.0


def test_compare_duplicate_class_names_use_indexed_save_names(tiny_clf_split):
    from eigenkit.qml import QSVC

    results = compare(
        models=[
            QSVC(feature_map="Angle", reps=1, provider="numpy"),
            QSVC(feature_map="AngleCNOT", reps=2, provider="numpy"),
        ],
        dataset=tiny_clf_split,
        metrics=["accuracy"],
        n_seeds=1,
    )
    model_names = {r["model"] for r in results.raw}
    assert "QSVC_0" in model_names
    assert "QSVC_1" in model_names
    assert "QSVC" not in model_names
    assert set(results.fitted_models.keys()) == {"QSVC_0", "QSVC_1"}


def test_compare_with_qsvc_does_not_import_torch():
    """compare() takes BaseQuantumEstimator from the torch-free qml tier, so a
    QSVC benchmark must not import torch. Run in a fresh interpreter and check
    torch is never loaded."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        from sklearn.svm import SVC
        from eigenkit import QSVC
        from eigenkit.benchmark import compare
        result = compare({"QSVC": QSVC(), "SVC": SVC()}, dataset="iris", n_seeds=1,
                         validation_size=0.0)
        assert set(result.to_dataframe()["model"]) == {"QSVC", "SVC"}
        assert "torch" not in sys.modules, "compare() imported torch"
        print("ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_timer_is_public():
    from eigenkit.benchmark import timer

    with timer() as elapsed:
        sum(range(1000))
    assert elapsed["elapsed"] >= 0.0
