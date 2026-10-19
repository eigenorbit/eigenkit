"""Round-trip and format tests for .eigenkit model persistence."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from eigenkit import (
    HQNC,
    HQNR,
    QSVC,
    QSVR,
    CustomQuantumClassifier,
    CustomQuantumRegressor,
    NumpyConfig,
    QuantumTimeSeriesClassifier,
    QuantumTimeSeriesRegressor,
    RecurrentHQNC,
    RecurrentHQNR,
    all_z,
    configs,
    load_model,
)
from eigenkit.configs import Encoding, Variational
from eigenkit.qml._serialization import (
    SerializationError,
    check_format_version,
    deserialize_provider_param,
    resolve_bundle_path,
    save_bundle,
    serialize_provider_param,
    to_json_safe,
)


def _multiclass_clf_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=40)
    return X[:30], y[:30], X[30:], y[30:]


def test_hqnc_save_load_roundtrip(tmp_path):
    X_tr, y_tr, X_te, _ = _multiclass_clf_data()
    model = HQNC(
        in_features=4,
        depth=1,
        n_classes=3,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    pred_before = model.predict(X_te)

    bundle_path = tmp_path / "hqnc"
    model.save(bundle_path)
    loaded = HQNC.load(bundle_path)
    np.testing.assert_array_equal(pred_before, loaded.predict(X_te))


def test_recurrent_hqnr_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2, 2)).astype(np.float32)
    y = X[:, -1, 0].astype(np.float32)
    X_tr, y_tr, X_te = X[:15], y[:15], X[15:]

    model = RecurrentHQNR(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    pred_before = model.predict(X_te)

    bundle_path = tmp_path / "recurrent_hqnr.eigenkit"
    model.save(bundle_path)
    loaded = RecurrentHQNR.load(bundle_path)
    np.testing.assert_allclose(pred_before, loaded.predict(X_te), rtol=1e-5, atol=1e-5)


def test_recurrent_hqnc_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2, 2)).astype(np.float32)
    y = np.where(X[:, -1, 0] > 0, "up", "down")
    X_tr, y_tr, X_te = X[:15], y[:15], X[15:]

    model = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    pred_before = model.predict(X_te)

    bundle_path = tmp_path / "recurrent_hqnc.eigenkit"
    model.save(bundle_path)
    loaded = RecurrentHQNC.load(bundle_path)
    np.testing.assert_array_equal(pred_before, loaded.predict(X_te))


def test_hqnr_save_load_roundtrip(tmp_path, reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    model = HQNR(
        in_features=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    pred_before = model.predict(X_te)

    bundle_path = tmp_path / "hqnr.eigenkit"
    model.save(bundle_path)
    loaded = HQNR.load(bundle_path)
    np.testing.assert_allclose(pred_before, loaded.predict(X_te), rtol=1e-5, atol=1e-5)


def test_qsvc_save_load_roundtrip(tmp_path, clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    model = QSVC(feature_map="Angle", reps=1, provider="numpy")
    model.fit(X_tr, y_tr)
    pred_before = model.predict(X_te)

    bundle_path = tmp_path / "qsvc.eigenkit"
    model.save(bundle_path)
    loaded = QSVC.load(bundle_path)
    np.testing.assert_array_equal(pred_before, loaded.predict(X_te))


def test_a_failed_save_keeps_the_previous_bundle(tmp_path):
    bundle_path = tmp_path / "model.eigenkit"
    save_bundle(bundle_path, metadata={"v": 1}, params={}, circuit_spec=None, state={})
    before = bundle_path.read_bytes()
    with pytest.raises(TypeError):
        save_bundle(
            bundle_path,
            metadata={"v": 2},
            params={"x": object()},
            circuit_spec=None,
            state={},
        )
    assert bundle_path.read_bytes() == before


def test_qsvr_save_load_roundtrip(tmp_path, reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    model = QSVR(feature_map="Angle", reps=1, provider="numpy")
    model.fit(X_tr, y_tr)
    pred_before = model.predict(X_te)

    bundle_path = tmp_path / "qsvr.eigenkit"
    model.save(bundle_path)
    loaded = QSVR.load(bundle_path)
    np.testing.assert_allclose(pred_before, loaded.predict(X_te), rtol=1e-5, atol=1e-5)


def test_custom_classifier_save_load_roundtrip(tmp_path, clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    blocks = [Encoding(in_features=4, rotation="RY"), Variational(depth=1, rotation="RY")]
    model = CustomQuantumClassifier(
        blocks=blocks,
        observables=all_z(4),
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    pred_before = model.predict(X_te)

    bundle_path = tmp_path / "custom_clf.eigenkit"
    model.save(bundle_path)
    loaded = CustomQuantumClassifier.load(bundle_path)
    np.testing.assert_array_equal(pred_before, loaded.predict(X_te))


def test_custom_regressor_save_load_roundtrip(tmp_path, reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    blocks = [Encoding(in_features=4, rotation="RY"), Variational(depth=1, rotation="RY")]
    model = CustomQuantumRegressor(
        blocks=blocks,
        observables=all_z(4),
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    pred_before = model.predict(X_te)

    bundle_path = tmp_path / "custom_reg.eigenkit"
    model.save(bundle_path)
    loaded = CustomQuantumRegressor.load(bundle_path)
    np.testing.assert_allclose(pred_before, loaded.predict(X_te), rtol=1e-5, atol=1e-5)


def test_provider_override_on_load(tmp_path, clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    model = HQNC(
        in_features=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    bundle_path = tmp_path / "hqnc_provider.eigenkit"
    model.save(bundle_path)

    loaded = HQNC.load(bundle_path, provider=NumpyConfig())
    assert loaded.provider == NumpyConfig()
    assert loaded.predict(X_te).shape == (len(X_te),)


@pytest.mark.parametrize("estimator_cls", [QSVC, QSVR])
def test_load_model_without_class_svm(tmp_path, clf_data, reg_data, estimator_cls):
    if estimator_cls is QSVC:
        X_tr, y_tr, X_te, _ = clf_data
        model = QSVC(feature_map="Angle", reps=1, provider="numpy")
        model.fit(X_tr, y_tr)
        pred_before = model.predict(X_te)
        rtol, atol = None, None
    else:
        X_tr, y_tr, X_te, _ = reg_data
        model = QSVR(feature_map="Angle", reps=1, provider="numpy")
        model.fit(X_tr, y_tr)
        pred_before = model.predict(X_te)
        rtol, atol = 1e-5, 1e-5

    bundle_path = tmp_path / f"{estimator_cls.__name__}_load_model.eigenkit"
    model.save(bundle_path)
    loaded = load_model(bundle_path)
    assert isinstance(loaded, estimator_cls)
    if rtol is None:
        np.testing.assert_array_equal(pred_before, loaded.predict(X_te))
    else:
        np.testing.assert_allclose(
            pred_before, loaded.predict(X_te), rtol=rtol, atol=atol
        )


def test_train_losses_persisted_after_load(tmp_path, clf_data):
    X_tr, y_tr, _, _ = clf_data
    model = HQNC(
        in_features=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    losses_before = list(model.train_losses_)

    bundle_path = tmp_path / "hqnc_train_losses.eigenkit"
    model.save(bundle_path)
    loaded = HQNC.load(bundle_path)
    assert loaded.train_losses_ == losses_before
    assert len(loaded.train_losses_) == 2


def test_load_model_without_class(tmp_path, clf_data):
    X_tr, y_tr, X_te, _ = clf_data
    model = HQNC(
        in_features=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    pred_before = model.predict(X_te)

    bundle_path = tmp_path / "hqnc_load_model.eigenkit"
    model.save(bundle_path)
    loaded = load_model(bundle_path)
    assert isinstance(loaded, HQNC)
    np.testing.assert_array_equal(pred_before, loaded.predict(X_te))


def test_bundle_format_no_pickle(tmp_path, clf_data):
    X_tr, y_tr, _, _ = clf_data
    model = HQNC(
        in_features=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    bundle_path = tmp_path / "bundle.eigenkit"
    model.save(bundle_path)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        names = set(zf.namelist())
        assert "metadata.json" in names
        assert "params.json" in names
        assert "state/weights.pt" in names
        assert "state/label_classes.json" in names
        assert not any(n.endswith(".pkl") or n.endswith(".joblib") for n in names)


def test_qsvc_joblib_bytes_are_npy_not_joblib(tmp_path, clf_data):
    """State file for SVC must be .npy (wrapped bytes), not a raw .joblib."""
    X_tr, y_tr, _, _ = clf_data
    model = QSVC(provider="numpy")
    model.fit(X_tr, y_tr)
    model.save(tmp_path / "qsvc.eigenkit")
    with zipfile.ZipFile(tmp_path / "qsvc.eigenkit") as zf:
        names = zf.namelist()
        assert "state/svc_joblib.npy" in names
        assert not any(n.endswith(".joblib") for n in names)


def test_qsvr_joblib_bytes_are_npy_not_joblib(tmp_path, reg_data):
    """State file for SVR must be .npy (wrapped bytes), not a raw .joblib."""
    X_tr, y_tr, _, _ = reg_data
    model = QSVR(provider="numpy")
    model.fit(X_tr, y_tr)
    model.save(tmp_path / "qsvr.eigenkit")
    with zipfile.ZipFile(tmp_path / "qsvr.eigenkit") as zf:
        names = zf.namelist()
        assert "state/svr_joblib.npy" in names
        assert not any(n.endswith(".joblib") for n in names)


def test_resolve_bundle_path_appends_suffix_for_dotted_stems(tmp_path):
    assert resolve_bundle_path(tmp_path / "run_v1.0") == tmp_path / "run_v1.0.eigenkit"
    assert resolve_bundle_path(tmp_path / "run_v1.9") == tmp_path / "run_v1.9.eigenkit"
    assert (
        resolve_bundle_path(tmp_path / "run_v1.0.eigenkit")
        == tmp_path / "run_v1.0.eigenkit"
    )


def test_save_bundle_does_not_collapse_dotted_stems(tmp_path):
    save_bundle(
        tmp_path / "run_v1.0",
        metadata={"class_name": "Test"},
        params={},
        circuit_spec=None,
        state={},
    )
    save_bundle(
        tmp_path / "run_v1.9",
        metadata={"class_name": "Test"},
        params={},
        circuit_spec=None,
        state={},
    )

    assert (tmp_path / "run_v1.0.eigenkit").is_file()
    assert (tmp_path / "run_v1.9.eigenkit").is_file()
    assert not (tmp_path / "run_v1.eigenkit").exists()


def test_load_accepts_path_without_bundle_suffix(tmp_path, clf_data):
    from eigenkit.torch.estimators.hqnc import HQNC

    X_tr, y_tr, X_te, _ = clf_data
    model = HQNC(
        in_features=4,
        depth=1,
        epochs=1,
        provider="numpy",
        show_progress=False,
    )
    model.fit(X_tr, y_tr)
    base = tmp_path / "hqnc_no_suffix"
    model.save(base)
    assert base.with_suffix(".eigenkit").is_file()
    loaded = HQNC.load(base)
    np.testing.assert_array_equal(model.predict(X_te), loaded.predict(X_te))


def test_label_classes_numpy_int64_serializes(tmp_path):
    """LabelEncoder.classes_ yields numpy int64 — must round-trip via JSON."""
    from sklearn.preprocessing import LabelEncoder

    from eigenkit.qml._serialization import save_bundle

    le = LabelEncoder()
    le.fit([0, 1, 2])
    save_bundle(
        tmp_path / "labels.eigenkit",
        metadata={"class_name": "Test"},
        params={},
        circuit_spec=None,
        state={"label_classes": list(le.classes_)},
    )


def test_serialization_error_on_unsupported_type(tmp_path):
    with pytest.raises(SerializationError, match="Cannot serialize"):
        save_bundle(
            tmp_path / "unused.eigenkit",
            metadata={"class_name": "Test"},
            params={},
            circuit_spec=None,
            state={"bad": lambda x: x},
        )


def test_benchmark_save_dir(tmp_path):
    from eigenkit.benchmark import compare

    compare(
        [HQNC(in_features=4, depth=1, epochs=1, provider="numpy", show_progress=False)],
        dataset="iris",
        n_seeds=1,
        save_dir=tmp_path,
    )
    assert (tmp_path / "HQNC_seed0.eigenkit").is_file()


def test_custom_block_serializes_successfully(tmp_path):
    from eigenkit.core.builder import CircuitBuilder
    from eigenkit.qml._serialization import (
        deserialize_circuit_spec,
        serialize_circuit_spec,
    )

    builder = CircuitBuilder(n_qubits=2)
    builder.h(0)
    block = builder.to_block()

    spec = serialize_circuit_spec([block], None)
    blocks, obs = deserialize_circuit_spec(spec)
    assert len(blocks) == 1
    assert blocks[0].n_qubits == block.n_qubits
    assert blocks[0].gates == block.gates
    assert blocks[0].n_weights == block.n_weights
    assert blocks[0].n_inputs == block.n_inputs


def test_check_format_version_raises_for_unsupported_version():
    with pytest.raises(SerializationError, match="format_version=2"):
        check_format_version({"format_version": 2})


def test_check_format_version_passes_for_version_1():
    check_format_version({"format_version": 1})


def test_check_format_version_rejects_a_missing_version():
    with pytest.raises(SerializationError, match="format_version=None"):
        check_format_version({})


def test_public_serialization_api_importable():
    assert callable(to_json_safe)
    assert callable(serialize_provider_param)
    assert callable(deserialize_provider_param)
    assert callable(check_format_version)


@pytest.mark.parametrize("config_cls", ["IQMConfig", "QiskitRuntimeConfig"])
def test_enable_gradients_is_not_saved(config_cls):
    """A shared bundle must not opt whoever loads it in to paid gradients."""
    import eigenkit.configs

    config = getattr(eigenkit.configs, config_cls)(enable_gradients=True)
    payload = serialize_provider_param(config)
    assert "enable_gradients" not in payload
    assert deserialize_provider_param(payload).enable_gradients is False


@pytest.mark.parametrize(
    ("loader_cls", "actual_class"),
    [
        (QuantumTimeSeriesClassifier, "QuantumTimeSeriesRegressor"),
        (QuantumTimeSeriesRegressor, "QuantumTimeSeriesClassifier"),
    ],
)
def test_timeseries_explicit_load_rejects_wrong_wrapper(
    monkeypatch, loader_cls, actual_class
):
    import eigenkit.torch.estimators.timeseries as timeseries_module

    params = {
        "estimator_class": "HQNR",
        "estimator_params": {},
        "in_size": 2,
        "shift": 1,
        "autoregressive": False,
    }

    def fake_load_bundle(path):
        return (
            {"class_name": actual_class, "format_version": 1},
            params,
            None,
            {},
        )

    monkeypatch.setattr(timeseries_module, "load_bundle", fake_load_bundle)

    with pytest.raises(TypeError, match=f"{loader_cls.__name__}.load"):
        loader_cls.load("wrong.eigenkit")

    assert params["estimator_class"] == "HQNR"


@pytest.mark.parametrize(
    ("loader_cls", "class_name", "extra_state"),
    [
        (QuantumTimeSeriesRegressor, "QuantumTimeSeriesRegressor", {}),
        (QuantumTimeSeriesClassifier, "QuantumTimeSeriesClassifier", {"classes": [0, 1]}),
    ],
)
def test_timeseries_explicit_load_restores_matching_wrapper(
    monkeypatch, loader_cls, class_name, extra_state
):
    import eigenkit.torch.estimators.timeseries as timeseries_module

    class FakeInner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def _restore_state(self, state):
            self.restored_state = state

    def fake_load_bundle(path):
        return (
            {"class_name": class_name, "format_version": 1},
            {
                "estimator_class": "FakeInner",
                "estimator_params": {"provider": "pennylane"},
                "in_size": 2,
                "shift": 1,
                "autoregressive": False,
            },
            {"blocks": []},
            {"estimator_weights": [1, 2], "train_losses_": [0.25], **extra_state},
        )

    monkeypatch.setattr(timeseries_module, "load_bundle", fake_load_bundle)
    monkeypatch.setattr(
        timeseries_module,
        "deserialize_circuit_spec",
        lambda spec: (["block"], ["observable"]),
    )
    monkeypatch.setattr(timeseries_module, "get_estimator_class", lambda name: FakeInner)

    model = loader_cls.load("model.eigenkit", provider="numpy")

    assert isinstance(model, loader_cls)
    assert model.estimator.kwargs == {
        "provider": "numpy",
        "blocks": ["block"],
        "observables": ["observable"],
    }
    assert model.estimator.restored_state == {"weights": [1, 2]}
    assert model.train_losses_ == [0.25]
    if extra_state:
        np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_qtsr_save_load_roundtrip(tmp_path, ts_data):
    from eigenkit import HQNR, QuantumTimeSeriesRegressor

    ts = ts_data  # shape (50,) sine wave
    X = ts.reshape(-1, 1)  # (50, 1)
    y = ts

    inner = HQNR(
        in_features=5,  # in_size * n_features after QTSR flattens
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model = QuantumTimeSeriesRegressor(estimator=inner, in_size=5, shift=1)
    model.fit(X, y)
    pred_before = model.predict(X[:20])

    bundle_path = tmp_path / "qtsr.eigenkit"
    model.save(bundle_path)

    loaded = QuantumTimeSeriesRegressor.load(bundle_path)
    np.testing.assert_allclose(pred_before, loaded.predict(X[:20]), rtol=1e-5, atol=1e-5)


def test_qtsr_bundle_contains_format_version(tmp_path, ts_data):
    import json

    from eigenkit import HQNR, QuantumTimeSeriesRegressor

    ts = ts_data
    X = ts.reshape(-1, 1)
    y = ts

    inner = HQNR(
        in_features=5,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model = QuantumTimeSeriesRegressor(estimator=inner, in_size=5, shift=1)
    model.fit(X, y)

    bundle_path = tmp_path / "qtsr_meta.eigenkit"
    model.save(bundle_path)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        meta = json.loads(zf.read("metadata.json"))
    assert meta.get("format_version") == 1
    assert meta.get("estimator_class") == "HQNR"


def test_qtsc_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2)).astype(np.float32)
    y = np.where(X[:, 0] > 0, "up", "down")

    inner = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model = QuantumTimeSeriesClassifier(estimator=inner, in_size=3, shift=1)
    model.fit(X, y)
    pred_before = model.predict(X[:15])

    bundle_path = tmp_path / "qtsc.eigenkit"
    model.save(bundle_path)

    loaded = QuantumTimeSeriesClassifier.load(bundle_path)
    np.testing.assert_array_equal(pred_before, loaded.predict(X[:15]))
    np.testing.assert_array_equal(pred_before, load_model(bundle_path).predict(X[:15]))


def test_qtsc_bundle_contains_format_version(tmp_path):
    import json

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2)).astype(np.float32)
    y = np.where(X[:, 0] > 0, "up", "down")

    inner = RecurrentHQNC(
        seq_feature_dim=2,
        rnn_hidden_dim=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model = QuantumTimeSeriesClassifier(estimator=inner, in_size=3, shift=1)
    model.fit(X, y)

    bundle_path = tmp_path / "qtsc_meta.eigenkit"
    model.save(bundle_path)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        meta = json.loads(zf.read("metadata.json"))
    assert meta.get("format_version") == 1
    assert meta.get("estimator_class") == "RecurrentHQNC"


def test_qtsc_autoregressive_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2)).astype(np.float32)
    y = np.where(X[:, 0] > 0, "up", "down")

    # autoregressive=True appends 1 extra channel → seq_feature_dim = 2 + 1 = 3
    inner = RecurrentHQNC(
        seq_feature_dim=3,
        rnn_hidden_dim=4,
        depth=1,
        epochs=2,
        provider="numpy",
        show_progress=False,
    )
    model = QuantumTimeSeriesClassifier(
        estimator=inner, in_size=3, shift=1, autoregressive=True
    )
    model.fit(X, y)
    pred_before = model.predict(X[:15], y_history=y[:15])

    bundle_path = tmp_path / "qtsc_ar.eigenkit"
    model.save(bundle_path)

    loaded = QuantumTimeSeriesClassifier.load(bundle_path)
    pred_after = loaded.predict(X[:15], y_history=y[:15])

    np.testing.assert_array_equal(pred_before, pred_after)
    assert hasattr(loaded, "_history_encoder")


def test_qtsc_serialize_error_on_unsupported_inner():
    from eigenkit.torch.estimators.timeseries import QuantumTimeSeriesClassifier

    class _MinimalEstimator:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(len(X), self.classes_[0])

    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.where(X[:, 0] > 15, "high", "low")
    model = QuantumTimeSeriesClassifier(estimator=_MinimalEstimator(), in_size=2, shift=1)
    model.fit(X, y)

    with pytest.raises(SerializationError):
        model._serialize_params()


@pytest.mark.parametrize("name", sorted({"NumpyConfig", *configs._LAZY_CONFIGS}))
def test_every_provider_config_round_trips(name):
    """Every config `save()` accepts must also load back."""
    try:
        cls = getattr(configs, name)
    except ImportError as error:
        pytest.skip(f"{name} needs an SDK that is not installed: {error}")
    config = cls()
    assert deserialize_provider_param(serialize_provider_param(config)) == config
