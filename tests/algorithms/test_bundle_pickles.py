"""`.eigenkit` bundles store scikit-learn objects without pickle.

Bundles written before format_version 2 hold those objects as joblib pickles.
They load only with `trusted=True`, because unpickling can run code.
"""

from __future__ import annotations

import io
import json
import zipfile

import joblib
import numpy as np
import pytest
import skops.io

from eigenkit import QSVC, QSVR, QuantumTimeSeriesRegressor, load_model
from eigenkit.benchmark import load_checkpoint
from eigenkit.configs import Encoding, Variational
from eigenkit.qml import BaseQuantumEstimator, QuantumReservoirClassifier
from eigenkit.qml._serialization import SerializationError


def _fitted(name):
    rng = np.random.default_rng(0)
    if name == "QSVC":
        X = rng.standard_normal((20, 2))
        y = np.arange(20) % 3
        return QSVC(probability=True, provider="numpy").fit(X, y), X
    if name == "QSVR":
        X = rng.standard_normal((20, 2))
        return QSVR(provider="numpy").fit(X, X[:, 0]), X
    if name == "QRC":
        X = rng.standard_normal((6, 5, 2))
        blocks = [
            Encoding(in_features=2, rotation="RY"),
            Variational(depth=2, rotation="RY"),
        ]
        model = QuantumReservoirClassifier(blocks).fit(X, (X[:, :, 0] > 0).astype(int))
        return model, X
    ts = np.sin(np.linspace(0, 4 * np.pi, 30))
    model = QuantumTimeSeriesRegressor(QSVR(provider="numpy"), in_size=3, shift=1)
    return model.fit(ts.reshape(-1, 1), ts), ts.reshape(-1, 1)


ESTIMATORS = ["QSVC", "QSVR", "QRC", "QTSR(QSVR)"]


def _entries(path):
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _write(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def _npy(data: bytes) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.frombuffer(data, dtype=np.uint8))
    return buf.getvalue()


def _save_v1(model, path):
    """Save `model` as format_version 1 did: scikit-learn objects as joblib pickles."""
    model.save(path)
    entries = _entries(path)
    metadata = json.loads(entries["metadata.json"])
    metadata["format_version"] = 1
    entries["metadata.json"] = json.dumps(metadata)
    for name in [n for n in entries if n.endswith("_skops.npy")]:
        blob = np.load(io.BytesIO(entries.pop(name))).tobytes()
        obj = skops.io.loads(blob, trusted=skops.io.get_untrusted_types(data=blob))
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        entries[name.replace("_skops.npy", "_joblib.npy")] = _npy(buf.getvalue())
    _write(path, entries)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_a_saved_bundle_holds_no_pickles(name, tmp_path):
    model, X = _fitted(name)
    model.save(tmp_path / "m.eigenkit")

    entries = _entries(tmp_path / "m.eigenkit")
    assert not [n for n in entries if "joblib" in n]
    assert json.loads(entries["metadata.json"])["format_version"] == 2
    loaded = load_model(tmp_path / "m.eigenkit")
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_qsvc_probabilities_survive_a_round_trip(tmp_path):
    model, X = _fitted("QSVC")
    model.save(tmp_path / "m.eigenkit")
    loaded = QSVC.load(tmp_path / "m.eigenkit")
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))


LOADERS = {
    "Class.load": lambda cls, path, **kw: cls.load(path, **kw),
    "BaseQuantumEstimator.load": lambda cls, path, **kw: BaseQuantumEstimator.load(
        path, **kw
    ),
    "load_model": lambda cls, path, **kw: load_model(path, **kw),
    "load_checkpoint": lambda cls, path, **kw: load_checkpoint(path, **kw),
}


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("name", ESTIMATORS)
def test_a_v1_bundle_with_pickles_loads_only_when_trusted(name, loader, tmp_path):
    if loader == "BaseQuantumEstimator.load" and name == "QTSR(QSVR)":
        pytest.skip("the time-series wrapper has its own load()")
    model, X = _fitted(name)
    path = tmp_path / "m.eigenkit"
    _save_v1(model, path)
    load = LOADERS[loader]

    with pytest.raises(SerializationError, match=r"trusted=True.*save"):
        load(type(model), path)

    loaded = load(type(model), path, trusted=True)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_saving_a_trusted_v1_bundle_again_drops_the_pickles(tmp_path):
    model, X = _fitted("QSVC")
    _save_v1(model, tmp_path / "old.eigenkit")

    QSVC.load(tmp_path / "old.eigenkit", trusted=True).save(tmp_path / "new.eigenkit")

    assert not [n for n in _entries(tmp_path / "new.eigenkit") if "joblib" in n]
    np.testing.assert_allclose(
        QSVC.load(tmp_path / "new.eigenkit").predict_proba(X), model.predict_proba(X)
    )


class _RunsOnUnpickle:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_a_refused_bundle_runs_none_of_its_pickles(tmp_path):
    model, _ = _fitted("QSVR")
    path = tmp_path / "m.eigenkit"
    _save_v1(model, path)
    marker = tmp_path / "pwned"
    buf = io.BytesIO()
    joblib.dump(_RunsOnUnpickle(marker), buf)
    entries = _entries(path)
    entries["state/svr_joblib.npy"] = _npy(buf.getvalue())
    _write(path, entries)

    with pytest.raises(SerializationError):
        load_model(path)
    assert not marker.exists()


def test_a_bundle_cannot_widen_the_trusted_types(tmp_path):
    model, _ = _fitted("QSVR")
    path = tmp_path / "m.eigenkit"
    model.save(path)
    entries = _entries(path)
    entries["state/svr_skops.npy"] = _npy(skops.io.dumps(_RunsOnUnpickle("x")))
    _write(path, entries)

    with pytest.raises(Exception, match="(?i)untrusted"):
        QSVR.load(path)
    with pytest.raises(Exception, match="(?i)untrusted"):
        QSVR.load(path, trusted=True)
