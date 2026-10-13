"""Secret config fields never reach a saved model.

A `.eigenkit` bundle is a deflated zip, so a secret written to `params.json` would
not show up in the raw file bytes: the scan below reads the raw bytes and every
decompressed member, and a positive control (a non-secret marker that must be
found) keeps it from passing vacuously. The joblib path is covered too, since
`benchmark.compare()` pickles built-in sklearn models with it.
"""

from __future__ import annotations

import dataclasses
import io
import zipfile

import numpy as np
import pytest


pytest.importorskip("sklearn")
import joblib  # noqa: E402
from sklearn.base import clone  # noqa: E402

from eigenkit import QSVC  # noqa: E402
from eigenkit.configs import IQMConfig, QiskitRuntimeConfig  # noqa: E402
from eigenkit.qml._serialization import (  # noqa: E402
    deserialize_provider_param,
    serialize_provider_param,
)


SECRET = "ibm-secret-2f8c1d"
MARKER = "crn:marker-7a1e"


def _bundle_texts(path) -> list[bytes]:
    """The raw file bytes, then the decompressed bytes of every zip member."""
    with open(path, "rb") as fh:
        texts = [fh.read()]
    with zipfile.ZipFile(path) as zf:
        texts.extend(zf.read(name) for name in zf.namelist())
    return texts


def _fitted_qsvc() -> QSVC:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 2))
    y = np.array([0, 1] * 4)
    return QSVC(feature_map="Angle", reps=1, provider="numpy").fit(X, y)


class _Holder:
    """Stands in for a model holding a provider config."""

    def __init__(self, provider):
        self.provider = provider


def test_serialize_provider_param_drops_the_token():
    cfg = QiskitRuntimeConfig(token=SECRET, instance=MARKER)
    payload = serialize_provider_param(cfg)

    assert payload["__provider_config__"] == "QiskitRuntimeConfig"
    assert payload["instance"] == MARKER
    assert "token" not in payload
    expected = dataclasses.asdict(cfg)
    del expected["token"]
    # Not secret, but never saved either: see test_enable_gradients_is_not_saved
    # and test_bundle_does_not_store_where_a_provider_connects.
    for name in ("enable_gradients", "service_kwargs", "backend_instance"):
        del expected[name]
    assert {k: v for k, v in payload.items() if k != "__provider_config__"} == expected


def test_bundle_carries_no_ibm_token(tmp_path):
    model = _fitted_qsvc()
    model.set_params(provider=QiskitRuntimeConfig(token=SECRET, instance=MARKER))
    path = tmp_path / "qsvc.eigenkit"
    model.save(path)

    texts = _bundle_texts(path)
    assert all(SECRET.encode() not in t for t in texts)
    # Positive control: the scanner does read inside the deflated members.
    assert any(MARKER.encode() in t for t in texts[1:])
    with zipfile.ZipFile(path) as zf:
        assert SECRET.encode() not in zf.read("metadata.json")


def test_bundle_loads_with_token_none(tmp_path, monkeypatch):
    model = _fitted_qsvc()
    model.set_params(provider=QiskitRuntimeConfig(token=SECRET, instance=MARKER))
    path = tmp_path / "qsvc.eigenkit"
    model.save(path)

    # QSVC rebuilds its kernel on load, which compiles against the provider and
    # would open an IBM Runtime service. Only the restored parameters matter here.
    monkeypatch.setattr(QSVC, "_restore_state", lambda self, state: None)
    loaded = QSVC.load(path)
    assert loaded.provider.token is None
    assert loaded.provider.instance == MARKER


def test_bundle_carries_no_iqm_token(tmp_path):
    model = _fitted_qsvc()
    model.set_params(provider=IQMConfig(token=SECRET, shots=10))
    path = tmp_path / "qsvc_iqm.eigenkit"
    model.save(path)

    assert all(SECRET.encode() not in t for t in _bundle_texts(path))
    # The bundle loads back, with the token left to the environment.
    from eigenkit.qml import QSVC

    loaded = QSVC.load(path)
    assert isinstance(loaded.provider, IQMConfig)
    assert loaded.provider.token is None and loaded.provider.shots == 10


def test_joblib_checkpoint_carries_no_token():
    holder = _Holder(QiskitRuntimeConfig(token=SECRET, instance=MARKER))
    buf = io.BytesIO()
    joblib.dump({"model": holder}, buf)

    assert SECRET.encode() not in buf.getvalue()
    assert MARKER.encode() in buf.getvalue()


def test_sklearn_clone_keeps_the_token():
    model = QSVC(provider=QiskitRuntimeConfig(token=SECRET))
    assert clone(model).provider.token == SECRET


def test_bundle_does_not_store_where_a_provider_connects():
    ibm = serialize_provider_param(
        QiskitRuntimeConfig(service_kwargs={"url": "https://example.invalid"})
    )
    iqm = serialize_provider_param(IQMConfig(url="https://example.invalid"))
    assert "service_kwargs" not in ibm
    assert "url" not in iqm


def test_bundle_cannot_redirect_the_loaders_credentials():
    """A crafted bundle naming a host must not make load() connect there with
    the token from the loader's environment."""
    payload = {
        "__provider_config__": "IQMConfig",
        "url": "http://127.0.0.1:9/",
        "provider_kwargs": {"proxies": {"https": "http://127.0.0.1:9"}},
        "quantum_computer": "emerald",
    }
    with pytest.warns(UserWarning, match="cannot choose where a provider connects"):
        config = deserialize_provider_param(payload)
    assert config.url is None
    assert config.provider_kwargs == {}
    assert config.quantum_computer == "emerald"
