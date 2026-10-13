"""Secret config fields: pickling drops them, copying keeps them.

`QiskitRuntimeConfig.token` and `IQMConfig.token` are marked secret. A pickle
(and so a joblib checkpoint) of either config, or of anything holding one, must
not carry the token, while `copy.copy`, `copy.deepcopy` and therefore
`sklearn.base.clone` must keep it so that cross-validation still authenticates.

Both config modules import without their vendor SDKs, so this file runs on a
core install.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle
from dataclasses import dataclass, field

import pytest

from eigenkit.configs import BraketConfig, IQMConfig, QiskitRuntimeConfig
from eigenkit.core.secret_fields import SecretFieldsMixin, secret_field_names


SECRET = "tok-secret-9c41e7b2"


def _configs():
    return [
        QiskitRuntimeConfig(token=SECRET, instance="crn:x"),
        IQMConfig(token=SECRET, url="https://example.invalid/"),
    ]


_IDS = ["qiskit-runtime", "iqm"]


class _Holder:
    """Stands in for an estimator holding a config, as joblib pickles it."""

    def __init__(self, provider):
        self.provider = provider


@pytest.mark.parametrize("cfg", _configs(), ids=_IDS)
def test_pickle_drops_the_token(cfg):
    data = pickle.dumps(cfg)
    assert SECRET.encode() not in data
    assert pickle.loads(data) == dataclasses.replace(cfg, token=None)


@pytest.mark.parametrize("cfg", _configs(), ids=_IDS)
def test_pickle_keeps_every_other_field(cfg):
    restored = pickle.loads(pickle.dumps(cfg))
    for f in dataclasses.fields(cfg):
        if f.name != "token":
            assert getattr(restored, f.name) == getattr(cfg, f.name)


@pytest.mark.parametrize("cfg", _configs(), ids=_IDS)
def test_nested_pickle_drops_the_token(cfg):
    data = pickle.dumps(_Holder(cfg))
    assert SECRET.encode() not in data
    assert pickle.loads(data).provider.token is None


@pytest.mark.parametrize("cfg", _configs(), ids=_IDS)
def test_copy_keeps_the_token(cfg):
    clone = copy.copy(cfg)
    assert clone == cfg
    assert clone.token == SECRET


@pytest.mark.parametrize("cfg", _configs(), ids=_IDS)
def test_deepcopy_keeps_the_token(cfg):
    clone = copy.deepcopy(cfg)
    assert clone == cfg
    assert clone is not cfg
    assert clone.token == SECRET


@pytest.mark.parametrize("cfg", _configs(), ids=_IDS)
def test_deepcopy_goes_through_the_memo(cfg):
    first, second = copy.deepcopy([cfg, cfg])
    assert first is second
    assert first.token == SECRET


@pytest.mark.parametrize("cfg", _configs(), ids=_IDS)
def test_repr_and_str_hide_the_token(cfg):
    assert SECRET not in repr(cfg)
    assert SECRET not in str(cfg)


@pytest.mark.parametrize("cfg", _configs(), ids=_IDS)
def test_token_is_the_only_secret_field(cfg):
    assert secret_field_names(cfg) == frozenset({"token"})


def test_config_without_secrets_has_no_secret_fields():
    assert secret_field_names(BraketConfig()) == frozenset()


def test_secret_field_names_rejects_a_non_dataclass():
    with pytest.raises(TypeError):
        secret_field_names(object())


@pytest.mark.parametrize("cls", [QiskitRuntimeConfig, IQMConfig])
def test_token_defaults_to_none(cls):
    # Unset, the backends keep resolving the token from the environment.
    assert cls().token is None
    assert {f.name: f.default for f in dataclasses.fields(cls)}["token"] is None


@dataclass(frozen=True)
class _FactoryCfg(SecretFieldsMixin):
    name: str = "x"
    headers: dict = field(default_factory=dict, repr=False, metadata={"secret": True})


def test_secret_with_a_default_factory_takes_a_fresh_default_on_load():
    restored = pickle.loads(pickle.dumps(_FactoryCfg(headers={"auth": SECRET})))
    assert restored == _FactoryCfg()
    assert restored.headers == {}
