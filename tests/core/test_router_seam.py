"""The `router` provider seam to eigenorbit-router-client.

eigenkit registers `router` lazily when the client package is importable, and
the client registers the same name itself when it is imported. Both loaders
build the client's `RouterBackend`, so the import order must not change what
`get("router")` does. The client is not a test dependency: these tests write a
small stand-in package with the client's shape into a temporary directory and
run each check in a fresh interpreter, so the result does not depend on what the
test session has already imported.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]

_NO_DEVICE = (
    "provider='router' needs a device: pass provider=RouterConfig(device='braket/sv1')"
)

# The client's public shape: a backend that refuses to be built without a
# device, a config whose API key is a secret field, and the client's own
# registration of `router`, run last on import.
_FAKE_CLIENT = f"""
from dataclasses import KW_ONLY, dataclass, field

from eigenkit.core.provider import register_lazy
from eigenkit.core.secret_fields import SecretFieldsMixin


class RouterBackend:
    def __init__(self, config=None):
        if config is None:
            raise ValueError({_NO_DEVICE!r})
        self.config = config


@dataclass(frozen=True)
class RouterConfig(SecretFieldsMixin):
    device: str
    _: KW_ONLY
    api_key: str | None = field(default=None, repr=False, metadata={{"secret": True}})
    base_url: str | None = None

    def make_backend(self):
        return RouterBackend(config=self)


register_lazy("router", lambda: RouterBackend())
"""

_CIRCUIT = """
import eigenkit as ek

def router_circuit():
    return ek.Circuit(
        [ek.Encoding(in_features=2), ek.Variational(depth=1, rotation="RY")],
        ek.all_z(2),
        provider="router",
        interface="numpy",
    )
"""


@pytest.fixture
def fake_client(tmp_path: Path) -> Path:
    """Write the stand-in client package and return the directory to put on the path."""
    package = tmp_path / "eigenorbit_router_client"
    package.mkdir()
    (package / "__init__.py").write_text(_FAKE_CLIENT)
    return tmp_path


def _run(code: str, pythonpath: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_router_is_not_offered_without_the_client():
    # A `None` entry in sys.modules makes the package unimportable, and
    # `find_spec` then reports it missing, whatever the environment holds.
    result = _run(
        """
        import sys
        sys.modules["eigenorbit_router_client"] = None
        from eigenkit.core.provider import get, registered_names
        assert "router" not in registered_names(), registered_names()
        try:
            get("router")
        except KeyError as error:
            assert "pip install eigenkit[router]" in str(error), str(error)
        else:
            raise AssertionError("router resolved without the client")
        """
    )
    assert result.returncode == 0, result.stderr


def test_import_eigenkit_registers_router_without_importing_the_client(fake_client):
    result = _run(
        """
        import sys
        import eigenkit  # noqa: F401
        from eigenkit.core.provider import registered_names
        assert "router" in registered_names(), registered_names()
        loaded = sorted(
            m for m in ("eigenorbit_router_client", "httpx") if m in sys.modules
        )
        assert not loaded, f"import eigenkit pulled: {loaded}"
        """,
        fake_client,
    )
    assert result.returncode == 0, result.stderr


def test_router_without_a_device_names_router_config_eigenkit_first(fake_client):
    result = _run(
        _CIRCUIT
        + """
from eigenkit.core.provider import get

for attempt in range(2):
    try:
        get("router")
    except ValueError as error:
        assert "RouterConfig(device=" in str(error), str(error)
    else:
        raise AssertionError("router built a backend without a device")

try:
    router_circuit()
except ValueError as error:
    assert "RouterConfig(device=" in str(error), str(error)
else:
    raise AssertionError("Circuit accepted provider='router' without a device")
""",
        fake_client,
    )
    assert result.returncode == 0, result.stderr


def test_router_without_a_device_names_router_config_client_first(fake_client):
    # The client registers `router` on import as well; whichever registration
    # wins, the lookup must fail the same way.
    result = _run(
        """
import eigenorbit_router_client  # noqa: F401
"""
        + _CIRCUIT
        + """
from eigenkit.core.provider import get, registered_names

assert "router" in registered_names(), registered_names()
try:
    get("router")
except ValueError as error:
    assert "RouterConfig(device=" in str(error), str(error)
else:
    raise AssertionError("router built a backend without a device")

try:
    router_circuit()
except ValueError as error:
    assert "RouterConfig(device=" in str(error), str(error)
else:
    raise AssertionError("Circuit accepted provider='router' without a device")
""",
        fake_client,
    )
    assert result.returncode == 0, result.stderr


def test_router_extra_pins_the_client_and_is_part_of_all():
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    assert extras["router"] == ["eigenorbit-router-client>=0.1,<1"]
    assert any("router" in requirement for requirement in extras["all"]), extras["all"]


def test_configs_router_config_is_the_clients_class(fake_client):
    result = _run(
        """
        import eigenkit.configs as configs
        import eigenorbit_router_client
        from eigenkit.configs import RouterConfig

        assert RouterConfig is eigenorbit_router_client.RouterConfig
        assert "RouterConfig" not in configs._LAZY_CONFIGS
        assert "RouterConfig" not in configs.__all__
        """,
        fake_client,
    )
    assert result.returncode == 0, result.stderr


def test_configs_router_config_names_the_extra_without_the_client():
    result = _run(
        """
        import sys
        sys.modules["eigenorbit_router_client"] = None
        try:
            from eigenkit.configs import RouterConfig  # noqa: F401
        except ImportError as error:
            assert "pip install eigenkit[router]" in str(error), str(error)
            assert not isinstance(error, ModuleNotFoundError), repr(error)
        else:
            raise AssertionError("RouterConfig imported without the client")
        """
    )
    assert result.returncode == 0, result.stderr


def test_configs_star_import_does_not_need_the_client():
    result = _run(
        """
        import sys
        sys.modules["eigenorbit_router_client"] = None
        from eigenkit.configs import *  # noqa: F403
        assert "NumpyConfig" in globals()
        assert "RouterConfig" not in globals()
        """
    )
    assert result.returncode == 0, result.stderr


def test_configs_router_config_keeps_a_missing_dependency_of_the_client(tmp_path):
    # A dependency missing inside the installed client is a different fault from
    # a missing client, and its error must reach the caller unchanged.
    package = tmp_path / "eigenorbit_router_client"
    package.mkdir()
    (package / "__init__.py").write_text("import a_dependency_that_is_missing\n")
    result = _run(
        """
        try:
            from eigenkit.configs import RouterConfig  # noqa: F401
        except ModuleNotFoundError as error:
            assert error.name == "a_dependency_that_is_missing", repr(error)
        else:
            raise AssertionError("RouterConfig imported from a broken client")
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_configs_unknown_name_is_still_an_attribute_error():
    import eigenkit.configs as configs

    with pytest.raises(AttributeError, match="has no attribute 'NoSuchConfig'"):
        configs.NoSuchConfig  # noqa: B018


def test_bundle_cannot_send_the_router_key_to_a_host_it_names(fake_client):
    # A bundle is data from whoever shared it; loading builds the router backend
    # with the loader's API key, which must only go to the URL the loader sets.
    result = _run(
        """
        import warnings
        import eigenorbit_router_client
        from eigenkit.qml._serialization import deserialize_provider_param

        payload = {
            "__provider_config__": "RouterConfig",
            "base_url": "https://attacker.example",
            "device": "braket/sv1",
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = deserialize_provider_param(payload)
        messages = [str(warning.message) for warning in caught]
        assert any(
            "cannot choose where a provider connects" in message
            and "base_url" in message
            for message in messages
        ), messages
        assert isinstance(config, eigenorbit_router_client.RouterConfig)
        assert config.base_url is None
        assert config.api_key is None
        assert config.device == "braket/sv1"
        """,
        fake_client,
    )
    assert result.returncode == 0, result.stderr


def test_bundle_cannot_make_the_loader_run_under_its_own_key(fake_client):
    # A key in the bundle would send the loader's jobs and inputs to the bundle
    # author's account; the provider must read the loader's own key instead.
    result = _run(
        """
        import warnings
        import eigenorbit_router_client
        from eigenkit.qml._serialization import deserialize_provider_param

        payload = {
            "__provider_config__": "RouterConfig",
            "api_key": "eo_live_bundle_author_key",
            "device": "braket/sv1",
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = deserialize_provider_param(payload)
        messages = [str(warning.message) for warning in caught]
        assert any(
            "cannot carry credentials" in message and "api_key" in message
            for message in messages
        ), messages
        assert isinstance(config, eigenorbit_router_client.RouterConfig)
        assert config.api_key is None
        assert config.device == "braket/sv1"
        """,
        fake_client,
    )
    assert result.returncode == 0, result.stderr


def test_bundle_stores_neither_the_router_key_nor_its_url(fake_client):
    result = _run(
        """
        import json
        from eigenkit.configs import RouterConfig
        from eigenkit.qml._serialization import serialize_provider_param

        config = RouterConfig(
            device="braket/sv1",
            api_key="a-router-key-to-keep-out",
            base_url="https://router.example",
        )
        payload = serialize_provider_param(config)
        assert payload == {
            "__provider_config__": "RouterConfig",
            "device": "braket/sv1",
        }, payload
        text = json.dumps(payload)
        assert "a-router-key-to-keep-out" not in text
        assert "router.example" not in text
        """,
        fake_client,
    )
    assert result.returncode == 0, result.stderr


def test_saved_model_with_a_router_provider_loads(fake_client, tmp_path):
    # The model is fitted on the reference simulator, then given a router
    # provider and saved; the bundle is edited to name a host, as a shared
    # bundle could, and loads with the router config but without the host.
    # Loading rebuilds the kernel on the provider's backend, and the stand-in
    # client has no working one, so the reference simulator stands in for it.
    result = _run(
        f"""
        import json
        import warnings
        import zipfile

        import numpy as np

        import eigenorbit_router_client
        from eigenkit.configs import NumpyConfig
        from eigenkit.qml import QSVC

        eigenorbit_router_client.RouterConfig.make_backend = (
            lambda self: NumpyConfig().make_backend()
        )

        rng = np.random.default_rng(0)
        X = rng.uniform(size=(8, 2))
        y = np.array([0, 1] * 4)
        model = QSVC(feature_map="Angle", reps=1, provider="numpy").fit(X, y)
        predictions = model.predict(X)
        model.provider = eigenorbit_router_client.RouterConfig(
            device="braket/sv1",
            api_key="a-router-key-to-keep-out",
            base_url="https://router.example",
        )
        bundle_path = {str(tmp_path / "router.eigenkit")!r}
        model.save(bundle_path)

        with zipfile.ZipFile(bundle_path) as archive:
            entries = {{name: archive.read(name) for name in archive.namelist()}}
        params = json.loads(entries["params.json"])
        assert params["provider"] == {{
            "__provider_config__": "RouterConfig",
            "device": "braket/sv1",
        }}, params["provider"]
        params["provider"]["base_url"] = "https://attacker.example"
        entries["params.json"] = json.dumps(params)
        with zipfile.ZipFile(bundle_path, "w") as archive:
            for name, data in entries.items():
                archive.writestr(name, data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = QSVC.load(bundle_path)
        assert any(
            "cannot choose where a provider connects" in str(warning.message)
            for warning in caught
        )
        assert isinstance(loaded.provider, eigenorbit_router_client.RouterConfig)
        assert loaded.provider.device == "braket/sv1"
        assert loaded.provider.base_url is None
        assert loaded.provider.api_key is None
        np.testing.assert_array_equal(loaded.predict(X), predictions)
        """,
        fake_client,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_provider_config_lists_router_config():
    from eigenkit.qml._serialization import (
        SerializationError,
        deserialize_provider_param,
    )

    with pytest.raises(SerializationError, match="RouterConfig"):
        deserialize_provider_param({"__provider_config__": "NoSuchConfig"})
