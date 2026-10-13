"""Tests for Provider protocol, registry, validation, parameter-shift."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest


# These two backend modules import their SDK at module scope, so naming them
# here would make the file uncollectable on a core install. Skip instead.
pytest.importorskip("pennylane")
pytest.importorskip("qiskit")

from eigenkit.core.backends.pennylane_backend import (
    PennyLaneBackend,  # noqa: E402
)
from eigenkit.core.backends.qiskit_backend import QiskitBackend  # noqa: E402
from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
from eigenkit.core.ir import CircuitSpec, Gate, Observable
from eigenkit.core.observables import all_z
from eigenkit.core.provider import (
    _EXTRA_HINT,
    _LAZY,
    _REGISTRY,
    Capabilities,
    get,
    parameter_shift_jacobian,
    register,
    register_lazy,
    registered_names,
    validate_circuit,
)


_HAS_QISKIT = importlib.util.find_spec("qiskit") is not None


class TestRegistry:
    def test_registered_names(self):
        names = registered_names()
        assert "pennylane" in names
        if _HAS_QISKIT:
            assert "qiskit" in names

    def test_get_returns_backend(self):
        backend = get("pennylane")
        assert backend.name == "pennylane"

    def test_get_missing_raises(self):
        with pytest.raises(KeyError, match="no provider"):
            get("does-not-exist")

    def test_re_register_overwrites(self):
        """Registering the same name twice takes the most recent one."""
        custom = PennyLaneBackend()
        register("pennylane", custom)
        assert get("pennylane") is custom

    def test_custom_name(self):
        lightning = PennyLaneBackend()
        register("pennylane-lightning", lightning)
        assert get("pennylane-lightning") is lightning
        assert "pennylane-lightning" in registered_names()

    def test_register_accepts_config(self):
        """register() can take a config directly — it materializes the backend."""
        from eigenkit.core import PennyLaneConfig

        register("pl-alt", PennyLaneConfig(device_name="default.qubit"))
        backend = get("pl-alt")
        assert backend.config.device_name == "default.qubit"


class TestValidation:
    def test_accepts_supported_gates(self):
        ir = CircuitSpec(
            n_qubits=1,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("H", (0,)),),
        )
        # Should not raise
        validate_circuit(ir, PennyLaneBackend().capabilities)

    def test_rejects_unsupported_gate(self):
        ir = CircuitSpec(
            n_qubits=1,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("TOFFOLI", (0,), ()),),
        )
        with pytest.raises(ValueError, match="TOFFOLI"):
            validate_circuit(ir, PennyLaneBackend().capabilities)

    def test_rejects_too_many_qubits(self):
        capabilities = Capabilities(
            supported_gates=frozenset({"H"}),
            supports_batching=False,
            supports_shots=False,
            max_qubits=2,
        )
        ir = CircuitSpec(
            n_qubits=5,
            n_weights=0,
            n_inputs=0,
            gates=(Gate("H", (0,)),),
        )
        with pytest.raises(ValueError, match="5 qubits"):
            validate_circuit(ir, capabilities)


class TestCapabilities:
    def test_pennylane_declaration(self):
        caps = PennyLaneBackend().capabilities
        assert "RX" in caps.supported_gates
        assert "CNOT" in caps.supported_gates
        assert caps.max_qubits is None

    def test_qiskit_declaration(self):
        caps = QiskitBackend().capabilities
        assert "RX" in caps.supported_gates


@pytest.mark.skipif(not _HAS_QISKIT, reason="requires qiskit extra")
class TestParameterShiftJacobian:
    def test_shape(self):
        ir = assemble(
            [
                AngleEncoding(in_features=2),
                BasicVariational(depth=1, rotation="RY"),
            ]
        )
        observables = all_z(2)
        compiled = QiskitBackend().compile(ir, observables)
        weights = np.array([0.3, 0.7], dtype=np.float64)
        inputs = np.array([0.1, 0.2], dtype=np.float64)
        jacobian = parameter_shift_jacobian(compiled, weights, inputs)
        assert jacobian.shape == (2, 2)  # (n_observables, n_weights)

    def test_matches_finite_difference(self):
        ir = assemble(
            [
                AngleEncoding(in_features=2),
                BasicVariational(depth=1, rotation="RY"),
            ]
        )
        observables = all_z(2)
        compiled = QiskitBackend().compile(ir, observables)
        weights = np.array([0.3, 0.7], dtype=np.float64)
        inputs = np.array([0.1, 0.2], dtype=np.float64)

        analytic = parameter_shift_jacobian(compiled, weights, inputs)

        # Numerical reference: central difference with a small epsilon
        eps = 1e-5
        numerical = np.zeros_like(analytic)
        for i in range(len(weights)):
            plus = weights.copy()
            plus[i] += eps
            minus = weights.copy()
            minus[i] -= eps
            numerical[:, i] = (
                compiled.run(plus, inputs) - compiled.run(minus, inputs)
            ) / (2 * eps)

        assert np.allclose(analytic, numerical, atol=1e-6)


class TestBackendNamesOverride:
    def test_pennylane_custom_name(self):
        backend = PennyLaneBackend(name="my-pennylane")
        assert backend.name == "my-pennylane"

    def test_qiskit_custom_name(self):
        backend = QiskitBackend(name="my-qiskit")
        assert backend.name == "my-qiskit"


@pytest.mark.skipif(not _HAS_QISKIT, reason="requires qiskit extra")
class TestQiskitShotsNotWired:
    def test_shots_raises(self):
        from eigenkit.core.backends.qiskit_backend import QiskitConfig

        backend = QiskitBackend(config=QiskitConfig(shots=100))
        ir = CircuitSpec(n_qubits=1, n_weights=0, n_inputs=0, gates=(Gate("H", (0,)),))
        with pytest.raises(NotImplementedError, match="shots"):
            backend.compile(ir, [Observable.z(0)])


class TestLazyBackendWithoutItsPackage:
    """A registered backend whose SDK is missing must say so, and stay registered.

    A failed import must not deregister the backend for the rest of the
    process. Every attempt reports the missing package; none claims the name is
    not registered at all — a different problem, and not a true one.
    """

    @staticmethod
    def _register_broken(name):
        """Register `name` with a loader that fails the way a missing SDK does."""

        def loader():
            raise ModuleNotFoundError("No module named 'not_installed_sdk'")

        register_lazy(name, loader)

    def test_the_error_names_the_missing_package(self, monkeypatch):
        monkeypatch.setitem(_EXTRA_HINT, "brokenbackend", "install with `pip install x`")
        self._register_broken("brokenbackend")
        try:
            with pytest.raises(ImportError, match="registered but its package is not"):
                get("brokenbackend")
        finally:
            _LAZY.pop("brokenbackend", None)

    def test_the_error_carries_the_extra_hint(self, monkeypatch):
        monkeypatch.setitem(_EXTRA_HINT, "brokenbackend", "install with `pip install x`")
        self._register_broken("brokenbackend")
        try:
            with pytest.raises(ImportError, match=r"pip install x"):
                get("brokenbackend")
        finally:
            _LAZY.pop("brokenbackend", None)

    def test_an_installed_package_that_fails_to_load_is_not_called_missing(self):
        """On Linux ARM, Qiskit is installed yet fails with a static TLS error once
        PyTorch and scikit-learn are loaded; "not installed" would send users to
        reinstall."""

        def loader():
            raise ImportError(
                "_accelerate.abi3.so: cannot allocate memory in static TLS block"
            )

        register_lazy("brokenbackend", loader)
        try:
            with pytest.raises(
                ImportError, match="installed but its package failed"
            ) as e:
                get("brokenbackend")
            assert "GLIBC_TUNABLES=glibc.rtld.optional_static_tls" in str(e.value)
        finally:
            _LAZY.pop("brokenbackend", None)

    def test_the_cudaq_probe_tells_installed_from_missing(self, monkeypatch):
        """The same split for CUDA-Q, whose bundled OpenMP runtime hits the limit."""
        from eigenkit.core.backends.cudaquantum_backend import _require_cudaq

        class Unloadable:  # cudaq is installed, but its shared library cannot load
            def find_spec(self, name, path=None, target=None):
                if name == "cudaq":
                    raise ImportError(
                        "libgomp.so: cannot allocate memory in static TLS block"
                    )

        monkeypatch.delitem(sys.modules, "cudaq", raising=False)
        monkeypatch.setattr(sys, "meta_path", [Unloadable(), *sys.meta_path])
        with pytest.raises(
            ImportError, match="installed but failed to import.*GLIBC_TUNABLES"
        ):
            _require_cudaq()
        monkeypatch.setitem(sys.modules, "cudaq", None)  # import fails as "not found"
        with pytest.raises(ImportError, match="cuda-quantum is not installed"):
            _require_cudaq()

    def test_the_registration_survives_a_failed_load(self):
        self._register_broken("brokenbackend")
        try:
            with pytest.raises(ImportError):
                get("brokenbackend")
            assert "brokenbackend" in registered_names()
            # The second attempt repeats the same true message rather than
            # inventing an unknown-name error.
            with pytest.raises(ImportError, match="registered but its package is not"):
                get("brokenbackend")
        finally:
            _LAZY.pop("brokenbackend", None)

    def test_a_backend_that_installs_later_resolves_without_a_restart(self):
        state = {"installed": False}

        def loader():
            if not state["installed"]:
                raise ModuleNotFoundError("No module named 'not_installed_sdk'")
            return get("numpy")

        register_lazy("laterbackend", loader)
        try:
            with pytest.raises(ImportError):
                get("laterbackend")
            state["installed"] = True
            assert get("laterbackend") is get("numpy")
        finally:
            _LAZY.pop("laterbackend", None)
            _REGISTRY.pop("laterbackend", None)

    def test_an_unknown_name_is_still_a_key_error(self):
        with pytest.raises(KeyError, match="no provider registered"):
            get("no-such-backend")

    def test_pennylane_has_an_install_hint(self):
        # PennyLane is its own extra, not part of a bare install, so asking for
        # it there must say what to install.
        assert "pennylane" in _EXTRA_HINT


class TestInstallHintsAreActionable:
    """A failed provider lookup must tell the user something that works.

    `cudaq` is never registered by import — it is opt-in, and the user has to
    call `register(...)` themselves. A hint naming only the extra would lead to
    installing a package and meeting the identical error. Every lazily
    registered name needs a hint too.
    """

    def test_every_hint_names_a_command(self):
        for name, hint in _EXTRA_HINT.items():
            assert "pip install" in hint or "register(" in hint, name

    def test_the_opt_in_backends_say_so(self):
        for name in ("cudaq",):
            assert "register(" in _EXTRA_HINT[name], (
                f"{name} is never registered by import, so its hint must say "
                f"how to register it, not only how to install it"
            )

    def test_every_lazily_registered_name_has_a_hint(self):
        # A name that resolves only when its SDK is importable will be missing
        # for someone; a bare KeyError is a guaranteed issue report.
        missing = sorted(set(_LAZY) - set(_EXTRA_HINT) - {"numpy"})
        assert missing == [], f"lazily registered without an install hint: {missing}"

    def test_an_unknown_name_still_lists_what_is_registered(self):
        with pytest.raises(KeyError, match="registered:"):
            get("definitely-not-a-backend")
