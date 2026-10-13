"""The core must import and run with no framework installed.

These tests spawn a *fresh* interpreter so the check is valid regardless of
what the current test session has already imported (the full suite loads
torch; a subprocess does not inherit that). They guard the lazy-import
architecture: `import eigenkit` / `import eigenkit.core`
must pull no torch / pennylane / qiskit / braket, and the numpy provider must
run without them.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest


_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
_HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_top_import_is_framework_free():
    result = _run(
        """
        import sys
        import eigenkit  # noqa: F401
        heavy = sorted(
            m for m in ("torch", "pennylane", "qiskit", "braket") if m in sys.modules
        )
        assert not heavy, f"import eigenkit pulled: {heavy}"
        """
    )
    assert result.returncode == 0, result.stderr


def test_core_import_is_framework_free():
    result = _run(
        """
        import sys
        import eigenkit.core  # noqa: F401
        heavy = sorted(
            m for m in ("torch", "pennylane", "qiskit", "braket") if m in sys.modules
        )
        assert not heavy, f"import eigenkit.core pulled: {heavy}"
        """
    )
    assert result.returncode == 0, result.stderr


def test_numpy_circuit_runs_without_frameworks():
    result = _run(
        """
        import sys
        import numpy as np
        import eigenkit as ek
        circuit = ek.Circuit(
            [ek.Encoding(in_features=2), ek.Variational(depth=1, rotation="RY")],
            ek.all_z(2),
            provider="numpy",
            interface="numpy",
        )
        # Two qubits, depth 1 -> two weights. `Circuit` checks lengths, so
        # extra weights are refused rather than read past in silence.
        out = circuit(np.array([0.1, 0.2]), np.array([0.5, -0.3]))
        assert out.shape == (2,), out.shape
        heavy = sorted(
            m for m in ("torch", "pennylane", "qiskit", "braket") if m in sys.modules
        )
        assert not heavy, f"running the numpy provider pulled: {heavy}"
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not _HAS_PENNYLANE, reason="pennylane extra not installed")
def test_pennylane_backend_numpy_path_is_torch_free():
    # The PennyLane backend has a numpy `run` path, and the `pennylane` / `qml`
    # extras do not install torch — so importing the backend and running a
    # circuit on the numpy interface must not pull torch. (torch IS installed
    # in this env; we assert it is not *loaded*, which would catch a regression
    # to a module-level `import torch` in pennylane_backend.)
    result = _run(
        """
        import sys
        import numpy as np
        import eigenkit as ek
        circuit = ek.Circuit(
            [ek.Encoding(in_features=2), ek.Variational(depth=1)],
            ek.all_z(2),
            provider="pennylane",
            interface="numpy",
        )
        circuit(np.zeros(circuit.spec.n_weights), np.array([0.1, 0.2]))
        assert "torch" not in sys.modules, "the pennylane numpy path pulled torch"
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not _HAS_SKLEARN, reason="qml tier needs the sklearn extra")
def test_qml_tier_is_torch_free():
    # The qml tier (kernels + SVMs) may use scikit-learn, but a torch-free
    # install must still be able to import and run it: the kernels run on the
    # numpy interface.
    result = _run(
        """
        import sys
        import numpy as np
        from eigenkit.qml import QSVC, QuantumKernel  # noqa: F401
        assert "torch" not in sys.modules, "qml import pulled torch"
        kernel = QuantumKernel(n_features=2, feature_map="Angle", reps=1, provider="numpy")
        kernel.kernel_value(np.array([0.3, 0.7]), np.array([0.3, 0.7]))
        assert "torch" not in sys.modules, "running the kernel pulled torch"
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not _HAS_TORCH, reason="torch extra not installed")
def test_accessing_torch_estimator_is_lazy_then_loads_torch():
    # Sanity check the other direction: the torch ring is NOT loaded on import,
    # but accessing a torch estimator pulls torch on demand.
    result = _run(
        """
        import sys
        import eigenkit as ek
        assert "torch" not in sys.modules
        _ = ek.HQNC
        assert "torch" in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


def test_hhl_and_iqae_import_without_scikit_learn():
    """HHL and IQAE need only the core; the sklearn-based portfolio estimator
    beside them must not be imported along with them."""
    result = _run(
        """
        import sys
        for blocked in ("sklearn", "torch", "pennylane"):
            sys.modules[blocked] = None
        import numpy as np
        from eigenkit import HHL, IQAE  # noqa: F401
        result = HHL().solve(np.diag([1.0, 2.0]), np.array([1.0, 0.0]))
        assert result is not None
        """
    )
    assert result.returncode == 0, result.stderr


def test_a_missing_extra_names_the_install_command():
    """A name whose extra is missing says what to install, and the configs
    stay importable without their SDKs."""
    result = _run(
        """
        import sys
        for blocked in ("sklearn", "torch", "pennylane"):
            sys.modules[blocked] = None
        import eigenkit
        from eigenkit import PennyLaneConfig  # noqa: F401
        for name, extra in (
            ("HQNC", "torch"),
            ("QSVC", "qml"),
            ("PortfolioRiskEstimator", "qml"),
            ("benchmark", "benchmark"),
        ):
            try:
                getattr(eigenkit, name)
            except ModuleNotFoundError as error:
                assert f"pip install 'eigenkit[{extra}]'" in str(error), str(error)
            else:
                raise AssertionError(f"{name} imported without its extra")
        """
    )
    assert result.returncode == 0, result.stderr
