"""Suite-wide fixtures.

The provider registry is process-global, so a test that registers an opt-in
backend changes what every later test sees; left alone, that would leak for the
rest of the session. Snapshotting here covers every directory, so no test can
leak a registration into another -- `cudaq` is the one backend registered by
hand, and a lazy backend materialized mid-test is restored too.
"""

from __future__ import annotations

import pytest

from eigenkit.core.provider import _LAZY, _REGISTRY


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Snapshot and restore the provider registry around each test so custom
    registrations (and lazy backends materialized during a test) don't leak.

    Both the eager registry and the lazy-loader table are saved; the snapshot
    captures the package-load defaults (``numpy`` eager + ``pennylane`` /
    ``qiskit`` / ``braket`` lazy), so restoring returns to a clean state."""
    saved_registry = dict(_REGISTRY)
    saved_lazy = dict(_LAZY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved_registry)
    _LAZY.clear()
    _LAZY.update(saved_lazy)
