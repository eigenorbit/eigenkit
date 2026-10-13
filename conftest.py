import os


# PyTorch and CUDA Quantum each bundle their own OpenMP runtime. On macOS this
# causes libomp.dylib to be initialised twice, which triggers an abort() inside
# OpenMP unless we allow it.  KMP_DUPLICATE_LIB_OK must be set before either
# library is imported, so it lives here — the earliest point pytest touches.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# qiskit_aer's threaded simulator (used by FakeBackends) segfaults under
# torch's OpenMP runtime once enough state has accumulated in the pytest
# session. Pinning to a single thread sidesteps the conflict; we only
# pay the cost in tests that actually go through Aer (Qiskit Runtime
# fake backend).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("QISKIT_NUM_PROCS", "1")

# cudaq >= 0.15 puts its LLVM/MLIR into the global symbol scope, and triton
# (torch's compiler on Linux) bundles another LLVM that crashes in dlopen if it
# loads after them. Test modules import cudaq at collection, before
# the backend's own guard can order the two, so load triton here first.
try:
    import triton  # noqa: F401
except ImportError:
    pass


def pytest_configure(config):
    """Register custom markers + default-skip the QPU-only ones.

    Tests marked `@pytest.mark.qpu` consume real cloud / QPU compute time
    and require credentials. Run them explicitly with `pytest -m qpu`.
    """
    config.addinivalue_line(
        "markers",
        "qpu: requires real QPU/cloud access; skipped by default",
    )
    config.addinivalue_line(
        "markers",
        "gpu: requires CUDA (torch.cuda.is_available()); integration GPU smoke tests",
    )


def pytest_collection_modifyitems(config, items):
    """Skip QPU tests unless `-m qpu` asks for them, and `pennylane` tests when
    the extra is missing. A case parametrized with the provider `"pennylane"`
    counts as a `pennylane` test.

    PennyLane is its own extra, not part of `torch`, so a suite run against a
    `torch` install has to pass without it. Marking the tests that use it
    explicitly, rather than turning its ImportError into a skip, keeps a
    library path that starts needing PennyLane unasked a failure.
    """
    import importlib.util

    import pytest

    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec and any(
            isinstance(v, str) and v == "pennylane" for v in callspec.params.values()
        ):
            item.add_marker(pytest.mark.pennylane)

    if importlib.util.find_spec("pennylane") is None:
        skip_pennylane = pytest.mark.skip(reason="pennylane extra not installed")
        for item in items:
            if item.get_closest_marker("pennylane") is not None:
                item.add_marker(skip_pennylane)

    selected = config.getoption("-m", default=None)
    if selected and "qpu" in selected:
        return
    skip = pytest.mark.skip(reason="QPU/cloud test; run with `-m qpu` to enable")
    for item in items:
        if "qpu" in item.keywords:
            item.add_marker(skip)


def pytest_sessionfinish(session, exitstatus):
    """Bypass the libc atexit queue when running on Linux with cudaq loaded.

    libcudaq.so registers a C-level atexit handler that double-frees during
    glibc's `__run_exit_handlers` once the new torch / qiskit-aer destructors
    have run. The crash happens after `Py_Finalize` returns, so pytest's
    exit code reaches the runner as 139 (SIGSEGV) even though every test
    passed and coverage XML was written. `os._exit` skips the libc atexit
    queue entirely, sidestepping the bad handler. Drop this when upstream
    cudaq stops free()-ing already-freed memory.
    """
    import sys

    if sys.platform != "linux":
        return
    os._exit(exitstatus)
