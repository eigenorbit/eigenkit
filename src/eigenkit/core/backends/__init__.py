"""Registration of the backends.

The NumPy reference simulator only needs NumPy, and it is registered when this
package is imported, because it is the default and is always available. The
other backends, apart from CUDA Quantum, are registered *lazily*: naming one
imports nothing, and its software development kit (PyTorch, PennyLane, Qiskit
or Braket) is imported only when a circuit first uses that provider. This keeps
`import eigenkit` and `import eigenkit.core` free of these large dependencies.

`"pennylane"`, `"qiskit"`, `"braket"`, `"router"` and the other optional
backends are registered only when their extra is installed. `cudaq` is not
registered automatically: once it is installed, register it explicitly:

```python
from eigenkit import CudaQuantumBackend, register

register("cudaq", CudaQuantumBackend())
```

The backend classes and the configuration dataclasses are re-exported lazily
through `__getattr__`. For example, `from eigenkit.core import PennyLaneConfig`
imports the PennyLane adapter only when the name is accessed.
"""

import importlib
import importlib.util

from ..provider import register, register_lazy
from .numpy_backend import NumpyConfig, NumpySimulator


# Pure numpy, no optional deps — always available.
register("numpy", NumpySimulator())


def _loader(module_name: str, class_name: str):
    """Build a zero-arg loader that imports a backend adapter on first use."""

    def load():
        module = importlib.import_module(f"{__name__}.{module_name}")
        return getattr(module, class_name)()

    return load


if importlib.util.find_spec("pennylane") is not None:
    register_lazy("pennylane", _loader("pennylane_backend", "PennyLaneBackend"))
if importlib.util.find_spec("qiskit") is not None:
    register_lazy("qiskit", _loader("qiskit_backend", "QiskitBackend"))
# qiskit-aer ships in the [qiskit] extra: statevector (exact) + MPS (bond-truncated) methods.
if importlib.util.find_spec("qiskit_aer") is not None:
    register_lazy("qiskit-aer", _loader("qiskit_aer_backend", "QiskitAerBackend"))

    def _qiskit_aer_mps():
        from .qiskit_aer_backend import QiskitAerConfig

        return QiskitAerConfig(method="matrix_product_state").make_backend()

    register_lazy("qiskit-aer-mps", _qiskit_aer_mps)
if importlib.util.find_spec("qsimcirq") is not None:
    register_lazy("qsim", _loader("qsim_backend", "QsimBackend"))


# IQM hardware through iqm-client's own Qiskit adapter (not through Braket).
# `iqm` is a namespace package shared by several IQM distributions, so probe
# the adapter module itself rather than the namespace; it also needs qiskit,
# which the [iqm] extra pulls in.
def _has_iqm_qiskit() -> bool:
    """Whether `iqm.qiskit_iqm` is importable.

    `find_spec` on a dotted name imports the parent package, and raises
    `ModuleNotFoundError` rather than returning None when the parent is absent
    — which is the usual case here, since iqm-client is an optional extra.
    """
    try:
        return importlib.util.find_spec("iqm.qiskit_iqm") is not None
    except ModuleNotFoundError:
        return False


if importlib.util.find_spec("qiskit") is not None and _has_iqm_qiskit():
    register_lazy("iqm", _loader("iqm_backend", "IQMBackend"))
if importlib.util.find_spec("braket") is not None:
    register_lazy("braket", _loader("braket_backend", "BraketBackend"))
if importlib.util.find_spec("tensorcircuit") is not None:
    register_lazy(
        "tensorcircuit", _loader("tensorcircuit_backend", "TensorCircuitBackend")
    )
if importlib.util.find_spec("qulacs") is not None:
    register_lazy("qulacs", _loader("qulacs_backend", "QulacsBackend"))


# EigenOrbit Router through eigenorbit-router-client (the `router` extra). The
# backend lives in that package, not under this one, so it gets its own loader.
# `find_spec` on a top-level name locates the package without importing it, so
# `import eigenkit` pulls neither the client nor httpx.
def _router_backend():
    return importlib.import_module("eigenorbit_router_client").RouterBackend()


if importlib.util.find_spec("eigenorbit_router_client") is not None:
    register_lazy("router", _router_backend)


# Lazy class/config re-exports: name -> defining module (under this package).
_LAZY_EXPORTS = {
    "BraketBackend": "braket_backend",
    "BraketConfig": "braket_backend",
    "BraketJobHandle": "braket_backend",
    "CudaQuantumBackend": "cudaquantum_backend",
    "CudaQuantumConfig": "cudaquantum_backend",
    "IQMBackend": "iqm_backend",
    "IQMConfig": "iqm_backend",
    "IQMJobHandle": "iqm_backend",
    "PennyLaneBackend": "pennylane_backend",
    "PennyLaneConfig": "pennylane_backend",
    "QiskitBackend": "qiskit_backend",
    "QiskitConfig": "qiskit_backend",
    "QiskitAerBackend": "qiskit_aer_backend",
    "QiskitAerConfig": "qiskit_aer_backend",
    "QsimBackend": "qsim_backend",
    "QsimConfig": "qsim_backend",
    "QiskitRuntimeBackend": "qiskit_runtime_backend",
    "QiskitRuntimeConfig": "qiskit_runtime_backend",
    "QiskitRuntimeJobHandle": "qiskit_runtime_backend",
    "QulacsBackend": "qulacs_backend",
    "QulacsConfig": "qulacs_backend",
    "TensorCircuitBackend": "tensorcircuit_backend",
    "TensorCircuitConfig": "tensorcircuit_backend",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        module = importlib.import_module(f"{__name__}.{module_name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NumpyConfig",
    "NumpySimulator",
    *_LAZY_EXPORTS,
]
