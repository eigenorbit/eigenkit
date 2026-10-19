"""The blocks that make up a circuit, and the configurations of the backends.

Blocks are combined into a circuit with `Circuit(blocks=[...])`:

- `Encoding` encodes classical features as rotation angles.
- `Variational` applies, in each layer, one trainable rotation to every qubit,
  followed by a ladder of controlled-NOT (CNOT) gates.
- `StrongVariational` applies, in each layer, trainable `RZ`, `RY` and `RZ`
  rotations to every qubit, followed by a ladder of CNOT gates.
- `Reuploading` encodes the input several times, between trainable layers. The
  encoding can have a trainable scale.

A configuration selects a backend and holds its settings. Pass it to
`Circuit(provider=...)`, or register a name for it. `NumpyConfig` is documented
below, and the other configurations are documented with their backends:

- `NumpyConfig`: the reference simulator, which needs no extra dependencies
  (`provider="numpy"`).
- [`PennyLaneConfig`][eigenkit.core.backends.pennylane_backend.PennyLaneConfig]
- [`QiskitConfig`][eigenkit.core.backends.qiskit_backend.QiskitConfig]
- [`QiskitAerConfig`][eigenkit.core.backends.qiskit_aer_backend.QiskitAerConfig]:
  the Qiskit Aer simulator, with the statevector and the matrix product state
  methods.
- [`QiskitRuntimeConfig`][eigenkit.core.backends.qiskit_runtime_backend.QiskitRuntimeConfig]
- [`IQMConfig`][eigenkit.core.backends.iqm_backend.IQMConfig]
- [`BraketConfig`][eigenkit.core.backends.braket_backend.BraketConfig]
- [`CudaQuantumConfig`][eigenkit.core.backends.cudaquantum_backend.CudaQuantumConfig]
- [`TensorCircuitConfig`][eigenkit.core.backends.tensorcircuit_backend.TensorCircuitConfig]
- [`QulacsConfig`][eigenkit.core.backends.qulacs_backend.QulacsConfig]
- [`QsimConfig`][eigenkit.core.backends.qsim_backend.QsimConfig]
- `RouterConfig`: the EigenOrbit Router, which runs circuits on real quantum
  hardware with one EigenOrbit API key. It is the class of
  eigenorbit-router-client and needs the `router` extra
  (`pip install eigenkit[router]`).

The blocks and `NumpyConfig` are imported with this module, because they only
need NumPy. The other configurations are imported when they are first used, so
importing this module does not import PyTorch, PennyLane, Qiskit or Braket.

The quantities to measure are set with `Circuit(observables=...)`. They are not
a block.

Example:
    ```python
    import numpy as np
    from eigenkit import Circuit, all_z
    from eigenkit.configs import Encoding, Variational, NumpyConfig

    circuit = Circuit(
        blocks=[Encoding(in_features=3), Variational(depth=2)],
        observables=all_z(3),
        provider=NumpyConfig(),
    )

    print(circuit.spec.n_qubits, circuit.spec.n_weights)  # 3 6
    print(circuit(np.zeros(circuit.spec.n_weights), np.zeros(3)))  # [1. 1. 1.]
    ```

    To change the backend, you change one argument. `provider=PennyLaneConfig(...)`
    runs the same blocks on PennyLane, and `provider="numpy"` selects the same
    reference simulator by its registered name.
"""

import importlib

from .core.backends.numpy_backend import NumpyConfig
from .core.blocks import AngleEncoding as Encoding
from .core.blocks import BasicVariational as Variational
from .core.blocks import Reuploading, StrongVariational
from .core.provider import _EXTRA_HINT


# config name -> defining backend module (under eigenkit.core.backends)
_LAZY_CONFIGS = {
    "BraketConfig": "braket_backend",
    "CudaQuantumConfig": "cudaquantum_backend",
    "IQMConfig": "iqm_backend",
    "PennyLaneConfig": "pennylane_backend",
    "QiskitAerConfig": "qiskit_aer_backend",
    "QiskitConfig": "qiskit_backend",
    "QiskitRuntimeConfig": "qiskit_runtime_backend",
    "QsimConfig": "qsim_backend",
    "QulacsConfig": "qulacs_backend",
    "TensorCircuitConfig": "tensorcircuit_backend",
}


# Configs that live in another distribution: config name -> (module, extra).
# They stay out of `_LAZY_CONFIGS` and `__all__`, so a star import and the
# lazy-config roster never need that distribution.
_EXTERNAL_CONFIGS = {"RouterConfig": ("eigenorbit_router_client", "router")}


def __getattr__(name: str):
    external = _EXTERNAL_CONFIGS.get(name)
    if external is not None:
        module_name, extra = external
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            raise ImportError(
                f"{name} needs {module_name} ({_EXTRA_HINT[extra]})"
            ) from exc
        return getattr(module, name)
    module_name = _LAZY_CONFIGS.get(name)
    if module_name is not None:
        module = importlib.import_module(f"eigenkit.core.backends.{module_name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Encoding",
    "Variational",
    "StrongVariational",
    "Reuploading",
    "NumpyConfig",
    *_LAZY_CONFIGS,
]
