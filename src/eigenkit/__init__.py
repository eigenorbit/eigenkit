"""eigenkit: a lightweight numpy core for variational quantum
circuits, with optional rings for torch (differentiable nn.Module circuits +
estimators) and pluggable backends.

Importing this package is cheap: only the numpy/scipy/pydantic core loads
eagerly. Everything heavier — the torch ring (estimators, layers, the
nn.Module circuit), the backend SDKs (pennylane, qiskit, braket, ...), and
the sklearn-based `datasets`/`benchmark` subpackages — is resolved lazily on
first access via the module-level ``__getattr__`` below. So a chemistry or
optimization user who only touches the numpy path never imports torch.

``eigenkit.__version__`` goes through the same hook: it reports the
version of the installed distribution, read from its metadata on first access.

A name whose extra is missing raises ``ModuleNotFoundError`` with the
``pip install`` command that brings it. One caveat follows: ``from eigenkit
import *`` resolves every name in ``__all__``, so a star import needs every
extra (``pip install 'eigenkit[all]'``) and pulls torch, sklearn and pennylane.
Import the names you use instead.
"""

from . import configs
from .circuit import Circuit
from .configs import (
    Encoding,
    NumpyConfig,
    Reuploading,
    StrongVariational,
    Variational,
)
from .core import (
    CircuitBuilder,
    CostWarning,
    JobHandle,
    NumpySimulator,
    Observable,
    VariationalResult,
    X,
    Y,
    Z,
    adjoint,
    all_z,
    assemble,
    from_string,
    minimize_expectation,
    register,
    registered_names,
)


# Lazily-resolved public names: name -> module that defines it. Accessing any
# of these triggers the import (and, for the torch names, pulls torch); a bare
# `import eigenkit` does not.
_LAZY_SUBMODULES = {
    "algorithms",
    "benchmark",
    "chemistry",
    "datasets",
    "optimization",
    "qml",
    "reservoir",
    "torch",
}
_LAZY_NAMES = {
    # qml tier (sklearn, torch-free) — kernels, SVMs, estimator base, loader
    "QSVC": "eigenkit.qml",
    "QSVR": "eigenkit.qml",
    "QuantumKernel": "eigenkit.qml",
    "KernelJob": "eigenkit.qml",
    "align_kernel": "eigenkit.qml",
    "KernelAlignmentResult": "eigenkit.qml",
    "spsa": "eigenkit.core.optimizers",
    "SPSAResult": "eigenkit.core.optimizers",
    "QuantumReservoirClassifier": "eigenkit.qml",
    "QuantumReservoirRegressor": "eigenkit.qml",
    "BaseQuantumEstimator": "eigenkit.qml",
    "BaseQuantumClassifier": "eigenkit.qml",
    "BaseQuantumRegressor": "eigenkit.qml",
    "load_model": "eigenkit.qml",
    # general quantum algorithms — HHL (linear systems), IQAE (amplitude
    # estimation), portfolio risk (VaR/CVaR)
    "HHL": "eigenkit.algorithms",
    "HHLResult": "eigenkit.algorithms",
    "IQAE": "eigenkit.algorithms",
    "IQAEResult": "eigenkit.algorithms",
    "PortfolioRiskEstimator": "eigenkit.algorithms",
    "PortfolioRiskResult": "eigenkit.algorithms",
    # torch ring — hybrid quantum-classical estimators
    "HQNC": "eigenkit.torch.estimators",
    "HQNR": "eigenkit.torch.estimators",
    "RecurrentHQNC": "eigenkit.torch.estimators",
    "RecurrentHQNR": "eigenkit.torch.estimators",
    "QuantumTimeSeriesClassifier": "eigenkit.torch.estimators",
    "QuantumTimeSeriesRegressor": "eigenkit.torch.estimators",
    "CustomQuantumClassifier": "eigenkit.torch.estimators",
    "CustomQuantumRegressor": "eigenkit.torch.estimators",
    "GenericClassifier": "eigenkit.torch.estimators",
    "GenericRegressor": "eigenkit.torch.estimators",
    # torch ring — layers
    "Basic": "eigenkit.torch.layers",
    "Reupload": "eigenkit.torch.layers",
    "Ensemble": "eigenkit.torch.layers",
    "MinMaxNorm": "eigenkit.torch.layers",
    # backend classes + async handles (resolved lazily through `core`)
    "BraketBackend": "eigenkit.core",
    "CudaQuantumBackend": "eigenkit.core",
    "IQMBackend": "eigenkit.core",
    "QiskitAerBackend": "eigenkit.core",
    "QiskitRuntimeBackend": "eigenkit.core",
    "QsimBackend": "eigenkit.core",
    "QulacsBackend": "eigenkit.core",
    "TensorCircuitBackend": "eigenkit.core",
    "BraketJobHandle": "eigenkit.core",
    "IQMJobHandle": "eigenkit.core",
    "QiskitRuntimeJobHandle": "eigenkit.core",
    # backend configs (resolved lazily through `configs`)
    "BraketConfig": "eigenkit.configs",
    "CudaQuantumConfig": "eigenkit.configs",
    "IQMConfig": "eigenkit.configs",
    "PennyLaneConfig": "eigenkit.configs",
    "QiskitAerConfig": "eigenkit.configs",
    "QiskitConfig": "eigenkit.configs",
    "QiskitRuntimeConfig": "eigenkit.configs",
    "QsimConfig": "eigenkit.configs",
    "QulacsConfig": "eigenkit.configs",
    "TensorCircuitConfig": "eigenkit.configs",
}


def __getattr__(name: str):
    import importlib

    if name == "__version__":
        # Read from the installed distribution metadata rather than duplicating
        # the number here, so it cannot drift from `pyproject.toml`. Resolved
        # lazily like everything else: importing `importlib.metadata` costs
        # ~50 ms, half again the cost of importing this package, and the module
        # docstring above promises a cheap import. Cached into the module
        # globals so only the first lookup pays.
        import importlib.metadata

        try:
            version = importlib.metadata.version("eigenkit")
        except importlib.metadata.PackageNotFoundError:
            # A source checkout on `sys.path` with nothing installed has no
            # distribution metadata to read. Report the attribute as absent
            # rather than letting `PackageNotFoundError` out: it is not an
            # `AttributeError`, so it would escape `hasattr()` and
            # `getattr(pkg, "__version__", default)` — the two calls a
            # compatibility check makes — instead of answering them. Inventing
            # a placeholder would be worse: a check would compare against it
            # and believe the answer.
            raise AttributeError(
                f"module {__name__!r} has no attribute '__version__': no "
                "distribution metadata for 'eigenkit'. It is read "
                "from the installed distribution, so an uninstalled source "
                "checkout has no version to report; `pip install -e .` gives "
                "it one."
            ) from None
        globals()["__version__"] = version
        return version
    if name in _LAZY_SUBMODULES:
        return _import_with_hint(f"{__name__}.{name}", name, attribute=False)
    source = _LAZY_NAMES.get(name)
    if source is not None:
        return _import_with_hint(source, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The extra that brings each optional dependency a lazy name can need. The
# subpackages with an extra of their own name it instead.
_EXTRA_FOR_DEPENDENCY = {
    "matplotlib": "viz",
    "pandas": "benchmark",
    "pennylane": "pennylane",
    "perceval": "perceval",
    "sklearn": "qml",
    "torch": "torch",
    "tqdm": "torch",
}
_EXTRA_FOR_SUBPACKAGE = {"benchmark": "benchmark", "datasets": "datasets"}
_DISTRIBUTION = {"sklearn": "scikit-learn"}


def _import_with_hint(module: str, name: str, attribute: bool = True):
    import importlib

    try:
        loaded = importlib.import_module(module)
        # The attribute too: a subpackage may itself load it lazily.
        return getattr(loaded, name) if attribute else loaded
    except ModuleNotFoundError as error:
        missing = (error.name or "").partition(".")[0]
        if missing not in _EXTRA_FOR_DEPENDENCY:
            raise
        subpackage = module.split(".")[1] if "." in module else ""
        extra = _EXTRA_FOR_SUBPACKAGE.get(subpackage, _EXTRA_FOR_DEPENDENCY[missing])
        raise ModuleNotFoundError(
            f"eigenkit.{name} needs {_DISTRIBUTION.get(missing, missing)}, which is "
            "not installed. Install "
            f"it with `pip install 'eigenkit[{extra}]'` (or 'eigenkit[all]' for "
            "every extra).",
            name=error.name,
        ) from error


__all__ = [
    # subpackages
    "benchmark",
    "configs",
    "datasets",
    "qml",
    # estimators (qml + torch rings)
    "HQNC",
    "HQNR",
    "QSVC",
    "QSVR",
    "QuantumReservoirClassifier",
    "QuantumReservoirRegressor",
    "RecurrentHQNC",
    "RecurrentHQNR",
    "QuantumTimeSeriesClassifier",
    "QuantumTimeSeriesRegressor",
    "CustomQuantumClassifier",
    "CustomQuantumRegressor",
    "QuantumKernel",
    "KernelJob",
    "align_kernel",
    "KernelAlignmentResult",
    "GenericClassifier",
    "GenericRegressor",
    "BaseQuantumEstimator",
    "BaseQuantumClassifier",
    "BaseQuantumRegressor",
    "load_model",
    # general quantum algorithms
    "HHL",
    "HHLResult",
    "IQAE",
    "IQAEResult",
    "PortfolioRiskEstimator",
    "PortfolioRiskResult",
    # core class
    "Circuit",
    # high-level layers
    "Basic",
    "Reupload",
    "Ensemble",
    "MinMaxNorm",
    # block classes
    "Encoding",
    "Variational",
    "StrongVariational",
    "Reuploading",
    # device configs
    "BraketConfig",
    "CudaQuantumConfig",
    "IQMConfig",
    "NumpyConfig",
    "PennyLaneConfig",
    "QiskitAerConfig",
    "QiskitConfig",
    "QiskitRuntimeConfig",
    "QsimConfig",
    "QulacsConfig",
    "TensorCircuitConfig",
    # backend classes (for opt-in registration / advanced users)
    "BraketBackend",
    "CudaQuantumBackend",
    "IQMBackend",
    "NumpySimulator",
    "QiskitAerBackend",
    "QiskitRuntimeBackend",
    "QsimBackend",
    "QulacsBackend",
    "TensorCircuitBackend",
    # async submission for QPU/cloud backends
    "BraketJobHandle",
    "IQMJobHandle",
    "QiskitRuntimeJobHandle",
    "JobHandle",
    "CostWarning",
    # observable construction
    "Observable",
    "X",
    "Y",
    "Z",
    "all_z",
    "from_string",
    # circuit construction
    "CircuitBuilder",
    "assemble",
    # transforms and registry
    "adjoint",
    "register",
    "registered_names",
    # variational optimization (VQE / QAOA engine)
    "minimize_expectation",
    "VariationalResult",
    "spsa",
    "SPSAResult",
]
