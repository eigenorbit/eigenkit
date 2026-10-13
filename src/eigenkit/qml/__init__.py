"""Quantum machine learning: kernels, support vector machines, the estimator base
classes compatible with scikit-learn, and saving and loading models.

This package needs scikit-learn, NumPy and the core of the library, but not
PyTorch: the quantum kernels run on the NumPy interface. The hybrid estimators
based on PyTorch (`HQNC`, `HQNR` and others) are in `eigenkit.torch.estimators`,
and they build on the base classes exported here.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from eigenkit.core.circuit import ProviderSpec

from ._alignment import (
    KernelAlignmentResult,
    align_kernel,
    kernel_target_alignment,
    svc_dual_loss,
)
from ._base import (
    BaseQuantumClassifier,
    BaseQuantumEstimator,
    BaseQuantumRegressor,
)
from ._kernels import KernelJob, QuantumKernel, project_psd
from ._registry import get_estimator_class, register_estimator
from ._serialization import load_bundle
from .qrc import QuantumReservoirClassifier, QuantumReservoirRegressor
from .qsvc import QSVC
from .qsvr import QSVR


def load_model(
    path: str | Path,
    *,
    provider: ProviderSpec | None = None,
    model: Any = None,
) -> BaseQuantumEstimator:
    """Load any saved model. The class of the model is read from `metadata.json`.

    Load only bundles that you trust. A bundle can contain joblib pickles, and
    unpickling runs code, as with `torch.load` or `joblib.load`.

    Args:
        path: The path to the `.eigenkit` bundle.
        provider: A provider that replaces the saved one. It is passed to the
            `load()` method of the estimator.
        model: The `nn.Module` architecture, for checkpoints of `GenericRegressor`
            and `GenericClassifier`. It is ignored for the other estimators.

    Returns:
        The loaded estimator, an instance of the class named in the bundle.
    """
    # Ensure estimator classes are registered before looking one up by name.
    # QSVC/QSVR register on the imports above; the torch ring and the portfolio
    # estimator register on import — best-effort so a torch-free install can
    # still load sklearn-tier models.
    try:
        import eigenkit.torch.estimators  # noqa: F401
    except ImportError:
        pass
    import eigenkit.algorithms.portfolio_risk  # noqa: F401

    metadata, _, _, _ = load_bundle(path)
    cls = get_estimator_class(metadata["class_name"])
    sig = inspect.signature(cls.load)
    kwargs: dict[str, Any] = {"provider": provider}
    if model is not None and "model" in sig.parameters:
        kwargs["model"] = model
    return cls.load(path, **kwargs)


__all__ = [
    "BaseQuantumEstimator",
    "BaseQuantumClassifier",
    "BaseQuantumRegressor",
    "QSVC",
    "QSVR",
    "QuantumReservoirClassifier",
    "QuantumReservoirRegressor",
    "QuantumKernel",
    "KernelJob",
    "project_psd",
    "align_kernel",
    "KernelAlignmentResult",
    "kernel_target_alignment",
    "svc_dual_loss",
    "get_estimator_class",
    "register_estimator",
    "load_model",
]
