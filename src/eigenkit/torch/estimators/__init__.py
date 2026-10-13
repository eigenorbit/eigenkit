"""The PyTorch estimators: differentiable hybrid quantum-classical models.

The estimator base classes compatible with scikit-learn, the quantum kernels, the
support vector machines and the saving and loading of models are in a separate
package, `eigenkit.qml`, which does not need PyTorch. The base classes and
`load_model` are also exported here for convenience, because the hybrid estimators
build on them.
"""

from eigenkit.qml import (
    BaseQuantumClassifier,
    BaseQuantumEstimator,
    BaseQuantumRegressor,
    load_model,
)

from .custom import CustomQuantumClassifier, CustomQuantumRegressor
from .generic import GenericClassifier, GenericRegressor
from .hqnc import HQNC
from .hqnr import HQNR
from .recurrent_hqnc import RecurrentHQNC
from .recurrent_hqnr import RecurrentHQNR
from .timeseries import QuantumTimeSeriesClassifier, QuantumTimeSeriesRegressor


__all__ = [
    "BaseQuantumEstimator",
    "BaseQuantumClassifier",
    "BaseQuantumRegressor",
    # hybrid quantum-classical estimators
    "HQNC",
    "HQNR",
    "RecurrentHQNC",
    "RecurrentHQNR",
    "CustomQuantumClassifier",
    "CustomQuantumRegressor",
    "GenericClassifier",
    "GenericRegressor",
    "QuantumTimeSeriesClassifier",
    "QuantumTimeSeriesRegressor",
    "load_model",
]
