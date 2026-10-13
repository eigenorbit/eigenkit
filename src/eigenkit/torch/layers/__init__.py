"""Quantum layers for the standard patterns.

- `Basic`: one `Encoding`, one `Variational` block and a measurement.
- `Reupload`: standard data reuploading, optionally with a trainable scale.
- `Ensemble`: `N` independent circuits, one for each chunk of the input.
- `MinMaxNorm`: min-max normalisation of each feature, in the style of
  `BatchNorm`.
"""

from .basic import Basic
from .ensemble import Ensemble
from .minmax import MinMaxNorm
from .reupload import Reupload


__all__ = [
    "Basic",
    "Ensemble",
    "MinMaxNorm",
    "Reupload",
]
