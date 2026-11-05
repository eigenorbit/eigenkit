r"""Orthogonal networks built from RBS (Givens) rotations.

An RBS gate rotates the two states \(|01\rangle\) and \(|10\rangle\) of a pair of
wires by an angle. A network of RBS gates keeps the number of 1s fixed, so on the
unary basis it acts as an orthogonal matrix, and it stays orthogonal for every value
of its angles. A `Topology` fixes which pairs the gates act on: the butterfly,
pyramid and X layouts of Cherrat et al. (Quantum 2024), one diagonal of X, a
brick wall, or any layout of your own.

This package uses only NumPy and belongs to the framework-free core of the
library. `orthogonal_matrix` gives the dense matrix of a layout, `RBSNetwork` and
`UnaryLoader` give the quantum circuit, and `to_circuit` turns trained angles into a
circuit that runs on any provider.

The PyTorch modules (`Orthogonal`, `OrthogonalConv1x1`, `PartitionedOrthogonal`)
and the fine-tuning adapters live in `eigenkit.orthogonal.torch`, which needs the
`torch` extra. Their names are also available here, and are imported the first
time one is accessed, so importing this package does not import PyTorch. They are
listed in `__all__`, so `from eigenkit.orthogonal import *` *does* import PyTorch.

Example:
    ```python
    from eigenkit.orthogonal import identity_angles, orthogonal_matrix, pyramid

    t = pyramid(4)
    orthogonal_matrix(t, identity_angles(t))  # the 4 x 4 identity
    ```
"""

from .blocks import (
    LoaderLayout,
    RBSNetwork,
    UnaryLoader,
    loader_depth,
    unary_loader_angles,
    unary_probabilities,
    unary_readout,
)
from .circuit import to_block, to_circuit
from .matrix import identity_angles, orthogonal_matrix, orthogonal_matrix_jacobian
from .topology import (
    Topology,
    TopologyName,
    brick,
    butterfly,
    butterfly_pairs,
    diagonal,
    pyramid,
    topology,
    x,
)


_TORCH_NAMES = (
    "LLNHead",
    "MaxMin",
    "Orthogonal",
    "OrthogonalAdapter",
    "OrthogonalConv1x1",
    "PartitionedOrthogonal",
    "apply_orthogonal_adapters",
    "cert_ce",
    "certified_accuracy",
    "certified_radius",
    "fold_bn",
    "lipschitz_bound",
    "merge_adapters",
    "rotate_weight",
    "trainable_parameters",
)

__all__ = [
    "LoaderLayout",
    "RBSNetwork",
    "Topology",
    "TopologyName",
    "UnaryLoader",
    "brick",
    "butterfly",
    "butterfly_pairs",
    "diagonal",
    "identity_angles",
    "loader_depth",
    "orthogonal_matrix",
    "orthogonal_matrix_jacobian",
    "pyramid",
    "to_block",
    "to_circuit",
    "topology",
    "unary_loader_angles",
    "unary_probabilities",
    "unary_readout",
    "x",
    *_TORCH_NAMES,
]


def __getattr__(name):
    # The torch ring is imported on first access, so `import eigenkit.orthogonal`
    # stays torch-free.
    if name in _TORCH_NAMES:
        try:
            from . import torch as ring
        except ModuleNotFoundError as error:
            if (error.name or "").partition(".")[0] != "torch":
                raise
            raise ModuleNotFoundError(
                f"eigenkit.orthogonal.{name} needs torch, which is not installed. "
                "Install it with `pip install 'eigenkit[torch]'` (or 'eigenkit[all]' "
                "for every extra).",
                name=error.name,
            ) from error
        return getattr(ring, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
