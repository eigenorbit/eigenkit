"""The PyTorch ring of `eigenkit.orthogonal`: trainable orthogonal modules and
orthogonal fine-tuning adapters.

It needs the `torch` extra (`pip install 'eigenkit[torch]'`). The NumPy part of
`eigenkit.orthogonal` loads this package only when one of its names is accessed.
"""

from .adapters import (
    OrthogonalAdapter,
    apply_orthogonal_adapters,
    fold_bn,
    merge_adapters,
    trainable_parameters,
)
from .functional import rotate_weight, unary_loader_angles
from .modules import (
    MAX_MATERIALISE,
    Orthogonal,
    OrthogonalConv1x1,
    PartitionedOrthogonal,
    binary_parts,
    mixing_permutation,
)
from .robust import (
    Certificate,
    LLNHead,
    MaxMin,
    cert_ce,
    certified_accuracy,
    certified_radius,
    head_distances,
    lipschitz_bound,
    sigma_max,
)


__all__ = [
    "Certificate",
    "LLNHead",
    "MAX_MATERIALISE",
    "MaxMin",
    "Orthogonal",
    "OrthogonalAdapter",
    "OrthogonalConv1x1",
    "PartitionedOrthogonal",
    "apply_orthogonal_adapters",
    "binary_parts",
    "cert_ce",
    "certified_accuracy",
    "certified_radius",
    "fold_bn",
    "head_distances",
    "lipschitz_bound",
    "merge_adapters",
    "mixing_permutation",
    "rotate_weight",
    "sigma_max",
    "trainable_parameters",
    "unary_loader_angles",
]
