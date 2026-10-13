"""The data-reuploading layer.

It encodes the input `depth` times, with new variational weights between the
encodings. Optionally, it adds to each encoding rotation a trainable scale for
each pair of layer and feature (Schuld et al. 2021).

In this layer, `in_features == n_qubits`.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from eigenkit.configs import Reuploading
from eigenkit.core import Observable
from eigenkit.core.circuit import ProviderSpec
from eigenkit.core.provider import Provider

from ..circuit import TorchCircuit
from ._dtype import match_input_dtype
from ._expressivity import warn_if_degenerate


_BASIS_TO_OBSERVABLE = {
    "X": Observable.x,
    "Y": Observable.y,
    "Z": Observable.z,
}


class Reupload(nn.Module):
    """The data-reuploading layer (Perez-Salinas 2020; Schuld 2021).

    It supports two patterns:

    - Standard reuploading (the default). The same input is encoded `depth`
      times, with new variational weights between the encodings.
    - Trainable feature scaling (`trainable_scale=True`). Each encoding rotation
      becomes `R(lambda * x)`, with a trainable `lambda` for each pair of layer
      and feature.

    Args:
        n_qubits: The number of qubits, which is also the number of input
            features.
        depth: The number of times the input is encoded.
        enc_rotation: The encoding rotation: `"RX"`, `"RY"` or `"RZ"`. Default:
            `"RY"`.
        var_rotation: The variational rotation. It is used only when
            `variational_type="basic"`. Default: `"RY"`.
        variational_type: `"basic"` or `"strong"`. Default: `"basic"`.
        basis: The Pauli measurement basis: `"X"`, `"Y"` or `"Z"`. Default: `"Z"`.
        trainable_scale: If True, each encoding rotation gets a trainable scale
            for each pair of layer and feature. The CUDA Quantum backend does not
            support it, because its `QuakeValue` arithmetic is limited. Default:
            `False`.
        provider: The backend that runs the circuit. Default: `"numpy"`.

    Example:
        ```python
        import torch
        from eigenkit import Reupload

        layer = Reupload(n_qubits=4, depth=3, trainable_scale=True)
        out = layer(torch.randn(8, 4))  # shape (8, 4)
        ```
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        enc_rotation: Literal["RX", "RY", "RZ"] = "RY",
        var_rotation: Literal["RX", "RY", "RZ"] = "RY",
        variational_type: Literal["basic", "strong"] = "basic",
        basis: Literal["X", "Y", "Z"] = "Z",
        trainable_scale: bool = False,
        provider: ProviderSpec = "numpy",
    ):
        super().__init__()
        if basis not in _BASIS_TO_OBSERVABLE:
            raise ValueError(f"basis must be 'X', 'Y', or 'Z', got {basis!r}")

        self.n_qubits = n_qubits
        self.in_features = n_qubits
        self.out_features = n_qubits

        warn_if_degenerate(
            layer_name="Reupload",
            enc_rotation=enc_rotation,
            var_rotation=var_rotation,
            variational_type=variational_type,
            basis=basis,
            n_qubits=n_qubits,
            n_encodings=depth,
        )

        block = Reuploading(
            in_features=n_qubits,
            depth=depth,
            encoding_rotation=enc_rotation,
            variational_rotation=var_rotation,
            variational_type=variational_type,
            trainable_scale=trainable_scale,
        )
        observables = [_BASIS_TO_OBSERVABLE[basis](q) for q in range(n_qubits)]

        self.circuit = TorchCircuit(
            blocks=[block],
            observables=observables,
            provider=provider,
        )

    @property
    def provider(self) -> Provider:
        """The backend that the circuit of this layer was compiled for.

        The constructor takes `provider=` in the same forms as `Circuit` (a
        registered name, a configuration or a backend object) and turns it into
        one `Provider`. This property returns it. The registered name is
        `provider.name`.

        Returns:
            provider (Provider): The backend that the circuit runs on.
        """
        return self.circuit.provider

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layer.

        Args:
            x (torch.Tensor): The input batch, of shape `(batch, in_features)`.

        Returns:
            out (torch.Tensor): The output of the layer for that batch.
        """
        return match_input_dtype(self.circuit(x), x)
