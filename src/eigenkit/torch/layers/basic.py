"""The basic quantum layer: one `Encoding`, one `Variational` block and a measurement.

It maps `n_qubits` input features to `n_qubits` expectation values, one for each
qubit, in the chosen Pauli basis. For data reuploading or ensembles, see
`Reupload` and `Ensemble`.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from eigenkit.configs import Encoding, StrongVariational, Variational
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


class Basic(nn.Module):
    """One `Encoding`, one `Variational` block and a measurement on each qubit.

    Args:
        n_qubits: The number of qubits, which is also the number of input
            features.
        depth: The number of variational layers.
        enc_rotation: The encoding rotation: `"RX"`, `"RY"` or `"RZ"`. Default:
            `"RX"`.
        var_rotation: The variational rotation. It is used only when
            `variational_type="basic"`. The default, `"RY"`, is off both the
            `"RX"` encoding axis and the `"Z"` measurement axis, and this is what
            lets the weights change the output. Default: `"RY"`.
        variational_type: `"basic"`, for one rotation and a controlled-NOT (CNOT)
            ladder in each layer, or `"strong"`, for `RZ`, `RY` and `RZ`
            rotations and a CNOT ladder in each layer. Default: `"basic"`.
        basis: The Pauli measurement basis: `"X"`, `"Y"` or `"Z"`. Default: `"Z"`.
        provider: The backend that runs the circuit. Default: `"numpy"`.

    Example:
        ```python
        import torch
        from eigenkit import Basic

        layer = Basic(n_qubits=4, depth=2)
        out = layer(torch.randn(8, 4))  # shape (8, 4)
        ```
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        enc_rotation: Literal["RX", "RY", "RZ"] = "RX",
        var_rotation: Literal["RX", "RY", "RZ"] = "RY",
        variational_type: Literal["basic", "strong"] = "basic",
        basis: Literal["X", "Y", "Z"] = "Z",
        provider: ProviderSpec = "numpy",
    ):
        super().__init__()
        if basis not in _BASIS_TO_OBSERVABLE:
            raise ValueError(f"basis must be 'X', 'Y', or 'Z', got {basis!r}")

        self.n_qubits = n_qubits
        self.in_features = n_qubits
        self.out_features = n_qubits

        warn_if_degenerate(
            layer_name="Basic",
            enc_rotation=enc_rotation,
            var_rotation=var_rotation,
            variational_type=variational_type,
            basis=basis,
            n_qubits=n_qubits,
            n_encodings=1,
        )

        if variational_type == "basic":
            variational = Variational(depth=depth, rotation=var_rotation)
        elif variational_type == "strong":
            variational = StrongVariational(depth=depth)
        else:
            raise ValueError(
                f"variational_type must be 'basic' or 'strong', got {variational_type!r}"
            )

        observables = [_BASIS_TO_OBSERVABLE[basis](q) for q in range(n_qubits)]

        self.circuit = TorchCircuit(
            blocks=[Encoding(in_features=n_qubits, rotation=enc_rotation), variational],
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
