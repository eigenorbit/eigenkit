"""An ensemble of independent quantum circuits, one for each chunk of the input.

It splits the input into chunks of `n_qubits` features and runs one independent
circuit on each chunk, each with its own weights. The outputs are concatenated
along the feature dimension.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import cat, nn, split

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


class Ensemble(nn.Module):
    """An ensemble of independent quantum circuits, one for each chunk of the input.

    Args:
        in_features: The total number of input features. It must be a multiple of
            `n_qubits`.
        n_qubits: The number of qubits of each circuit.
        depth: The number of variational layers of each circuit.
        enc_rotation: The encoding rotation: `"RX"`, `"RY"` or `"RZ"`. Default:
            `"RX"`.
        var_rotation: The variational rotation. It is used only when
            `variational_type="basic"`. Default: `"RY"`.
        variational_type: `"basic"` or `"strong"`. Default: `"strong"`.
        basis: The Pauli measurement basis: `"X"`, `"Y"` or `"Z"`. Default: `"Z"`.
        provider: The backend that runs the circuits. Default: `"numpy"`.

    Example:
        ```python
        import torch
        from eigenkit import Ensemble

        layer = Ensemble(in_features=20, n_qubits=4, depth=5)
        out = layer(torch.randn(128, 20))  # shape (128, 20)
        ```
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int,
        depth: int,
        enc_rotation: Literal["RX", "RY", "RZ"] = "RX",
        var_rotation: Literal["RX", "RY", "RZ"] = "RY",
        variational_type: Literal["basic", "strong"] = "strong",
        basis: Literal["X", "Y", "Z"] = "Z",
        provider: ProviderSpec = "numpy",
    ):
        super().__init__()
        if in_features % n_qubits != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by n_qubits ({n_qubits})"
            )
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.q_layers = in_features // n_qubits
        self.out_features = self.q_layers * self.n_qubits

        # Validate before warning, as `Basic` does, so an invalid `basis` does
        # not draw a degenerate-ansatz warning about a layer that is about to
        # raise anyway.
        if basis not in _BASIS_TO_OBSERVABLE:
            raise ValueError(f"basis must be 'X', 'Y', or 'Z', got {basis!r}")

        warn_if_degenerate(
            layer_name="Ensemble",
            enc_rotation=enc_rotation,
            var_rotation=var_rotation,
            variational_type=variational_type,
            basis=basis,
            n_qubits=n_qubits,
            n_encodings=1,
        )
        observables = [_BASIS_TO_OBSERVABLE[basis](q) for q in range(n_qubits)]

        self.circuits = nn.ModuleList()
        for _ in range(self.q_layers):
            encoding = Encoding(in_features=n_qubits, rotation=enc_rotation)
            if variational_type == "basic":
                variational = Variational(depth=depth, rotation=var_rotation)
            elif variational_type == "strong":
                variational = StrongVariational(depth=depth)
            else:
                raise ValueError(
                    f"variational_type must be 'basic' or 'strong', "
                    f"got {variational_type!r}"
                )
            self.circuits.append(
                TorchCircuit(
                    blocks=[encoding, variational],
                    observables=observables,
                    provider=provider,
                )
            )

    @property
    def provider(self) -> Provider:
        """The backend that the circuits of this layer were compiled for.

        The constructor takes `provider=` in the same forms as `Circuit` (a
        registered name, a configuration or a backend object) and turns it into
        one `Provider`. This property returns it. The registered name is
        `provider.name`.

        Returns:
            provider (Provider): The backend that the circuits run on.
        """
        return self.circuits[0].provider

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layer.

        Args:
            x (torch.Tensor): The input batch, of shape `(batch, in_features)`.

        Returns:
            out (torch.Tensor): The output of the layer for that batch.
        """
        squeeze_back = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_back = True
        # `split` neither complains about a wider input nor explains a narrower
        # one: extra columns would be dropped in silence with a plausible-looking
        # result, while too few would surface as `IndexError: tuple index out of
        # range` from the comprehension below.
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Ensemble(in_features={self.in_features}) got an input with "
                f"{x.shape[1]} features. Every circuit takes {self.n_qubits} "
                f"of them, so the input must have exactly "
                f"{self.in_features} columns."
            )
        chunks = split(x, self.n_qubits, dim=1)
        outputs = [circuit(chunks[i]) for i, circuit in enumerate(self.circuits)]
        result = match_input_dtype(cat(outputs, dim=1), x)
        if squeeze_back:
            result = result.squeeze(0)
        return result
