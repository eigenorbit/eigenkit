r"""The orthogonal quantum layer: a unary loader, a network of RBS gates, and the
unary readout.

The layer loads each input vector \(x\) into the unary basis, applies an orthogonal
network \(W(\theta)\) of RBS gates, and returns the probability of each wire,
\((Wx)_i^2 / \lVert x \rVert^2\). Its circuit is the one of
`eigenkit.orthogonal.to_circuit`, with trainable angles. See `eigenkit.orthogonal`
for the layouts and the matrix form.
"""

from __future__ import annotations

import torch
from torch import nn

from eigenkit.core.circuit import ProviderSpec
from eigenkit.core.provider import Provider
from eigenkit.orthogonal.blocks import (
    LoaderLayout,
    RBSNetwork,
    UnaryLoader,
    unary_readout,
)
from eigenkit.orthogonal.topology import Topology, TopologyName
from eigenkit.orthogonal.topology import topology as named_topology
from eigenkit.orthogonal.torch.functional import _unit_scale, unary_loader_angles
from eigenkit.orthogonal.torch.modules import _pair_layers, _topology_matrix

from ..circuit import TorchCircuit
from ._dtype import match_input_dtype


class OrthogonalQNN(nn.Module):
    r"""A quantum orthogonal layer: \(x \mapsto (Wx)^2 / \lVert x \rVert^2\) on
    `n_qubits` wires.

    A `UnaryLoader` loads \(x / \lVert x \rVert\) with \(n - 1\) RBS gates, for
    \(n\) = `n_qubits`, whose angles are computed from \(x\); an `RBSNetwork`
    applies the orthogonal matrix \(W(\theta)\) of `topology`; and the `Z`
    observable of each wire gives the probability
    \(p_i = (1 - \langle Z_i \rangle) / 2\). The outputs of a row sum to 1.

    The output values come from the circuit, and so does the gradient with respect
    to the angles. The gradient with respect to the input is the exact derivative
    of \((Wx)^2 / \lVert x \rVert^2\), with \(W\) built from the current angles: the
    loader angles are not differentiable where \(x\) has zeros (after a ReLU, or a
    one-hot input), while the output is. On a noiseless simulator the two agree; on
    noisy hardware the input gradient is that of the ideal circuit.

    A row that is all zeros has no direction to load, so
    \((Wx)^2 / \lVert x \rVert^2\) is not defined for it. Such a row, which a ReLU in
    front of the layer can produce, gives the uniform distribution \(1/n\) on
    every wire, and no gradient
    flows back to it. The other rows of the batch are not affected.

    The circuit simulates \(2^n\) amplitudes, so the layer is practical up to
    about 20 qubits. The matrix form, `eigenkit.orthogonal.torch.Orthogonal`,
    computes the same \(W\) classically at any width.

    Args:
        n_qubits (int): The number of wires, which is the number of input features
            and of outputs.
        topology (TopologyName | Topology): The layout of the network: a name
            (`"butterfly"`, `"pyramid"`, `"x"`, `"diagonal"`) or a `Topology` on
            `n_qubits` wires. Default: `"pyramid"`.
        loader (LoaderLayout): The layout of the loader: `"diagonal"` (any width,
            adjacent wires), `"semi_diagonal"` (even width) or `"parallel"`
            (power-of-two width). Default: `"diagonal"`.
        init (str): `"random"` for standard normal angles, `"identity"` for zero
            angles, which make `W` the identity. Default: `"random"`.
        provider (ProviderSpec): The backend that runs the circuit. Default:
            `"numpy"`.

    Example:
        ```python
        import torch
        from eigenkit import OrthogonalQNN

        layer = OrthogonalQNN(n_qubits=4, topology="pyramid")
        out = layer(torch.randn(8, 4))  # shape (8, 4), each row sums to 1
        ```
    """

    def __init__(
        self,
        n_qubits: int,
        topology: TopologyName | Topology = "pyramid",
        loader: LoaderLayout = "diagonal",
        init: str = "random",
        provider: ProviderSpec = "numpy",
    ):
        super().__init__()
        if isinstance(topology, Topology):
            if topology.n != n_qubits:
                raise ValueError(
                    f"topology is on {topology.n} wires, but n_qubits={n_qubits}"
                )
            layout = topology
        else:
            layout = named_topology(topology, n_qubits)
        if init not in ("random", "identity"):
            raise ValueError(f"init must be 'random' or 'identity', got {init!r}")
        n_qubits = layout.n

        self.n_qubits = n_qubits
        self.in_features = n_qubits
        self.out_features = n_qubits
        self.topology = layout
        self.loader = loader
        self._init = init
        # The pair indices of the closed form, built once per device rather than
        # on every forward pass. A plain dict, not buffers: nothing to save, and
        # nothing for `to_empty()` to leave uninitialised.
        self._layers_by_device: dict[torch.device, list] = {}

        weights = (
            torch.zeros(layout.n_angles, dtype=torch.float64)
            if init == "identity"
            else None
        )
        self.circuit = TorchCircuit(
            blocks=[UnaryLoader(n_qubits, layout=loader), RBSNetwork(layout)],
            observables=unary_readout(n_qubits),
            provider=provider,
            init=weights,
        )

    @property
    def provider(self) -> Provider:
        """The backend that the circuit of this layer was compiled for.

        Returns:
            provider (Provider): The backend that the circuit runs on.
        """
        return self.circuit.provider

    def reset_parameters(self) -> None:
        """Initialise the angles again by the rule of `init`.

        `init="random"` draws new standard normal angles from the global
        generator (`circuit.reset_parameters()`); `init="identity"` sets them to
        zero, so the layer starts at `W = I` again.

        The estimators call `reset_parameters` on every module through
        `model.apply`, which resets `circuit` first and then this layer. With
        `init="random"` the angles are therefore drawn twice, and the second draw
        is kept: the result still depends only on the seed, and the generator moves
        on by `2 * n_angles` numbers rather than `n_angles`.
        """
        if self._init == "identity":
            with torch.no_grad():
                self.circuit.weights.zero_()
        else:
            self.circuit.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Run the layer.

        Args:
            x (torch.Tensor): One input `(n_qubits,)` or a batch
                `(batch, n_qubits)`. A row of zeros gives the uniform distribution
                (see the class docstring).

        Returns:
            out (torch.Tensor): The probabilities \((Wx)_i^2 / \lVert x \rVert^2\), of
                the shape
                of `x`.
        """
        xd = x.to(torch.float64)
        # A zero row cannot be loaded. It is replaced by a placeholder for the
        # computation, and its output is set to the uniform distribution at the
        # end, so it neither raises nor sends a NaN gradient into the batch.
        zero = (xd == 0).all(dim=-1, keepdim=True)
        safe = torch.where(zero, torch.ones_like(xd), xd)
        angles = unary_loader_angles(safe.detach(), self.loader)
        p = (1.0 - self.circuit(angles)) / 2.0
        theta = self.circuit.weights.detach().to(torch.float64)
        layers = self._layers_by_device.get(theta.device)
        if layers is None:
            layers = self._layers_by_device[theta.device] = _pair_layers(
                self.topology, theta.device
            )
        W = _topology_matrix(self.topology, theta, layers)
        # Scaled by a power of two first, which is exact and leaves the ratio as
        # it is, so neither square overflows or underflows.
        xs = _unit_scale(safe)
        smooth = (xs @ W.t()) ** 2 / (xs**2).sum(dim=-1, keepdim=True)
        # Adds exactly zero, so the value stays the circuit's; only the input
        # gradient comes from the closed form.
        out = p + (smooth - smooth.detach())
        out = torch.where(zero, torch.full_like(out, 1.0 / self.n_qubits), out)
        return match_input_dtype(out, x)

    def extra_repr(self) -> str:
        return (
            f"n_qubits={self.n_qubits}, topology={self.topology.name}, "
            f"loader={self.loader}"
        )
