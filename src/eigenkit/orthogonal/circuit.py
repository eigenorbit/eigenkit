r"""Turn the angles of an orthogonal network into a quantum circuit.

The angles can come from anywhere, for example from a network trained classically
with the PyTorch modules. `to_block` fixes them as `Constant`s in an `RBSNetwork`,
and `to_circuit` puts a `UnaryLoader` in front and the unary readout behind, so the
circuit runs on any provider, on a simulator or on hardware.

A circuit simulates a statevector of \(2^n\) amplitudes, so this form is limited to
about 20 wires. The matrix form, `orthogonal_matrix`, is the same map restricted to
the unary basis and scales to thousands of wires.

Example:
    ```python
    import numpy as np

    from eigenkit.orthogonal import (
        orthogonal_matrix,
        pyramid,
        to_circuit,
        unary_loader_angles,
        unary_probabilities,
    )

    t = pyramid(4)
    theta = np.random.default_rng(0).normal(size=t.n_angles)
    circuit = to_circuit(t, theta)
    x = np.array([1.0, 2.0, -0.5, 0.3])
    p = unary_probabilities(circuit(np.zeros(0), unary_loader_angles(x)))
    # p == (orthogonal_matrix(t, theta) @ x) ** 2 / (x @ x)
    ```
"""

from __future__ import annotations

import numpy as np

from ..core.blocks import assemble
from ..core.circuit import Circuit, ProviderSpec
from .blocks import LoaderLayout, RBSNetwork, UnaryLoader, unary_readout
from .matrix import _as_angles
from .topology import Topology


def to_block(topology: Topology, theta) -> RBSNetwork:
    """An `RBSNetwork` block with fixed angles.

    Args:
        topology (Topology): The gate layout.
        theta (array-like): The angles, of shape `(topology.n_angles,)`, in the
            order of `topology.pairs()`.

    Returns:
        RBSNetwork: A block that takes no weights.

    Raises:
        ValueError: If `theta` is not a real vector of `topology.n_angles` angles.
    """
    angles = _as_angles(topology, theta).astype(np.float64)
    return RBSNetwork(topology, tuple(angles.tolist()))


def to_circuit(
    topology: Topology,
    theta,
    loader: LoaderLayout | None = "diagonal",
    provider: ProviderSpec = "numpy",
) -> Circuit:
    r"""A runnable circuit: a unary loader, the network with fixed angles, and the
    unary readout.

    The circuit takes no weights. With a loader, its inputs are the \(n - 1\)
    loader angles of `unary_loader_angles(x, loader)`, and its outputs are
    \(\langle Z_i \rangle\), from which `unary_probabilities` gives
    \((Wx)_i^2 / \lVert x \rVert^2\).

    Args:
        topology (Topology): The gate layout.
        theta (array-like): The angles, of shape `(topology.n_angles,)`.
        loader (LoaderLayout | None): The layout of the `UnaryLoader`, or `None` for
            a circuit with the network alone, which starts from
            \(|0 \cdots 0\rangle\). The `"diagonal"` default works for every width
            and uses adjacent wires only. Default: `"diagonal"`.
        provider (ProviderSpec): The backend, as for `Circuit`: a name, a config or
            a backend instance. Default: `"numpy"`.

    Returns:
        Circuit: The compiled NumPy-interface circuit.
    """
    blocks = [] if loader is None else [UnaryLoader(topology.n, layout=loader)]
    blocks.append(to_block(topology, theta))
    return Circuit(
        assemble(blocks), observables=unary_readout(topology.n), provider=provider
    )
