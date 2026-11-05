r"""The orthogonal matrix of a network of RBS gates, in NumPy.

On the two states \(|01\rangle\) and \(|10\rangle\) of its wires \((i, j)\), an RBS
gate acts as the Givens rotation (Cherrat et al., Eq. 1)

\[
\mathrm{RBS}(\theta) =
\begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}
\]

Restricted to the unary basis, where exactly one wire is 1, a network of RBS
gates is therefore the orthogonal matrix \(W(\theta)\) built here: the amplitude
of \(e_r\) after the circuit, started in \(e_c\), is \(W_{rc}\).

Example:
    ```python
    import numpy as np

    from eigenkit.orthogonal import butterfly, orthogonal_matrix

    t = butterfly(8)
    W = orthogonal_matrix(t, np.random.default_rng(0).normal(size=t.n_angles))
    np.allclose(W.T @ W, np.eye(8))  # True
    ```
"""

from __future__ import annotations

import numpy as np

from .topology import Topology


def identity_angles(topology: Topology) -> np.ndarray:
    """The angles that give the identity matrix: all zeros.

    Args:
        topology (Topology): The layout.

    Returns:
        np.ndarray: A float64 array of shape `(topology.n_angles,)`.
    """
    return np.zeros(topology.n_angles)


def _as_angles(topology: Topology, theta) -> np.ndarray:
    theta = np.asarray(theta)
    if np.iscomplexobj(theta):
        raise ValueError("angles must be real, got a complex array")
    if theta.dtype != np.float32:
        theta = theta.astype(np.float64, copy=False)
    if theta.shape != (topology.n_angles,):
        raise ValueError(
            f"{topology.name} on {topology.n} wires needs {topology.n_angles} angles, "
            f"got shape {theta.shape}"
        )
    return theta


def _layer_indices(topology: Topology) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.array([i for i, _ in layer], dtype=np.int64),
            np.array([j for _, j in layer], dtype=np.int64),
        )
        for layer in topology.layers
    ]


def _apply_layer(W, i_idx, j_idx, c, s):
    # In place: W[i_idx] and W[j_idx] are copies, so both rows are updated from the
    # rows before the update. Copying W instead made sparse layouts O(n^3).
    Wi, Wj = W[i_idx], W[j_idx]
    W[i_idx] = c * Wi + s * Wj
    W[j_idx] = -s * Wi + c * Wj
    return W


def orthogonal_matrix(topology: Topology, theta) -> np.ndarray:
    r"""Build the \(n \times n\) orthogonal matrix of a network of RBS gates.

    Numerical contract: starting from `W = eye(n)`, each layer, in order, takes
    the next `k` angles (`k` = the number of pairs in the layer), with
    `c = cos(θ)[:, None]` and `s = sin(θ)[:, None]`, and replaces the rows
    `W[i] <- c*W[i] + s*W[j]` and `W[j] <- -s*W[i] + c*W[j]` for all its pairs at
    once, from the rows before the update. With `reflection=True` the last row is
    negated at the end. These are the operations, in this order, of the torch
    module of the same topology, so both give the same matrix.

    The cost is \(O(\mathrm{depth} \cdot n^2)\).

    Args:
        topology (Topology): The layout.
        theta (array-like): The angles, of shape `(topology.n_angles,)`, in the
            order of `topology.pairs()`. A float32 array gives a float32 matrix,
            and any other input a float64 one.

    Returns:
        np.ndarray: The orthogonal matrix, of shape `(n, n)`.

    Raises:
        ValueError: If `theta` has the wrong shape or is complex.
    """
    theta = _as_angles(topology, theta)
    W = np.eye(topology.n, dtype=theta.dtype)
    at = 0
    for i_idx, j_idx in _layer_indices(topology):
        k = i_idx.size
        c = np.cos(theta[at : at + k])[:, None]
        s = np.sin(theta[at : at + k])[:, None]
        at += k
        W = _apply_layer(W, i_idx, j_idx, c, s)
    if topology.reflection:
        W[-1] = -W[-1]
    return W


def orthogonal_matrix_jacobian(topology: Topology, theta) -> np.ndarray:
    r"""The derivative of `orthogonal_matrix` with respect to each angle.

    It is exact: the derivative of one RBS gate is spliced between the product of
    the layers before it and the product of the layers after it. The cost is
    \(O(n_{\mathrm{angles}} \, n^2 + \mathrm{depth} \cdot n^3)\).

    Args:
        topology (Topology): The layout.
        theta (array-like): The angles, of shape `(topology.n_angles,)`.

    Returns:
        np.ndarray: A float64 array of shape `(n_angles, n, n)`, where entry `k`
        is \(\partial W / \partial \theta_k\).

    Raises:
        ValueError: If `theta` has the wrong shape or is complex.
    """
    theta = _as_angles(topology, theta).astype(np.float64)
    n = topology.n
    layers = _layer_indices(topology)
    prefixes = [np.eye(n)]
    gates = []
    at = 0
    for i_idx, j_idx in layers:
        k = i_idx.size
        c = np.cos(theta[at : at + k])[:, None]
        s = np.sin(theta[at : at + k])[:, None]
        gates.append((at, c, s))
        at += k
        prefixes.append(_apply_layer(prefixes[-1].copy(), i_idx, j_idx, c, s))
    suffix = np.eye(n)
    if topology.reflection:
        suffix[-1, -1] = -1.0
    jac = np.empty((topology.n_angles, n, n))
    for (i_idx, j_idx), (at, c, s), before in zip(
        reversed(layers), reversed(gates), reversed(prefixes[:-1]), strict=True
    ):
        Pi, Pj = before[i_idx], before[j_idx]
        row_i = -s * Pi + c * Pj
        row_j = -c * Pi - s * Pj
        jac[at : at + i_idx.size] = (
            suffix[:, i_idx].T[:, :, None] * row_i[:, None, :]
            + suffix[:, j_idx].T[:, :, None] * row_j[:, None, :]
        )
        suffix = _apply_layer(suffix.T, i_idx, j_idx, c, -s).T
    return jac
