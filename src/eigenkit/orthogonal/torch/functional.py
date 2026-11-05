"""Functional forms of the orthogonal operations.

Example:
    ```python
    import torch

    from eigenkit.orthogonal.torch import Orthogonal, rotate_weight

    conv = torch.nn.Conv2d(16, 32, 3)
    R = Orthogonal(32, init="random", std=1.0).matrix()
    W = rotate_weight(R, conv.weight)  # same shape, same singular values
    ```
"""

from __future__ import annotations

from typing import Literal

import torch

from ..blocks import LoaderLayout, _check_loader, _loader_plan


Side = Literal["out", "in"]


def _unit_scale(batch: torch.Tensor) -> torch.Tensor:
    # Scale each row by a power of two that brings its largest entry into
    # [0.5, 1). The scaling is exact, so the angles keep every bit wherever the
    # norm neither overflows nor underflows, and stay finite where it would. The
    # factor is a constant in two halves, so neither overflows the dtype, and the
    # backward pass multiplies by the same powers of two (torch.ldexp would
    # build them in float32 there).
    largest = batch.detach().abs().amax(dim=-1, keepdim=True)
    _, exponent = torch.frexp(largest)
    half = exponent // 2
    one = torch.ones_like(largest)
    return batch * torch.ldexp(one, -half) * torch.ldexp(one, half - exponent)


def unary_loader_angles(
    x: torch.Tensor, layout: LoaderLayout = "diagonal"
) -> torch.Tensor:
    r"""The input angles of `UnaryLoader` that load \(x / \lVert x \rVert\),
    differentiable in `x`.

    The PyTorch form of `eigenkit.orthogonal.unary_loader_angles`: the same gate
    plan, with `torch.atan2` on the signed component or the norm of the wires each
    gate keeps and passes on. Each row is first scaled by a power of two, which
    is exact, so the angles do not depend on the scale of `x`, at any finite
    scale.

    Args:
        x (torch.Tensor): One vector `(n,)` or a batch `(B, n)`.
        layout (LoaderLayout): The loader layout. Default: `"diagonal"`.

    Returns:
        torch.Tensor: The angles, `(n - 1,)` or `(B, n - 1)`, in the dtype of `x`
        (float64 for an integer `x`).

    Raises:
        ValueError: If `x` is not 1-D or 2-D, a row is zero or not finite, or the
            layout does not accept the width.
    """
    if x.dim() not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D, got shape {tuple(x.shape)}")
    if not x.is_floating_point():
        x = x.to(torch.float64)
    batch = x.unsqueeze(0) if x.dim() == 1 else x
    n = batch.shape[1]
    _check_loader(n, layout)
    if not bool(torch.isfinite(batch).all()):
        raise ValueError("cannot load a vector with NaN or infinite entries")
    if bool((batch == 0).all(dim=1).any()):
        raise ValueError("cannot load a zero vector")
    batch = _unit_scale(batch)

    def value(wires):
        if len(wires) == 1:
            return batch[:, wires[0]]
        return torch.linalg.vector_norm(batch[:, list(wires)], dim=1)

    angles = []
    for layer in _loader_plan(n, layout)[1]:
        for step in layer:
            sign = -1.0 if step.receiver > step.owner else 1.0
            angles.append(torch.atan2(sign * value(step.passed), value(step.keep)))
    out = torch.stack(angles, dim=1)
    return out[0] if x.dim() == 1 else out


def rotate_weight(R: torch.Tensor, W: torch.Tensor, side: Side = "out") -> torch.Tensor:
    r"""Rotate the output or the input channels of a weight tensor by an orthogonal
    matrix.

    The weight is read as the matrix `W.reshape(out, in·k·k)`. Rotating its rows
    (`side="out"`, \(RW\)) or its input channels (`side="in"`,
    \(W (R^\top \otimes I)\)) keeps
    every singular value of that matrix, so the operator norm, and the Lipschitz
    constant, of the layer are unchanged. For a convolution `R` acts pixel by
    pixel, so the same holds for the convolution operator.

    Numerical contract for `side="out"`:
    `(R @ W.reshape(W.shape[0], -1)).reshape(W.shape)`, which is `R @ W` for a 2-D
    weight. `side="in"` computes
    `torch.einsum("oi...,ji->oj...", W, R).contiguous()`, which is `W @ R.T` for a
    2-D weight.

    Args:
        R (torch.Tensor): An orthogonal matrix, `(out, out)` for `side="out"` or
            `(in, in)` for `side="in"`.
        W (torch.Tensor): A weight `(out, in)` or `(out, in, *kernel)`.
        side (Side): `"out"` or `"in"`. Default: `"out"`.

    Returns:
        torch.Tensor: The rotated weight, of the shape of `W`.

    Raises:
        ValueError: If `side` is unknown or `R` does not match the axis.
    """
    if side == "out":
        if R.shape != (W.shape[0], W.shape[0]):
            raise ValueError(
                f"R of shape {tuple(R.shape)} cannot rotate the {W.shape[0]} outputs"
            )
        return (R @ W.reshape(W.shape[0], -1)).reshape(W.shape)
    if side == "in":
        if W.dim() < 2 or R.shape != (W.shape[1], W.shape[1]):
            raise ValueError(
                f"R of shape {tuple(R.shape)} cannot rotate the inputs of a weight "
                f"of shape {tuple(W.shape)}"
            )
        # contiguous: the einsum result is laid out differently from W, which would
        # send a convolution down another kernel and change its last bits.
        return torch.einsum("oi...,ji->oj...", W, R).contiguous()
    raise ValueError(f"side must be 'out' or 'in', got {side!r}")
