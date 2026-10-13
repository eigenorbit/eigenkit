"""Return the output in the dtype of the input, without destroying the result.

A quantum layer computes expectation values, which are real numbers in `[-1, 1]`.
The layers cast this result back to the dtype of the input, so that they work
with the precision of the surrounding `nn.Module` objects: a `float32` input
gives a `float32` output.

If the cast were always applied, an integer input tensor would truncate every
expectation value to `0` without any warning. The layer would return only zeros,
the gradients would vanish, and training would finish without learning anything.
PyTorch itself does not do this (`sin` of an integer tensor gives a float tensor,
it does not truncate), so this module does not do it either.
"""

from __future__ import annotations

import torch


def match_input_dtype(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Cast `out` to the dtype of `x` when that dtype can hold it.

    Args:
        out (torch.Tensor): The expectation values computed by the layer.
        x (torch.Tensor): The input that the layer was called with.

    Returns:
        `out` in the dtype of `x` when `x` is floating point, and `out` unchanged
            otherwise. An integer or boolean input is promoted to a floating point
            output, as everywhere else in PyTorch.
    """
    if x.is_floating_point():
        return out.to(x.dtype)
    return out
