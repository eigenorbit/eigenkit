r"""Certified robustness for networks with a known Lipschitz constant.

If a network \(f = h \circ g\) has a feature map \(g\) that is \(L\)-Lipschitz in
the L2 norm, and its head \(h\) is linear with rows \(w_k\), then the logit gap
\(z_p - z_j\) between the predicted class \(p\) and any other class \(j\) changes
by at most \(L \, \lVert w_p - w_j \rVert \, \lVert \delta \rVert\) under an input
perturbation \(\delta\). So no perturbation of norm below

\[
r = \min_{j \ne p} \frac{z_p - z_j}{L \, \lVert w_p - w_j \rVert}
\]

can change the prediction: \(r\) is a certified radius. The tools here compute
it, train towards it (`cert_ce`), and build the pieces that keep \(L\) known:
`LLNHead` (unit-norm rows), `MaxMin` (a 1-Lipschitz activation that keeps the
norm) and `lipschitz_bound` (an upper bound on \(L\) for a chain of layers). An
orthogonal adapter keeps every singular value of the weight it rotates, so it
leaves `lipschitz_bound` unchanged.

Numerical contracts. Each function documents the exact torch operations it runs,
and the tests pin them bit for bit.

Example:
    ```python
    import torch

    from eigenkit.orthogonal.torch import LLNHead, certified_accuracy, certified_radius

    head = LLNHead(16, 3)
    logits = head(torch.randn(8, 16))
    cert = certified_radius(logits, head.get_weight(), L=1.0)
    labels = torch.randint(0, 3, (8,))
    certified_accuracy(cert.pred == labels, cert.radius, eps=0.1)
    ```
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import OrthogonalAdapter
from .modules import Orthogonal, PartitionedOrthogonal


Pairing = Literal["adjacent", "halves"]


class MaxMin(nn.Module):
    r"""The MaxMin activation: sort the channels within disjoint pairs.

    It is 1-Lipschitz and keeps the norm of its input, because it only permutes
    values. It is not distance-preserving: it maps `(1, 0)` and `(0, 1)` to the
    same point (Anil, Lucas & Grosse, ICML 2019). It acts on dim 1.

    Two pairings are in use and give different functions:

    - `"adjacent"` (GroupSort) pairs channels \(2k\) and \(2k + 1\) and writes the
      minimum first. Numerical contract: `g = x.reshape(b, c // 2, 2, *rest)`,
      `lo = torch.minimum(g[:, :, 0], g[:, :, 1])`, `hi = torch.maximum(...)`,
      then `torch.stack((lo, hi), dim=2).reshape(x.shape)`.
    - `"halves"` (BRONet's MinMax) pairs channel \(k\) with \(k + C/2\) and writes the
      maximum of each pair in the first half. Numerical contract:
      `a, b = torch.split(x, c // 2, dim=1)`, `index = a > b`, then
      `torch.cat([torch.where(index, a, b), torch.where(index, b, a)], dim=1)`.

    Args:
        pairing (Pairing): `"adjacent"` or `"halves"`. Default: `"adjacent"`.
    """

    def __init__(self, pairing: Pairing = "adjacent") -> None:
        super().__init__()
        if pairing not in ("adjacent", "halves"):
            raise ValueError(f"pairing must be 'adjacent' or 'halves', got {pairing!r}")
        self.pairing = pairing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation.

        Args:
            x (torch.Tensor): A tensor `(B, C, ...)` with an even `C`.

        Returns:
            torch.Tensor: The output, of the shape of `x`.

        Raises:
            ValueError: If `C` is odd.
        """
        b, c = x.shape[0], x.shape[1]
        if c % 2:
            raise ValueError(f"channels must be even, got {c}")
        if self.pairing == "adjacent":
            g = x.reshape(b, c // 2, 2, *x.shape[2:])
            lo = torch.minimum(g[:, :, 0], g[:, :, 1])
            hi = torch.maximum(g[:, :, 0], g[:, :, 1])
            return torch.stack((lo, hi), dim=2).reshape(x.shape)
        a, b = torch.split(x, c // 2, dim=1)
        index = a > b
        return torch.cat([torch.where(index, a, b), torch.where(index, b, a)], dim=1)

    def extra_repr(self) -> str:
        return f"pairing={self.pairing}"


class LLNHead(nn.Module):
    r"""A linear head with unit-norm rows (last-layer normalisation).

    The rows \(w_k\) of the weight are normalised at every call, so the
    certificate can use the distances \(\lVert w_p - w_j \rVert\) directly. The bias
    cancels in every logit difference, so the certificate does not see it.

    Numerical contract:
    `F.linear(h, F.normalize(self.linear.weight, dim=1), self.linear.bias)`. The
    parameters are `linear.weight` and `linear.bias`.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of classes.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def get_weight(self) -> torch.Tensor:
        """The weight with unit-norm rows, `F.normalize(self.linear.weight, dim=1)`.

        Returns:
            torch.Tensor: A `(out_features, in_features)` tensor.
        """
        return F.normalize(self.linear.weight, dim=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the head.

        Args:
            h (torch.Tensor): Features `(B, in_features)`.

        Returns:
            torch.Tensor: Logits `(B, out_features)`.
        """
        return F.linear(h, self.get_weight(), self.linear.bias)


class Certificate(NamedTuple):
    """The prediction, the top-1 margin and the certified radius of every row.

    Attributes:
        pred (torch.Tensor): The predicted class of each row, `long`.
        margin (torch.Tensor): The gap between the two largest logits, float64.
        radius (torch.Tensor): The certified L2 radius, float64; `inf` when no
            other class can overtake the prediction.
    """

    pred: torch.Tensor
    margin: torch.Tensor
    radius: torch.Tensor


def head_distances(head_weight: torch.Tensor) -> torch.Tensor:
    r"""The distances \(\lVert w_i - w_j \rVert\) between the rows of a head weight,
    in float64.

    Numerical contract: `w = head_weight.detach().double()`, then
    `torch.cdist(w, w)`.

    Args:
        head_weight (torch.Tensor): The head weight `(K, D)`.

    Returns:
        torch.Tensor: The `(K, K)` matrix of distances.
    """
    w = head_weight.detach().double()
    return torch.cdist(w, w)


@torch.no_grad()
def certified_radius(
    logits: torch.Tensor, head_weight: torch.Tensor, L: float
) -> Certificate:
    """The certified L2 radius of every row of `logits`, for a feature map with
    Lipschitz constant `L` and a linear head with weight `head_weight`.

    Numerical contract, in float64 on the CPU: `z = logits.double()`,
    `d = head_distances(head_weight)`, `pred = z.argmax(1)`,
    `gap = z.gather(1, pred[:, None]) - z`, `dist = d[pred]`,
    `r = gap / (L * dist.clamp_min(1e-300))`; then `r` is set to `inf` on the
    predicted class and wherever `dist == 0` (a class with the same row can never
    overtake the prediction), and the radius is `r.min(1).values`. The margin is
    the difference of the two largest logits, from `z.topk(2, dim=1)`.

    Args:
        logits (torch.Tensor): Logits `(B, K)`, with `K >= 2`.
        head_weight (torch.Tensor): The head weight `(K, D)`, such as
            `LLNHead.get_weight()`.
        L (float): The Lipschitz constant of the feature map in front of the head.

    Returns:
        Certificate: The prediction, the margin and the radius of every row.
    """
    z = logits.detach().double().cpu()
    d = head_distances(head_weight.cpu())
    pred = z.argmax(1)
    gap = z.gather(1, pred[:, None]) - z
    dist = d[pred]
    r = gap / (L * dist.clamp_min(1e-300))
    r[torch.arange(len(pred)), pred] = math.inf
    r[dist == 0] = math.inf
    top2 = z.topk(2, dim=1).values
    return Certificate(pred, top2[:, 0] - top2[:, 1], r.min(1).values)


def certified_accuracy(correct: torch.Tensor, radius: torch.Tensor, eps: float) -> float:
    """The fraction of rows that are correct and certified beyond `eps`.

    Numerical contract: `float(correct.float().mean())` for `eps == 0`, which is
    the plain accuracy, and `float((correct & (radius > eps)).float().mean())`
    otherwise.

    Args:
        correct (torch.Tensor): A boolean tensor, `pred == labels`.
        radius (torch.Tensor): The certified radii, from `certified_radius`.
        eps (float): The L2 radius to certify.

    Returns:
        float: The certified accuracy at `eps`.
    """
    if eps == 0:
        return float(correct.float().mean())
    return float((correct & (radius > eps)).float().mean())


def _pair_dist(head_weight: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (head_weight[y].unsqueeze(1) - head_weight.unsqueeze(0)).norm(dim=-1)


def cert_ce(
    logits: torch.Tensor,
    y: torch.Tensor,
    head_weight: torch.Tensor,
    L: float,
    eps: float,
    tau: float = 4.0,
) -> torch.Tensor:
    r"""A certification-aware cross-entropy.

    Every wrong logit is raised by the most a perturbation of norm `eps` could
    raise it against the true class,
    \(L \cdot \mathrm{eps} \cdot \lVert w_y - w_j \rVert\), and the result is scored
    with a cross-entropy at temperature `1/tau`. Minimising it pushes the margins
    beyond `eps`.

    Numerical contract: `d = (head_weight[y].unsqueeze(1) -
    head_weight.unsqueeze(0)).norm(dim=-1)`, `z = logits + L * eps * d`, then
    `F.cross_entropy(tau * z, y)`. The row of the true class has `d = 0`.

    Args:
        logits (torch.Tensor): Logits `(B, K)`.
        y (torch.Tensor): The labels `(B,)`.
        head_weight (torch.Tensor): The head weight `(K, D)`, in the graph if the
            head trains.
        L (float): The Lipschitz constant of the feature map.
        eps (float): The radius to train for.
        tau (float): The scale of the logits. Default: `4.0`.

    Returns:
        torch.Tensor: The loss, a scalar.
    """
    z = logits + L * eps * _pair_dist(head_weight, y)
    return F.cross_entropy(tau * z, y)


def sigma_max(w: torch.Tensor) -> torch.Tensor:
    """The largest singular value of a matrix, through its Gram matrix.

    Exact and differentiable. Numerical contract:
    `torch.linalg.eigvalsh(w @ w.t()).amax().clamp_min(0).sqrt()`.

    Args:
        w (torch.Tensor): A matrix `(m, n)`.

    Returns:
        torch.Tensor: A scalar tensor.
    """
    return torch.linalg.eigvalsh(w @ w.t()).amax().clamp_min(0).sqrt()


_ONE_LIPSCHITZ = (
    MaxMin,
    nn.ReLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.Flatten,
    nn.Identity,
)
_ROTATIONS = (Orthogonal, PartitionedOrthogonal)


def _matrix_norm(weight: torch.Tensor) -> float:
    # sigma_max(W) = sigma_max(Wᵀ): the Gram matrix is built on the smaller side,
    # so a layer with 3000 outputs and 2 inputs needs a 2 x 2 eigenproblem.
    w = weight.detach().cpu().double()
    w = w.reshape(w.shape[0], -1)
    return float(sigma_max(w.t() if w.shape[0] > w.shape[1] else w))


def _conv_factor(conv: nn.Conv2d) -> float:
    # The bound needs every input pixel to enter each kernel offset at most once.
    # Reflect and replicate padding copy border pixels, so they can break it, and
    # circular padding wraps a pixel around twice once 2p > d(k - 1).
    if conv.padding_mode not in ("zeros", "circular"):
        raise ValueError(
            f"cannot bound a convolution with padding_mode={conv.padding_mode!r}; "
            "lipschitz_bound supports 'zeros' and 'circular' padding"
        )
    if conv.padding_mode == "circular":
        reach = [d * (k - 1) for d, k in zip(conv.dilation, conv.kernel_size)]
        if conv.padding == "valid":
            total = [0, 0]
        elif conv.padding == "same":
            total = reach
        else:
            total = [2 * p for p in conv.padding]
        if any(t > r for t, r in zip(total, reach)):
            raise ValueError(
                f"cannot bound a circular convolution with padding={conv.padding} "
                f"wider than dilation*(kernel_size - 1)/2 = {[r / 2 for r in reach]}: "
                "the padding wraps an input pixel under one kernel offset twice"
            )
    return math.sqrt(conv.kernel_size[0] * conv.kernel_size[1])


def _layer_bound(module: nn.Module) -> float:
    if isinstance(module, OrthogonalAdapter):
        bound = _matrix_norm(module.weight)
        return (
            bound * _conv_factor(module.base)
            if isinstance(module.base, nn.Conv2d)
            else bound
        )
    if isinstance(module, _ROTATIONS):
        return 1.0
    if isinstance(module, LLNHead):
        return _matrix_norm(module.get_weight())
    if isinstance(module, nn.Linear):
        return _matrix_norm(module.weight)
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            raise ValueError("grouped convolutions are not supported")
        return _matrix_norm(module.weight) * _conv_factor(module)
    if isinstance(module, nn.LeakyReLU) and abs(module.negative_slope) <= 1:
        return 1.0
    if isinstance(module, _ONE_LIPSCHITZ):
        return 1.0
    raise ValueError(
        f"cannot bound the Lipschitz constant of {type(module).__name__}; "
        "lipschitz_bound takes a chain of Linear, Conv2d, orthogonal and "
        "1-Lipschitz layers"
    )


def _chain(module: nn.Module):
    if isinstance(module, nn.Sequential):
        for child in module:
            yield from _chain(child)
    else:
        yield module


def lipschitz_bound(model: nn.Module) -> float:
    r"""An upper bound on the L2 Lipschitz constant of a chain of layers.

    The bound is the product of a bound for every layer of an `nn.Sequential`
    (nested ones are flattened): the largest singular value of the weight of an
    `nn.Linear`, an `LLNHead` (its normalised weight) or an `OrthogonalAdapter`
    (its rotated weight); for an `nn.Conv2d`, that of the weight read as a matrix
    `(out, in·kh·kw)`, times \(\sqrt{k_h k_w}\), which bounds the convolution for
    any stride and dilation with `"zeros"` padding, and with `"circular"` padding
    no wider than `dilation * (kernel_size - 1) / 2` on each axis (wider circular
    padding, `"reflect"` and `"replicate"` feed a pixel to one kernel offset more
    than once and raise a `ValueError`); 1 for the
    orthogonal modules and for `MaxMin`, `ReLU`, `LeakyReLU` (slope at most 1),
    `Tanh`, `Sigmoid`, `Flatten` and `Identity`. The singular values are computed
    with `sigma_max` in float64, on the weight matrix or its transpose, whichever
    has fewer rows. A 1x1 convolution and a linear layer are bounded
    exactly.

    An orthogonal adapter keeps every singular value of the weight it rotates, so
    adapting a chain leaves its bound unchanged.

    Args:
        model (nn.Module): An `nn.Sequential`, or a single layer.

    Returns:
        float: The bound.

    Raises:
        ValueError: If a layer is not one of the above, or a convolution pads with
            `"reflect"`, `"replicate"`, or `"circular"` wider than
            `dilation * (kernel_size - 1) / 2`.
    """
    bound = 1.0
    for layer in _chain(model):
        bound *= _layer_bound(layer)
    return bound
