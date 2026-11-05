"""A reference implementation that the orthogonal module must reproduce.

It defines:

- _check_width, num_butterfly_angles, butterfly_pairs, build_butterfly_matrix,
  ButterflyOrthogonal;
- binary_parts, mixing_permutation, PartitionedButterfly;
- fold_bn;
- sigma_max, GroupSort;
- minmax, LLNHead;
- head_distances, radii, certified_accuracy;
- _pair_dist, cert_ce.

The tests run these and the library in the same process and compare the results
bit for bit, so any change in the order of operations shows up on every torch
version.
"""

# ruff: noqa: E741
# fmt: off
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_width(channels: int) -> None:
    if not isinstance(channels, int) or channels < 2 or channels & (channels - 1):
        raise ValueError(
            f"butterfly width must be a power of two and at least 2, got {channels!r}. "
            "Widths are never padded silently: pick a power-of-two channel count."
        )


def num_butterfly_angles(channels: int) -> int:
    """(C / 2) * log2(C) -- the gate count of the butterfly circuit.

    Table 1 of Cherrat et al., "Quantum Vision Transformers" (Quantum 2024).
    For C = 64 that is 192 angles, against 4096 for a dense 64x64 matrix.
    """
    _check_width(channels)
    return (channels // 2) * int(round(math.log2(channels)))


def butterfly_pairs(channels: int, level: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Index pairs for one level: (i, j) with j = i XOR 2**level and i < j.

    The Cooley-Tukey FFT pattern. Every index appears in exactly one pair, so
    the rotations within a level act on disjoint pairs and commute.
    """
    _check_width(channels)
    m = int(round(math.log2(channels)))
    if not 0 <= level < m:
        raise ValueError(f"level must be in [0, {m}), got {level}")
    bit = 1 << level
    idx = torch.arange(channels, dtype=torch.long)
    i_idx = idx[(idx & bit) == 0]
    return i_idx, i_idx | bit


def build_butterfly_matrix(theta: torch.Tensor, channels: int, levels) -> torch.Tensor:
    """Materialise W(theta) by left-multiplying the identity by each level.

    Uses out-of-place ``index_copy`` so autograd sees a clean graph. The only
    Python loop is over the log2(C) levels, never over the C/2 gates in a level.
    """
    expected = sum(int(i.numel()) for i, _ in levels)
    if theta.numel() != expected:
        raise ValueError(f"circuit needs {expected} angles, got {theta.numel()}")
    W = torch.eye(channels, device=theta.device, dtype=theta.dtype)
    at = 0
    for i_idx, j_idx in levels:
        k = i_idx.numel()
        c = torch.cos(theta[at:at + k]).unsqueeze(1)
        s = torch.sin(theta[at:at + k]).unsqueeze(1)
        at += k
        Wi, Wj = W.index_select(0, i_idx), W.index_select(0, j_idx)
        W = W.index_copy(0, i_idx, c * Wi + s * Wj) \
             .index_copy(0, j_idx, -s * Wi + c * Wj)
    return W


class ButterflyOrthogonal(nn.Module):
    """Orthogonal 1x1 convolution parameterised by the angles of a Givens circuit.

    A Givens rotation is orthogonal for *every* value of its angle, and a product
    of orthogonal matrices is orthogonal -- so there is no constraint to enforce,
    no projection step and no penalty term. The angles train with plain Adam and
    the matrix is orthogonal at every step by construction.

    On quantum hardware each rotation is one RBS gate; acting on unary states an
    RBS gate *is* a Givens rotation, so this is the classical simulation of a
    quantum orthogonal layer and the angle count is the gate count. See
    `analysis.ipynb` §1 for the references.

    Implementation is "materialise, then GEMM": W depends only on theta, not on
    the data, so it is built once per forward pass and applied as an ordinary 1x1
    convolution. It is rebuilt every training step -- caching it would starve the
    angles of gradients -- and cached only when autograd is off.
    """

    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        _check_width(channels)
        self.channels = channels
        self.num_levels = int(round(math.log2(channels)))
        self.num_angles = num_butterfly_angles(channels)
        # Small angles: W starts near the identity, which trains more stably
        # than a uniformly random orthogonal matrix.
        self.theta = nn.Parameter(torch.randn(self.num_angles) * 0.1)
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        for l in range(self.num_levels):
            i_idx, j_idx = butterfly_pairs(channels, l)
            self.register_buffer(f"i_index_{l}", i_idx, persistent=False)
            self.register_buffer(f"j_index_{l}", j_idx, persistent=False)
        self._cache_key = None
        self._cache_W = None
        self._frozen_W = None

    def _levels(self):
        return [(getattr(self, f"i_index_{l}"), getattr(self, f"j_index_{l}"))
                for l in range(self.num_levels)]

    def matrix(self) -> torch.Tensor:
        """The materialised C x C orthogonal matrix."""
        if self._frozen_W is not None:
            return self._frozen_W
        # Cache only when autograd is off: a cached graph-carrying W would leak
        # memory and silently reuse a stale graph across steps.
        cacheable = not self.training and not torch.is_grad_enabled()
        key = (self.theta._version, self.theta.device, self.theta.dtype)
        if cacheable and self._cache_key == key and self._cache_W is not None:
            return self._cache_W
        W = build_butterfly_matrix(self.theta, self.channels, self._levels())
        if cacheable:
            self._cache_key, self._cache_W = key, W
        return W

    def freeze_matrix(self) -> None:
        """Snapshot W and reuse it, detached, until unfrozen.

        Needed for input-space attacks: PGD wants gradients w.r.t. the *input*
        while the weights are fixed, which is not the condition the eval cache
        covers. Also turns the per-step materialisation into a single matmul.
        """
        with torch.no_grad():
            self._frozen_W = self.matrix().detach()

    def unfreeze_matrix(self) -> None:
        self._frozen_W = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.matrix()
        return F.conv2d(x, W.reshape(self.channels, self.channels, 1, 1), self.bias)

    def singular_values(self) -> tuple[float, float]:
        """(sigma_max, sigma_min). Both must be 1: this is checked, not assumed."""
        with torch.no_grad():
            s = torch.linalg.svdvals(self.matrix().cpu().double())
            return float(s.max()), float(s.min())

    def orthogonality_error(self) -> float:
        """max |W^T W - I|, the quantity checked after training."""
        with torch.no_grad():
            W = self.matrix().cpu().double()
            return float((W.t() @ W - torch.eye(self.channels, dtype=W.dtype)).abs().max())

    def extra_repr(self) -> str:
        return (f"channels={self.channels}, angles={self.num_angles}, "
                f"levels={self.num_levels}")


def binary_parts(width: int) -> list[int]:
    """The powers of two that sum to `width`, largest first: 588 -> [512, 64, 8, 4]."""
    if not isinstance(width, int) or width < 1:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    return [1 << b for b in range(width.bit_length() - 1, -1, -1) if width >> b & 1]


def mixing_permutation(width: int, seed: int) -> torch.Tensor:
    """A fixed random permutation of the channels (CPU generator, reproducible)."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(width, generator=g)


class PartitionedButterfly(nn.Module):
    """C x C orthogonal matrix from `stages` block-diagonal butterflies over the
    binary parts of C, with a fixed channel permutation between consecutive stages.

    Trainable: the angles only (no bias). Acts on dim 1 of its input, so it is a
    1x1 convolution on (B, C, H, W) and a linear map on (B, C).
    """

    def __init__(self, width: int, stages: int = 2, perm_seed: int = 0):
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be at least 1")
        self.width, self.stages, self.perm_seed = width, stages, perm_seed
        self.parts = binary_parts(width)
        rotating = [p for p in self.parts if p >= 2]
        self.blocks = nn.ModuleList(
            nn.ModuleList(ButterflyOrthogonal(p, bias=False) for p in rotating)
            for _ in range(stages))
        with torch.no_grad():
            for stage in self.blocks:
                for b in stage:
                    b.theta.zero_()                # R = I at initialisation
        for s in range(stages - 1):
            self.register_buffer(f"perm_{s}", mixing_permutation(width, perm_seed * 1000 + s),
                                 persistent=False)

    @property
    def num_angles(self) -> int:
        return sum(b.num_angles for stage in self.blocks for b in stage)

    def _stage_matrix(self, s: int) -> torch.Tensor:
        mats = [b.matrix() for b in self.blocks[s]]
        if self.parts[-1] == 1:                    # odd width: the last channel stays
            mats.append(torch.ones(1, 1, device=mats[0].device if mats else None,
                                   dtype=mats[0].dtype if mats else torch.float32))
        return torch.block_diag(*mats)

    def matrix(self) -> torch.Tensor:
        """The materialised C x C orthogonal matrix R."""
        R = self._stage_matrix(0)
        for s in range(1, self.stages):
            perm = getattr(self, f"perm_{s - 1}")
            B = self._stage_matrix(s)
            # P^T B P with P = I[perm]: (P^T B P)[a, b] = B[q[a], q[b]], q = argsort(perm)
            inv = torch.argsort(perm)
            R = B[inv][:, inv] @ R
        return R

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        R = self.matrix().to(z.dtype)
        if z.dim() == 2:
            return z @ R.t()
        return torch.einsum("oc,bc...->bo...", R, z)

    def orthogonality_error(self) -> float:
        """max |R^T R - I|."""
        with torch.no_grad():
            R = self.matrix().cpu().double()
            return float((R.t() @ R - torch.eye(self.width, dtype=R.dtype)).abs().max())

    def extra_repr(self) -> str:
        return (f"width={self.width}, parts={self.parts}, stages={self.stages}, "
                f"angles={self.num_angles}, perm_seed={self.perm_seed}")


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """(W_eff, b_eff), float64: the eval-mode conv + BN as one convolution,
    W_eff = W * gamma / sqrt(var + eps), b_eff = beta - gamma * mean / sqrt(var + eps)."""
    scale = bn.weight.detach().double().cpu() / torch.sqrt(
        bn.running_var.detach().double().cpu() + bn.eps)
    w = conv.weight.detach().double().cpu() * scale.view(-1, 1, 1, 1)
    b = bn.bias.detach().double().cpu() - scale * bn.running_mean.detach().double().cpu()
    return w, b


def sigma_max(w: torch.Tensor) -> torch.Tensor:
    """Largest singular value of a matrix, via its Gram matrix.

    Exact and differentiable. The Gram matrix is tiny for the readout
    (num_classes x num_classes), and for a square mixer this is the same cost as
    an SVD but stays on the accelerator.
    """
    return torch.linalg.eigvalsh(w @ w.t()).amax().clamp_min(0).sqrt()


class GroupSort(nn.Module):
    """MaxMin: sort channels within disjoint pairs.

    1-Lipschitz, and norm preserving because sorting only permutes values. Not
    distance preserving, though: it maps (1,0) and (0,1) to the same point, so
    the network is 1-Lipschitz but not bi-Lipschitz. (Anil, Lucas & Grosse,
    ICML 2019.)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        if c % 2:
            raise ValueError(f"channels must be even, got {c}")
        g = x.reshape(b, c // 2, 2, *x.shape[2:])
        lo = torch.minimum(g[:, :, 0], g[:, :, 1])
        hi = torch.maximum(g[:, :, 0], g[:, :, 1])
        return torch.stack((lo, hi), dim=2).reshape(x.shape)


def minmax(x: torch.Tensor) -> torch.Tensor:
    """BRONet's MinMax (models/layers/activation.py), dim 1: split channels in two
    halves a, b; output [max(a, b), min(a, b)]. 1-Lipschitz and norm preserving."""
    a, b = torch.split(x, x.shape[1] // 2, dim=1)
    index = a > b
    return torch.cat([torch.where(index, a, b), torch.where(index, b, a)], dim=1)


class LLNHead(nn.Module):
    """Linear layer with unit-norm rows (BRONet's LLN head); bias allowed -- it
    cancels in every logit difference, so the certificate does not see it."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def get_weight(self) -> torch.Tensor:
        return F.normalize(self.linear.weight, dim=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.get_weight(), self.linear.bias)


def head_distances(head_weight: torch.Tensor) -> torch.Tensor:
    """(K, K) matrix of ||w_i - w_j|| for unit-norm rows."""
    w = head_weight.detach().double()
    return torch.cdist(w, w)


@torch.no_grad()
def radii(logits: torch.Tensor, head_weight: torch.Tensor, L: float) -> dict:
    """pred, top-1 margin, and the certified radius of every row of `logits`."""
    z = logits.detach().double().cpu()
    d = head_distances(head_weight.cpu())
    pred = z.argmax(1)
    gap = z.gather(1, pred[:, None]) - z                      # z_p - z_j
    dist = d[pred]                                            # ||w_p - w_j||
    r = gap / (L * dist.clamp_min(1e-300))
    r[torch.arange(len(pred)), pred] = math.inf
    r[dist == 0] = math.inf                                   # identical rows: never beats p
    top2 = z.topk(2, dim=1).values
    return {"pred": pred, "margin": top2[:, 0] - top2[:, 1], "radius": r.min(1).values}


def certified_accuracy(correct: torch.Tensor, radius: torch.Tensor, eps: float) -> float:
    if eps == 0:
        return float(correct.float().mean())
    return float((correct & (radius > eps)).float().mean())


def _pair_dist(head_weight: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, K): ||w_y - w_j|| for each row's label y."""
    return (head_weight[y].unsqueeze(1) - head_weight.unsqueeze(0)).norm(dim=-1)


def cert_ce(logits, y, head_weight, L: float, eps: float, tau: float = 4.0):
    z = logits + L * eps * _pair_dist(head_weight, y)
    return F.cross_entropy(tau * z, y)
# fmt: on
