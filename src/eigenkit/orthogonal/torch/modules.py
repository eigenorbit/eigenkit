"""PyTorch modules for orthogonal networks: the angles are an `nn.Parameter`, and the
orthogonal matrix is rebuilt from them, so it stays orthogonal at every training step.

Numerical contracts. Each module documents the exact sequence of torch operations
it runs, and the tests pin it bit for bit. Changing one of them is a breaking
change, because it changes the last bits of every result and so the trajectory of
any training run.

Example:
    ```python
    import torch

    from eigenkit.orthogonal.torch import Orthogonal

    rotation = Orthogonal(8, "butterfly", init="random", std=0.5)
    x = torch.randn(4, 8)
    y = rotation(x)  # x @ R.T
    torch.allclose(y.norm(dim=1), x.norm(dim=1))  # True
    ```
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..topology import Topology, TopologyName, _as_int
from ..topology import topology as _named_topology


Init = Literal["identity", "random"]
Mode = Literal["materialise", "apply"]

MAX_MATERIALISE = 2**31
r"""The largest \(\mathrm{depth} \cdot n^2\) for which `mode="materialise"` builds
the dense matrix."""


def _resolve(n: int, topology: TopologyName | Topology) -> Topology:
    if isinstance(topology, Topology):
        if topology.n != n:
            raise ValueError(f"topology is on {topology.n} wires, but n={n}")
        return topology
    return _named_topology(topology, n)


def _materialise(theta: torch.Tensor, layers, n: int, reflection: bool) -> torch.Tensor:
    W = torch.eye(n, device=theta.device, dtype=theta.dtype)
    at = 0
    for i_idx, j_idx in layers:
        k = i_idx.numel()
        c = torch.cos(theta[at : at + k]).unsqueeze(1)
        s = torch.sin(theta[at : at + k]).unsqueeze(1)
        at += k
        Wi, Wj = W.index_select(0, i_idx), W.index_select(0, j_idx)
        W = W.index_copy(0, i_idx, c * Wi + s * Wj).index_copy(0, j_idx, -s * Wi + c * Wj)
    if reflection:
        W = torch.cat((W[:-1], -W[-1:]))
    return W


def _pair_layers(
    topology: Topology, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The `(i, j)` index tensors of every layer of `topology`, on `device`."""
    pairs = topology.pairs()
    i_all = torch.tensor([i for i, _ in pairs], dtype=torch.long, device=device)
    j_all = torch.tensor([j for _, j in pairs], dtype=torch.long, device=device)
    layers, at = [], 0
    for layer in topology.layers:
        layers.append((i_all[at : at + len(layer)], j_all[at : at + len(layer)]))
        at += len(layer)
    return layers


def _topology_matrix(
    topology: Topology,
    theta: torch.Tensor,
    layers: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    # `layers` are the index tensors of `_pair_layers(topology, theta.device)`;
    # a caller that builds many matrices passes them in instead of rebuilding them.
    if layers is None:
        layers = _pair_layers(topology, theta.device)
    return _materialise(theta, layers, topology.n, topology.reflection)


def _apply_layers(theta: torch.Tensor, layers, x: torch.Tensor, reflection: bool):
    at = 0
    for i_idx, j_idx in layers:
        k = i_idx.numel()
        c = torch.cos(theta[at : at + k]).to(x.dtype)
        s = torch.sin(theta[at : at + k]).to(x.dtype)
        at += k
        xi, xj = x.index_select(-1, i_idx), x.index_select(-1, j_idx)
        x = x.index_copy(-1, i_idx, c * xi + s * xj).index_copy(
            -1, j_idx, -s * xi + c * xj
        )
    if reflection:
        x = torch.cat((x[..., :-1], -x[..., -1:]), dim=-1)
    return x


def _check_dim(x: torch.Tensor, dim: int) -> str:
    if x.dim() == 2 and dim in (-1, 1):
        return "last"
    if x.dim() >= 3 and dim == 1:
        return "channels"
    raise ValueError(
        f"rotate a 2-D input along its last axis, or an input of 3 or more "
        f"dimensions along dim=1; got shape {tuple(x.shape)} with dim={dim}"
    )


def _apply_matrix(R: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
    if _check_dim(x, dim) == "last":
        return x @ R.t()
    return torch.einsum("oc,bc...->bo...", R, x)


class _Rotation(nn.Module):
    """Freezing, the `frozen()` context, `forward(x, dim)` and the diagnostics,
    shared by every module that provides `matrix()` and `n`."""

    n: int

    def __init__(self) -> None:
        super().__init__()
        self._frozen_W: torch.Tensor | None = None

    def matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def freeze_matrix(self) -> None:
        """Store `matrix().detach()` and return it from `matrix()` until
        `unfreeze_matrix()`.

        Gradients still flow to the input, but not to the angles. This is the
        setting of an input-space attack, and it turns the per-step build of the
        matrix into one matrix product.
        """
        with torch.no_grad():
            self._frozen_W = self.matrix().detach()

    def unfreeze_matrix(self) -> None:
        """Drop the stored matrix, so `matrix()` follows the angles again."""
        self._frozen_W = None

    @contextlib.contextmanager
    def frozen(self) -> Iterator[_Rotation]:
        """A context in which the matrix is frozen (see `freeze_matrix`).

        A matrix that is already frozen when the context opens stays frozen, as it
        is, when it closes: only a freeze made by the context is undone.

        Yields:
            The module itself.
        """
        was_frozen = self._frozen_W is not None
        if not was_frozen:
            self.freeze_matrix()
        try:
            yield self
        finally:
            if not was_frozen:
                self.unfreeze_matrix()

    def _fixed_buffers(self) -> dict[str, torch.Tensor]:
        return {}

    def _register_fixed_buffers(self) -> None:
        # Built on the CPU, where the values are defined, then placed with the
        # angles: on the meta device too, so a meta-built module runs on meta.
        reference = next(self.parameters(), None)
        device = reference.device if reference is not None else torch.device("cpu")
        for name, value in self._fixed_buffers().items():
            self.register_buffer(name, value.to(device), persistent=False)

    def _apply(self, fn, recurse=True):
        # The fixed buffers (pair indices, permutations) are not saved, so after
        # `to_empty()` no checkpoint could restore them: rebuild them where they
        # now live. The frozen matrix is not a buffer, so `.to()`, `.double()` and
        # the like would otherwise leave it behind; it follows the angles' device
        # and dtype rather than going through `fn`, which for `to_empty()` would
        # swap it for uninitialised memory.
        super()._apply(fn, recurse)
        for name, value in self._fixed_buffers().items():
            setattr(self, name, value.to(getattr(self, name).device))
        reference = next(self.parameters(), None)
        if self._frozen_W is not None and reference is not None:
            self._frozen_W = self._frozen_W.to(
                device=reference.device, dtype=reference.dtype
            )
        return self

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        r"""Rotate `x` along one axis: \(y = Rx\) on that axis.

        Numerical contract: `R = self.matrix().to(x.dtype)`; a 2-D `x` along its
        last axis gives `x @ R.t()`, and an `x` of 3 or more dimensions along
        `dim=1` gives `torch.einsum("oc,bc...->bo...", R, x)`.

        Args:
            x (torch.Tensor): A 2-D batch `(B, n)`, or a tensor `(B, n, ...)` such
                as a feature map `(B, n, H, W)`.
            dim (int): The axis to rotate: `-1` (or `1`) for a 2-D input, `1` for
                a larger one. Default: `-1`.

        Returns:
            torch.Tensor: The rotated tensor, of the shape of `x`.

        Raises:
            ValueError: For any other combination of shape and `dim`.
        """
        R = self.matrix().to(x.dtype)
        return _apply_matrix(R, x, dim)

    def singular_values(self) -> tuple[float, float]:
        """The largest and smallest singular values of the matrix, in float64.

        Both are 1 for an orthogonal matrix; this checks it rather than assumes it.

        Returns:
            tuple[float, float]: `(sigma_max, sigma_min)`.
        """
        with torch.no_grad():
            s = torch.linalg.svdvals(self.matrix().cpu().double())
            return float(s.max()), float(s.min())

    def orthogonality_error(self) -> float:
        r"""\(\max \lvert W^\top W - I \rvert\), computed in float64.

        Returns:
            float: The largest deviation from orthogonality.
        """
        with torch.no_grad():
            W = self.matrix().cpu().double()
            return float((W.t() @ W - torch.eye(self.n, dtype=W.dtype)).abs().max())


class Orthogonal(_Rotation):
    r"""An \(n \times n\) orthogonal matrix parameterised by the angles of a network of
    RBS (Givens) rotations.

    A Givens rotation is orthogonal for every angle, and a product of orthogonal
    matrices is orthogonal, so the matrix is orthogonal at every training step with
    no projection and no penalty. On quantum hardware each rotation is one RBS gate,
    so this is the classical simulation of a quantum orthogonal layer.

    The `state_dict` holds only `theta`. The pair indices are buffers that are not
    saved, so `.to(device)` moves them and a checkpoint stays small. They are
    rebuilt from the topology whenever the module moves, so a module built on the
    `meta` device, then `to_empty()` and `load_state_dict()`, works.

    Numerical contracts:

    - Initialisation: `init="identity"` sets `theta = torch.zeros(n_angles)` and
      draws no random numbers; `init="random"` sets
      `theta = torch.randn(n_angles) * std`, one call on the global generator,
      made before any other random draw of the module.
    - The matrix: starting from `W = torch.eye(n)` in the dtype and on the device
      of `theta`, each layer takes the next `k` angles,
      `c = cos(theta[at:at+k]).unsqueeze(1)`, `s = sin(...).unsqueeze(1)`, and sets
      `W = W.index_copy(0, i, c*Wi + s*Wj).index_copy(0, j, -s*Wi + c*Wj)` with
      `Wi = W.index_select(0, i)`, `Wj = W.index_select(0, j)`. With a reflected
      topology the last row is negated at the end.
    - Caching: the matrix is cached only when
      `not self.training and not torch.is_grad_enabled()`, under the key
      `(theta._version, theta.device, theta.dtype)`, and it is reused only while
      `theta` still holds the values it was built from (`torch.equal`), so a
      write through `theta.data` cannot leave it stale. The cache decides only
      when the matrix is rebuilt, never its values. On the `meta` device nothing
      is cached. A frozen matrix (`freeze_matrix`) takes precedence.

    `mode="materialise"` builds the dense matrix, at a cost of
    \(O(\mathrm{depth} \cdot n^2)\), and applies it as one matrix product. It is
    refused when \(\mathrm{depth} \cdot n^2\) exceeds `MAX_MATERIALISE`
    (\(2^{31}\)), which rules out a pyramid wider than 1024. `mode="apply"`
    applies the rotations to the input layer by layer, at a cost of
    \(O(\mathrm{depth} \cdot n \cdot B)\) for a batch of \(B\), and never builds the
    matrix in `forward`; it has no contract with the materialised path, whose last
    bits it does not reproduce.

    Args:
        n (int): The width.
        topology (TopologyName | Topology): A layout name, or a `Topology` on `n`
            wires. Default: `"butterfly"`.
        init (Init): `"identity"` for zero angles, `"random"` for normal angles of
            standard deviation `std`. Default: `"identity"`.
        std (float): The standard deviation of the random angles. Default: `0.1`.
        mode (Mode): `"materialise"` or `"apply"`. Default: `"materialise"`.
    """

    def __init__(
        self,
        n: int,
        topology: TopologyName | Topology = "butterfly",
        *,
        init: Init = "identity",
        std: float = 0.1,
        mode: Mode = "materialise",
    ) -> None:
        super().__init__()
        layout = _resolve(n, topology)
        if init not in ("identity", "random"):
            raise ValueError(f"init must be 'identity' or 'random', got {init!r}")
        if mode not in ("materialise", "apply"):
            raise ValueError(f"mode must be 'materialise' or 'apply', got {mode!r}")
        if mode == "materialise" and layout.depth * layout.n**2 > MAX_MATERIALISE:
            raise ValueError(
                f"materialising {layout.name} on {layout.n} wires costs depth·n² = "
                f"{layout.depth * layout.n**2:,} > {MAX_MATERIALISE:,}; "
                "use mode='apply'"
            )
        if init == "identity":
            theta = torch.zeros(layout.n_angles)
        else:
            theta = torch.randn(layout.n_angles) * std
        self.theta = nn.Parameter(theta)
        self.topology = layout
        self.n = layout.n
        self.mode = mode
        self._init = init
        self._std = std
        self._register_fixed_buffers()
        self._layer_sizes = tuple(len(layer) for layer in layout.layers)
        self._drop_cache()

    @property
    def n_angles(self) -> int:
        """The number of angles, one per RBS gate."""
        return self.theta.numel()

    def reset_parameters(self) -> None:
        """Initialise the angles again, by the rule of `init`.

        `init="identity"` sets them to zero and draws nothing; `init="random"`
        draws `torch.randn(n_angles) * std` from the global generator, as the
        constructor does, and copies it into `theta` in place. The estimators call
        this after seeding, so `random_state` controls the random angles, and an
        identity start stays the identity.
        """
        with torch.no_grad():
            if self._init == "identity":
                self.theta.zero_()
            else:
                self.theta.copy_(torch.randn(self.n_angles) * self._std)

    def _drop_cache(self) -> None:
        self._cache_key = None
        self._cache_W = None
        self._cache_theta = None

    def _apply(self, fn, recurse=True):
        # A cached matrix is not a buffer, so it would stay on the device the
        # module left; the next eval call rebuilds it where the angles now are.
        super()._apply(fn, recurse)
        self._drop_cache()
        return self

    def _fixed_buffers(self) -> dict[str, torch.Tensor]:
        pairs = self.topology.pairs()
        with torch.device("cpu"):
            return {
                "pair_i": torch.tensor([i for i, _ in pairs], dtype=torch.long),
                "pair_j": torch.tensor([j for _, j in pairs], dtype=torch.long),
            }

    def _layers(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out, at = [], 0
        for k in self._layer_sizes:
            out.append((self.pair_i[at : at + k], self.pair_j[at : at + k]))
            at += k
        return out

    def matrix(self) -> torch.Tensor:
        r"""The materialised \(n \times n\) orthogonal matrix (see the class contract).

        Returns:
            torch.Tensor: The matrix, in the dtype and on the device of `theta`.
        """
        if self._frozen_W is not None:
            return self._frozen_W
        # A cached matrix that carries a graph would leak memory and reuse a stale
        # graph, so the cache is used only with autograd off. A meta tensor has no
        # values to check, and torch.equal has no meta kernel.
        cacheable = (
            not self.training and not torch.is_grad_enabled() and not self.theta.is_meta
        )
        key = (self.theta._version, self.theta.device, self.theta.dtype)
        if (
            cacheable
            and self._cache_key == key
            and self._cache_W is not None
            and torch.equal(self._cache_theta, self.theta)
        ):
            return self._cache_W
        W = _materialise(self.theta, self._layers(), self.n, self.topology.reflection)
        if cacheable:
            self._cache_key, self._cache_W = key, W
            self._cache_theta = self.theta.detach().clone()
        return W

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        r"""Rotate `x` along one axis: \(y = Rx\) on that axis.

        In `mode="materialise"`, or with a frozen matrix: `R =
        self.matrix().to(x.dtype)`, then `x @ R.t()` for a 2-D `x` along its last
        axis and `torch.einsum("oc,bc...->bo...", R, x)` for an `x` of 3 or more
        dimensions along `dim=1`. In `mode="apply"` the rotations act on `x`
        directly.

        Args:
            x (torch.Tensor): A 2-D batch `(B, n)`, or a tensor `(B, n, ...)`.
            dim (int): `-1` (or `1`) for a 2-D input, `1` for a larger one.
                Default: `-1`.

        Returns:
            torch.Tensor: The rotated tensor, of the shape of `x`.

        Raises:
            ValueError: For any other combination of shape and `dim`.
        """
        if self.mode == "materialise" or self._frozen_W is not None:
            return super().forward(x, dim)
        if _check_dim(x, dim) == "last":
            return _apply_layers(self.theta, self._layers(), x, self.topology.reflection)
        moved = x.movedim(1, -1)
        out = _apply_layers(self.theta, self._layers(), moved, self.topology.reflection)
        return out.movedim(-1, 1)

    def extra_repr(self) -> str:
        return (
            f"n={self.n}, topology={self.topology.name}, angles={self.n_angles}, "
            f"depth={self.topology.depth}, mode={self.mode}"
        )


class OrthogonalConv1x1(Orthogonal):
    """An orthogonal 1x1 convolution: `Orthogonal` applied to the channels of a
    feature map, plus an optional bias.

    Numerical contract: `forward(x) = F.conv2d(x, W.reshape(n, n, 1, 1), bias)`
    with `W = self.matrix().to(x.dtype)` and `bias = self.bias.to(x.dtype)`, as
    `Orthogonal.forward` casts its matrix to the dtype of the input. The bias
    parameter is created after `theta`, so the `state_dict` holds `theta` and
    `bias`.

    Args:
        n (int): The number of channels.
        topology (TopologyName | Topology): A layout name, or a `Topology` on `n`
            wires. Default: `"butterfly"`.
        init (Init): `"identity"` or `"random"`. Default: `"identity"`.
        std (float): The standard deviation of the random angles. Default: `0.1`.
        bias (bool): Whether to add a trainable bias, initialised to zero.
            Default: `True`.
        mode (Mode): `"materialise"` or `"apply"`. Default: `"materialise"`.
    """

    def __init__(
        self,
        n: int,
        topology: TopologyName | Topology = "butterfly",
        *,
        init: Init = "identity",
        std: float = 0.1,
        bias: bool = True,
        mode: Mode = "materialise",
    ) -> None:
        super().__init__(n, topology, init=init, std=std, mode=mode)
        self.bias = nn.Parameter(torch.zeros(self.n)) if bias else None

    def reset_parameters(self) -> None:
        """Initialise the angles again by the rule of `init` (see
        `Orthogonal.reset_parameters`), and the bias to zero, as the constructor
        does."""
        super().reset_parameters()
        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the orthogonal 1x1 convolution.

        Args:
            x (torch.Tensor): A feature map of shape `(B, n, H, W)`, or `(n, H, W)`
                without a batch axis, as `F.conv2d` takes.

        Returns:
            torch.Tensor: The output, of the shape of `x`.
        """
        if self.mode == "apply" and self._frozen_W is None:
            # Without a batch axis, dim=1 would be H rather than the channels.
            unbatched = x.dim() == 3
            out = super().forward(x.unsqueeze(0) if unbatched else x, dim=1)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype).view(1, -1, 1, 1)
            return out.squeeze(0) if unbatched else out
        W = self.matrix().to(x.dtype)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.conv2d(x, W.reshape(self.n, self.n, 1, 1), bias)


def binary_parts(width: int) -> list[int]:
    """The powers of two that sum to `width`, largest first: 588 gives
    `[512, 64, 8, 4]`.

    Args:
        width (int): A positive integer.

    Returns:
        list[int]: The binary digits of `width`, as powers of two.

    Raises:
        ValueError: If `width` is not a positive integer.
    """
    width = _as_int(width, "width", 1)
    return [1 << b for b in range(width.bit_length() - 1, -1, -1) if width >> b & 1]


def mixing_permutation(width: int, random_state: int) -> torch.Tensor:
    """A fixed random permutation of `width` channels, from its own CPU generator.

    Numerical contract:
    `torch.randperm(width, generator=torch.Generator().manual_seed(random_state))`.
    The global generator is not touched.

    Args:
        width (int): The number of channels.
        random_state (int): The seed of the generator.

    Returns:
        torch.Tensor: A permutation of `0 .. width - 1`, as a `long` tensor.
    """
    g = torch.Generator().manual_seed(int(random_state))
    return torch.randperm(width, generator=g)


class PartitionedOrthogonal(_Rotation):
    r"""An orthogonal \(\mathrm{width} \times \mathrm{width}\) matrix for any
    width, from orthogonal blocks on the binary parts of the width, mixed by fixed
    permutations.

    The butterfly needs a power-of-two width. This module splits the width into its
    binary parts (\(588 = 512 + 64 + 8 + 4\)), rotates every part of size 2 or more
    with its own `Orthogonal`, and repeats this over `stages` stages, \(S\) of them,
    with a fixed channel permutation \(P_s\) in between:

    \[
    R = (P_S^\top B_S P_S) \cdots (P_2^\top B_2 P_2) \, B_1,
    \qquad B_s = \operatorname{blockdiag}(\text{parts of stage } s)
    \]

    Every factor is orthogonal, so \(R\) is too. All angles start at zero, so
    \(R = I\) at initialisation. A part of size 1 (an odd width) is left as it is.

    Numerical contracts:

    - Construction: stages outer, parts inner, one `Orthogonal(part, topology)`
      with `init="identity"` per part of size 2 or more, so no random numbers are
      drawn; then, for `s = 0 .. stages - 2`, the buffers `perm_s =
      mixing_permutation(width, perm_seed * 1000 + s)` and its inverse
      `inv_s = torch.argsort(perm_s)`, not saved in the `state_dict` and rebuilt
      the same way whenever the module moves (so `to_empty()` after a
      `meta`-device build works).
    - Stage matrix: `torch.block_diag(*part_matrices)`, with `torch.ones(1, 1)` in
      the dtype and on the device of the other parts appended for a part of size 1.
    - The matrix: `R = stage_0`; then for `s >= 1`, `R = B_s[inv][:, inv] @ R` with
      `inv = inv_{s-1}`, the stored `torch.argsort(perm_{s-1})`.
    - `forward(x, dim)` follows the contract of `Orthogonal.forward` in
      `mode="materialise"`.

    The `state_dict` keys are `blocks.<stage>.<part>.theta`.

    Args:
        width (int): The width, at least 2.
        stages (int): The number of block-diagonal stages, at least 1. Default: `2`.
        perm_seed (int): The seed of the mixing permutations. Default: `0`.
        topology (TopologyName): The layout of every part. Default: `"butterfly"`.
    """

    def __init__(
        self,
        width: int,
        stages: int = 2,
        perm_seed: int = 0,
        topology: TopologyName = "butterfly",
    ) -> None:
        super().__init__()
        width = _as_int(width, "width", 2)
        stages = _as_int(stages, "stages", 1)
        self.n = width
        self.stages = stages
        self.perm_seed = perm_seed
        self.parts = binary_parts(width)
        rotating = [p for p in self.parts if p >= 2]
        self.blocks = nn.ModuleList(
            nn.ModuleList(Orthogonal(p, topology) for p in rotating)
            for _ in range(stages)
        )
        self._register_fixed_buffers()

    def _fixed_buffers(self) -> dict[str, torch.Tensor]:
        # The inverse of each permutation is stored with it, so `matrix()` does not
        # sort on every call. A permutation has exactly one inverse, so storing it
        # gives the same indices as sorting where the matrix is built.
        with torch.device("cpu"):
            perms = [
                mixing_permutation(self.n, self.perm_seed * 1000 + s)
                for s in range(self.stages - 1)
            ]
            return {
                **{f"perm_{s}": perm for s, perm in enumerate(perms)},
                **{f"inv_{s}": torch.argsort(perm) for s, perm in enumerate(perms)},
            }

    @property
    def n_angles(self) -> int:
        """The number of angles over every stage and part."""
        return sum(b.n_angles for stage in self.blocks for b in stage)

    def _stage_matrix(self, s: int) -> torch.Tensor:
        mats = [b.matrix() for b in self.blocks[s]]
        if self.parts[-1] == 1:
            mats.append(torch.ones(1, 1, device=mats[0].device, dtype=mats[0].dtype))
        return torch.block_diag(*mats)

    def matrix(self) -> torch.Tensor:
        r"""The materialised \(\mathrm{width} \times \mathrm{width}\) orthogonal matrix
        (see the class contract).

        Returns:
            torch.Tensor: The matrix.
        """
        if self._frozen_W is not None:
            return self._frozen_W
        R = self._stage_matrix(0)
        for s in range(1, self.stages):
            inv = getattr(self, f"inv_{s - 1}")
            B = self._stage_matrix(s)
            R = B[inv][:, inv] @ R
        return R

    def extra_repr(self) -> str:
        return (
            f"width={self.n}, parts={self.parts}, stages={self.stages}, "
            f"angles={self.n_angles}, perm_seed={self.perm_seed}"
        )
