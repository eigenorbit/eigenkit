r"""Orthogonal fine-tuning: adapt a pretrained layer by rotating its weight.

An `OrthogonalAdapter` keeps the pretrained layer frozen and trains only the angles
of an orthogonal matrix \(R\), applied as \(W' = RW\) (or \(W' = WR^\top\) on the
inputs). \(R\) starts at the identity, so the adapted model starts at the
pretrained function, and \(RW\) has exactly the singular values of \(W\) at every
step: the Lipschitz constant of the layer never changes. `merge()` folds \(R\) into
the weight, so the adapter costs nothing at inference.

Example:
    ```python
    import torch.nn as nn

    from eigenkit.orthogonal.torch import (
        apply_orthogonal_adapters,
        merge_adapters,
        trainable_parameters,
    )

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10))
    apply_orthogonal_adapters(model, ["0", "2"], topology="pyramid")
    params = list(trainable_parameters(model))  # the angles only
    merge_adapters(model)  # plain nn.Linear layers again
    ```
"""

from __future__ import annotations

import copy
from typing import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..topology import Topology, TopologyName
from .functional import Side, rotate_weight
from .modules import Orthogonal, PartitionedOrthogonal


def _is_power_of_two(n: int) -> bool:
    return n >= 2 and not n & (n - 1)


class OrthogonalAdapter(nn.Module):
    r"""A frozen `nn.Linear` or `nn.Conv2d` whose weight is rotated by a trainable
    orthogonal matrix.

    `side="out"` gives \(W' = RW\), with \(R\) on the output channels;
    `side="in"` gives \(W' = WR^\top\), with \(R\) on the input channels. The bias of the
    base layer is kept as it is. The weight of a convolution is read as the matrix
    `(out, in·k·k)` (see `rotate_weight`).

    With the butterfly layout and a width that is not a power of two, the rotation
    is a `PartitionedOrthogonal` when `partition` is `None` (the default) or `True`.
    Any other layout takes any width it accepts, and `partition=True` uses the
    partitioned form for it too. The angles take the dtype and the device of the
    base weight.

    An adapter cannot be re-initialised (see `reset_parameters`), so a model with
    adapters is trained with a training loop of your own: the estimators
    `GenericClassifier` and `GenericRegressor` re-initialise every layer in
    `fit()` and refuse it.

    Args:
        base (nn.Linear | nn.Conv2d): The pretrained layer. Its parameters are
            frozen in place.
        topology (TopologyName | Topology): The layout of the rotation.
            Default: `"butterfly"`.
        side (Side): `"out"` or `"in"`. Default: `"out"`.
        partition (bool | None): Whether to use `PartitionedOrthogonal`; `None`
            chooses it for a butterfly whose width is not a power of two.
            Default: `None`.
        stages (int): The stages of a partitioned rotation. Default: `2`.
        perm_seed (int): The permutation seed of a partitioned rotation.
            Default: `0`.
    """

    def __init__(
        self,
        base: nn.Linear | nn.Conv2d,
        topology: TopologyName | Topology = "butterfly",
        side: Side = "out",
        partition: bool | None = None,
        stages: int = 2,
        perm_seed: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(base, (nn.Linear, nn.Conv2d)):
            raise TypeError(
                f"base must be nn.Linear or nn.Conv2d, got {type(base).__name__}"
            )
        if isinstance(base, nn.Conv2d) and base.groups != 1:
            raise ValueError("grouped convolutions are not supported")
        if side not in ("out", "in"):
            raise ValueError(f"side must be 'out' or 'in', got {side!r}")
        width = base.weight.shape[0] if side == "out" else base.weight.shape[1]
        if partition is None:
            partition = topology == "butterfly" and not _is_power_of_two(width)
        if partition and isinstance(topology, Topology):
            raise ValueError("a partitioned rotation takes a layout name, not a Topology")
        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base
        self.side = side
        if partition:
            self.rotation = PartitionedOrthogonal(width, stages, perm_seed, topology)
        else:
            self.rotation = Orthogonal(width, topology)
        self.rotation.to(device=base.weight.device, dtype=base.weight.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """The adapted weight `rotate_weight(R, base.weight, side)`.

        A property, like the `weight` of the layer it replaces, so code that reads
        `layer.weight` directly (`nn.MultiheadAttention` reads `out_proj.weight`)
        gets the adapted weight and trains the angles. It is rebuilt from the
        angles at every read.

        Returns:
            torch.Tensor: A tensor of the shape of `base.weight`.
        """
        return rotate_weight(self.rotation.matrix(), self.base.weight, self.side)

    @property
    def bias(self) -> torch.Tensor | None:
        """The bias of the base layer, `base.bias`, which is not rotated.

        Returns:
            torch.Tensor | None: The bias, or `None` if the base layer has none.
        """
        return self.base.bias

    def reset_parameters(self) -> None:
        """Refuse to re-initialise: an adapter starts from its pretrained weight.

        `model.apply(lambda m: m.reset_parameters())`, which the estimators
        `GenericClassifier` and `GenericRegressor` run at the start of `fit()`,
        resets the children of a module before the module itself, so by the time
        this method runs `base` already holds a new random weight. Training would
        then rotate that random weight instead of the pretrained one, without any
        error. Raising here stops it. Train an adapted model with a training loop
        of your own, as in the fine-tuning example.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            "an OrthogonalAdapter cannot be re-initialised: it adapts the "
            "pretrained weight of its base layer, which reset_parameters() would "
            "replace with a random one. The estimators (GenericClassifier, "
            "GenericRegressor) re-initialise every layer in fit(), so train an "
            "adapted model with your own training loop instead"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the base layer with the adapted weight.

        Args:
            x (torch.Tensor): The input of the base layer.

        Returns:
            torch.Tensor: The output of the base layer with weight `self.weight`.
        """
        if isinstance(self.base, nn.Linear):
            return F.linear(x, self.weight, self.base.bias)
        return self.base._conv_forward(x, self.weight, self.base.bias)

    def merge(self) -> nn.Linear | nn.Conv2d:
        """A copy of the base layer with the rotation folded into its weight.

        Its parameters are frozen, like those of the base layer.

        Returns:
            nn.Linear | nn.Conv2d: A plain layer that computes what the adapter does.
        """
        merged = copy.deepcopy(self.base)
        with torch.no_grad():
            merged.weight.copy_(self.weight)
        return merged


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


def apply_orthogonal_adapters(
    model: nn.Module,
    target_modules: Iterable[str] | Callable[[str, nn.Module], bool],
    topology: TopologyName | Topology = "butterfly",
    side: Side = "out",
    freeze_rest: bool = True,
    **adapter_kwargs,
) -> nn.Module:
    """Replace layers of a model by `OrthogonalAdapter`s, in place.

    Args:
        model (nn.Module): The model to adapt.
        target_modules (str | Iterable[str] | Callable[[str, nn.Module], bool]):
            The qualified name of the layer to adapt, or the names of several
            layers (as in `named_modules()`), or a predicate on the name and the
            module. Only `nn.Linear` and `nn.Conv2d` layers can be adapted.
        topology (TopologyName | Topology): The layout of every rotation.
            Default: `"butterfly"`.
        side (Side): `"out"` or `"in"`. Default: `"out"`.
        freeze_rest (bool): Whether to freeze every parameter that is not an angle
            of an adapter. Default: `True`.
        **adapter_kwargs (Any): Passed to `OrthogonalAdapter`, such as `partition`.

    Returns:
        nn.Module: The same model.

    Raises:
        KeyError: If a name is not a module of the model.
        TypeError: If a named module is not `nn.Linear` or `nn.Conv2d`.
        ValueError: If nothing matches, or a named module is inside an adapter.
    """
    modules = dict(model.named_modules())
    if callable(target_modules):
        names = [
            name
            for name, module in modules.items()
            if name
            and isinstance(module, (nn.Linear, nn.Conv2d))
            and not any(
                isinstance(modules.get(p), OrthogonalAdapter) for p in _ancestors(name)
            )
            and target_modules(name, module)
        ]
    else:
        names = (
            [target_modules] if isinstance(target_modules, str) else list(target_modules)
        )
        for name in names:
            if name not in modules or not name:
                raise KeyError(f"model has no submodule {name!r}")
            if any(
                isinstance(modules.get(p), OrthogonalAdapter) for p in _ancestors(name)
            ):
                raise ValueError(
                    f"{name!r} is inside an OrthogonalAdapter; adapt the adapter's "
                    "layer once, not the layer inside it"
                )
            if not isinstance(modules[name], (nn.Linear, nn.Conv2d)):
                raise TypeError(
                    f"{name!r} is a {type(modules[name]).__name__}; only nn.Linear "
                    "and nn.Conv2d can be adapted"
                )
    if not names:
        raise ValueError("no module matched target_modules")
    for name in names:
        parent, child = _parent(model, name)
        adapter = OrthogonalAdapter(modules[name], topology, side, **adapter_kwargs)
        setattr(parent, child, adapter)
    if freeze_rest:
        keep = {
            id(p)
            for m in model.modules()
            if isinstance(m, OrthogonalAdapter)
            for p in m.rotation.parameters()
        }
        for p in model.parameters():
            if id(p) not in keep:
                p.requires_grad_(False)
    return model


def _ancestors(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:k]) for k in range(1, len(parts))]


def merge_adapters(model: nn.Module) -> nn.Module:
    """Replace every `OrthogonalAdapter` of a model by its merged layer, in place.

    Args:
        model (nn.Module): A model with adapters. If the model is itself an
            adapter, its merged layer is returned.

    Returns:
        nn.Module: The model, with plain layers where the adapters were.
    """
    if isinstance(model, OrthogonalAdapter):
        return model.merge()
    adapters = [
        name for name, m in model.named_modules() if isinstance(m, OrthogonalAdapter)
    ]
    for name in adapters:
        parent, child = _parent(model, name)
        setattr(parent, child, getattr(parent, child).merge())
    return model


def trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """The parameters that require a gradient, for the optimiser.

    Args:
        model (nn.Module): The model.

    Yields:
        nn.Parameter: Each parameter with `requires_grad=True`.
    """
    return (p for p in model.parameters() if p.requires_grad)


def fold_bn(
    conv: nn.Conv1d | nn.Conv2d | nn.Conv3d,
    bn: nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold an eval-mode BatchNorm into the convolution before it.

    Numerical contract, all in float64 on the CPU:
    `scale = bn.weight / sqrt(bn.running_var + bn.eps)`,
    `w = conv.weight * scale.view(-1, 1, 1, 1)` (one `1` per axis of the weight
    after the first, so `view(-1, 1, 1)` for a `Conv1d`) and
    `b = bn.bias - scale * bn.running_mean`. A convolution with a bias of its own
    gives `b = bn.bias - scale * (bn.running_mean - conv.bias)` instead. A
    BatchNorm with `affine=False` counts as `weight = 1` and `bias = 0`. Cast the
    results to the dtype you need.

    Args:
        conv (nn.Conv1d | nn.Conv2d | nn.Conv3d): The convolution.
        bn (nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d): The BatchNorm
            that follows it, in eval mode, with one feature per output channel
            of `conv`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The folded weight, of the shape of
        `conv.weight`, and bias, float64 on the CPU.

    Raises:
        TypeError: If `conv` is not `nn.Conv1d`, `nn.Conv2d` or `nn.Conv3d` (a
            transposed convolution keeps its output channels on another axis).
        ValueError: If the BatchNorm keeps no running statistics, or its number
            of features is not the number of output channels of `conv`.
    """
    if not isinstance(conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        raise TypeError(
            f"conv must be nn.Conv1d, nn.Conv2d or nn.Conv3d, got {type(conv).__name__}"
        )
    if bn.num_features != conv.out_channels:
        raise ValueError(
            f"bn has {bn.num_features} features, but conv has {conv.out_channels} "
            "output channels"
        )
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError(
            "fold_bn needs running statistics; this BatchNorm was built with "
            "track_running_stats=False"
        )
    n = bn.num_features
    gamma = torch.ones(n, dtype=torch.float64) if bn.weight is None else bn.weight
    beta = torch.zeros(n, dtype=torch.float64) if bn.bias is None else bn.bias
    scale = gamma.detach().double().cpu() / torch.sqrt(
        bn.running_var.detach().double().cpu() + bn.eps
    )
    w = conv.weight.detach().double().cpu()
    w = w * scale.view(-1, *[1] * (w.dim() - 1))
    mean = bn.running_mean.detach().double().cpu()
    if conv.bias is not None:
        mean = mean - conv.bias.detach().double().cpu()
    b = beta.detach().double().cpu() - scale * mean
    return w, b
