import warnings

import torch
import torch.nn as nn


class MinMaxNorm(nn.Module):
    """Min-max normalisation of each feature, with running statistics.

    It rescales each feature to `[min_val, max_val]`. In training mode, it uses
    the minimum and maximum of each batch, and it updates an exponential moving
    average of these statistics. In evaluation mode, it uses the running
    statistics. A training batch of a single row has no range, so it uses the
    running statistics too, and if no batch with a range has been seen yet, it
    returns the row unchanged with a warning. Training and evaluation modes work
    as in the `BatchNorm` and `LayerNorm` layers of PyTorch, but the output is
    rescaled to a fixed interval instead of standardised.

    It is useful to prepare the inputs of quantum encoding rotations, which
    expect angles in a bounded range such as `[0, π]`.

    Args:
        feature_dim: The number of features in the last dimension of the input.
            The inputs have shape `(batch_size, feature_dim)`.
        min_val: The lower bound of the target interval. Default: `0.0`.
        max_val: The upper bound of the target interval. Default: `π`.
        momentum: The momentum factor of the exponential moving average of the
            running statistics. Default: `0.1`.

    Example:
        ```python
        import torch
        from eigenkit import Ensemble, MinMaxNorm

        norm = MinMaxNorm(feature_dim=16, min_val=0.0, max_val=torch.pi)
        layer = Ensemble(in_features=16, n_qubits=4, depth=5)
        out = layer(norm(torch.randn(32, 16)))
        ```
    """

    def __init__(
        self,
        feature_dim: int,
        min_val: float = 0.0,
        max_val: float = torch.pi,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.momentum = momentum
        self.feature_dim = feature_dim

        self.register_buffer("running_min", torch.zeros(1, feature_dim))
        self.register_buffer("running_max", torch.ones(1, feature_dim))
        self.register_buffer(
            "any_batch_tracked",
            torch.tensor(False, dtype=torch.bool),
        )

    def forward(self, x):
        """Run the layer.

        Args:
            x (torch.Tensor): The input batch, of shape `(batch, feature_dim)`.

        Returns:
            out (torch.Tensor): The output of the layer for that batch.
        """
        if self.training:
            batch_min = x.min(dim=0, keepdim=True)[0]
            batch_max = x.max(dim=0, keepdim=True)[0]

            if x.shape[0] == 1 and not self.any_batch_tracked:
                # One row and no history: there is no range to normalise
                # against, and any answer here is arbitrary. Say so rather than
                # mapping the row to `min_val` in silence, and leave the
                # statistics untracked so a later real batch can establish
                # them.
                warnings.warn(
                    "MinMaxNorm saw a single-sample batch before any batch "
                    "with a range, so it has no statistics to normalise "
                    "against and is passing this batch through unchanged. Use "
                    "a batch size above 1, or call the layer in eval mode once "
                    "statistics exist.",
                    UserWarning,
                    stacklevel=2,
                )
                return x
            # Only a batch that carries a range updates the statistics. One
            # row has `batch_min == batch_max` on every feature, so folding it
            # into the EMA drags both bounds towards that row's value and
            # narrows the stored range -- over epochs the range converges on
            # that single row, distorting eval-mode normalisation silently and
            # data-dependently. The trailing batch hits this whenever
            # `n_samples % batch_size == 1`, which is the same case the
            # no-history branch above declines to learn from.
            if x.shape[0] > 1:
                if not self.any_batch_tracked:
                    self.running_min.copy_(batch_min.detach())
                    self.running_max.copy_(batch_max.detach())
                    self.any_batch_tracked.fill_(True)
                else:
                    with torch.no_grad():
                        # `copy_` into the buffers rather than rebinding them:
                        # assignment replaces the registered tensor, so a
                        # single float64 batch would change `running_min`'s
                        # dtype for good.
                        self.running_min.copy_(
                            (1 - self.momentum) * self.running_min
                            + self.momentum * batch_min
                        )
                        self.running_max.copy_(
                            (1 - self.momentum) * self.running_max
                            + self.momentum * batch_max
                        )

            # Standard BN practice: use current-batch statistics while
            # training -- except when there are not enough of them. A batch of
            # one has batch_min == batch_max on every feature, the clamped
            # denominator takes over, and every input maps to `min_val`: the
            # row's information is destroyed rather than normalised. That fires
            # on the trailing batch whenever `n_samples % batch_size == 1`, so
            # it is data-dependent and silent. Fall back to the running
            # statistics, which is what eval mode uses.
            if x.shape[0] > 1:
                current_min = batch_min
                current_max = batch_max
            else:
                current_min = self.running_min
                current_max = self.running_max
        else:
            current_min = self.running_min
            current_max = self.running_max

        denom = (current_max - current_min).clamp(min=1e-6)
        x_norm = (x - current_min) / denom
        x_scaled = x_norm * (self.max_val - self.min_val) + self.min_val
        # Hard clamp guards against phase wrapping on unseen outliers.
        return x_scaled.clamp(self.min_val, self.max_val)
