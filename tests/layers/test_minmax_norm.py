"""Tests for MinMaxNorm: feature-wise BatchNorm-style min/max normalization.

The layer mirrors PyTorch's BatchNorm in spirit: training mode computes
batch statistics and updates a running EMA; eval mode uses the running
stats.
"""

from __future__ import annotations

import math

import torch

from eigenkit import MinMaxNorm


class TestForward:
    def test_default_range_is_zero_to_pi(self):
        layer = MinMaxNorm(feature_dim=4)
        layer.train()
        # Construct input with known feature-wise min/max
        x = torch.tensor(
            [
                [0.0, -1.0, 2.0, 5.0],
                [10.0, 1.0, 4.0, 5.0],
            ],
            dtype=torch.float32,
        )
        out = layer(x)
        # First column: [0, 10] → [0, π]; second: [-1, 1] → [0, π]; etc.
        assert out.shape == x.shape
        assert torch.all(out >= 0.0)
        assert torch.all(out <= math.pi + 1e-5)

    def test_custom_range(self):
        layer = MinMaxNorm(feature_dim=2, min_val=-1.0, max_val=1.0)
        layer.train()
        x = torch.tensor([[0.0, 10.0], [10.0, 20.0]], dtype=torch.float32)
        out = layer(x)
        assert torch.all(out >= -1.0)
        assert torch.all(out <= 1.0)

    def test_shape_preserved(self):
        layer = MinMaxNorm(feature_dim=8)
        layer.train()
        x = torch.randn(32, 8)
        assert layer(x).shape == (32, 8)


class TestRunningStats:
    def test_running_stats_update_during_training(self):
        layer = MinMaxNorm(feature_dim=2, momentum=0.5)
        layer.train()
        x1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        layer(x1)
        # After first batch, running_min/max reflect this batch.
        assert torch.allclose(layer.running_min.flatten(), torch.tensor([0.0, 0.0]))
        assert torch.allclose(layer.running_max.flatten(), torch.tensor([1.0, 1.0]))

        # Second batch with different range
        x2 = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
        layer(x2)
        # Running stats update via EMA
        assert layer.running_max[0, 0].item() == 0.5 * 1.0 + 0.5 * 3.0  # = 2.0

    def test_eval_uses_running_stats(self):
        layer = MinMaxNorm(feature_dim=1, min_val=0.0, max_val=1.0)
        layer.train()
        # Train on data ranging [0, 10]
        layer(torch.tensor([[0.0], [10.0]]))
        layer.eval()
        # In eval mode, even an outlier gets clamped to the [0, 1] target
        # (uses running min=0, max=10 for scaling).
        out = layer(torch.tensor([[5.0]]))
        assert torch.allclose(out, torch.tensor([[0.5]]))

    def test_eval_clamps_out_of_range(self):
        layer = MinMaxNorm(feature_dim=1, min_val=0.0, max_val=1.0)
        layer.train()
        layer(torch.tensor([[0.0], [10.0]]))
        layer.eval()
        # 100 > running_max=10, but the layer hard-clamps the output.
        out = layer(torch.tensor([[100.0]]))
        assert out.item() == 1.0  # clamped to max_val


class TestStateDict:
    def test_running_stats_in_state_dict(self):
        layer = MinMaxNorm(feature_dim=4)
        sd = layer.state_dict()
        assert "running_min" in sd
        assert "running_max" in sd
        assert "any_batch_tracked" in sd
