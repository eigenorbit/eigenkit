"""What the layers do with inputs that are not quite what they expect.

Three silent wrong answers are guarded here, all of the same shape: a layer
accepting something it should question and returning a plausible-looking
number.

  * An integer input tensor must not truncate every expectation -- all in
    [-1, 1] -- to zero, which would leave the quantum layer emitting all-zeros
    and training on nothing.
  * `MinMaxNorm` must not normalise a single-row batch by its own range, which
    is empty and would map every feature to `min_val`. That case arises on the
    trailing batch whenever `n_samples % batch_size == 1`.
  * `Ensemble` must not split a wider input into fixed-width chunks and drop
    the trailing columns without a word.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from eigenkit.torch.layers import (
    Basic,
    Ensemble,
    MinMaxNorm,
    Reupload,
)


LAYERS = [
    (Basic, dict(n_qubits=2, depth=1), 2),
    (Reupload, dict(n_qubits=2, depth=1), 2),
    (Ensemble, dict(in_features=2, n_qubits=2, depth=1), 2),
]
IDS = [c.__name__ for c, _, _ in LAYERS]


class TestInputDtype:
    @pytest.mark.parametrize("cls,kwargs,n_features", LAYERS, ids=IDS)
    def test_an_integer_input_does_not_truncate_the_output(self, cls, kwargs, n_features):
        out = cls(**kwargs)(torch.ones(1, n_features, dtype=torch.int64))
        assert out.is_floating_point(), "expectations truncated to an integer dtype"
        assert (out != 0).any(), "every expectation collapsed to zero"

    @pytest.mark.parametrize("cls,kwargs,n_features", LAYERS, ids=IDS)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_a_float_input_keeps_its_precision(self, cls, kwargs, n_features, dtype):
        out = cls(**kwargs)(torch.ones(1, n_features, dtype=dtype))
        assert out.dtype == dtype


class TestMinMaxNormSmallBatches:
    def test_a_single_row_batch_does_not_collapse_after_warmup(self):
        layer = MinMaxNorm(feature_dim=3)
        layer.train()
        layer(torch.tensor([[1.0, 5.0, 9.0], [3.0, 7.0, 11.0]]))
        out = layer(torch.tensor([[2.0, 6.0, 10.0]]))
        assert out.unique().numel() > 1 or not torch.allclose(
            out, torch.full_like(out, layer.min_val)
        ), "a one-row batch was normalised to a constant"

    def test_a_single_row_first_batch_warns_instead_of_collapsing(self):
        layer = MinMaxNorm(feature_dim=3)
        layer.train()
        with pytest.warns(UserWarning, match="single-sample batch"):
            out = layer(torch.tensor([[1.0, 5.0, 9.0]]))
        torch.testing.assert_close(out, torch.tensor([[1.0, 5.0, 9.0]]))
        assert not bool(layer.any_batch_tracked), (
            "a batch with no range must not become the running statistics"
        )

    def test_a_later_real_batch_still_establishes_statistics(self):
        layer = MinMaxNorm(feature_dim=3)
        layer.train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layer(torch.tensor([[1.0, 5.0, 9.0]]))
        layer(torch.tensor([[1.0, 5.0, 9.0], [3.0, 7.0, 11.0]]))
        assert bool(layer.any_batch_tracked)

    def test_a_single_row_batch_leaves_the_running_statistics_alone(self):
        """The output is only half of it: an EMA update on a batch whose min
        equals its max would pull both bounds towards that row with each
        one-row batch and narrow the stored range."""
        layer = MinMaxNorm(feature_dim=2)
        layer.train()
        layer(torch.tensor([[0.0, 0.0], [10.0, 10.0]]))
        before_min = layer.running_min.clone()
        before_max = layer.running_max.clone()

        layer(torch.tensor([[5.0, 5.0]]))

        torch.testing.assert_close(layer.running_min, before_min)
        torch.testing.assert_close(layer.running_max, before_max)

    def test_repeated_single_rows_do_not_converge_the_range(self):
        layer = MinMaxNorm(feature_dim=1)
        layer.train()
        layer(torch.tensor([[0.0], [10.0]]))
        for _ in range(50):
            layer(torch.tensor([[5.0]]))
        assert float(layer.running_min) == 0.0
        assert float(layer.running_max) == 10.0

    def test_a_single_row_batch_still_normalises_against_history(self):
        layer = MinMaxNorm(feature_dim=1, min_val=0.0, max_val=1.0)
        layer.train()
        layer(torch.tensor([[0.0], [10.0]]))
        out = layer(torch.tensor([[5.0]]))
        torch.testing.assert_close(out, torch.tensor([[0.5]]))

    def test_a_float64_batch_does_not_change_the_buffer_dtype(self):
        layer = MinMaxNorm(feature_dim=2)
        layer.train()
        layer(torch.randn(4, 2))
        layer(torch.randn(4, 2, dtype=torch.float64))
        assert layer.running_min.dtype == torch.float32
        assert layer.running_max.dtype == torch.float32


class TestEnsembleFeatureCount:
    def test_extra_columns_are_rejected_not_dropped(self):
        layer = Ensemble(in_features=8, n_qubits=2, depth=1)
        with pytest.raises(ValueError, match="12 features"):
            layer(torch.randn(3, 12))

    def test_too_few_columns_names_the_problem(self):
        layer = Ensemble(in_features=8, n_qubits=2, depth=1)
        with pytest.raises(ValueError, match="in_features=8"):
            layer(torch.randn(3, 4))

    def test_the_right_count_still_works(self):
        layer = Ensemble(in_features=8, n_qubits=2, depth=1)
        assert layer(torch.randn(3, 8)).shape == (3, 8)

    def test_an_invalid_basis_matches_its_siblings(self):
        with pytest.raises(ValueError, match="basis must be"):
            Ensemble(in_features=2, n_qubits=2, depth=1, basis="W")
        with pytest.raises(ValueError, match="basis must be"):
            Basic(n_qubits=2, depth=1, basis="W")
