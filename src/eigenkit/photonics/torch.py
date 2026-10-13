"""The PyTorch interface for photonic circuits (experimental).

`PhotonicCircuit` is a PyTorch `nn.Module` that stores the weights of the circuit
as an `nn.Parameter`. With it, `loss.backward()` computes the exact photonic
parameter-shift gradient. It uses the same `torch.autograd.Function` bridge as the
qubit backends that use the parameter-shift rule (see
`eigenkit/torch/interface.py`). This is what makes photonic circuits *trainable*
for quantum machine learning with the usual PyTorch tools, such as the optimisers
and the combination with classical layers.

PyTorch is imported at the top of this module. The module is imported only when
the PyTorch interface is requested, so the rest of the photonics package (and
`import eigenkit`) imports neither PyTorch nor Perceval.

Scope: one observable that is diagonal in the Fock basis, so the output is a
single number, and one sample per forward pass. Outputs with several values or
several observables, batches, and gradients computed by PyTorch's own automatic
differentiation are not supported yet.
"""

from __future__ import annotations

import torch

from .backends.perceval_backend import PercevalBackend
from .gradients import _check_observable
from .ir import PhotonicCircuitSpec, PhotonicObservable


class _PhotonicExpectation(torch.autograd.Function):
    """Bridges the numpy photonic executor into torch autograd: forward runs the
    Perceval simulation; backward returns the photonic parameter-shift jacobian."""

    @staticmethod
    def forward(ctx, weights, inputs, compiled, observable):
        w = weights.detach().cpu().numpy().astype(float)
        x = inputs.detach().cpu().numpy().astype(float)
        value = compiled.expectation(w, x, observable)
        ctx.compiled = compiled
        ctx.observable = observable
        ctx.save_for_backward(weights, inputs)
        return weights.new_tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        weights, inputs = ctx.saved_tensors
        w = weights.detach().cpu().numpy().astype(float)
        x = inputs.detach().cpu().numpy().astype(float)
        need_weight_grad, need_input_grad = ctx.needs_input_grad[:2]
        # Each skipped side saves its whole share of the 2N-term shift evaluations
        # (e.g. data-encoding phases when the inputs don't require grad).
        grad_w, grad_x = ctx.compiled.jacobian(
            w,
            x,
            ctx.observable,
            method="parameter_shift",
            need_weight_grad=need_weight_grad,
            need_input_grad=need_input_grad,
        )
        return (
            grad_output * weights.new_tensor(grad_w) if need_weight_grad else None,
            grad_output * inputs.new_tensor(grad_x) if need_input_grad else None,
            None,
            None,
        )


class PhotonicCircuit(torch.nn.Module):
    """A trainable linear-optical circuit, as a PyTorch `nn.Module`.

    It stores the circuit's `weights` as an `nn.Parameter`. `forward(inputs)`
    returns the expectation value of `observable`, connected to PyTorch's automatic
    differentiation.

    Args:
        spec (PhotonicCircuitSpec): The circuit to train.
        observable (PhotonicObservable): The readout, a single observable that is
            diagonal in the Fock basis.
        backend_name (str): The Perceval backend, as for `PercevalBackend`.
            Default: `"SLOS"`.
        init_weights (Sequence[float] | torch.Tensor | None): The initial weights,
            of length `spec.n_weights`. The default is zeros.
        dtype (torch.dtype): The data type of the weights. The default is
            `torch.float64`, to match the precision of the executor.

    Raises:
        ValueError: If `init_weights` does not have the shape `(spec.n_weights,)`,
            if `observable` can never match the output of the circuit (a mode
            outside the circuit, or a target with the wrong length or number of
            photons), or if `backend_name` is unknown or only samples.
    """

    def __init__(
        self,
        spec: PhotonicCircuitSpec,
        observable: PhotonicObservable,
        *,
        backend_name: str = "SLOS",
        init_weights=None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        _check_observable(spec, observable)  # fail before training, not as a 0 gradient
        self.spec = spec
        self.observable = observable
        self._compiled = PercevalBackend(backend_name).compile(spec)
        if init_weights is None:
            init = torch.zeros(spec.n_weights, dtype=dtype)
        else:
            init = torch.as_tensor(init_weights, dtype=dtype).clone()
            if init.shape != (spec.n_weights,):
                raise ValueError(
                    f"init_weights has shape {tuple(init.shape)} but the spec needs "
                    f"({spec.n_weights},)"
                )
        self.weights = torch.nn.Parameter(init)

    def forward(self, inputs=None):
        """Return the expectation value of `observable` for one sample.

        The value is connected to PyTorch's automatic differentiation.

        Args:
            inputs (Sequence[float] | torch.Tensor | None): The 1-D input vector,
                of length `spec.n_inputs`. The default is zeros.
        """
        if inputs is None:
            inputs = torch.zeros(self.spec.n_inputs, dtype=self.weights.dtype)
        else:
            inputs = torch.as_tensor(inputs, dtype=self.weights.dtype)
        return _PhotonicExpectation.apply(
            self.weights, inputs, self._compiled, self.observable
        )
