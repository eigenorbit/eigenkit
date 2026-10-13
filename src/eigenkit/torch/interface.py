"""The PyTorch interface, which makes the NumPy core differentiable in PyTorch.

This module is the PyTorch *adapter* of the interface model (see
`eigenkit.core.interfaces`). The core `Circuit` computes the forward values and
the jacobians with NumPy. This module wraps them into `torch.autograd.Function`
objects, so that the automatic differentiation of PyTorch works through a
quantum circuit. When it is imported, the module registers itself as the
`"torch"` interface.

Nothing here is part of the core, and importing this module needs PyTorch. This
is intended: PyTorch is an optional layer on top of the NumPy core.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.interfaces import register_interface
from ..core.provider import _batch_pair


def batch_call_torch(run_one, weights, inputs):
    """Loop over the first axis of a batch and keep the PyTorch autograd graph.

    It is the PyTorch version of `core.provider.batch_call`. It lives in the
    PyTorch package and not in the core, so that the core never imports PyTorch.
    The backends that differentiate with PyTorch themselves (PennyLane) use it.

    Args:
        run_one (Callable): A function that evaluates a single sample.
        weights (torch.Tensor): The parameter vector, 1-D or batched.
        inputs (torch.Tensor): The input features, 1-D or batched.

    Returns:
        stacked (torch.Tensor): The results for each sample, stacked. They are
            still differentiable.
    """
    B, w_is_batch, x_is_batch = _batch_pair(weights, inputs)
    if B is None:
        return run_one(weights, inputs)
    outputs = []
    for i in range(B):
        w_i = weights[i] if w_is_batch else weights
        x_i = inputs[i] if x_is_batch else inputs
        try:
            outputs.append(run_one(w_i, x_i))
        except Exception as exc:
            raise RuntimeError(f"failed on batch sample index {i}") from exc
    return torch.stack(outputs)


class _NoGradFn(torch.autograd.Function):
    """Forward-only torch op for backends (e.g. QPUs) that don't support
    gradients. Forward returns detached values wrapped as a torch tensor;
    `.backward()` raises with a message pointing at the
    simulator-train-then-deploy workflow.
    """

    @staticmethod
    def forward(ctx, compiled, weights: torch.Tensor, inputs: torch.Tensor):
        ctx.compiled = compiled
        weights_np = weights.detach().cpu().numpy().astype(np.float64)
        inputs_np = inputs.detach().cpu().numpy().astype(np.float64)
        output = compiled.run(weights_np, inputs_np)
        return torch.tensor(output, dtype=weights.dtype, device=weights.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise RuntimeError(
            "QPU/cloud backends with gradient_method='none' do not support "
            "automatic differentiation. Train on a simulator (provider="
            "'numpy' or 'pennylane'), then deploy frozen weights to "
            "the QPU for inference. To train on the hardware anyway, set "
            "enable_gradients=True in the device config; each backward pass "
            "then costs 2 * n_weights paid jobs."
        )


class _ParameterShiftFn(torch.autograd.Function):
    """Wraps a parameter-shift backend as a differentiable torch op.

    Forward returns `(n_obs,)` for 1-D inputs or `(B, n_obs)` for 2-D.
    Backward dispatches to the naive `parameter_shift_jacobian` when
    every gate angle is a direct `WeightRef` and only weight gradients
    are needed, or to the chain-rule-aware path when any gate wraps a
    weight in Scale/Multiply/Sum/Negate or when input gradients are
    requested. Batched inputs loop over samples in backward; weight
    gradient sums across samples, input gradient stacks per-sample.
    """

    @staticmethod
    def forward(
        ctx,
        circuit,
        weights: torch.Tensor,
        inputs: torch.Tensor,
    ):
        ctx.circuit = circuit
        ctx.save_for_backward(weights, inputs)
        weights_np = weights.detach().cpu().numpy().astype(np.float64)
        inputs_np = inputs.detach().cpu().numpy().astype(np.float64)
        output = circuit.compiled.run(weights_np, inputs_np)
        return torch.tensor(output, dtype=weights.dtype, device=weights.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weights, inputs = ctx.saved_tensors
        weights_np = weights.detach().cpu().numpy().astype(np.float64)
        inputs_np = inputs.detach().cpu().numpy().astype(np.float64)
        need_input_grad = ctx.needs_input_grad[2]
        # Frozen quantum weights: skip their shifts, which on hardware are
        # paid jobs whose result would be thrown away.
        need_weight_grad = ctx.needs_input_grad[1]
        # Build matmul tensors in grad_output's dtype so the product
        # works regardless of weights/inputs dtype (e.g. when inputs
        # come from a float32 RNN feeding a float64 quantum block).
        # Cast the result back to each parameter's dtype before returning.
        matmul_dtype = grad_output.dtype

        if inputs_np.ndim == 1:
            weight_jac, input_jac = ctx.circuit._gradient_jacobians(
                weights_np,
                inputs_np,
                need_input_grad=need_input_grad,
                need_weight_grad=need_weight_grad,
            )
            weights_grad: torch.Tensor | None = None
            if need_weight_grad and weight_jac is not None:
                weight_jac_t = torch.tensor(
                    weight_jac, dtype=matmul_dtype, device=weights.device
                )
                weights_grad = (grad_output @ weight_jac_t).to(weights.dtype)
            inputs_grad: torch.Tensor | None = None
            if need_input_grad and input_jac is not None:
                input_jac_t = torch.tensor(
                    input_jac, dtype=matmul_dtype, device=inputs.device
                )
                inputs_grad = (grad_output @ input_jac_t).to(inputs.dtype)
            return (None, weights_grad, inputs_grad)

        # Batched jacobians: weight_jac is (B, n_obs, n_weights) and input_jac
        # (B, n_obs, n_inputs). Backends with a batch-capable jacobian (the
        # numpy adjoint) compute these in one sweep; the rest loop internally.
        # The weight gradient sums over the batch; the input gradient stays
        # per-sample.
        weight_jac, input_jac = ctx.circuit._gradient_jacobians(
            weights_np,
            inputs_np,
            need_input_grad=need_input_grad,
            need_weight_grad=need_weight_grad,
        )
        grad_output_m = grad_output.to(matmul_dtype)
        weights_grad = None
        if need_weight_grad and weight_jac is not None:
            weight_jac_t = torch.tensor(
                weight_jac, dtype=matmul_dtype, device=weights.device
            )
            weights_grad = torch.einsum("bo,bow->w", grad_output_m, weight_jac_t).to(
                weights.dtype
            )
        inputs_grad = None
        if need_input_grad and input_jac is not None:
            input_jac_t = torch.tensor(
                input_jac, dtype=matmul_dtype, device=inputs.device
            )
            inputs_grad = torch.einsum("bo,boi->bi", grad_output_m, input_jac_t).to(
                inputs.dtype
            )
        return (None, weights_grad, inputs_grad)


class TorchInterface:
    """The `"torch"` differentiation interface.

    It chooses how to run a call of a core `Circuit`:

    - When the backend's `gradient_method` is `"native"` and it provides
      `run_torch`, PyTorch differentiates through the simulator itself.
    - When the backend computes no gradients (`gradient_method="none"`), the
      forward values are wrapped in an operation that has no backward pass.
    - Otherwise, the call goes through a `torch.autograd.Function` whose backward
      pass asks the core circuit for the jacobians. The core uses the backend's
      own jacobian when there is one (for example the adjoint method of the NumPy
      simulator), and the parameter-shift rule otherwise.
    """

    name = "torch"

    def run_circuit(self, circuit, weights: torch.Tensor, inputs):
        """Run the circuit with PyTorch, so that PyTorch can differentiate the result.

        This is the PyTorch side of the `Interface` protocol. It uses the backend's
        own differentiation when there is one, and otherwise it wraps the NumPy
        path in a `torch.autograd.Function`.

        Args:
            circuit (Circuit): The core circuit to run.
            weights (torch.Tensor): The trainable parameters of the layer.
            inputs (torch.Tensor): The input features, 1-D or batched.

        Returns:
            values (torch.Tensor): The expectation values, which are
                differentiable.
        """
        if inputs is None:
            inputs = self._default_inputs(circuit)
        if inputs.ndim not in (1, 2):
            raise ValueError(
                f"inputs must be 1D (single sample) or 2D (batch), got ndim={inputs.ndim}"
            )
        # The length guard belongs after the default is resolved, and it is the
        # core's, not this adapter's -- `Circuit.__call__` cannot run it before
        # dispatching here because this interface owns that default. Without it
        # the torch path would have both failure modes the numpy path guards
        # against: a too-wide input truncated in silence, a too-narrow one
        # surfacing as `IndexError` from inside the backend. That path is the one
        # `TorchCircuit`, `Basic` and `Reupload` all take.
        circuit._check_parameter_lengths(weights, inputs)
        method = circuit.provider.gradient_method
        if method == "native" and hasattr(circuit.compiled, "run_torch"):
            output = circuit.compiled.run_torch(weights, inputs)
            # Match the device-handling contract of the other paths
            # (_NoGradFn / _ParameterShiftFn return on weights.device).
            # A backend may emit tensors on its own device; rebase so a
            # CPU-built model can consume the result.
            if output.device != weights.device:
                output = output.to(weights.device)
        elif method == "none":
            output = _NoGradFn.apply(circuit.compiled, weights, inputs)
        else:
            output = _ParameterShiftFn.apply(circuit, weights, inputs)
        if not circuit._scalar:
            return output
        if inputs.ndim == 1:
            return output[0]
        return output[:, 0]

    @staticmethod
    def _default_inputs(circuit) -> torch.Tensor:
        if circuit.spec.n_inputs != 0:
            raise ValueError(
                f"circuit expects {circuit.spec.n_inputs} input value(s) but "
                f"none were provided"
            )
        return torch.empty(0, dtype=torch.float64)


register_interface("torch", TorchInterface())
