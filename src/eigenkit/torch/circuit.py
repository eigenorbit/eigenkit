"""The PyTorch `nn.Module` circuit, which is the trainable quantum layer for users.

`TorchCircuit` stores its trainable weights as an `nn.Parameter` and passes the
execution to the core `Circuit` with the PyTorch interface. It can therefore be
used inside any PyTorch model (`nn.Sequential` or your own `nn.Module`), and it
trains with a standard PyTorch optimiser. The gradients are computed with the
method that the active backend declares: the backend's own automatic
differentiation, the parameter-shift rule or the adjoint method.

The top-level `Circuit(..., interface="torch")` function returns this class. It
accepts a list of blocks, a `CircuitSpec` that is already assembled, or a
`CircuitBuilder`.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import nn

from .. import core
from ..core.blocks import Block
from ..core.builder import CircuitBuilder
from ..core.circuit import ObservableLike, ProviderSpec
from ..core.ir import CircuitSpec
from ..core.provider import Provider


_BlocksOrSpec = Union[Sequence[Block], CircuitSpec, CircuitBuilder]


class TorchCircuit(nn.Module):
    """A quantum layer that stores trainable weights and runs on any registered backend.

    It runs through the PyTorch interface, so that PyTorch can compute the
    gradients of its output.

    Args:
        blocks: The circuit. It can be a list of blocks to assemble, a `CircuitSpec`
            that is already assembled, or a `CircuitBuilder`. A builder is built
            when the layer is created, so gates added to it afterwards are not part
            of this layer.
        observables: What to measure. `None` measures `Z` on every qubit and returns
            one value per qubit. A single `Observable` or `PauliExpr` returns a
            single value, and a list of them returns one value for each. Default:
            `None`.
        provider: The backend that runs the circuit. It can be a registered name, a
            configuration instance, or a provider object. Any backend works. A
            backend that computes gradients itself avoids the fallback to the
            parameter-shift rule. Default: `"numpy"`.
        init: The initial weights, of shape `(n_weights,)`. They are random when
            `init` is not given. Default: `None`.
    """

    def __init__(
        self,
        blocks: _BlocksOrSpec,
        observables: ObservableLike = None,
        provider: ProviderSpec = "numpy",
        init: torch.Tensor | None = None,
    ):
        super().__init__()
        if isinstance(blocks, CircuitBuilder):
            spec = blocks.build()
        elif isinstance(blocks, CircuitSpec):
            spec = blocks
        else:
            spec = core.assemble(blocks)
        self._observables: ObservableLike = observables
        self._inner = core.Circuit(
            spec,
            observables=observables,
            provider=provider,
            interface="torch",
        )
        if init is None:
            init = torch.randn(spec.n_weights, dtype=torch.float64)
        self.weights = nn.Parameter(init)

    def reset_parameters(self) -> None:
        """Draw new circuit weights from the current PyTorch random number generator.

        Every `nn.Module` that PyTorch provides defines this method. The estimators
        call `model.apply(reset_parameters)` after setting the seed, so that
        `random_state` controls the whole initialisation. Without this method,
        the estimators would skip the layer: two fits with different values of
        `random_state` would give different classical weights but exactly the
        same quantum weights, and a user who restarts with new seeds to escape a
        barren plateau would change only the part of the model that is not the
        circuit.
        """
        with torch.no_grad():
            self.weights.copy_(
                torch.randn(
                    self.weights.shape,
                    dtype=self.weights.dtype,
                    device=self.weights.device,
                )
            )

    @property
    def spec(self) -> CircuitSpec:
        """The circuit that this layer runs.

        It is the [`CircuitSpec`][eigenkit.core.ir.CircuitSpec] assembled from
        `blocks`, or the one that was passed in already assembled."""
        return self._inner.spec

    @property
    def provider(self) -> Provider:
        """The backend that this circuit was compiled for.

        It is the same as [`Circuit.provider`][eigenkit.core.circuit.Circuit],
        which turns the `provider=` argument (a registered name, a configuration
        or a backend object) into one `Provider`. The registered name is
        `provider.name`. To run the same circuit on another backend, use
        `with_provider()`, which returns a new instance and leaves this one
        unchanged."""
        return self._inner.provider

    @property
    def n_qubits(self) -> int:
        """The number of qubits of the circuit in `spec`."""
        return self._inner.spec.n_qubits

    @property
    def out_features(self) -> int:
        """The number of observables, or `1` for a single observable."""
        return 1 if self._inner._scalar else len(self._inner.observables)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Run the layer.

        Args:
            x (torch.Tensor): The input batch, of shape `(batch, in_features)`.

        Returns:
            The output of the layer for that batch.
        """
        return self._inner(self.weights, x)

    def submit(self, x: torch.Tensor | None = None):
        """Submit a job without waiting for it, and return a `JobHandle`.

        Only the backends that provide submission support it (Braket, Qiskit
        Runtime and IQM).

        Args:
            x (torch.Tensor | None): One 1-D input sample. `None` works only for a
                circuit that takes no inputs, and it raises a `ValueError` for any
                other circuit. Default: `None`.

        Returns:
            handle (JobHandle): A handle for the submitted job.
        """
        return self._inner.submit(self.weights, x)

    def dry_run_info(self) -> dict:
        """Describe the prepared circuit in a dictionary, without submitting it.

        It is useful to check the size of the circuit, the gate counts after
        transpilation, the number of shots and the number of circuits that each call
        runs, before you submit to a cloud simulator or a quantum processing unit
        (QPU). It does not estimate a price. It is available only on the backends that provide it.
        Currently, these are Braket (on its local simulator as well as on cloud
        devices), Qiskit Runtime and IQM. Other backends raise a
        `NotImplementedError`.
        """
        return self._inner.dry_run_info()

    def zero_state_probability(self, x: torch.Tensor | None = None) -> float:
        """Return the probability of measuring the all-zeros state.

        It uses the current weights and the given input. It is useful for fidelity
        kernels and swap tests.

        It takes one sample, and its result is not differentiable. The core reads
        the amplitude through the backend's access to the state and returns a plain
        `float`, so PyTorch does not record it for automatic differentiation. A
        batch raises an error, because a single number cannot hold a batch axis.
        For a batch, call this method once for each row, or use
        `Circuit.zero_state_probabilities`.

        Args:
            x (torch.Tensor | None): The input features of one sample, as a 1-D
                tensor. `None` works only for a circuit that takes no inputs, and
                it raises a `ValueError` for any other circuit. Default: `None`.

        Returns:
            The all-zeros probability, a float in `[0, 1]`.

        Raises:
            ValueError: If `x` is 2-D.
        """
        return self._inner.zero_state_probability(self.weights, x)

    def sample(self, x: torch.Tensor | None = None) -> dict[str, int]:
        """Return the raw measurement histogram for the current weights and input.

        The histogram maps each bitstring to its number of shots. In each
        bitstring, qubit `0` is the most significant bit.

        It is the second step of the workflow in which you train on a simulator
        and then run on hardware: you train the layer, and then you read what the
        device measured. It takes one sample, and its result is not
        differentiable, because the counts are integers that PyTorch does not
        record for automatic differentiation.

        Args:
            x (torch.Tensor | None): The input features of one sample, as a 1-D
                tensor. `None` works only for a circuit that takes no inputs, and
                it raises a `ValueError` for any other circuit. Default: `None`.

        Returns:
            counts (dict[str, int]): The number of shots for each measured
                bitstring.

        Raises:
            NotImplementedError: On backends that do not sample.
            ValueError: If `x` is 2-D.
        """
        return self._inner.sample(self.weights, x)

    def draw(self, *, decimals: int = 3) -> str:
        """Return a text drawing of the circuit, over several lines.

        The drawing does not depend on the provider: it is built from the
        `CircuitSpec`, so it is the same whatever the active backend is. Print the
        returned string directly.

        Example:
            ```python
            from eigenkit import Circuit, Encoding, Variational

            blocks = [Encoding(in_features=2), Variational(depth=1)]
            circuit = Circuit(blocks=blocks, interface="torch")
            print(circuit.draw())
            # q_0: ──RY(x_0)──RY(w_0)──●──
            #                          │
            # q_1: ──RY(x_1)──RY(w_1)──X──
            ```

        Args:
            decimals: The number of significant digits shown for a numeric gate
                angle. Default: `3`.

        Returns:
            The drawing, as a single string that contains newlines.
        """
        return self._inner.draw(decimals=decimals)

    def draw_mpl(self, **kwargs):
        """Draw the circuit as a matplotlib figure, and return `(fig, ax)`.

        The picture does not depend on the provider, because it is built from the
        `CircuitSpec` and the observables. matplotlib is an optional dependency.
        Install it with `eigenkit[viz]`. The keyword arguments are passed to
        [`draw_mpl()`][eigenkit.core.draw_mpl.draw_mpl].
        """
        return self._inner.draw_mpl(**kwargs)

    def with_provider(
        self,
        provider: ProviderSpec,
        *,
        copy_weights: bool = True,
        init: torch.Tensor | None = None,
    ) -> "TorchCircuit":
        """Return a new `TorchCircuit` for another `provider`.

        The new circuit has the same `spec` and the same `observables` argument as
        this instance, and it is compiled for the given `provider`.

        `provider` accepts the same forms as in `__init__()`: a registered name
        (a `str`), a configuration object with a `make_backend()` method (for
        example `PennyLaneConfig`), or a backend instance.

        The gradients follow the `gradient_method` of the new backend. The forward
        values should match across simulators up to numerical tolerance, but not
        exactly.

        Args:
            provider: The backend for the new circuit.
            copy_weights: If True, the new circuit starts from a copy of
                `self.weights`. It is ignored when `init` is given. Default: `True`.
            init: An optional weight tensor of shape `(spec.n_weights,)`. When it is
                given, it takes precedence over `copy_weights`. Default: `None`.

        Returns:
            The new `TorchCircuit`.
        """
        if init is not None:
            new_init = init
        elif copy_weights:
            new_init = self.weights.detach().clone()
        else:
            new_init = None
        return TorchCircuit(
            blocks=self.spec,
            observables=self._observables,
            provider=provider,
            init=new_init,
        )
