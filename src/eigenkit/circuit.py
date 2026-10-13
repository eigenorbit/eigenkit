"""The top-level `Circuit` function, which builds a circuit for the chosen interface.

`Circuit(...)` returns a different type of circuit depending on `interface`, as
in PennyLane:

- With `interface="numpy"`, the default, it returns the core NumPy `Circuit`.
  This circuit does not store weights: you pass the `weights` at each call. The
  variational optimisation tools and the `chemistry` and `optimization` packages
  use this form. With the default `provider="numpy"`, the circuit runs on the
  built-in reference simulator, so `pip install eigenkit` with no extra is enough
  to build and run circuits.
- With `interface="torch"`, it returns a
  [`TorchCircuit`][eigenkit.torch.circuit.TorchCircuit]. This is a PyTorch
  `nn.Module` that stores trainable weights and can be used inside PyTorch
  models. It comes from `eigenkit.torch`, which needs the `torch` extra. PyTorch
  is imported only when this interface is requested, so an environment without
  PyTorch never imports it.

Both `interface` and `provider` default to NumPy. To use PyTorch, pass
`interface="torch"`. To use a faster simulator or hardware, pass a provider such
as `provider="pennylane"` or `provider="qiskit"`.

Example:
    ```python
    import numpy as np
    from eigenkit import Circuit, Encoding, Variational, all_z

    blocks = [Encoding(in_features=4), Variational(depth=2, rotation="RY")]
    circuit = Circuit(blocks=blocks, observables=all_z(4))
    weights = np.zeros(circuit.spec.n_weights)
    values = circuit(weights, np.array([0.1, 0.2, 0.3, 0.4]))  # one value per qubit

    # A differentiable PyTorch module on PennyLane (torch and pennylane extras):
    layer = Circuit(
        blocks=blocks, observables=all_z(4), interface="torch", provider="pennylane"
    )
    ```
"""

from __future__ import annotations

from typing import Literal, Sequence, Union

from . import core
from .core.builder import CircuitBuilder
from .core.circuit import Circuit as _NumpyCircuit
from .core.circuit import ObservableLike, ProviderSpec
from .core.ir import CircuitSpec


_BlocksOrSpec = Union[Sequence["core.blocks.Block"], CircuitSpec, CircuitBuilder]  # noqa: F821
_Interface = Literal["torch", "numpy"]


def Circuit(
    blocks: _BlocksOrSpec,
    observables: ObservableLike = None,
    provider: ProviderSpec = "numpy",
    interface: _Interface = "numpy",
    init=None,
):
    """Build a circuit and compile it for a backend.

    This is the entry point for the rest of the library: you give it blocks, and it
    returns a circuit that you can call. The type of the circuit depends on
    `interface`. The NumPy interface returns a core `Circuit`, and the PyTorch
    interface returns a `TorchCircuit`.

    Args:
        blocks: The circuit. It can be a list of blocks to assemble, a `CircuitSpec`
            that is already assembled, or a `CircuitBuilder`. A builder is built when
            `Circuit` is called, so gates added to it afterwards are not part of this
            circuit. A builder can also take the place of a block in the list.
        observables: What to measure. `None` measures `Z` on every qubit and returns
            one value per qubit. A single `Observable` or `PauliExpr` returns a
            single value, and a list of them returns one value for each. Default:
            `None`.
        provider: The backend that runs the circuit. It can be a registered name
            (`"numpy"`, `"pennylane"`, `"qiskit"` and others), a configuration
            instance such as `PennyLaneConfig(...)`, or a backend object. Default:
            `"numpy"`.
        interface: `"numpy"` returns a circuit that takes the weights at each call
            and does not depend on any framework. `"torch"` returns a
            `TorchCircuit`, which is a PyTorch `nn.Module` that stores the weights.
            Default: `"numpy"`.
        init (torch.Tensor | None): The initial weights of the PyTorch circuit, of
            shape `(n_weights,)`. They are random when `init` is not given. The
            NumPy interface stores no weights, so passing `init` with it raises a
            `ValueError`. Default: `None`.

    Returns:
        circuit (Circuit | TorchCircuit): The circuit for the requested `interface`.

    Raises:
        ValueError: If `interface` is neither `"numpy"` nor `"torch"`, or if `init`
            is passed with the NumPy interface.

    Example:
        ```python
        import numpy as np
        from eigenkit import Circuit, Encoding, Variational, all_z

        circuit = Circuit(
            blocks=[Encoding(in_features=4), Variational(depth=2)],
            observables=all_z(4),
        )
        weights = np.zeros(circuit.spec.n_weights)
        values = circuit(weights, np.random.uniform(0, np.pi, size=4))
        ```
    """
    if isinstance(blocks, CircuitBuilder):
        blocks = blocks.build()
    if interface == "torch":
        from .core.interfaces import get_interface

        get_interface("torch")  # raises with the install hint when torch is missing
        from .torch.circuit import TorchCircuit

        return TorchCircuit(blocks, observables=observables, provider=provider, init=init)
    if interface == "numpy":
        if init is not None:
            raise ValueError(
                "`init` only applies to interface='torch'; the numpy circuit "
                "takes weights at call time, not as owned parameters"
            )
        spec = blocks if isinstance(blocks, CircuitSpec) else core.assemble(blocks)
        return _NumpyCircuit(
            spec, observables=observables, provider=provider, interface="numpy"
        )
    raise ValueError(f"unknown interface {interface!r}; use 'torch' or 'numpy'")
