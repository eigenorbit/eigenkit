"""Differentiation interfaces, which connect the core to automatic differentiation.

The core [`Circuit`][eigenkit.core.circuit.Circuit] computes forward values and
analytic NumPy jacobians (with the parameter-shift rule, the chain rule or the
adjoint method). An *interface* connects this NumPy code to the automatic
differentiation of a framework. It wraps the NumPy `run` and jacobians of the
circuit into a differentiable operation of that framework (a
`torch.autograd.Function` now, a `jax.custom_vjp` later), and it converts between
the framework and NumPy.

NumPy needs no interface, because the core returns arrays directly. Frameworks
register themselves when their package is imported. Adding JAX therefore takes a
new adapter module and one
[`register_interface()`][eigenkit.core.interfaces.register_interface] call. The
core never changes and never imports a framework.

`get_interface("torch")` imports the PyTorch package (`eigenkit.torch`) the first
time it is needed, and raises a clear error if PyTorch is not installed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Interface(Protocol):
    """An adapter from the NumPy core to the automatic differentiation of a framework.

    `run_circuit` receives the core `Circuit` (for `compiled`,
    `provider.gradient_method`, `observables` and the jacobian helpers), and the
    `weights` and `inputs` as tensors of the framework. It returns a
    differentiable output, also as a tensor of the framework.
    """

    name: str

    def run_circuit(self, circuit, weights, inputs):
        """Run `circuit` on tensors of the framework and return a differentiable result.

        An implementation that builds its own default `inputs` (for a circuit that
        takes none) must call `circuit._check_parameter_lengths(weights, inputs)`
        once that default exists. The core cannot run this check before it passes
        the call here, and without it an input of the wrong size can be cut short
        without any error.

        Args:
            circuit (Circuit): The core circuit being run.
            weights (Any): The weights, as a tensor of the framework.
            inputs (Any): The inputs, as a tensor of the framework.

        Returns:
            result (Any): A differentiable result, as a tensor of the framework.
        """
        ...


_INTERFACES: dict[str, Interface] = {}

# Packages that register an interface on import. Used by `get_interface` to
# turn a string into an import-and-retry before giving up.
_LAZY_PACKAGES = {"torch": "eigenkit.torch"}


def register_interface(name: str, interface: Interface) -> None:
    """Register the interface of a framework under `name` (for example `"torch"`).

    Args:
        name (str): The name to register.
        interface (Interface): The adapter to register.
    """
    _INTERFACES[name] = interface


def registered_interfaces() -> list[str]:
    """Return the names of the interfaces registered so far.

    An interface appears only once its package has been imported, so this lists
    what is loaded now, not everything that is installed.

    Example:
        ```python
        from eigenkit.core.interfaces import registered_interfaces

        print(registered_interfaces())   # []

        import eigenkit.torch  # registers itself on import
        print(registered_interfaces())   # ['torch']
        ```

    Returns:
        The registered interface names.
    """
    return list(_INTERFACES)


def get_interface(name: str) -> Interface:
    """Return the interface registered under `name`, importing its package if needed.

    Args:
        name (str): The registered name.

    Returns:
        The registered interface.

    Raises:
        ImportError: If the package of the interface is known but its framework
            is not installed. The message says how to install it.
        KeyError: If no interface is registered under `name`.
    """
    if name not in _INTERFACES:
        package = _LAZY_PACKAGES.get(name)
        if package is not None:
            try:
                __import__(package)  # importing it triggers self-registration
            except ImportError as exc:
                raise ImportError(
                    f"interface {name!r} requires the optional '{name}' "
                    f"dependency. Install it (e.g. `pip install "
                    f"eigenkit[{name}]`)."
                ) from exc
    if name not in _INTERFACES:
        raise KeyError(
            f"no interface registered under {name!r}; registered: {sorted(_INTERFACES)}"
        )
    return _INTERFACES[name]
