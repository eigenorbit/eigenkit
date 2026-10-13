"""The registry of estimator classes, used to load `.eigenkit` bundles."""

from __future__ import annotations


_REGISTRY: dict[str, type] = {}


def register_estimator(cls):
    """Register an estimator class under its `__name__`.

    It is used as a class decorator.

    Args:
        cls (type): The estimator class to register.

    Returns:
        (type): The same class, unchanged.
    """
    _REGISTRY[cls.__name__] = cls
    return cls


def get_estimator_class(name: str) -> type:
    """Return the estimator class registered under `name`.

    Args:
        name: The class name, as stored in the `metadata.json` of a bundle.

    Returns:
        The registered class.

    Raises:
        ValueError: If no class is registered under `name`. The module that
            defines the class must be imported before `load()` is called.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown estimator class: {name!r}. "
            f"Known: {sorted(_REGISTRY)}. "
            "Make sure the module is imported before calling load()."
        )
    return _REGISTRY[name]
