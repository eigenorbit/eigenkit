"""Secret configuration fields, which are kept in memory and never saved.

A configuration dataclass marks a field that holds a credential with
`metadata={"secret": True}`, and with `repr=False`, so that the field stays out of
`repr()`, `str()` and the `metadata.json` of a saved model. Two things respect
the mark:

- `SecretFieldsMixin` makes pickling drop those fields. joblib checkpoints and
  process pools also use pickle, so they drop them as well. Copies inside one
  process (`copy.copy`, `copy.deepcopy`, and therefore `sklearn.base.clone`) keep
  them, so a grid search can still authenticate.
- The `.eigenkit` model serialiser writes a configuration with
  `dataclasses.asdict()`, which ignores both `repr=False` and the pickle hooks, so
  it explicitly drops every name in `secret_field_names()`.

After a load, a secret field holds its default value (usually `None`), and the
backend falls back to its own way of finding credentials, usually an environment
variable.

Example:
    ```python
    import pickle
    from dataclasses import dataclass, field

    from eigenkit.core.secret_fields import (
        SecretFieldsMixin,
        secret_field_names,
    )


    @dataclass(frozen=True)
    class MyConfig(SecretFieldsMixin):
        device: str = "sim"
        token: str | None = field(default=None, repr=False, metadata={"secret": True})


    cfg = MyConfig(token="s3cr3t")
    print(secret_field_names(cfg))            # frozenset({'token'})
    print(cfg)                                # MyConfig(device='sim')
    print(pickle.loads(pickle.dumps(cfg)).token)  # None
    ```
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any


def secret_field_names(config: Any) -> frozenset[str]:
    """Return the names of the dataclass fields that are marked secret.

    A field is secret when the `"secret"` entry of its metadata is true.

    Args:
        config: The dataclass instance, or dataclass type, to inspect.

    Returns:
        The names of the secret fields. The set is empty when the dataclass has
            none.

    Raises:
        TypeError: If `config` is not a dataclass.
    """
    if not dataclasses.is_dataclass(config):
        raise TypeError(f"expected a dataclass, got {type(config).__name__}")
    return frozenset(
        f.name for f in dataclasses.fields(config) if f.metadata.get("secret")
    )


def _field_default(f: dataclasses.Field) -> Any:
    """The value a field takes when it is not supplied."""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


class SecretFieldsMixin:
    """A mixin for frozen configuration dataclasses that hold credentials.

    Pickling (and therefore joblib) drops every secret field, and unpickling gives
    each of them its default value. Copying (`copy.copy`, `copy.deepcopy`,
    `sklearn.base.clone`) keeps every field, including the secret ones.

    Mark a field secret with
    `field(default=None, repr=False, metadata={"secret": True})`, and put the
    mixin first among the bases of a `@dataclass(frozen=True)` class.
    """

    def __getstate__(self) -> dict[str, Any]:
        """Return every field that is not secret, for pickle."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not f.metadata.get("secret")
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the pickled fields and set every secret field to its default."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        for f in dataclasses.fields(self):
            if f.metadata.get("secret"):
                object.__setattr__(self, f.name, _field_default(f))

    def __copy__(self):
        """Return a shallow copy that keeps the secret fields."""
        return dataclasses.replace(self)

    def __deepcopy__(self, memo: dict[int, Any]):
        """Return a deep copy that keeps the secret fields."""
        new = object.__new__(type(self))
        memo[id(self)] = new
        for f in dataclasses.fields(self):
            object.__setattr__(new, f.name, copy.deepcopy(getattr(self, f.name), memo))
        return new
