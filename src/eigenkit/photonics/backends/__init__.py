"""The photonic backends, which are executors that implement `Executor`.

`Executor` is the protocol defined in `photonics.executor`.
"""

from .perceval_backend import PercevalBackend, PercevalCompiled


__all__ = ["PercevalBackend", "PercevalCompiled"]
