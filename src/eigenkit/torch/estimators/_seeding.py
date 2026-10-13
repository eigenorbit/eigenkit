"""Keep the seeding of an estimator from changing the random state of the caller.

`torch.manual_seed` sets the random number generator (RNG) of the whole process:
the one of the central processing unit (CPU) and the one of every CUDA device. So
an estimator that calls it while it builds a model also reseeds the random stream
of the caller. For example, if you seed PyTorch with your own value and then
create an `HQNC`, your next `torch.randn` does not follow the sequence that you
asked for.

`isolated_rng` prevents that without changing what the estimator does. For a call
that seeds, it saves the RNG state when the call starts and restores it when the
call ends. The `torch.manual_seed` inside the call still controls everything that
happens during the call (the initialisation of the weights and the shuffling of
the `DataLoader` after it), so the models are bit-identical to the models built
without it. The only difference is that the seed does not leak out of the call.

A call with `random_state=None` never seeds, so there is nothing to contain, and
it is left as it is. It draws from the random stream of the caller and advances
it, so two such fits differ, as an unseeded fit should.

The RNG state of the CUDA devices is also saved and restored, because
`torch.manual_seed` seeds them as well. Without them, the CPU stream would be
fixed but the stream of the graphics processing unit (GPU) would still be
overwritten. This costs one initialisation of the CUDA context the first time
(measured at `4.5 ms` on a machine with one GPU) and about `0.08 ms` for each
call after that, compared with a `fit()` that trains for seconds.
"""

from __future__ import annotations

import functools

import torch


def _forked_devices() -> list[int]:
    """Return the CUDA devices whose RNG state has to be saved and restored.

    Returns:
        Every visible CUDA device index, or an empty list with no CUDA.
    """
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def isolated_rng(method):
    """Run a method on a private copy of the random number generator (RNG) state.

    Decorate every entry point that seeds PyTorch (`fit()` and the `load()`
    path), not the builder that it calls. Then the seed stays in force for the
    whole call, and the caller gets their random stream back afterwards.

    Only a call that seeds is isolated. With `random_state=None`, the estimator
    never calls `torch.manual_seed`, so there is no leak to contain. Restoring
    the state that such a call used would also make every unseeded `fit()` start
    from the same place and return the same model, which is the opposite of what
    `None` asks for.

    Args:
        method: The method to wrap.

    Returns:
        The wrapped method. For the seeded calls, which change the RNG state, it
            restores the RNG state of PyTorch on the central processing unit
            (CPU) and on the CUDA devices when it returns.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "random_state", None) is None:
            return method(self, *args, **kwargs)
        with torch.random.fork_rng(devices=_forked_devices()):
            return method(self, *args, **kwargs)

    return wrapper
