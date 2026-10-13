"""The one state-preparation unitary every backend uses.

`AmplitudeEncoding` asks for a unitary whose first column is the requested
state, so that `U|0...0> = |amps>`. That leaves the remaining `2**k - 1`
columns free, and any orthonormal completion satisfies the request — from
`|0...0>`.

It stops being free the moment the target register is in any other state.
Completions chosen independently — a QR factorisation, a modified
Gram-Schmidt over the computational basis, or an SDK's own state-preparation
synthesis — agree from `|0...0>` and disagree everywhere else, so the same
circuit would return a different value on each backend, with nothing raising.

So the completion is fixed here, once, and every backend applies this matrix
as an ordinary unitary rather than calling its SDK's state-preparation
primitive. The cost is that the SDKs' native state-prep synthesis is unused
for this gate; the gain is that a `CircuitSpec` means the same thing
everywhere, which is the contract the whole provider layer rests on.
"""

from __future__ import annotations

import numpy as np


def state_prep_unitary(amps) -> np.ndarray:
    r"""Return the canonical unitary `U` with `U|0...0> = |amps>`.

    The first column is `amps`, normalized. The rest are the completion from
    a QR factorisation of the identity with `amps` substituted into column 0,
    with each column rescaled by the phase of its `R` diagonal so that column
    0 comes back exactly `amps` rather than `amps` up to a phase.

    Args:
        amps (np.ndarray | Sequence[complex]): The state to prepare, as a flat
            array-like of length `2**k`. Normalized here if it is not already.

    Returns:
        The `2**k` x `2**k` unitary, `complex128`.

    Raises:
        ValueError: If `amps` is all-zero, which names no state.
    """
    psi = np.asarray(amps, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm == 0.0:  # pragma: no cover - the AmplitudeEncoding block rejects this
        raise ValueError("AmplitudeEncoding state is all-zero")
    psi = psi / norm
    seed = np.eye(psi.shape[0], dtype=np.complex128)
    seed[:, 0] = psi
    q, r = np.linalg.qr(seed)
    phases = np.diagonal(r).copy()
    phases[phases == 0.0] = 1.0
    return q * (phases / np.abs(phases))
