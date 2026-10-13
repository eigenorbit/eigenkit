"""Hamiltonian-simulation helpers for the honest (hardware-faithful) HHL path.

The reference HHL path applies ``e^{iAt}`` as a dense ``QubitUnitary`` from
``scipy.linalg.expm`` — cheap to simulate, impossible to synthesise on hardware.  The
honest path instead writes the Hermitian ``A`` as a sum of Pauli strings
``A = Σ_k c_k P_k`` so that ``e^{iAt}`` can be Trotterised into elementary gates (see
[`TrotterBlock`][eigenkit.core.blocks.TrotterBlock]).

The term count is the honest cost metric, and it follows sparsity **in the Pauli
basis**, which is not sparsity of the matrix: a non-zero at ``(i, j)`` feeds the strings
whose X/Y support is ``i XOR j``, so the count tracks the number of *distinct* offsets.
A band has only the ``log2 N`` patterns ``2^k − 1``; a scattered pattern exhausts them,
and a random diagonal still needs ``N`` because the Z-string coefficients are the
Walsh–Hadamard transform of the diagonal.  So "sparse ⇒ few terms" is false;
"constructed as a short Pauli sum ⇒ few terms" is what holds.  HHL 2009's premise is a
different input model (``s``-sparse, row-computable, via a sparse-access oracle) whose
cost analysis does not transfer here.
"""

from __future__ import annotations

import itertools

import numpy as np


# Single-qubit Pauli matrices, keyed by the symbol used in a Pauli string.
_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string_matrix(pauli_string: str) -> np.ndarray:
    """Dense matrix of a Pauli string.

    ``pauli_string[i]`` acts on qubit ``i``; the operator is the Kronecker product in qubit
    order, qubit 0 being the most significant factor (the library's big-endian layout).

    Args:
        pauli_string: Non-empty string over ``{I, X, Y, Z}``.

    Returns:
        A ``2^n × 2^n`` complex matrix, ``n = len(pauli_string)``.
    """
    if not pauli_string:
        raise ValueError("pauli_string must be non-empty")
    matrix = np.array([[1.0 + 0j]])
    for symbol in pauli_string:
        try:
            factor = _PAULI_MATRICES[symbol]
        except KeyError:
            raise ValueError(
                f"invalid Pauli symbol {symbol!r}; expected one of I, X, Y, Z"
            ) from None
        matrix = np.kron(matrix, factor)
    return matrix


def pauli_decompose(A: np.ndarray, rtol: float = 1e-10) -> list[tuple[complex, str]]:
    """Decompose a Hermitian matrix into a weighted sum of Pauli strings.

    The ``4^n`` Pauli strings are orthogonal under the Hilbert–Schmidt product
    (``Tr(P_k P_l) = 2^n δ_kl``), so ``c_k = Tr(P_k A) / 2^n``.  For Hermitian ``A`` every
    ``c_k`` is real up to numerical noise; the imaginary part is returned so callers can
    assert it is negligible.

    Pruning is **relative** to ``max|A_ij|``, so the term list — and every circuit built
    from it — is invariant under ``A → αA``.  An absolute threshold would make the honest
    path depend on the physical units of ``A``.

    Args:
        A: A Hermitian ``2^n × 2^n`` matrix (array-like).
        rtol: Relative pruning tolerance; terms with ``|c_k| <= rtol · max|A_ij|`` go.

    Returns:
        ``(coeff, pauli_string)`` pairs in lexicographic order over ``I < X < Y < Z``.

        !!! warning
            The *order* of this list is part of the numerical contract of
            [`TrotterBlock`][eigenkit.core.blocks.TrotterBlock]: the first-order Trotter
            error depends on the order the factors are applied in, so reordering these
            terms changes honest-mode results at a fixed ``trotter_steps``.  Pinned by a
            regression test; do not change it casually.

    Raises:
        ValueError: If ``A`` is not a square 2-D matrix, its dimension is not a power of
            two, or it is not Hermitian to within ``1e-8`` relative to its scale.
    """
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A.shape}")
    dim = int(A.shape[0])
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"A dimension must be a power of two, got {dim}")
    n_qubits = dim.bit_length() - 1
    scale = float(np.max(np.abs(A)))
    # Relative to the matrix, with no floor under it.  A floor such as `max(scale, 1.0)`
    # would hold the tolerance at 1e-8 however small A is, so every matrix with entries
    # at or below that would pass whatever its structure — `1e-9 * [[0, 1], [0, 0]]`
    # among them — and `TrotterBlock` would drop the imaginary parts and evolve
    # something else.  An all-zero A still passes: `atol=0` with `rtol=0` compares
    # exactly, and it is equal to its own conjugate transpose.
    if not np.allclose(A, A.conj().T, atol=1e-8 * scale, rtol=0):
        raise ValueError(
            "A is not Hermitian; pauli_decompose expects a Hermitian operator "
            "(the honest HHL path only simulates Hermitian A)."
        )
    threshold = rtol * scale

    coefficients = _pauli_coefficients(A)

    terms: list[tuple[complex, str]] = []
    for symbols in itertools.product("IXYZ", repeat=n_qubits):
        pauli_string = "".join(symbols)
        x_mask, z_mask = _string_masks(pauli_string)
        coeff = complex(coefficients[x_mask, z_mask])
        if abs(coeff) > threshold:
            terms.append((coeff, pauli_string))
    return terms


def _string_masks(pauli_string: str) -> tuple[int, int]:
    """``(x, z)`` bit masks of a Pauli string in the library's big-endian layout.

    ``x`` marks the qubits carrying X or Y, ``z`` those carrying Z or Y.  Qubit 0 is the
    most significant Kronecker factor, matching `pauli_string_matrix()`, so it owns bit
    ``n−1``, which is the layout `_pauli_coefficients()` indexes its table by.
    """
    n = len(pauli_string)
    x_mask = z_mask = 0
    for k, symbol in enumerate(pauli_string):
        bit = 1 << (n - 1 - k)
        if symbol in ("X", "Y"):
            x_mask |= bit
        if symbol in ("Z", "Y"):
            z_mask |= bit
    return x_mask, z_mask


def _walsh_hadamard(values: np.ndarray) -> np.ndarray:
    """Unnormalised Walsh–Hadamard transform along the last axis.

    ``out[..., z] = Σ_r (−1)^popcount(r & z) · values[..., r]``, in
    ``O(N log N)`` per row instead of ``O(N²)``.
    """
    out = np.array(values, dtype=complex, copy=True)
    length = out.shape[-1]
    step = 1
    while step < length:
        out = out.reshape(*out.shape[:-1], length // (2 * step), 2, step)
        lower, upper = out[..., 0, :].copy(), out[..., 1, :].copy()
        out[..., 0, :] = lower + upper
        out[..., 1, :] = lower - upper
        out = out.reshape(*out.shape[:-3], length)
        step *= 2
    return out


def _pauli_coefficients(A: np.ndarray) -> np.ndarray:
    """All ``4^n`` Pauli coefficients of ``A`` as a table indexed by ``[x, z]``.

    A Pauli string has exactly **one** non-zero per row — at column ``r ⊕ x``, with value
    ``(−i)^{n_Y} · (−1)^{popcount(r & z)}`` where ``n_Y = popcount(x & z)``.  So the
    Hilbert–Schmidt trace collapses to a single sum over rows::

        Tr(P·A) = (−i)^{n_Y} · Σ_r (−1)^{popcount(r & z)} · A[r ⊕ x, r]

    and that sum, taken over *all* ``z`` at once for a fixed ``x``, is a Walsh–Hadamard
    transform of the ``x``-shifted diagonal of ``A``.  The table therefore costs
    ``O(N² log N)``, against the ``O(N⁴)`` of forming each of the ``4^n`` dense Pauli
    matrices and contracting it, which is what bounds the reachable system size.
    """
    dim = int(A.shape[0])
    rows = np.arange(dim)
    # shifted[x, r] = A[r ⊕ x, r] — the diagonal of A displaced by x.
    shifted = A[rows[None, :] ^ rows[:, None], rows[None, :]]
    table = _walsh_hadamard(shifted)
    # The (−i)^{n_Y} phase, with n_Y = popcount(x & z) over the table's own indices.
    n_y = np.bitwise_count(rows[:, None] & rows[None, :])
    table *= np.power(-1j, n_y)
    return table / dim


def _symplectic_bits(
    pauli_strings: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Pack Pauli strings into ``(x, z)`` bitmasks, one ``uint64`` per string.

    ``X → (1, 0)``, ``Z → (0, 1)``, ``Y → (1, 1)``, ``I → (0, 0)``.  Two strings commute
    iff ``popcount(x_P & z_Q) + popcount(x_Q & z_P)`` is even — the symplectic form.
    """
    n = len(pauli_strings[0]) if pauli_strings else 0
    if any(len(s) != n for s in pauli_strings):
        raise ValueError(
            "all Pauli strings must have the same length; a ragged list would be "
            "packed into one bit layout and silently yield wrong commutation parities"
        )
    if n > 64:
        raise ValueError(f"Pauli strings longer than 64 qubits are unsupported (got {n})")
    x_bits = np.zeros(len(pauli_strings), dtype=np.uint64)
    z_bits = np.zeros(len(pauli_strings), dtype=np.uint64)
    for i, s in enumerate(pauli_strings):
        x = z = 0
        for q, ch in enumerate(s):
            bit = 1 << q
            if ch in ("X", "Y"):
                x |= bit
            if ch in ("Z", "Y"):
                z |= bit
        x_bits[i] = np.uint64(x)
        z_bits[i] = np.uint64(z)
    return x_bits, z_bits


def anticommuting_weight_profile(
    pauli_terms: list[tuple[complex, str]] | tuple[tuple[float, str], ...],
) -> np.ndarray:
    """Per-term anticommuting suffix weights ``R_k`` of a Pauli sum.

    Returns ``R_k = |c_k| · Σ_{l > k, P_l anticommutes with P_k} |c_l|``, indexed in the
    order ``pauli_terms`` is given — the order
    [`TrotterBlock`][eigenkit.core.blocks.TrotterBlock] applies the factors in.  Two
    Pauli strings either commute or anticommute, and in the latter case
    ``‖[P_k, P_l]‖ = 2`` exactly, so ``R`` carries all the spectral information the leading
    Trotter bounds need: ``‖[Σ_{l>k} c_l P_l, c_k P_k]‖ ≤ 2 R_k``.

    Costs ``O(L² n / 64)`` word operations, negligible beside the decomposition that
    produced the terms.
    """
    coeffs = np.array([abs(complex(c)) for c, _ in pauli_terms], dtype=float)
    strings = [s for _, s in pauli_terms]
    n_terms = len(strings)
    if n_terms == 0:
        return np.zeros(0, dtype=float)
    x_bits, z_bits = _symplectic_bits(strings)

    result = np.zeros(n_terms, dtype=float)
    block = max(1, min(n_terms, 4_194_304 // max(n_terms, 1)))
    index = np.arange(n_terms)
    for start in range(0, n_terms, block):
        stop = min(start + block, n_terms)
        xk = x_bits[start:stop, None]
        zk = z_bits[start:stop, None]
        parity = np.bitwise_count(xk & z_bits[None, :]) + np.bitwise_count(
            x_bits[None, :] & zk
        )
        anticommutes = (parity & 1).astype(bool)
        # Keep only the strict upper triangle l > k.
        anticommutes &= index[None, :] > index[start:stop, None]
        result[start:stop] = coeffs[start:stop] * (anticommutes * coeffs[None, :]).sum(
            axis=1
        )
    return result


def trotter_drift_bound(
    pauli_terms: list[tuple[complex, str]] | tuple[tuple[float, str], ...],
    tau: float,
    order: int,
) -> float:
    """Upper bound on ``‖A_eff − A‖`` for one Trotter step of size ``tau``.

    Because the honest path holds ``tau`` fixed across every controlled power, QPE is
    *exact* phase estimation of the single step unitary ``W``, and the whole circuit is
    ideal HHL for the effective generator ``A_eff := Log(W)/(i·tau)``.  The entire Trotter
    error is that one spectral substitution, and this bounds its size.

    Both are leading-order product-formula bounds, of the commutator-scaling form
    Theorem 11 of Childs, Su, Tran, Wiebe and Zhu establishes (*Theory of Trotter Error
    with Commutator Scaling*, Phys. Rev. X 11, 011020, 2021; preprint arXiv:1912.08854).
    That theorem is stated up to an order of magnitude, so the constants below are not
    taken from it.  Written here for Pauli terms, where ``‖[P_k, P_l]‖ ∈ {0, 2}``:

    * ``order = 1``: ``τ · Γ₁``, ``Γ₁ = Σ_k R_k = Σ_{k<l, anticommuting} |c_k c_l|``.
    * ``order = 2``: ``τ² · Γ₂``, ``Γ₂ = ⅓ Σ_k S_k R_k + ⅙ Σ_k |c_k| R_k`` with suffix sums
      ``S_k = Σ_{l>k} |c_l|``.  The symmetric product is a palindrome, so the first-order
      commutator cancels and the leading error is ``O(τ²)``.

    These bound the *leading* Trotter term.  They also bound the full ``‖A_eff − A‖`` up to
    the Lipschitz constant of the principal logarithm, which is ~1 in the regime the caller
    drives ``tau`` into; outside it the number is an estimate, not a bound.  Both the
    constants and the upper-bound property are pinned by tests against ``‖A_eff − A‖``
    computed exactly on random Hermitian matrices.

    ``Γ₁`` is exactly invariant under permutations of ``pauli_terms`` — reordering flips the
    signs of the anticommuting pairs' contributions, never which pairs appear — so no
    reordering can improve the guarantee.  ``Γ₂`` contains suffix sums and is not invariant,
    which is a further reason the term order is pinned.

    Args:
        pauli_terms: ``(coeff, pauli_string)`` terms in **application order**.
        tau: Trotter step size (``time / steps``); only ``|tau|`` matters.
        order: 1 (Lie–Trotter) or 2 (symmetric Strang).

    Returns:
        A non-negative bound, exactly ``0.0`` when all terms mutually commute.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    coeffs = np.array([abs(complex(c)) for c, _ in pauli_terms], dtype=float)
    if coeffs.size == 0:
        return 0.0
    R = anticommuting_weight_profile(pauli_terms)
    if not R.any():
        return 0.0
    tau = abs(float(tau))
    if order == 1:
        return float(tau * R.sum())
    # Suffix sums S_k = Σ_{l>k} |c_l|.
    S = np.concatenate([np.cumsum(coeffs[::-1])[::-1][1:], [0.0]])
    gamma2 = float((S * R).sum() / 3.0 + (coeffs * R).sum() / 6.0)
    return tau * tau * gamma2
