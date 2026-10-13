"""Counts-based measurement: the shared core behind every sampling backend.

Real hardware does not hand back expectation values. It hands back a histogram
of measured bitstrings, and everything else is post-processing. IBM's
``SamplerV2`` and IQM's ``backend.run(...).result().get_counts()`` both return
exactly that, so the arithmetic that turns counts into the library's
``(n_obs,)`` expectation vector belongs in one place rather than once per
adapter.

Three steps, and each one is a function here:

1. **Plan.** [`plan_measurements`][eigenkit.core.backends._sampling.plan_measurements]
   sorts the Pauli terms of every observable into *measurement groups*. Two
   terms share a group when no qubit needs two different bases from them
   (qubit-wise commuting), so one circuit execution resolves all of them. This
   is what keeps a 4-qubit ``all_z()`` at one job instead of four.
2. **Rotate.** A group names a Pauli basis per qubit;
   [`rotation_gates`][eigenkit.core.backends._sampling.rotation_gates]
   turns that into the gates to append before measuring, since hardware only
   measures in Z.
3. **Reduce.** [`expectations_from_counts`][eigenkit.core.backends._sampling.expectations_from_counts]
   folds each group's histogram back into the per-observable values, and
   [`probabilities_from_counts`][eigenkit.core.backends._sampling.probabilities_from_counts]
   densifies a Z-basis histogram into the ``2**n`` distribution that QAOA reads
   its answer out of.

Pure numpy: no SDK import, no provider import, nothing that would drag a
backend dependency into the core. Adapters do the SDK-specific work (build the
circuit, submit it, pull the histogram out of whatever result object came back)
and call in here for the rest.

Bit order: both Qiskit-family SDKs key their histograms little-endian, so
``"001"`` means qubit 0 is excited. The library is qubit-0-MSB everywhere else,
so [`normalize_counts`][eigenkit.core.backends._sampling.normalize_counts]
reverses them once, at the boundary, and everything downstream reads left to
right as qubit 0 to qubit n-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..ir import Observable


# Densifying a histogram costs 2**n floats. 24 qubits is 128 MiB, already past
# what anyone wants returned from a method call by accident, and a device wide
# enough to need more has nothing like enough shots to populate it: 2**24 bins
# from even a million shots is noise. `sample()` stays available at any width --
# it is the sparse histogram, and its size is bounded by the shot count.
MAX_DENSE_QUBITS = 24

# Basis -> the gates that map its eigenbasis onto Z, which is the only basis
# hardware measures in. "I" marks a qubit no term in the group constrains; it
# is measured along with the rest (readout is per-qubit and simultaneous) and
# simply ignored when the parities are taken.
_ROTATIONS: dict[str, tuple[str, ...]] = {
    "I": (),
    "Z": (),
    "X": ("H",),
    # Y = S-dagger then H: S-dagger sends Y to X, H sends X to Z.
    "Y": ("Sdg", "H"),
}


@dataclass(frozen=True)
class TermPlan:
    """One Pauli term, addressed back to the observable it came from.

    Attributes:
        observable_index: Which observable of the compiled set this term
            contributes to.
        coeff: The term's real coefficient, carried through unchanged.
        qubits: The qubits whose measured bits set the term's sign. Their
            parity is what the estimator reduces over.
    """

    observable_index: int
    coeff: float
    qubits: tuple[int, ...]


@dataclass(frozen=True)
class MeasurementGroup:
    """A set of Pauli terms one circuit execution can resolve at once.

    Attributes:
        basis: One letter per qubit — `"X"`, `"Y"`, `"Z"`, or `"I"` for a
            qubit no term in this group constrains. Length is the circuit
            width.
        terms: The terms this group resolves, each pointing back at its
            observable.
    """

    basis: tuple[str, ...]
    terms: tuple[TermPlan, ...]

    @property
    def is_z_basis(self) -> bool:
        """True when no qubit needs a rotation, so the raw computational-basis
        histogram already answers this group."""
        return all(letter in ("I", "Z") for letter in self.basis)


def plan_measurements(
    observables: Sequence[Observable],
    n_qubits: int,
    *,
    group_commuting: bool = True,
) -> tuple[MeasurementGroup, ...]:
    """Sort every Pauli term of `observables` into qubit-wise-commuting groups.

    Each returned group is one circuit execution: rotate per `group.basis`,
    measure, and the histogram resolves every term the group carries. Grouping
    is greedy in the order the terms appear — first fit, new group otherwise —
    which is not the minimum number of groups in general, but is deterministic,
    cheap, and collapses the cases that actually show up (`all_z(n)` to one
    group, a QAOA cost Hamiltonian's Z and ZZ terms to one group).

    Args:
        observables (Sequence[Observable]): The observables to measure.
        n_qubits (int): Circuit width, which fixes the length of each
            group's basis.
        group_commuting (bool): Whether to pack compatible terms together.
            `False` gives one group per term: more circuit executions, but
            every term gets the full shot budget to itself, which is the
            comparison point when you want to attribute shot noise per term.

    Returns:
        groups (tuple[MeasurementGroup, ...]): One entry per circuit execution
            needed, in submission order.

    Raises:
        ValueError: If a term names a qubit outside the circuit.
    """
    # Bases are merged in place while grouping, then frozen into tuples at the
    # end; a dict per group keeps the conflict check to a lookup per Pauli.
    bases: list[list[str]] = []
    members: list[list[TermPlan]] = []
    for observable_index, observable in enumerate(observables):
        for term in observable.terms:
            required: dict[int, str] = {}
            for letter, qubit in term.paulis:
                if not 0 <= qubit < n_qubits:
                    raise ValueError(
                        f"observable {observable_index} measures qubit {qubit}, "
                        f"outside the {n_qubits}-qubit circuit"
                    )
                required[qubit] = letter
            plan = TermPlan(
                observable_index=observable_index,
                coeff=float(term.coeff),
                qubits=tuple(sorted(required)),
            )
            placed = False
            if group_commuting:
                for basis, group_terms in zip(bases, members):
                    if all(basis[q] in ("I", letter) for q, letter in required.items()):
                        for qubit, letter in required.items():
                            basis[qubit] = letter
                        group_terms.append(plan)
                        placed = True
                        break
            if not placed:
                basis = ["I"] * n_qubits
                for qubit, letter in required.items():
                    basis[qubit] = letter
                bases.append(basis)
                members.append([plan])
    return tuple(
        MeasurementGroup(basis=tuple(basis), terms=tuple(group_terms))
        for basis, group_terms in zip(bases, members)
    )


def rotation_gates(basis: Sequence[str]) -> tuple[tuple[str, int], ...]:
    """The gates to append before measuring, for a group's `basis`.

    Hardware measures in Z only, so an X or Y factor has to be rotated onto Z
    first. Returned as `(gate_name, qubit)` pairs in application order, naming
    gates the way the library's IR does (`"H"`, `"Sdg"`); each adapter applies
    them with its own SDK.

    Args:
        basis (Sequence[str]): One letter per qubit, as carried by
            [`MeasurementGroup.basis`][eigenkit.core.backends._sampling.MeasurementGroup].

    Returns:
        gates (tuple[tuple[str, int], ...]): `(gate_name, qubit)` pairs, in
            the order they must be applied.
    """
    gates: list[tuple[str, int]] = []
    for qubit, letter in enumerate(basis):
        for name in _ROTATIONS[letter]:
            gates.append((name, qubit))
    return tuple(gates)


def normalize_counts(
    counts: Mapping[str, int], n_qubits: int, *, little_endian: bool = True
) -> dict[str, int]:
    """Rewrite a histogram into the library's qubit-0-MSB bitstring keys.

    Qiskit-family SDKs — IBM Runtime and IQM alike — key their counts
    little-endian, so `"001"` is qubit 0 excited, and separate classical
    registers come back space-separated. The library reads bitstrings left to
    right as qubit 0 to qubit n-1, so the keys are reversed once here rather
    than being second-guessed at every use.

    Args:
        counts (Mapping[str, int]): The SDK's histogram.
        n_qubits (int): Circuit width. Keys are left-padded to this length,
            which matters when an SDK trims leading zeros.
        little_endian (bool): Whether `counts` is keyed little-endian, which
            both supported SDKs are. Default: `True`.

    Returns:
        counts (dict[str, int]): The same histogram, keyed qubit-0-MSB.

    Raises:
        ValueError: If a key is wider than the circuit, or holds anything but
            `0` and `1` once separators are stripped.
    """
    normalized: dict[str, int] = {}
    for key, count in counts.items():
        bits = key.replace(" ", "")
        if set(bits) - {"0", "1"}:
            raise ValueError(f"unexpected characters in measurement key {key!r}")
        if len(bits) > n_qubits:
            raise ValueError(
                f"measurement key {key!r} is {len(bits)} bits wide, more than the "
                f"{n_qubits}-qubit circuit"
            )
        bits = bits.zfill(n_qubits)
        if little_endian:
            bits = bits[::-1]
        # Padding can collapse two SDK keys onto one; add rather than assign.
        normalized[bits] = normalized.get(bits, 0) + int(count)
    return normalized


def _as_bit_table(
    counts: Mapping[str, int], n_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a normalized histogram into `(bits, multiplicities)`.

    `bits` is `(n_distinct, n_qubits)` of 0/1 with column `q` holding qubit
    `q`; `multiplicities` is the matching shot count per row. Doing this once
    per group is what lets every term of the group reduce as one vectorized
    parity, instead of a Python loop over shots.
    """
    if not counts:
        return np.zeros((0, n_qubits), dtype=np.int8), np.zeros(0, dtype=np.float64)
    keys = list(counts)
    ragged = [key for key in keys if len(key) != n_qubits]
    if ragged:
        raise ValueError(
            f"measurement keys must be {n_qubits} bits wide; got {ragged[0]!r}. "
            f"Run the SDK's histogram through normalize_counts() first."
        )
    flat = np.frombuffer("".join(keys).encode("ascii"), dtype=np.uint8)
    table = (flat - ord("0")).reshape(len(keys), n_qubits).astype(np.int8)
    return table, np.array([counts[key] for key in keys], dtype=np.float64)


def expectations_from_counts(
    groups: Iterable[MeasurementGroup],
    counts_per_group: Sequence[Mapping[str, int]],
    n_observables: int,
    n_qubits: int,
) -> np.ndarray:
    """Reduce per-group histograms into one expectation value per observable.

    For a Pauli product `P` the estimate is the shot-weighted mean of
    `(-1)**parity`, where the parity is taken over the bits of the qubits `P`
    acts on; the observable's value is the coefficient-weighted sum of its
    terms. Shots are counted per group, so a group that came back empty or
    short does not silently scale its neighbours.

    Args:
        groups (Iterable[MeasurementGroup]): The plan these histograms answer,
            as returned by `plan_measurements`.
        counts_per_group (Sequence[Mapping[str, int]]): One histogram per
            group, in the same order, keyed qubit-0-MSB (run the SDK's output
            through `normalize_counts` first).
        n_observables (int): How many expectation values to return.
        n_qubits (int): Circuit width.

    Returns:
        values (np.ndarray): One expectation value per observable, shape
            `(n_observables,)`.

    Raises:
        ValueError: If there are not exactly as many histograms as groups, or
            a group's histogram holds no shots at all.
    """
    groups = tuple(groups)
    if len(counts_per_group) != len(groups):
        raise ValueError(
            f"expected one histogram per measurement group: got "
            f"{len(counts_per_group)} for {len(groups)} groups"
        )
    values = np.zeros(n_observables, dtype=np.float64)
    for group, counts in zip(groups, counts_per_group):
        bits, multiplicities = _as_bit_table(counts, n_qubits)
        shots = float(multiplicities.sum())
        if shots <= 0:
            raise ValueError(
                "a measurement group came back with no shots; the job most "
                "likely failed or was cancelled before it ran"
            )
        for term in group.terms:
            # (-1)**(number of 1s on the term's qubits), summed over shots.
            parity = bits[:, list(term.qubits)].sum(axis=1) & 1
            signs = 1.0 - 2.0 * parity
            values[term.observable_index] += term.coeff * float(
                signs @ multiplicities / shots
            )
    return values


def probabilities_from_counts(counts: Mapping[str, int], n_qubits: int) -> np.ndarray:
    """Densify a computational-basis histogram into the full distribution.

    Returns the `2**n_qubits` probabilities in qubit-0-MSB basis-state order,
    which is what `Circuit.probabilities()` promises and what QAOA decodes its
    answer from. Unmeasured basis states are exact zeros, not smoothed: with
    finite shots most of the vector genuinely is zero, and pretending otherwise
    would misreport how peaked the distribution is.

    Args:
        counts (Mapping[str, int]): A Z-basis histogram, keyed qubit-0-MSB.
        n_qubits (int): Circuit width.

    Returns:
        probabilities (np.ndarray): Shape `(2**n_qubits,)`, summing to 1.

    Raises:
        ValueError: If the circuit is wider than `MAX_DENSE_QUBITS`, or the
            histogram holds no shots.
    """
    if n_qubits > MAX_DENSE_QUBITS:
        raise ValueError(
            f"probabilities() would allocate 2**{n_qubits} floats; the sampling "
            f"backends cap this at {MAX_DENSE_QUBITS} qubits. Use sample() and "
            f"read the histogram directly — at this width the shot count cannot "
            f"populate a dense distribution anyway."
        )
    distribution = np.zeros(1 << n_qubits, dtype=np.float64)
    shots = 0
    for bits, count in counts.items():
        # Keys are qubit-0-MSB, which is exactly big-endian binary, so the
        # basis-state index is the key read as a base-2 integer.
        distribution[int(bits, 2)] += count
        shots += count
    if shots <= 0:
        raise ValueError(
            "the measurement job came back with no shots; it most likely "
            "failed or was cancelled before it ran"
        )
    return distribution / shots
