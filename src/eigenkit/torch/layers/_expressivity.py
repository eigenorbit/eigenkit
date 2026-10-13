"""Warn about ansatz settings whose weights or data cannot reach the output.

Every layer here builds a circuit of the same form: a rotation encoding, a
variational block of rotations followed by a controlled-NOT (CNOT) ladder, and a
Pauli measurement on each qubit. Some combinations of these three settings make
the circuit degenerate, which can be proved: either the variational weights
cancel exactly out of the expectation value, or the input does. Such a layer
runs, returns plausible numbers and trains the classical layers around it, so
nothing later in the model shows that the circuit has no effect.

Four separate mechanisms cause this, and this module tells them apart instead of
grouping them under a rule to "keep the axes distinct". That rule is neither
necessary nor sufficient. `RY`/`RY`/`Y` shares an axis and works on two qubits or
more, because the CNOT ladder does not preserve `Y`-strings. A single `RZ`
encoding has no effect whatever the other two axes are.

    1. An `RZ` encoding with a single encoding layer. The circuit starts in
       `|0...0>`, which is an eigenstate of every `RZ`, so the encoding only adds
       a phase. Layers that encode again between variational blocks are not
       affected, because their second encoding acts on a state that has already
       changed.

    2. The variational block commutes with the observable. `RZ` commutes with
       `Z`-strings and `RX` with `X`-strings, and the CNOT ladder maps `Z`-strings
       to `Z`-strings and `X`-strings to `X`-strings. So with
       `variational_type="basic"` and the rotation on the measurement axis, the
       weights cancel exactly. CNOT does *not* preserve `Y`-strings
       (`YI -> YX`, `IY -> ZY`), which is why the `Y` case is degenerate only on
       a single qubit, where there is no ladder.

    3. A circuit with only real gates, measured in `Y`. `RY` and CNOT are real
       matrices, so an `RY` encoding followed by `RY` rotations keeps the
       statevector real, and `<Y>` is exactly zero for a real state.

    4. A circuit with only diagonal gates. An `RZ` encoding with `RZ` rotations
       never leaves `|0...0>`, whatever the measurement basis is.

The rule in `degenerate_reason` below was found by trying every layer, number of
qubits, depth, rotation, basis and variational type, and comparing the result
with the measured gradients.
`tests/layers/test_degenerate_ansatz_warning.py` runs this search again: the
warning must be raised exactly when a gradient or the sensitivity to the data is
zero, so that neither a false warning nor a degenerate layer without a warning
can go unnoticed.
"""

from __future__ import annotations

import warnings


_AXIS = {"RX": "X", "RY": "Y", "RZ": "Z"}

# CNOT conjugation maps X-strings to X-strings and Z-strings to Z-strings, but
# sends Y-strings to mixed ones. So a shared axis only cancels the weights for
# X and Z — unless there is no ladder at all.
_LADDER_PRESERVES = ("X", "Z")


def degenerate_reason(
    *,
    enc_rotation: str,
    var_rotation: str,
    variational_type: str,
    basis: str,
    n_qubits: int,
    n_encodings: int,
) -> str | None:
    """Return why this configuration is degenerate, or `None` if it is not.

    Args:
        enc_rotation: The encoding rotation: `"RX"`, `"RY"` or `"RZ"`.
        var_rotation: The variational rotation: `"RX"`, `"RY"` or `"RZ"`. It is
            used only when `variational_type` is `"basic"`.
        variational_type: `"basic"` or `"strong"`.
        basis: The measurement basis: `"X"`, `"Y"` or `"Z"`.
        n_qubits: The number of qubits in one circuit. `1` means that there is no
            controlled-NOT (CNOT) ladder.
        n_encodings: The number of times the input is encoded into one circuit.
            It is `1` for a single encoding and `depth` for reuploading.

    Returns:
        A sentence that names the mechanism and how to avoid it, or `None`. It is
            also `None` when either rotation names no axis. The layers call this
            function before they build their blocks, and rejecting a wrong
            rotation is the job of the block. Reading `_AXIS` without this check
            would replace the `ValueError` of the block with a `KeyError` from a
            private module, and checking only the rotation that was valid would
            warn about a configuration that is rejected anyway.
    """
    enc_axis = _AXIS.get(enc_rotation)
    var_axis = _AXIS.get(var_rotation)
    if enc_axis is None or var_axis is None:
        return None
    is_basic = variational_type == "basic"

    if enc_axis == "Z" and var_axis == "Z" and is_basic:
        return (
            "every gate is diagonal (RZ encoding, RZ rotations, CNOT ladder), so "
            "the state never leaves |0...0> and the output is a constant. Encode "
            'on "RX" or "RY", and rotate off the measurement axis.'
        )

    if enc_axis == "Z" and n_encodings == 1:
        return (
            "the circuit starts in |0...0>, which every RZ leaves alone up to a "
            'phase, so the input never reaches the measurement. Use enc_rotation="RX" '
            'or "RY".'
        )

    if is_basic and var_axis == basis:
        ladder_cancels = basis in _LADDER_PRESERVES or n_qubits == 1
        encoding_rescues = n_encodings > 1 and enc_axis != basis
        if ladder_cancels and not encoding_rescues:
            return (
                f"the variational block commutes with the <{basis}> observable "
                f"({var_rotation} rotations and the CNOT ladder both preserve "
                f"{basis}-strings), so the weights cancel out of the expectation "
                "value and every gradient is exactly zero. Rotate off the "
                'measurement axis, or use variational_type="strong".'
            )

    if is_basic and enc_axis == "Y" and var_axis == "Y" and basis == "Y":
        return (
            "RY and CNOT are real matrices, so the statevector stays real and "
            "<Y> is identically zero on it. Measure in another basis, or move the "
            "encoding or the rotation off the Y axis."
        )

    return None


def warn_if_degenerate(
    *,
    layer_name: str,
    enc_rotation: str,
    var_rotation: str,
    variational_type: str,
    basis: str,
    n_qubits: int,
    n_encodings: int,
    stacklevel: int = 3,
) -> None:
    """Issue a `UserWarning` if this configuration cannot train.

    The layer is still built. The caller decides which circuit to run, and there
    are valid reasons to build a degenerate one on purpose: it is exactly the
    baseline for checking whether the quantum part contributes anything.
    """
    reason = degenerate_reason(
        enc_rotation=enc_rotation,
        var_rotation=var_rotation,
        variational_type=variational_type,
        basis=basis,
        n_qubits=n_qubits,
        n_encodings=n_encodings,
    )
    if reason is None:
        return
    warnings.warn(
        f"{layer_name}(enc_rotation={enc_rotation!r}, var_rotation={var_rotation!r}, "
        f"variational_type={variational_type!r}, basis={basis!r}) is degenerate: "
        f"{reason}",
        UserWarning,
        stacklevel=stacklevel,
    )
