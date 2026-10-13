"""Shared provider capability checks for algorithm preflight."""

from __future__ import annotations

from ..core.circuit import ProviderSpec
from ..core.ir import CircuitSpec
from ..core.provider import Capabilities, resolve_provider


def gate_adjoint_supported(capabilities: Capabilities, gate_name: str) -> bool:
    """Return whether *gate_name* adjoint is supported on this backend."""
    adjoint_gates = (
        capabilities.supported_adjoint_gates
        if capabilities.supported_adjoint_gates is not None
        else capabilities.supported_gates
    )
    return gate_name in adjoint_gates


def provider_configured_shots(provider: ProviderSpec) -> int | None:
    """Return the provider config shot count, or ``None`` for analytic mode."""
    backend = resolve_provider(provider)
    if not backend.capabilities.supports_shots:
        return None
    shots = getattr(getattr(backend, "config", None), "shots", None)
    if shots in (None, 0):
        return None
    return int(shots)


def check_required_gates(
    provider: ProviderSpec,
    required_gates: frozenset[str],
    *,
    algorithm: str,
    hint: str = "",
) -> None:
    """Raise ``ValueError`` when the backend lacks a required gate."""
    backend = resolve_provider(provider)
    missing = required_gates - backend.capabilities.supported_gates
    if not missing:
        return
    suffix = f" {hint}" if hint else ""
    raise ValueError(
        f"{algorithm} requires gate(s) {sorted(missing)} that are not supported "
        f"by the selected provider '{backend.name}'.{suffix}"
    )


def check_oracle_matrix_extraction(
    provider: ProviderSpec, *, algorithm: str = "IQAE"
) -> None:
    """Raise when a callable oracle cannot be converted via ``qml.matrix``."""
    backend = resolve_provider(provider)
    if backend.capabilities.supports_oracle_matrix:
        return
    raise ValueError(
        f"callable oracle requires provider='pennylane': the selected provider "
        f"'{backend.name}' does not support oracle matrix extraction "
        f"(capabilities.supports_oracle_matrix=True). "
        f"Convert the oracle to a CircuitSpec and pass it directly to "
        f"{algorithm}.solve(), or switch to provider='pennylane'."
    )


def check_amplitude_encoding_adjoint(
    oracle_spec: CircuitSpec,
    provider: ProviderSpec,
    *,
    algorithm: str = "IQAE",
) -> None:
    """Raise when the oracle uses AmplitudeEncoding without adjoint support."""
    if not any(g.name == "AmplitudeEncoding" for g in oracle_spec.gates):
        return
    backend = resolve_provider(provider)
    if gate_adjoint_supported(backend.capabilities, "AmplitudeEncoding"):
        return
    raise ValueError(
        "The oracle circuit contains an AmplitudeEncoding gate. "
        f"The Grover iterate built by {algorithm} applies the oracle adjoint "
        "(A†), which requires adjoint(AmplitudeEncoding). "
        f"The selected provider '{backend.name}' does not declare this support "
        f"in capabilities.supported_adjoint_gates. "
        "Switch to provider='pennylane', or replace AmplitudeEncoding with "
        "QubitUnitary in the oracle (QubitUnitary adjoint is computed in the "
        "IR and works on all backends)."
    )


def check_tail_oracle_provider(provider: ProviderSpec) -> None:
    """Preflight for portfolio tail oracles (AmplitudeEncoding + IQAE gates)."""
    from .iqae import IQAE

    check_required_gates(
        provider,
        IQAE._REQUIRED_GATES,
        algorithm="IQAE",
        hint=(
            "IQAE uses QubitUnitary to wrap the oracle and X / MCZ / Z inside "
            "the Grover iterate."
        ),
    )
    backend = resolve_provider(provider)
    if not gate_adjoint_supported(backend.capabilities, "AmplitudeEncoding"):
        raise ValueError(
            "Portfolio risk estimation builds a tail oracle with "
            "AmplitudeEncoding. The Grover iterate requires "
            "adjoint(AmplitudeEncoding), which the selected provider "
            f"'{backend.name}' does not support. Switch to provider='pennylane', "
            "or call IQAE directly with a QubitUnitary-based CircuitSpec."
        )
