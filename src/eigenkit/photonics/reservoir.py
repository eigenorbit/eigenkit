"""Photonic reservoir computing, as a quantum extreme learning machine.

The reservoir is a *fixed* linear-optical interferometer, which is random and never
trained. At each timestep, the input is encoded into its phases, the photons pass
through it, and photon-number statistics are read out. This gives a nonlinear map
to many features. Only a classical linear readout is trained, so the executor only
has to evaluate the circuit, and no gradients go through the optics.

The parts that do not depend on the kind of hardware come from the qubit reservoir
package, `eigenkit.reservoir`. This module uses its `Stateless` policy, which gives
temporal context with a classical window of past inputs, and the readout can be
that package's `LinearReadout`, a ridge regression solved in closed form. The part
specific to photonics turns the input of each timestep into a vector of features
from a Fock distribution, with a photonic `Executor`.

Scope: no memory, or a classical window of inputs (`Stateless`). Carrying the
photonic *state* from one timestep to the next (a quantum memory, like the qubit
`Carry` policy) needs the executor to accept a Fock state as input, and is not
supported yet. Policies that carry state raise an error here.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from ..core.ir import InputRef, Scale, WeightRef
from ..reservoir import Stateless
from .backends.perceval_backend import PercevalBackend
from .ir import BeamSplitter, PhaseShifter, PhotonicCircuitSpec


def fock_basis(n_modes: int, n_photons: int) -> list[tuple[int, ...]]:
    """Return the sorted list of output Fock states.

    The list holds every way to place `n_photons` photons into `n_modes` modes.
    Passive linear optics conserves the number of photons, so these are exactly the
    outputs that the circuit can reach. They are the fixed columns of the feature
    vector.

    Args:
        n_modes (int): The number of optical modes.
        n_photons (int): The total number of photons.
    """
    # combinations_with_replacement yields each multiset of modes once, so the
    # count tuples are already distinct.
    states = []
    for combo in combinations_with_replacement(range(n_modes), n_photons):
        counts = [0] * n_modes
        for mode in combo:
            counts[mode] += 1
        states.append(tuple(counts))
    return sorted(states)


def fock_probability_features(dist: dict, basis: list[tuple[int, ...]]) -> np.ndarray:
    """Return the probability of every state in `basis`, as a dense vector.

    These are the detailed features of the reservoir.

    Args:
        dist (dict): A Fock distribution, as `{photon_counts_tuple: probability}`.
        basis (list[tuple[int, ...]]): The output states, in the order of the
            features, for example from `fock_basis`.
    """
    return np.array([dist.get(state, 0.0) for state in basis], dtype=np.float64)


def photon_number_features(dist: dict, n_modes: int) -> np.ndarray:
    """Return the expected number of photons `<n̂_m>` in each mode `m`.

    This readout is compact, and easy to measure with detectors.

    Args:
        dist (dict): A Fock distribution, as `{photon_counts_tuple: probability}`.
        n_modes (int): The number of optical modes.
    """
    feat = np.zeros(n_modes, dtype=np.float64)
    for state, prob in dist.items():
        for m in range(n_modes):
            feat[m] += prob * state[m]
    return feat


def random_reservoir_spec(
    n_modes: int,
    n_inputs: int,
    *,
    depth: int = 2,
    input_state: tuple[int, ...] | None = None,
    input_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[PhotonicCircuitSpec, np.ndarray]:
    """Build a fixed, random photonic reservoir and its frozen weights.

    A mixing layer comes first, so that the photons are spread over the modes. Then
    the `n_inputs` features are encoded as phases on the modes `0` to
    `n_inputs - 1` (each multiplied by `input_scale`), and `depth` more mixing
    layers follow. A mixing layer is a beam splitter on every pair of neighbouring
    modes, followed by a phase shifter on every mode. All their angles are random
    weights, drawn uniformly from `[0, 2π)`.

    Args:
        n_modes (int): The number of optical modes. It must be at least `2`.
        n_inputs (int): The number of features encoded at each step, with one phase
            each. It must be at least `1` and at most `n_modes - 1`.
        depth (int): The number of mixing layers after the encoding. It must be at
            least `1`. Default: `2`.
        input_state (tuple[int, ...] | None): The Fock state sent into the circuit.
            The default is a single photon in mode `0`. More photons give a larger
            space of Fock features.
        input_scale (float): The factor applied to each input before it is used as
            a phase. Default: `1.0`.
        rng (np.random.Generator | None): The source of the random weights of the
            reservoir. The default is a new, unseeded generator.

    Returns:
        spec (PhotonicCircuitSpec): The circuit of the reservoir.
        weights (np.ndarray): The **frozen** weights of the reservoir, which are
            never trained.

    Raises:
        ValueError: If `n_modes < 2`, `depth < 1`, `n_inputs` is outside
            `[1, n_modes - 1]`, or `input_state` holds no photon. In each of these
            cases, the features of the reservoir would ignore the input, or, when
            every mode encodes an input, would see only the differences between
            inputs.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_modes < 2:
        # With one mode there is no beam splitter to mix with, so the input phases
        # never reach a Fock-diagonal readout: the reservoir would ignore its input.
        raise ValueError(
            f"n_modes must be >= 2 (a reservoir needs mixing), got {n_modes}"
        )
    if input_state is None:
        input_state = (1,) + (0,) * (n_modes - 1)
    if sum(input_state) < 1:
        raise ValueError(f"input_state must carry at least one photon, got {input_state}")
    if not 1 <= n_inputs < n_modes:
        # Photon number is conserved, so a phase shift common to every mode is a
        # global phase: with all modes encoded the features see only input
        # differences, and with none they ignore the input.
        raise ValueError(
            f"n_inputs={n_inputs} must be in [1, n_modes - 1] (n_modes={n_modes})"
        )
    if depth < 1:
        # The input is encoded as phase shifts at the end of layer 0; phase shifters
        # are diagonal in the Fock basis, so without at least one *subsequent* mixing
        # layer they only add a global phase and the readout becomes input-independent.
        raise ValueError(f"depth must be >= 1 (needs mixing after encoding), got {depth}")

    components: list = []
    n_weights = 0
    # `depth + 1` random mixing layers, encoding the input *after the first* layer:
    # phase-encoding the definite-occupation input modes before any mixing applies only a
    # global phase (no observable effect), so the photons must be spread across modes first.
    for layer in range(depth + 1):
        for m in range(n_modes - 1):
            components.append(BeamSplitter(modes=(m, m + 1), theta=WeightRef(n_weights)))
            n_weights += 1
        for m in range(n_modes):
            components.append(PhaseShifter(mode=m, phi=WeightRef(n_weights)))
            n_weights += 1
        if layer == 0:
            for j in range(n_inputs):
                components.append(
                    PhaseShifter(mode=j, phi=Scale(InputRef(j), input_scale))
                )

    spec = PhotonicCircuitSpec(
        n_modes=n_modes,
        n_weights=n_weights,
        n_inputs=n_inputs,
        components=tuple(components),
        input_state=tuple(input_state),
    )
    weights = rng.uniform(0.0, 2.0 * np.pi, size=n_weights)
    return spec, weights


def run_photonic_reservoir(
    spec: PhotonicCircuitSpec,
    weights,
    sequences,
    *,
    executor=None,
    policy=None,
    features: str = "fock",
) -> np.ndarray:
    """Run the fixed reservoir over input sequences, and return the features.

    The reservoir `spec`, with its frozen `weights`, is evaluated at every timestep
    of `sequences`, of shape `(B, T, F)`. The result is the sequence of features, of
    shape `(B, T, n_features)`.

    Args:
        spec (PhotonicCircuitSpec): The reservoir, for example from
            `random_reservoir_spec`.
        weights (np.ndarray): The frozen weights of the reservoir.
        sequences (np.ndarray): The input sequences, of shape `(B, T, F)`.
        executor (Executor | None): The executor that runs each step. The default
            is `PercevalBackend()`.
        policy (Stateless | None): The policy whose `expand_inputs` turns each
            timestep into the circuit's `n_inputs` values (`window * F`). The
            default is `Stateless(window=1)`. Policies that carry the state from one
            step to the next are not supported for photonics yet, because they need
            a Fock state as the input of the executor.
        features (str): `"fock"` gives the probabilities of all the states of the
            Fock basis with the input number of photons. This is the default, and
            gives the most features. `"photon_number"` gives the expected number of
            photons `<n̂_m>` in each mode.

    Returns:
        The features, of shape `(B, T, n_features)`. `n_features` is the number of
            Fock states with `n_modes` modes and `spec.n_photons` photons for
            `"fock"`, and `n_modes` for `"photon_number"`.

    Raises:
        ValueError: If `features` is unknown, `sequences` is not 3-D, or the number
            of inputs per step that the policy gives does not match `spec.n_inputs`.
        NotImplementedError: If `policy` carries state from one step to the next.
    """
    # Checked before anything runs: on a cloud executor every sample is a paid job.
    if features not in ("fock", "photon_number"):
        raise ValueError(f"unknown features={features!r} (use 'fock'|'photon_number')")
    executor = executor or PercevalBackend()
    policy = policy or Stateless(window=1)
    if not hasattr(policy, "expand_inputs"):
        raise NotImplementedError(
            "photonic reservoirs currently support only stateless/windowed policies "
            "(quantum state carry-over needs Fock-state injection, not yet implemented)"
        )

    sequences = np.asarray(sequences, dtype=np.float64)
    if sequences.ndim != 3:
        raise ValueError(f"sequences must be 3-D (B, T, F), got ndim={sequences.ndim}")
    weights = np.asarray(weights, dtype=np.float64)
    batch, timesteps, _ = sequences.shape
    # The Fock basis has C(n_modes + N - 1, N) states; build it only when it is read.
    basis = fock_basis(spec.n_modes, spec.n_photons) if features == "fock" else None
    n_features = spec.n_modes if basis is None else len(basis)
    if batch * timesteps == 0:
        return np.zeros((batch, timesteps, n_features), dtype=np.float64)

    inputs_bt = np.asarray(policy.expand_inputs(sequences), dtype=np.float64)
    if inputs_bt.shape[-1] != spec.n_inputs:
        raise ValueError(
            f"policy produced {inputs_bt.shape[-1]} inputs/step but spec.n_inputs="
            f"{spec.n_inputs}"
        )

    compiled = executor.compile(spec)
    flat = inputs_bt.reshape(batch * timesteps, spec.n_inputs)
    rows = np.empty((batch * timesteps, n_features), dtype=np.float64)
    for row, x in enumerate(flat):
        dist = compiled.run(weights, x)
        if features == "fock":
            rows[row] = fock_probability_features(dist, basis)
        else:
            rows[row] = photon_number_features(dist, spec.n_modes)
    return rows.reshape(batch, timesteps, n_features)
