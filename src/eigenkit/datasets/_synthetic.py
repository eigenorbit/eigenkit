"""
Synthetic dataset generators for eigenkit.
"""

from __future__ import annotations

import numpy as np


def generate_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two half-circles that interleave, called moons.

    Args:
        n_samples: The number of points to generate.
        noise: The standard deviation of the Gaussian noise added to the data.
        random_state: The random seed, for reproducible results.

    Returns:
        The pair `(X, y)`, with `X` of shape `(n_samples, 2)` and `y` of shape
            `(n_samples,)`.
    """
    from sklearn.datasets import make_moons

    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def generate_circles(
    n_samples: int = 200,
    noise: float = 0.05,
    factor: float = 0.5,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two concentric circles.

    Args:
        n_samples: The number of points to generate.
        noise: The standard deviation of the Gaussian noise added to the data.
        factor: The ratio of the inner circle to the outer circle, between `0` and
            `1`.
        random_state: The random seed, for reproducible results.

    Returns:
        The pair `(X, y)`, with `X` of shape `(n_samples, 2)` and `y` of shape
            `(n_samples,)`.
    """
    from sklearn.datasets import make_circles

    return make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=random_state
    )


def generate_checkerboard(
    n_samples: int = 200,
    grid_size: int = 2,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset with a checkerboard pattern.

    The points are drawn uniformly in `[0, grid_size]²`. The label is
    `(floor(x) + floor(y)) % 2`, which gives regions of alternating classes.

    Args:
        n_samples: The number of points to generate.
        grid_size: The number of cells along each axis.
        random_state: The random seed, for reproducible results.

    Returns:
        The pair `(X, y)`, with `X` of shape `(n_samples, 2)` and `y` of shape
            `(n_samples,)`.
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, grid_size, size=(n_samples, 2))
    y = (np.floor(X[:, 0]).astype(int) + np.floor(X[:, 1]).astype(int)) % 2
    return X, y


def _zz_feature_map_spec(n_qubits: int, reps: int = 2):
    """Qiskit's ZZFeatureMap as a `CircuitSpec`: per repetition, a Hadamard
    layer, `P(2 x_q)` on every qubit, and `CX P(2 (pi - x_a)(pi - x_b)) CX` on
    every pair."""
    from ..core.builder import CircuitBuilder
    from ..core.ir import Constant, Multiply, Negate, Scale, Sum

    qc = CircuitBuilder(n_qubits)
    x = qc.add_inputs(n_qubits)
    pi = float(np.pi)
    for _ in range(reps):
        for q in range(n_qubits):
            qc.h(q)
            qc.phase_shift(Scale(x[q], 2.0), qubit=q)
        for a in range(n_qubits):
            for b in range(a + 1, n_qubits):
                angle = Scale(
                    Multiply(
                        Sum((Constant(pi), Negate(x[a]))),
                        Sum((Constant(pi), Negate(x[b]))),
                    ),
                    2.0,
                )
                qc.cnot(a, b)
                qc.phase_shift(angle, qubit=b)
                qc.cnot(a, b)
    return qc.build()


def generate_ad_hoc(
    n_samples: int = 200,
    n_qubits: int = 2,
    gap: float = 0.3,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the "ad hoc" dataset of Havlíček et al., Nature 567, 209 (2019).

    The dataset is designed so that a quantum kernel separates it. Each point `x` in
    `[0, 2π)^n` is encoded with the ZZ feature map, with two repetitions as in
    Qiskit's `ZZFeatureMap`. Its label is the sign of `⟨ψ(x)| V† Z^{⊗n} V |ψ(x)⟩`
    for a fixed random unitary `V`. The points closer than `gap` to the decision
    boundary are dropped, and the two classes are balanced. A kernel built from the
    same ZZ feature map classifies the data almost perfectly, and classical kernels
    usually do not.

    Args:
        n_samples: The number of points, split evenly between the two classes.
        n_qubits: The number of features, which is also the number of qubits of the
            circuit that generates the data.
        gap: The margin around the decision boundary, in `[0, 1)`.
        random_state: The seed of the unitary `V` and of the sampled points.

    Returns:
        The pair `(X, y)`, with `X` of shape `(n_samples, n_qubits)` and `y` of
            shape `(n_samples,)`. The labels are `0` and `1`.

    Raises:
        ValueError: If `gap` is not in `[0, 1)`, or if `n_qubits < 1`.
    """
    if not 0 <= gap < 1:
        raise ValueError(f"gap must be in [0, 1), got {gap}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    from scipy.stats import unitary_group

    from ..core.circuit import Circuit

    rng = np.random.default_rng(random_state)
    dim = 2**n_qubits
    V = unitary_group.rvs(dim, random_state=int(rng.integers(2**31)))
    parity = np.array([(-1) ** bin(i).count("1") for i in range(dim)], dtype=float)
    measured = V.conj().T @ np.diag(parity) @ V
    feature_map = Circuit(_zz_feature_map_spec(n_qubits), provider="numpy")

    wanted = (n_samples - n_samples // 2, n_samples // 2)
    kept: tuple[list, list] = ([], [])
    for _ in range(10_000):
        X = rng.uniform(0.0, 2.0 * np.pi, size=(max(256, n_samples), n_qubits))
        psi = feature_map.statevector(np.empty(0), X)
        value = np.real(np.einsum("bi,ij,bj->b", psi.conj(), measured, psi))
        for x, v in zip(X, value):
            if abs(v) <= gap:
                continue
            label = int(v > 0)
            if len(kept[label]) < wanted[label]:
                kept[label].append(x)
        if all(len(kept[c]) >= wanted[c] for c in (0, 1)):
            break
    else:  # pragma: no cover - needs an extreme gap
        raise ValueError("could not draw enough points; lower `gap`")
    X = np.array(kept[0] + kept[1])
    y = np.array([0] * wanted[0] + [1] * wanted[1])
    order = rng.permutation(len(y))
    return X[order], y[order]


def generate_regression(
    n_samples: int = 200,
    n_features: int = 4,
    noise: float = 0.1,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset with scikit-learn's `make_regression`.

    Args:
        n_samples: The number of samples to generate.
        n_features: The number of input features.
        noise: The standard deviation of the Gaussian noise on the output.
        random_state: The random seed, for reproducible results.

    Returns:
        The pair `(X, y)`, with `X` of shape `(n_samples, n_features)` and `y` of
            shape `(n_samples,)`, both of type `float32`.
    """
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.float32)


def generate_timeseries(
    n_samples: int = 300,
    kind: str = "sine",
    noise: float = 0.05,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 1-D time series, as the pair `(t, y)`.

    Args:
        n_samples: The number of time steps.
        kind: The shape of the series: `"sine"` for a sine wave with noise, or
            `"ar"` for an autoregressive process of order `1`, written AR(1).
        noise: The standard deviation of the Gaussian noise.
        random_state: The random seed, for reproducible results.

    Returns:
        The pair `(t, y)`, two arrays of shape `(n_samples,)` and type `float32`.
            `t` holds time values spaced evenly in `[0, 1]`, and `y` is the signal.
            The arrays are not cut into windows, because the algorithms do that.
    """
    if kind not in ("sine", "ar"):
        raise ValueError(f"Unknown kind: {kind!r}. Expected 'sine' or 'ar'")

    rng = np.random.default_rng(random_state)
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)

    if kind == "sine":
        y = np.sin(2 * np.pi * t) + rng.normal(0, noise, size=n_samples).astype(
            np.float32
        )
    else:  # "ar"
        noise_vals = rng.normal(0, noise, size=n_samples).astype(np.float32)
        y = np.zeros(n_samples, dtype=np.float32)
        for i in range(1, n_samples):
            y[i] = 0.9 * y[i - 1] + noise_vals[i]

    return t, y


GENERATORS: dict[str, callable] = {
    "make_moons": generate_moons,
    "make_circles": generate_circles,
    "make_checkerboard": generate_checkerboard,
    "generate_regression": generate_regression,
    "generate_sine_wave": lambda random_state=42: generate_timeseries(
        kind="sine", random_state=random_state
    ),
    "generate_ar_process": lambda random_state=42: generate_timeseries(
        kind="ar", random_state=random_state
    ),
}
