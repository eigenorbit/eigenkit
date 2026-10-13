"""Photonic QML: a tiny end-to-end binary classifier on a linear-optics circuit.

Exercises the *data-encoding* path of the photonics ring: 2-D features are encoded
as phase shifts (``InputRef``), a shallow trainable interferometer (``WeightRef``
beam-splitters / phases) mixes them with data re-uploading, and the expected
photon number in mode 0 is read out as the class probability. Training is plain
``torch.optim.Adam`` flowing through the photonic parameter-shift gradient via the
``PhotonicCircuit`` nn.Module — the same path any hybrid photonic QML model uses.

Run: ``python examples/photonics_classification.py``
"""

import numpy as np
import torch

from eigenkit.core.ir import InputRef, WeightRef
from eigenkit.photonics import (
    BeamSplitter,
    PhaseShifter,
    PhotonicCircuit,
    PhotonicCircuitSpec,
    PhotonNumber,
)


def make_spec(n_layers: int = 3) -> PhotonicCircuitSpec:
    """A 2-mode / 1-photon data-reuploading circuit. The two features are encoded
    at non-commuting points (each followed by a trainable beam-splitter) so the
    model depends on both features independently, not just their difference."""
    components = []
    n_weights = 0
    for _ in range(n_layers):
        components.append(PhaseShifter(mode=0, phi=InputRef(0)))
        components.append(BeamSplitter(modes=(0, 1), theta=WeightRef(n_weights)))
        n_weights += 1
        components.append(PhaseShifter(mode=0, phi=InputRef(1)))
        components.append(BeamSplitter(modes=(0, 1), theta=WeightRef(n_weights)))
        n_weights += 1
        components.append(PhaseShifter(mode=0, phi=WeightRef(n_weights)))
        n_weights += 1
    return PhotonicCircuitSpec(
        n_modes=2,
        n_weights=n_weights,
        n_inputs=2,
        components=tuple(components),
        input_state=(1, 0),
    )


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples = 30
    features = rng.uniform(0.0, np.pi, size=(n_samples, 2))
    labels = (features[:, 0] + features[:, 1] > np.pi).astype(float)  # diagonal split

    spec = make_spec(n_layers=3)
    circuit = PhotonicCircuit(
        spec,
        PhotonNumber(mode=0),  # <n̂_0> in [0, 1] read as P(class = 1)
        init_weights=rng.uniform(0.0, np.pi, size=spec.n_weights),
    )
    optimizer = torch.optim.Adam(circuit.parameters(), lr=0.15)

    def accuracy() -> float:
        with torch.no_grad():
            preds = np.array([circuit(torch.tensor(x)).item() for x in features]) > 0.5
        return float((preds == labels.astype(bool)).mean())

    print("epoch   loss     acc")
    for epoch in range(41):
        optimizer.zero_grad()
        loss = (
            sum(
                (circuit(torch.tensor(features[i])) - labels[i]) ** 2
                for i in range(n_samples)
            )
            / n_samples
        )
        if epoch % 8 == 0:
            print(f"{epoch:4d}   {loss.item():.4f}   {accuracy():.2f}")
        loss.backward()
        optimizer.step()
    print(f"final accuracy: {accuracy():.2f}")


if __name__ == "__main__":
    main()
