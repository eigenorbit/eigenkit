"""Run a photonic circuit on Quandela Cloud (real hardware or a cloud simulator).

Set your API key first — it is read from the environment (or the token saved with
``pcvl.RemoteConfig``), never hardcoded:

    export QUANDELA_TOKEN=...            # your Quandela Cloud token
    export QUANDELA_PLATFORM=sim:slos    # or e.g. qpu:ascella for real hardware
    python examples/quandela_cloud_hardware.py

Submits the Hong-Ou-Mandel circuit, samples it on the cloud (async submit + poll), and
prints the sampled Fock distribution next to the exact local SLOS result.
"""

import math
import os

import numpy as np

from eigenkit.core.ir import Constant
from eigenkit.photonics import (
    BeamSplitter,
    PercevalBackend,
    PhotonicCircuitSpec,
    QuandelaCloudBackend,
)


def main() -> None:
    platform = os.environ.get("QUANDELA_PLATFORM", "sim:slos")

    spec = PhotonicCircuitSpec(
        n_modes=2,
        n_weights=0,
        n_inputs=0,
        components=(BeamSplitter(modes=(0, 1), theta=Constant(math.pi / 2)),),
        input_state=(1, 1),
    )
    no_params = np.array([])

    compiled = QuandelaCloudBackend(platform, shots=20000).compile(spec)
    print("dry run:", compiled.dry_run_info())
    print(f"submitting Hong-Ou-Mandel to Quandela Cloud [{platform}] ...")
    handle = compiled.submit(no_params, no_params)
    print("job status:", handle.status())
    sampled = handle.result()  # blocks until the cloud job completes

    exact = PercevalBackend().compile(spec).run(no_params, no_params)
    print("\n fock state    sampled       exact")
    for state in sorted(set(sampled) | set(exact)):
        print(
            f"  {state}       {sampled.get(state, 0.0):.4f}       {exact.get(state, 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
