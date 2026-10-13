"""The photonics ring's lazy import boundary.

Deliberately *not* gated on perceval: the leak check is meaningful with or without the
SDK installed, so it runs in every environment, including the full-extras CI job where a
stray top-level ``import perceval`` would otherwise go unnoticed.
"""

import subprocess
import sys


def test_import_surface_is_framework_free():
    # Importing the photonics ring must NOT import perceval or torch (lazy boundary).
    # Compare on the top-level package so our own `...photonics.torch` submodule and
    # `...backends.perceval_backend` (top-level pkg `eigenkit`) don't match.
    code = (
        "import sys, eigenkit.photonics; "
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in ('perceval', 'torch')); "
        "assert not leaked, leaked"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
