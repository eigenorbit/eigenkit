"""The orthogonal package's lazy import boundary.

Not gated on torch: the leak check is meaningful with or without it installed, so it
runs in every environment, including the full-extras CI job.
"""

import subprocess
import sys


def test_import_surface_is_framework_free():
    code = (
        "import sys, eigenkit.orthogonal; "
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('torch', 'pennylane', 'qiskit', 'sklearn', 'perceval')); "
        "assert not leaked, leaked"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
