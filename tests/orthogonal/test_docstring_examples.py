"""Every example in a docstring of the orthogonal module runs as written, in a
fresh namespace, as a reader would copy it."""

import importlib
import re
import textwrap

import pytest


pytest.importorskip("torch")

MODULES = [
    "eigenkit.orthogonal",
    "eigenkit.orthogonal.topology",
    "eigenkit.orthogonal.matrix",
    "eigenkit.orthogonal.blocks",
    "eigenkit.orthogonal.circuit",
    "eigenkit.orthogonal.torch",
    "eigenkit.orthogonal.torch.modules",
    "eigenkit.orthogonal.torch.functional",
    "eigenkit.orthogonal.torch.adapters",
    "eigenkit.orthogonal.torch.robust",
    "eigenkit.torch.layers.orthogonal_qnn",
]


def _examples():
    out = []
    for name in MODULES:
        module = importlib.import_module(name)
        docs = [(name, module.__doc__ or "")]
        for attr, obj in vars(module).items():
            if getattr(obj, "__module__", None) == name and obj.__doc__:
                docs.append((f"{name}.{attr}", obj.__doc__))
        for where, doc in docs:
            for k, block in enumerate(re.findall(r"```python\n(.*?)```", doc, re.S)):
                out.append(pytest.param(block, id=f"{where}#{k}"))
    return out


@pytest.mark.parametrize("code", _examples())
def test_the_example_runs(code):
    exec(compile(textwrap.dedent(code), "<docstring>", "exec"), {})
