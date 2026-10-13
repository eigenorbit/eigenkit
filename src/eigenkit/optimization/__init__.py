"""Combinatorial optimisation with the core of the library.

Write a problem as an `Ising` or `QUBO` model, then solve it with `QAOA`, the
Quantum Approximate Optimization Algorithm. The ansatz uses the gates of the core,
the angles are optimised with `minimize_expectation`, and the solution is read from
the distribution of bitstrings of the optimised state. The package uses only NumPy
and SciPy, needs no PyTorch, and works with any backend.

Example:
    ```python
    from eigenkit.optimization import max_cut, QAOA

    problem = max_cut([(0, 1), (1, 2), (2, 0), (2, 3)])
    result = QAOA(layers=2).solve(problem)
    print(result.best_bitstring, result.best_energy)
    ```
"""

from .problems import QUBO, Ising, max_cut
from .qaoa import QAOA, OptimizationResult, qaoa_ansatz


__all__ = [
    "Ising",
    "QUBO",
    "max_cut",
    "QAOA",
    "OptimizationResult",
    "qaoa_ansatz",
]
