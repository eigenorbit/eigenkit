# References

This page collects the papers that the algorithm pages cite. The original papers state
the formal guarantees. The reviews and tutorials are there to build intuition.

## IQAE: amplitude estimation

These papers cover iterative quantum amplitude estimation (IQAE).

- **The original paper.** D. Grinko, J. Gacon, C. Zoufal, S. Woerner, *Iterative Quantum
  Amplitude Estimation*, npj Quantum Information (2021),
  [arXiv:1912.05559](https://arxiv.org/abs/1912.05559). This library follows its adaptive
  schedule and its construction of the confidence interval.
- **The background.** G. Brassard, P. Høyer, M. Mosca, A. Tapp, *Quantum Amplitude
  Amplification and Estimation* (2000),
  [arXiv:quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055). This paper gives the
  Grover operator and the formulation `a = sin²θ`.
- **The limitations.** Y. Suzuki et al., *Amplitude estimation without phase estimation*
  (2019), [arXiv:1904.10246](https://arxiv.org/abs/1904.10246); K. Miyamoto, *On the bias
  in iterative quantum amplitude estimation* (2023),
  [arXiv:2311.16560](https://arxiv.org/abs/2311.16560). These two papers discuss the
  limitations of the method.

## HHL: linear systems

These papers cover the Harrow-Hassidim-Lloyd (HHL) algorithm for systems of linear
equations.

- **The original paper.** A. Harrow, A. Hassidim, S. Lloyd, *Quantum algorithm for linear
  systems of equations*, Phys. Rev. Lett. (2009),
  [arXiv:0811.3171](https://arxiv.org/abs/0811.3171).
- **A review.** D. Dervovic et al., *Quantum linear systems algorithms: a primer* (2018),
  [arXiv:1802.08227](https://arxiv.org/abs/1802.08227).
- **A walkthrough.** A. Morrell et al., *Step-by-Step HHL Algorithm Walkthrough* (2021),
  [arXiv:2108.09004](https://arxiv.org/abs/2108.09004). It works through a numerical
  example at the level of detail that this implementation uses.
- **The limitations and later work.** A. Childs, R. Kothari, R. Somma, *Quantum algorithm
  for systems of linear equations with exponentially improved dependence on precision*
  (2017), [arXiv:1511.02306](https://arxiv.org/abs/1511.02306);
  M. E. S. Morales et al., *Quantum Linear System Solvers: A Survey of Algorithms and
  Applications* (2024), [arXiv:2411.02522](https://arxiv.org/abs/2411.02522).
- **Hamiltonian simulation.** These papers concern the evolution `e^{iAt}` that the
  `"honest"` mode builds. S. Lloyd, *Universal Quantum Simulators*, Science (1996),
  [DOI:10.1126/science.273.5278.1073](https://doi.org/10.1126/science.273.5278.1073);
  A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, S. Zhu, *Theory of Trotter Error with
  Commutator Scaling*, Phys. Rev. X 11, 011020 (2021),
  [DOI:10.1103/PhysRevX.11.011020](https://doi.org/10.1103/PhysRevX.11.011020). The
  preprint of the second paper, [arXiv:1912.08854](https://arxiv.org/abs/1912.08854), has
  the title *A Theory of Trotter Error*. Its Theorem 11 establishes the form of the error
  bound with commutator scaling, which `trotter_steps` and `trotter_order` rely on. This
  library writes that bound for Pauli terms, where every commutator norm is `0` or `2`.
  The theorem is stated only up to an order of magnitude, so the explicit constants that
  this library uses do not come from it. Instead, tests fix them by comparing the bound
  with `‖A_eff - A‖`, computed exactly.
- **The cost of the Pauli decomposition.** There are `4ⁿ` Pauli strings on `n` qubits, so
  a generic matrix has `Θ(4ⁿ)` terms. For the decomposition itself, L. Hantzko,
  L. Binkowski, S. Gupta, *Tensorized Pauli decomposition algorithm*, Phys. Scr. 99,
  085128 (2024), [arXiv:2310.13421](https://arxiv.org/abs/2310.13421), works by slicing
  and adding matrices instead of multiplying them. For tridiagonal matrices, the exact
  count is `(n+1)2ⁿ` in general (Proposition 1) and `(n+2)2ⁿ⁻¹` when the matrix is real
  and symmetric (Corollary 2). Both counts are in B. Arseniev, D. Guskov, R. Sengupta,
  J. Biamonte, I. Zacharov, *Tridiagonal matrix decomposition for Hamiltonian simulation
  on a quantum computer*, Phys. Rev. A 109, 052629 (2024),
  [arXiv:2310.00121](https://arxiv.org/abs/2310.00121). The second count gives the 576
  terms that the [scaling table](hhl.md#guarantees-and-limits) reports at `n = 7`. The
  following paper explains why the Pauli input model, which is a linear combination of
  unitaries (LCU), and the sparse-access input model are *distinct*: A. Gilyén, Y. Su,
  G. H. Low, N. Wiebe, *Quantum singular value transformation and beyond: exponential
  improvements for quantum matrix arithmetics*, STOC 2019,
  [arXiv:1806.01838](https://arxiv.org/abs/1806.01838).
- **The eigenvalue inversion as arithmetic.** The inversion would become a uniform family
  of circuits if the angle were computed from the clock register with reversible
  arithmetic, instead of being tabulated bucket by bucket. Circuits for the functions
  involved, including the arcsine, are in T. Häner, M. Roetteler, K. M. Svore,
  *Optimizing Quantum Circuits for Arithmetic* (2018),
  [arXiv:1805.12445](https://arxiv.org/abs/1805.12445).
- **The eigenvalue inversion as elementary gates.** M. Möttönen, J. J. Vartiainen,
  V. Bergholm, M. M. Salomaa, *Transformation of quantum states using uniformly
  controlled rotations*, Quantum Inf. Comput. 5, 467 (2005),
  [quant-ph/0407010](https://arxiv.org/abs/quant-ph/0407010). This paper gives the
  uniformly controlled rotation and its transform of the angles. The inversion stage of
  the `"honest"` mode is built from them, with `2^(n_l)` `RY` gates and `2^(n_l)`
  controlled-NOT (`CNOT`) gates. It needs no ancillas, and the assembled circuit contains
  no multi-controlled primitive anywhere.
- **The fine print.** S. Aaronson, *Read the fine print*, Nature Physics (2015),
  [DOI:10.1038/nphys3272](https://doi.org/10.1038/nphys3272). The preprint has the title
  *Quantum Machine Learning Algorithms: Read the Fine Print*. It names four caveats:
  preparing `|b⟩`, applying `e^{-iAt}`, the condition number, and the fact that the
  output is a state and not a vector.
- **Dense matrices.** L. Wossnig, Z. Zhao, A. Prakash, *A quantum linear system algorithm
  for dense matrices*, Phys. Rev. Lett. 120, 050502 (2018),
  [arXiv:1704.06174](https://arxiv.org/abs/1704.06174). For a dense matrix with a bounded
  spectral norm, it reports a quadratic speedup over the earlier quantum approaches. The
  method is built on singular value estimation and a specialised memory architecture.
  That machinery is out of the scope of this library.

## QFT: the quantum Fourier transform

These papers cover the quantum Fourier transform (QFT) and quantum phase estimation.

- **The original paper.** D. Coppersmith, *An approximate Fourier transform useful in
  quantum factoring*, IBM Research Report RC19642 (1994),
  [arXiv:quant-ph/0201067](https://arxiv.org/abs/quant-ph/0201067). It gives the circuit
  of Hadamard and controlled-phase gates that this block builds, and the approximate
  variant that drops the smallest rotations.
- **The paper that made it known.** P. W. Shor, *Polynomial-Time Algorithms for Prime
  Factorization and Discrete Logarithms on a Quantum Computer*, SIAM J. Comput. (1997),
  [arXiv:quant-ph/9508027](https://arxiv.org/abs/quant-ph/9508027).
- **Phase estimation.** A. Yu. Kitaev, *Quantum measurements and the Abelian Stabilizer
  Problem* (1995), [arXiv:quant-ph/9511026](https://arxiv.org/abs/quant-ph/9511026);
  R. Cleve, A. Ekert, C. Macchiavello, M. Mosca, *Quantum Algorithms Revisited*, Proc. R.
  Soc. Lond. A (1998),
  [arXiv:quant-ph/9708016](https://arxiv.org/abs/quant-ph/9708016). These papers give the
  construction that `QPEBlock` implements.
- **The limitations.** A. Barenco, A. Ekert, K.-A. Suominen, P. Törmä, *Approximate
  Quantum Fourier Transform and Decoherence* (1996),
  [arXiv:quant-ph/9601018](https://arxiv.org/abs/quant-ph/9601018), which studies what
  happens to the smallest phase rotations; Y. Nam, Y. Su, D. Maslov, *Approximate Quantum
  Fourier Transform with `O(n log n)` T gates* (2020),
  [arXiv:1803.04933](https://arxiv.org/abs/1803.04933).

## QAOA: combinatorial optimisation

These papers cover the Quantum Approximate Optimization Algorithm (QAOA).

- **The original paper.** E. Farhi, J. Goldstone, S. Gutmann, *A Quantum Approximate
  Optimization Algorithm* (2014), [arXiv:1411.4028](https://arxiv.org/abs/1411.4028).
- **A tutorial and a review.** A. Sturm, *Theory and Implementation of the Quantum
  Approximate Optimization Algorithm* (2023),
  [arXiv:2301.09535](https://arxiv.org/abs/2301.09535); K. Blekos et al., *A Review on
  Quantum Approximate Optimization Algorithm and its Variants* (2023),
  [arXiv:2306.09198](https://arxiv.org/abs/2306.09198).
- **The limitations.** M. Goh, *The Overlap Gap Property limits limit swapping in the
  QAOA* (2024), [arXiv:2404.06087](https://arxiv.org/abs/2404.06087).

## VQE: the variational quantum eigensolver

These papers cover the Variational Quantum Eigensolver (VQE).

- **The original paper.** A. Peruzzo, J. McClean et al., *A variational eigenvalue solver
  on a quantum processor*, Nature Communications (2014),
  [arXiv:1304.3061](https://arxiv.org/abs/1304.3061).
- **The hybrid framework.** J. R. McClean, J. Romero, R. Babbush, A. Aspuru-Guzik, *The
  theory of variational hybrid quantum-classical algorithms*, New J. Phys. (2016),
  [arXiv:1509.04279](https://arxiv.org/abs/1509.04279).
- **The mappings from fermions to qubits.** P. Jordan, E. Wigner, *Über das Paulische
  Äquivalenzverbot*, Zeitschrift für Physik 47 (1928), 631–651, which gives the original
  transformation; S. Bravyi, A. Kitaev, *Fermionic quantum computation* (2000),
  [arXiv:quant-ph/0003137](https://arxiv.org/abs/quant-ph/0003137); J. T. Seeley,
  M. J. Richard, P. J. Love, *The Bravyi-Kitaev transformation for quantum computation of
  electronic structure* (2012), [arXiv:1208.5986](https://arxiv.org/abs/1208.5986).
- **The ansatze.** J. Romero et al., *Strategies for quantum computing molecular energies
  using the unitary coupled cluster ansatz* (2018),
  [arXiv:1701.02691](https://arxiv.org/abs/1701.02691), is the reference for the unitary
  coupled cluster ansatz with single and double excitations (UCCSD). A. Kandala et al.,
  *Hardware-efficient Variational Quantum Eigensolver for Small Molecules and Quantum
  Magnets*, Nature (2017), [arXiv:1704.05018](https://arxiv.org/abs/1704.05018), is the
  reference for the hardware-efficient family.
- **A review and the limitations.** J. Tilly et al., *The Variational Quantum
  Eigensolver: a review of methods and best practices*, Physics Reports (2022),
  [arXiv:2111.05176](https://arxiv.org/abs/2111.05176). The papers on barren plateaus are
  listed under HQNN below.

## QSVC / QSVR: quantum kernels

These papers cover the quantum support vector classifier (QSVC) and the quantum support
vector regressor (QSVR), which use quantum kernels.

- **The original paper.** V. Havlíček et al., *Supervised learning with quantum enhanced
  feature spaces*, Nature (2019), [arXiv:1804.11326](https://arxiv.org/abs/1804.11326).
- **The kernel point of view.** M. Schuld, N. Killoran, *Quantum machine learning in
  feature Hilbert spaces* (2018), [arXiv:1803.07128](https://arxiv.org/abs/1803.07128).
- **The limitations.** S. Thanasilp et al., *Exponential concentration in quantum kernel
  methods* (2022), [arXiv:2208.11060](https://arxiv.org/abs/2208.11060); S. Jerbi et al.,
  *Quantum machine learning beyond kernel methods* (2023),
  [Nature Communications](https://www.nature.com/articles/s41467-023-36159-y).
- **Kernel alignment.** J. R. Glick et al., *Covariant quantum kernels for data with group
  structure*, Nature Physics (2024), [arXiv:2105.03406](https://arxiv.org/abs/2105.03406),
  is the reference for the `"svc"` loss of `align_kernel`. N. Cristianini et al., *On
  kernel-target alignment*, NeurIPS (2001), is the reference for the `"kta"` loss.
- **The optimiser.** J. C. Spall, *Multivariate stochastic approximation using a
  simultaneous perturbation gradient approximation*, IEEE Transactions on Automatic
  Control (1992), is the reference for simultaneous perturbation stochastic approximation
  (SPSA).

## HQNN: hybrid quantum neural networks

These papers cover hybrid quantum neural networks (HQNN).

- **A recent paper.** A. Abbas et al., *The power of quantum neural networks* (2021),
  [arXiv:2011.00027](https://arxiv.org/abs/2011.00027).
- **A historical review.** M. Schuld, I. Sinayskiy, F. Petruccione, *The quest for a
  Quantum Neural Network* (2014), [arXiv:1408.7005](https://arxiv.org/abs/1408.7005).
- **Trainability and data encoding.** J. McClean et al., *Barren plateaus in quantum
  neural network training landscapes* (2018),
  [arXiv:1803.11173](https://arxiv.org/abs/1803.11173); M. Schuld, R. Sweke, J. Meyer,
  *Effect of data encoding on the expressive power of variational quantum machine
  learning models* (2021), [arXiv:2008.08605](https://arxiv.org/abs/2008.08605).

## Quantum reservoir computing

- **The original paper.** K. Fujii, K. Nakajima, *Harnessing disordered quantum dynamics
  for machine learning*, Phys. Rev. Applied (2017),
  [arXiv:1602.08159](https://arxiv.org/abs/1602.08159). It uses quantum dynamics as the
  reservoir, with a trained linear readout.
- **The classical origins.** H. Jaeger, *The "echo state" approach to analysing and
  training recurrent neural networks*, GMD Report 148 (2001); W. Maass, T. Natschläger,
  H. Markram, *Real-time computing without stable states*, Neural Computation 14 (2002),
  2531–2560. Neither of these has an arXiv link, and this is not by accident: both are
  reports, not preprints.
- **Capacity and multiplexing.** K. Nakajima, K. Fujii, M. Negoro, K. Mitarai,
  M. Kitagawa, *Boosting computational power through spatial multiplexing in quantum
  reservoir computing*, Phys. Rev. Applied (2019),
  [arXiv:1803.04574](https://arxiv.org/abs/1803.04574).
- **A review.** P. Mujal, R. Martínez-Peña et al., *Opportunities in Quantum Reservoir
  Computing and Extreme Learning Machines*, Adv. Quantum Technol. (2021),
  [arXiv:2102.11831](https://arxiv.org/abs/2102.11831). It explains why dissipation, and
  not unitarity, is what gives a reservoir a memory that can be used.

## Cross-cutting foundations

These papers are used across several pages.

- **Data re-uploading.** A. Pérez-Salinas et al., *Data re-uploading for a universal
  quantum classifier* (2020), [arXiv:1907.02085](https://arxiv.org/abs/1907.02085). This
  paper is the basis of the `Reupload` layer.
- **Gradients by parameter shift.** K. Mitarai et al., *Quantum circuit learning* (2018),
  [arXiv:1803.00745](https://arxiv.org/abs/1803.00745); M. Schuld et al., *Evaluating
  analytic gradients on quantum hardware* (2019),
  [arXiv:1811.11184](https://arxiv.org/abs/1811.11184). These papers give the gradient
  method used when a backend has no native automatic differentiation.
