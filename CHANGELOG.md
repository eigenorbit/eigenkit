# Changelog

<!-- version list -->

## v1.2.2 (2026-11-09)

### Bug Fixes

- Cap qiskit below 3 on Python 3.14 ([#40](https://github.com/eigenorbit/eigenkit/pull/40),
  [`80bfdee`](https://github.com/eigenorbit/eigenkit/commit/80bfdeef849cd41a58d21a88237d37c8035b0d91))

### Documentation

- Cite the eigenkit paper on arXiv ([#42](https://github.com/eigenorbit/eigenkit/pull/42),
  [`c45b6bf`](https://github.com/eigenorbit/eigenkit/commit/c45b6bf9e845dcd8f355cd9efecc035be2b6dd01))

### Continuous Integration

- Run the test suite in parallel with pytest-xdist
  ([#43](https://github.com/eigenorbit/eigenkit/pull/43),
  [`059ac37`](https://github.com/eigenorbit/eigenkit/commit/059ac376513f3eb15fa827a4a983e4d559477e49))


## v1.2.1 (2026-11-05)

### Bug Fixes

- Require eigenorbit-router-client 0.2 in the router extra
  ([#38](https://github.com/eigenorbit/eigenkit/pull/38),
  [`bc0eb8e`](https://github.com/eigenorbit/eigenkit/commit/bc0eb8e955c6d7fc1260c16fd7953a30c5207649))

### Chores

- **release**: Order changelog sections by importance
  ([#37](https://github.com/eigenorbit/eigenkit/pull/37),
  [`a35c5ff`](https://github.com/eigenorbit/eigenkit/commit/a35c5ff4f6730f0be4a45cb7b0fc647cd3b3719d))


## v1.2.0 (2026-11-05)

### Features

- Add eigenkit.orthogonal (orthogonal networks, certified robustness, OrthogonalQNN)
  ([#32](https://github.com/eigenorbit/eigenkit/pull/32),
  [`a4bca92`](https://github.com/eigenorbit/eigenkit/commit/a4bca921bf9dd5a796dfe5488b632bfeabbd7781))

- Pass primitive options to the Qiskit Runtime primitives
  ([#31](https://github.com/eigenorbit/eigenkit/pull/31),
  [`65164a3`](https://github.com/eigenorbit/eigenkit/commit/65164a3ea09e57e0519a6c93e0737ea09ee1cd4f))

### Bug Fixes

- **orthogonal**: Exact certificate distances, single adapter names
  ([#34](https://github.com/eigenorbit/eigenkit/pull/34),
  [`8d617a2`](https://github.com/eigenorbit/eigenkit/commit/8d617a212b699c76caa5a93ae15cb89bb019b900))

### Documentation

- Write formulas in MathJax ([#35](https://github.com/eigenorbit/eigenkit/pull/35),
  [`f0275ea`](https://github.com/eigenorbit/eigenkit/commit/f0275ea2c30a221f7a05963b5c5f579322b3ecab))

### Continuous Integration

- Publish a docs preview for each pull request
  ([#33](https://github.com/eigenorbit/eigenkit/pull/33),
  [`a2ff611`](https://github.com/eigenorbit/eigenkit/commit/a2ff6115e8ae53c68bbd52b6e718a8bd3b2de32c))

- **deps**: Bump astral-sh/setup-uv from 10.3.2 to 10.3.3
  ([#23](https://github.com/eigenorbit/eigenkit/pull/23),
  [`bf13039`](https://github.com/eigenorbit/eigenkit/commit/bf130392e056fdc3e98b51c36e6867e87088b162))

### Chores

- **deps**: Bump amazon-braket-sdk, pytest and jax
  ([#36](https://github.com/eigenorbit/eigenkit/pull/36),
  [`54a9e34`](https://github.com/eigenorbit/eigenkit/commit/54a9e3467883456ff171ad4992f0b206abcc6fe1))

- **deps**: Bump vulnerable locked dependencies
  ([#27](https://github.com/eigenorbit/eigenkit/pull/27),
  [`8bf972b`](https://github.com/eigenorbit/eigenkit/commit/8bf972b978f2656ee0f64e75715ed0238d42b0bb))


## v1.1.0 (2026-10-19)

### Features

- Add the router provider for eigenorbit-router-client
  ([#18](https://github.com/eigenorbit/eigenkit/pull/18),
  [`80a2b10`](https://github.com/eigenorbit/eigenkit/commit/80a2b105129498c1ca756a8e5c4931a33656a2e4))

### Documentation

- Add the Qiskit Ecosystem badge ([#19](https://github.com/eigenorbit/eigenkit/pull/19),
  [`091edeb`](https://github.com/eigenorbit/eigenkit/commit/091edebc4ac3d0855eb1988ab8e1bcf7a62b96a8))


## v1.0.1 (2026-10-13)

### Bug Fixes

- Install cuda-quantum on Apple Silicon from a uv lock
  ([#16](https://github.com/eigenorbit/eigenkit/pull/16),
  [`44a826a`](https://github.com/eigenorbit/eigenkit/commit/44a826ad08ec55ecd15ed097187817d66414eb95))

- List the pennylane provider only when PennyLane is installed
  ([#14](https://github.com/eigenorbit/eigenkit/pull/14),
  [`6ff13ac`](https://github.com/eigenorbit/eigenkit/commit/6ff13ac150a7db59f31d4126f2517fc78575820a))

### Documentation

- Add link previews, structured data and robots.txt
  ([#17](https://github.com/eigenorbit/eigenkit/pull/17),
  [`63c17be`](https://github.com/eigenorbit/eigenkit/commit/63c17be38e8a97d3a918c91b7f21feb45531a07c))

- Describe eigenkit as an open-source Python library
  ([#15](https://github.com/eigenorbit/eigenkit/pull/15),
  [`898e48d`](https://github.com/eigenorbit/eigenkit/commit/898e48d64bf06e8fd77ebe3b885718bca90c04e1))

- Tidy the 1.0.0 changelog entry ([#13](https://github.com/eigenorbit/eigenkit/pull/13),
  [`007f2fd`](https://github.com/eigenorbit/eigenkit/commit/007f2fd24c0a0ab2e708c0759d48f7557e4d96b5))

### Continuous Integration

- **deps**: Bump astral-sh/setup-uv from 10.2.0 to 10.3.0
  ([#2](https://github.com/eigenorbit/eigenkit/pull/2),
  [`498700e`](https://github.com/eigenorbit/eigenkit/commit/498700e5817cefc87ee2a660e41c0aa94c3de4cd))


## v1.0.0 (2026-10-13)

### Features

- eigenkit 1.0.0 ([#1](https://github.com/eigenorbit/eigenkit/pull/1),
  [`babce1f`](https://github.com/eigenorbit/eigenkit/commit/babce1fa8c8fffe9c3894dbd4d24b3d13870ba25))
