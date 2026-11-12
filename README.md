# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/eigenorbit/eigenkit/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/eigenkit/\_\_init\_\_.py                           |       38 |       10 |        8 |        1 |     72% |145-154, 164, 193-199 |
| src/eigenkit/algorithms/\_\_init\_\_.py                |       11 |        2 |        2 |        1 |     77% |    41, 45 |
| src/eigenkit/algorithms/\_hamiltonian\_sim.py          |      119 |        0 |       46 |        0 |    100% |           |
| src/eigenkit/algorithms/\_provider\_checks.py          |       40 |        1 |       14 |        1 |     96% |       105 |
| src/eigenkit/algorithms/hhl.py                         |      436 |       15 |      164 |       12 |     96% |521, 539, 543-551, 685-\>687, 794-795, 804, 813, 825-826, 967, 1215, 1287, 1444 |
| src/eigenkit/algorithms/iqae.py                        |      233 |        9 |       76 |        7 |     95% |192-193, 232, 249, 352, 362, 447, 473, 578 |
| src/eigenkit/algorithms/portfolio\_risk.py             |      256 |       10 |       78 |       12 |     93% |166, 206, 217, 219, 334-\>345, 338, 501, 519-\>521, 553, 599, 606, 635, 778-\>781 |
| src/eigenkit/benchmark/\_\_init\_\_.py                 |      123 |        6 |       62 |        7 |     93% |102, 150, 245, 248, 250, 252, 282-\>291 |
| src/eigenkit/benchmark/\_baselines.py                  |       26 |        3 |        2 |        1 |     86% |14, 58, 65 |
| src/eigenkit/benchmark/\_io.py                         |       16 |        1 |        6 |        1 |     91% |        39 |
| src/eigenkit/benchmark/\_metrics.py                    |      120 |        8 |       50 |        5 |     92% |27-30, 103, 164, 188, 242, 259 |
| src/eigenkit/benchmark/\_results.py                    |      379 |       45 |      166 |       31 |     86% |35-36, 63, 71, 73, 75, 168-173, 250, 265, 274, 280, 286-\>285, 288, 298, 329, 419, 428, 475-\>471, 482, 485-486, 521-522, 532-\>534, 576-577, 616, 668, 705-706, 722-\>720, 733, 752, 766-767, 787-788, 789-\>792, 847-848, 853, 870, 875, 879, 903-904, 915 |
| src/eigenkit/benchmark/\_runner.py                     |      128 |       21 |       42 |        6 |     84% |41, 67, 69, 74, 85, 96-100, 102-\>121, 114-115, 119-120, 129-130, 183-189, 195-201 |
| src/eigenkit/chemistry/\_\_init\_\_.py                 |        9 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/chemistry/\_pauli.py                      |       60 |        1 |       24 |        2 |     96% |65-\>67, 111 |
| src/eigenkit/chemistry/ansatz.py                       |       51 |        1 |       22 |        1 |     97% |       151 |
| src/eigenkit/chemistry/exact.py                        |       18 |        0 |        4 |        0 |    100% |           |
| src/eigenkit/chemistry/fermion.py                      |       58 |        3 |       16 |        0 |     96% |64, 137, 155 |
| src/eigenkit/chemistry/interop.py                      |       18 |        0 |        4 |        0 |    100% |           |
| src/eigenkit/chemistry/mappings.py                     |       27 |        0 |        4 |        0 |    100% |           |
| src/eigenkit/chemistry/models.py                       |       16 |        0 |        4 |        0 |    100% |           |
| src/eigenkit/chemistry/molecule.py                     |       32 |        4 |        2 |        0 |     82% |     28-36 |
| src/eigenkit/chemistry/vqe.py                          |       51 |        0 |       14 |        0 |    100% |           |
| src/eigenkit/circuit.py                                |       23 |        2 |        8 |        2 |     87% |  122, 130 |
| src/eigenkit/configs.py                                |       25 |        8 |        6 |        1 |     65% |    97-106 |
| src/eigenkit/core/\_\_init\_\_.py                      |       16 |        0 |        2 |        0 |    100% |           |
| src/eigenkit/core/\_results.py                         |       23 |        0 |        6 |        0 |    100% |           |
| src/eigenkit/core/backends/\_\_init\_\_.py             |       47 |        5 |       20 |        9 |     79% |46-\>48, 48-\>51, 51-\>60, 55-57, 60-\>68, 77-78, 81-\>83, 83-\>85, 85-\>89, 89-\>97, 98, 101-\>106 |
| src/eigenkit/core/backends/\_kernel\_support.py        |       76 |        9 |       16 |        2 |     88% |28-29, 62-63, 67, 82, 96, 179-180 |
| src/eigenkit/core/backends/\_sampling.py               |      101 |        0 |       50 |        0 |    100% |           |
| src/eigenkit/core/backends/\_state\_prep.py            |       12 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/core/backends/braket\_backend.py          |      531 |       23 |      244 |       14 |     95% |126-127, 137-138, 262, 333-340, 541-\>537, 659, 663, 750, 753, 759, 790, 795, 1041-1044, 1098, 1286, 1303-\>1307, 1356, 1366 |
| src/eigenkit/core/backends/cudaquantum\_backend.py     |      187 |       20 |       90 |       13 |     87% |83-84, 156, 172, 193, 265, 274, 276, 282, 300-308, 351, 397, 438-440, 471, 475 |
| src/eigenkit/core/backends/iqm\_backend.py             |      259 |       20 |       68 |       11 |     90% |133-134, 298, 325, 337-338, 366-\>368, 370, 417-418, 422, 428, 530-\>549, 588-589, 627, 817, 872, 900, 922-924, 952-\>956 |
| src/eigenkit/core/backends/numpy\_backend.py           |      401 |       13 |      104 |        6 |     96% |147, 321-323, 344-346, 511, 809-810, 830-832 |
| src/eigenkit/core/backends/pennylane\_backend.py       |      289 |       23 |      136 |       13 |     89% |45-46, 211, 213, 219, 228-246, 419-\>434, 453-\>467, 464, 502, 591, 617-\>615, 622-\>620, 644, 674 |
| src/eigenkit/core/backends/qiskit\_aer\_backend.py     |      109 |        3 |       24 |        2 |     96% |83-84, 234-\>233, 243 |
| src/eigenkit/core/backends/qiskit\_backend.py          |      195 |        9 |       84 |        8 |     94% |78-79, 161, 218, 232-\>237, 258-259, 266-267, 336-\>335, 339-\>338, 353 |
| src/eigenkit/core/backends/qiskit\_runtime\_backend.py |      397 |       16 |      114 |        9 |     95% |213-214, 358-\>360, 447, 449, 472-473, 477, 542-543, 560, 571, 829, 907, 1132, 1151, 1261 |
| src/eigenkit/core/backends/qsim\_backend.py            |      138 |        5 |       56 |        3 |     96% |91, 121-122, 175, 218 |
| src/eigenkit/core/backends/qulacs\_backend.py          |      241 |       12 |       90 |       10 |     93% |121-122, 168, 256, 280-281, 365, 374, 383, 393-\>396, 398, 514, 518 |
| src/eigenkit/core/backends/tensorcircuit\_backend.py   |      107 |        5 |       28 |        2 |     95% |82-83, 124, 251-252 |
| src/eigenkit/core/blocks.py                            |      720 |       19 |      314 |        9 |     97% |120, 838, 844, 905, 907, 911, 1033, 1163-1164, 1166, 1289, 1309-1314, 1334-1336, 1405, 1417 |
| src/eigenkit/core/builder.py                           |      127 |        8 |       42 |        8 |     91% |244, 246, 267, 269, 300, 316, 335, 384 |
| src/eigenkit/core/circuit.py                           |      244 |        3 |      112 |        4 |     98% |483-\>481, 694, 702, 780 |
| src/eigenkit/core/draw.py                              |      138 |        3 |       70 |        2 |     98% |125-\>127, 211-212, 252 |
| src/eigenkit/core/draw\_mpl.py                         |      338 |        4 |      110 |        2 |     99% |236, 364-365, 680 |
| src/eigenkit/core/interfaces.py                        |       23 |        8 |        6 |        2 |     59% |94, 112-117, 123 |
| src/eigenkit/core/ir.py                                |      317 |        7 |      126 |        6 |     97% |330, 678, 714, 745, 1087-1088, 1109 |
| src/eigenkit/core/observables.py                       |       84 |        5 |       30 |        5 |     91% |123, 130, 137, 142, 152 |
| src/eigenkit/core/optimizers.py                        |       52 |        1 |       12 |        1 |     97% |       151 |
| src/eigenkit/core/provider.py                          |      172 |        7 |       58 |        7 |     94% |347, 532, 539, 544, 562, 680, 733 |
| src/eigenkit/core/secret\_fields.py                    |       31 |        1 |       14 |        1 |     96% |        79 |
| src/eigenkit/core/solver.py                            |        8 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/core/variational.py                       |       70 |        1 |       20 |        2 |     97% |250-\>252, 254 |
| src/eigenkit/core/wire.py                              |       32 |        0 |       12 |        0 |    100% |           |
| src/eigenkit/datasets/\_\_init\_\_.py                  |       77 |       15 |       26 |        4 |     78% |101, 103, 167-175, 185, 223-237 |
| src/eigenkit/datasets/\_cache.py                       |       52 |        3 |       14 |        1 |     94% |51-52, 128 |
| src/eigenkit/datasets/\_loaders.py                     |       58 |        7 |       22 |        4 |     86% |89-90, 101, 119, 136-138 |
| src/eigenkit/datasets/\_registry.py                    |       21 |        0 |        2 |        0 |    100% |           |
| src/eigenkit/datasets/\_synthetic.py                   |       78 |        1 |       26 |        1 |     98% |       146 |
| src/eigenkit/datasets/eda.py                           |      461 |       68 |      168 |       27 |     81% |156, 160, 163, 174, 180, 237-240, 278, 306, 326-\>330, 333-\>337, 399, 441, 453, 534, 545, 562, 577, 597-603, 675, 682-686, 752, 806-816, 879-887, 906-913, 929-931, 947-949, 966, 1035 |
| src/eigenkit/optimization/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/optimization/problems.py                  |       82 |        1 |       24 |        2 |     97% |91, 221-\>223 |
| src/eigenkit/optimization/qaoa.py                      |       59 |        2 |       16 |        1 |     93% |   183-187 |
| src/eigenkit/orthogonal/\_\_init\_\_.py                |       16 |        4 |        4 |        0 |     70% |   104-107 |
| src/eigenkit/orthogonal/blocks.py                      |      153 |        2 |       46 |        2 |     98% |  113, 368 |
| src/eigenkit/orthogonal/circuit.py                     |       14 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/orthogonal/matrix.py                      |       59 |        0 |       16 |        0 |    100% |           |
| src/eigenkit/orthogonal/topology.py                    |       97 |        1 |       28 |        1 |     98% |       152 |
| src/eigenkit/orthogonal/torch/\_\_init\_\_.py          |        5 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/orthogonal/torch/adapters.py              |      108 |        1 |       48 |        1 |     99% |       101 |
| src/eigenkit/orthogonal/torch/functional.py            |       45 |        3 |       22 |        3 |     91% |66, 75, 123 |
| src/eigenkit/orthogonal/torch/modules.py               |      243 |        4 |       66 |        1 |     98% |136, 173, 437, 500-\>502, 649 |
| src/eigenkit/orthogonal/torch/robust.py                |      115 |        3 |       40 |        2 |     97% |106, 326, 359 |
| src/eigenkit/photonics/\_\_init\_\_.py                 |       10 |        0 |        2 |        0 |    100% |           |
| src/eigenkit/photonics/backends/\_\_init\_\_.py        |        2 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/photonics/backends/perceval\_backend.py   |       73 |        3 |       18 |        2 |     95% |104-105, 239 |
| src/eigenkit/photonics/backends/perceval\_remote.py    |      108 |        9 |       20 |        0 |     93% |45-49, 167-171 |
| src/eigenkit/photonics/executor.py                     |       22 |        1 |        4 |        2 |     88% |142-\>149, 144 |
| src/eigenkit/photonics/gradients.py                    |       99 |        5 |       40 |        4 |     94% |92, 102, 134-135, 202 |
| src/eigenkit/photonics/ir.py                           |       70 |        0 |       22 |        0 |    100% |           |
| src/eigenkit/photonics/reservoir.py                    |       79 |        2 |       44 |        2 |     97% |  240, 251 |
| src/eigenkit/photonics/torch.py                        |       41 |        0 |        6 |        0 |    100% |           |
| src/eigenkit/qml/\_\_init\_\_.py                       |       27 |        2 |        2 |        0 |     93% |     70-71 |
| src/eigenkit/qml/\_alignment.py                        |       66 |        4 |       16 |        3 |     91% |207, 221-\>223, 248-250 |
| src/eigenkit/qml/\_base.py                             |       91 |        6 |       26 |        2 |     93% |185-\>188, 204, 236-237, 251, 255, 282 |
| src/eigenkit/qml/\_kernels.py                          |      319 |       27 |       96 |       12 |     89% |136, 260, 272, 288-\>290, 304-305, 381, 531-\>533, 546-559, 576, 644-651, 739-\>748, 750, 798-\>803, 836, 884, 891, 921 |
| src/eigenkit/qml/\_registry.py                         |        9 |        1 |        2 |        1 |     82% |        38 |
| src/eigenkit/qml/\_serialization.py                    |      199 |       18 |       78 |       13 |     89% |201, 203, 241, 253-254, 288, 348, 391, 394-395, 406, 410, 412, 414, 430, 445-446, 532 |
| src/eigenkit/qml/qrc.py                                |      149 |        4 |       16 |        2 |     96% |123, 218, 256, 371 |
| src/eigenkit/qml/qsvc.py                               |       71 |        1 |       10 |        1 |     98% |       102 |
| src/eigenkit/qml/qsvr.py                               |       49 |        1 |        4 |        1 |     96% |        93 |
| src/eigenkit/reservoir/\_\_init\_\_.py                 |        5 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/reservoir/driver.py                       |       31 |        1 |        8 |        1 |     95% |        57 |
| src/eigenkit/reservoir/features.py                     |       57 |        4 |       18 |        4 |     89% |47, 50, 60, 91 |
| src/eigenkit/reservoir/policies.py                     |       57 |        2 |        8 |        2 |     94% |  127, 280 |
| src/eigenkit/reservoir/readout.py                      |       28 |        1 |        6 |        1 |     94% |        33 |
| src/eigenkit/torch/\_\_init\_\_.py                     |        3 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/torch/circuit.py                          |       60 |        5 |       10 |        1 |     91% |65, 170, 223, 251, 261 |
| src/eigenkit/torch/estimators/\_\_init\_\_.py          |        9 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/torch/estimators/\_nn.py                  |       59 |        2 |       24 |        2 |     95% |  104, 121 |
| src/eigenkit/torch/estimators/\_seeding.py             |       15 |        1 |        4 |        1 |     89% |        44 |
| src/eigenkit/torch/estimators/\_training.py            |      112 |        8 |       52 |       10 |     89% |29, 62, 69, 200, 215, 227-\>248, 241-\>248, 249, 307, 319 |
| src/eigenkit/torch/estimators/custom.py                |      149 |        4 |       20 |        4 |     94% |43-45, 59-\>61, 214 |
| src/eigenkit/torch/estimators/generic.py               |      225 |        6 |       46 |        4 |     96% |201-\>203, 241, 315-316, 515-\>517, 563, 656-657 |
| src/eigenkit/torch/estimators/hqnc.py                  |       98 |        1 |        8 |        0 |     99% |       180 |
| src/eigenkit/torch/estimators/hqnr.py                  |       86 |        2 |        8 |        2 |     96% |154, 162-\>167, 222 |
| src/eigenkit/torch/estimators/recurrent\_hqnc.py       |      146 |        5 |       44 |        6 |     94% |131, 133, 175, 193-\>195, 310, 312 |
| src/eigenkit/torch/estimators/recurrent\_hqnr.py       |      160 |       16 |       48 |       14 |     86% |64, 172, 178, 180, 194, 241, 268, 272, 280-\>285, 330, 336, 367, 369, 371, 391, 393, 395 |
| src/eigenkit/torch/estimators/timeseries.py            |      336 |       20 |      128 |       26 |     90% |145, 157, 175, 183, 185, 187, 192, 231, 233, 339, 355-\>357, 417-\>420, 506, 516, 528, 538, 550, 571, 573, 579, 641, 696, 777-\>779, 779-\>781, 793-\>795, 849-\>852 |
| src/eigenkit/torch/interface.py                        |       95 |       11 |       30 |        3 |     87% |42-50, 228, 242 |
| src/eigenkit/torch/layers/\_\_init\_\_.py              |        6 |        0 |        0 |        0 |    100% |           |
| src/eigenkit/torch/layers/\_dtype.py                   |        6 |        0 |        2 |        0 |    100% |           |
| src/eigenkit/torch/layers/\_expressivity.py            |       27 |        0 |       14 |        0 |    100% |           |
| src/eigenkit/torch/layers/basic.py                     |       33 |        0 |        6 |        0 |    100% |           |
| src/eigenkit/torch/layers/ensemble.py                  |       50 |        1 |       16 |        1 |     97% |       104 |
| src/eigenkit/torch/layers/minmax.py                    |       39 |        0 |       10 |        0 |    100% |           |
| src/eigenkit/torch/layers/orthogonal\_qnn.py           |       58 |        1 |       10 |        0 |     99% |       195 |
| src/eigenkit/torch/layers/reupload.py                  |       29 |        0 |        2 |        0 |    100% |           |
| **TOTAL**                                              | **13007** |  **670** | **4378** |  **419** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/eigenorbit/eigenkit/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/eigenorbit/eigenkit/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eigenorbit/eigenkit/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/eigenorbit/eigenkit/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Feigenorbit%2Feigenkit%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/eigenorbit/eigenkit/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.