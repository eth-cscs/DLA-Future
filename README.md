[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.10518288.svg)](https://doi.org/10.5281/zenodo.10518288) [![pipeline status](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/4700071344751697/7514005670787789/badges/master/pipeline.svg)](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/4700071344751697/7514005670787789/-/commits/master) [![codecov](https://codecov.io/gh/eth-cscs/DLA-Future/branch/master/graph/badge.svg)](https://codecov.io/gh/eth-cscs/DLA-Future)

# Distributed Linear Algebra from the Future

DLA-Future is a distributed linear algebra library implemented using C++ `std::execution` [P2300](https://cplusplus.github.io/sender-receiver/execution.html) which provides:
- an asynchronous C++ interface,
- a synchronous C interface,
- a synchronous ScaLAPACK-like C interface (ScaLAPACK drop-in with a subset of ScaLAPACK arguments. E.g. workspace arguments are not present),
- a synchronous ScaLAPACK-like Fortran interface (See [DLA-Future-Fortran](https://github.com/eth-cscs/DLA-Future-Fortran)).

DLA-Future runs on x86 and ARM CPUs as well as NVIDIA and AMD GPUs.

Currently DLA-Future provides the following algorithms which are available in the following interfaces:
| ScaLAPACK Name | Algorithm | C++ API | C API | Scalapack C API |
|----------------|-----------|:-------:|:-----:|:---------------:|
| `P[S,D,C,Z]TRMM` | Triangular Matrix Multiplication | ✔️ | ❌ | ❌ |
| `P[C,Z]HEMM`, `P[S,D]SYMM` | Hermitian Matrix Multiplication | ✔️ | ❌ | ❌ |
| `P[S,D,C,Z]TRSM` | Triangular Solver | ✔️ | ❌ | ❌ |
| `P[S,D,C,Z]POTRF` | Cholesky Decomposition | ✔️ | ✔️ | ✔️ |
| `P[C,Z]HEEV*`, `P[S,D]SYEV*` | Symmetric / Hermitian eigenvalue solver | ✔️ | ✔️ | ✔️ |
| `P[C,Z]HEGV*`, `P[S,D]SYGV*` | Generalized Symmetric / Hermitian eigenvalue solver | ✔️ | ✔️ | ✔️ |

Notes:
- The C++ interface provides a non distributed interface as well.

## Getting started with DLAF

### Build

See [BUILD.md](BUILD.md).

### Link your program/library with DLAF

Using DLAF in a CMake project is extremely easy!

In the following, the variable `DLAF_INSTALL_PREFIX` is set to where DLAF is installed. In case you used spack for installing DLAF, you can easily set it with:

```bash
export DLAF_INSTALL_PREFIX=`spack location -i dla-future`
```

Then, you can configure your project with one of the following:

```bash
# By appending the value to the CMAKE_PREFIX_PATH
cmake -DCMAKE_PREFIX_PATH=${DLAF_INSTALL_PREFIX} ..

# ... or by setting DLAF_DIR
cmake -DDLAF_DIR="$DLAF_INSTALL_PREFIX/lib/cmake" ..
```

Then, it is just as simple as adding these directives in your `CMakeLists.txt`:

```
find_package(DLAF)
# ... and then for your executable/library target
target_link_libraries(<your_target> PRIVATE DLAF::DLAF)
```

### Documentation

- [Documentation of `master` branch](https://eth-cscs.github.io/DLA-Future/master/)
- [Documentation of `v0.7.2`](https://eth-cscs.github.io/DLA-Future/v0.7.2/)

See [DOCUMENTATION.md](DOCUMENTATION.md) for the documentation of older versions, or for the instructions to build it.

## Citing

If you are using DLA-Future, please cite the following paper in addition to this repository:

```
@InProceedings{10.1007/978-3-031-61763-8_13,
    author="Solc{\`a}, Raffaele
        and Simberg, Mikael
        and Meli, Rocco
        and Invernizzi, Alberto
        and Reverdell, Auriane
        and Biddiscombe, John",
    editor="Diehl, Patrick
        and Schuchart, Joseph
        and Valero-Lara, Pedro
        and Bosilca, George",
    title="DLA-Future: A Task-Based Linear Algebra Library Which Provides a GPU-Enabled Distributed Eigensolver",
    booktitle="Asynchronous Many-Task Systems and Applications",
    year="2024",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="135--141",
    isbn="978-3-031-61763-8"
}
```

## Acknowledgements

The development of DLAF library would not be possible without support of the following organizations (in alphabetic order):

|||
:---:|:---
<img height="50" src="./doc/images/logo-cineca.png"> | [**CINECA**](https://www.cineca.it/en)**: Cineca Consorzio Interuniversitario**
|||
<img height="50" src="./doc/images/logo-cscs.jpg"> | [**CSCS**](https://www.cscs.ch)**: Swiss National Supercomputing Centre**
|||
<img height="50" src="./doc/images/logo-eth.svg"> | [**ETH Zurich**](https://ethz.ch/en.html)**: Swiss Federal Institute of Technology Zurich**
|||
<img height="50" src="./doc/images/logo-pasc.png"> | [**PASC**](https://www.pasc-ch.org/)**: Platform for Advanced Scientific Computing**
|||
<img height="50" src="./doc/images/logo-prace.jpg"> | [**PRACE**](https://prace-ri.eu/)**: Partnership for Advanced Computing in Europe**<br/>As part of [IP6 WP8](https://prace-ri.eu/about/ip-projects/#PRACE6IP)
|||
