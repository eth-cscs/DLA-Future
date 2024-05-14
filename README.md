[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.10518288.svg)](https://doi.org/10.5281/zenodo.10518288) [![pipeline status](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/4700071344751697/7514005670787789/badges/master/pipeline.svg)](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/4700071344751697/7514005670787789/-/commits/master) [![codecov](https://codecov.io/gh/eth-cscs/DLA-Future/branch/master/graph/badge.svg)](https://codecov.io/gh/eth-cscs/DLA-Future)

# Distributed Linear Algebra from the Future

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
# By appending the value to the CMAKE_INSTALL_PREFIX
cmake -DCMAKE_INSTALL_PREFIX=${DLAF_INSTALL_PREFIX} ..

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

- [Documentation of `v0.1.0`](https://eth-cscs.github.io/DLA-Future/v0.1.0/)
- [Documentation of `v0.2.0`](https://eth-cscs.github.io/DLA-Future/v0.2.0/)
- [Documentation of `v0.2.1`](https://eth-cscs.github.io/DLA-Future/v0.2.1/)
- [Documentation of `v0.3.0`](https://eth-cscs.github.io/DLA-Future/v0.3.0/)
- [Documentation of `v0.3.1`](https://eth-cscs.github.io/DLA-Future/v0.3.1/)
- [Documentation of `v0.4.0`](https://eth-cscs.github.io/DLA-Future/v0.4.0/)
- [Documentation of `v0.4.1`](https://eth-cscs.github.io/DLA-Future/v0.4.1/)
- [Documentation of `master` branch](https://eth-cscs.github.io/DLA-Future/master/)

#### How to generate the documentation

The documentation can be built together with the project by enabling its generation with the flag `DLAF_BUILD_DOC=on` and then use the `doc` target to eventually generate it.

```bash
# from the build folder, if you have already configured the CMake project
cmake -DDLAF_BUILD_DOC=on .
make doc
```

Alternatively, the documentation can be generated independently by using `doc/Doxyfile.in`, which is a template configuration file in which you have to replace the text `${DLAF_SOURCE_DIR}` with the root folder of DLAF containing the source code (e.g. where you cloned this repository).

As a shortcut for this process a `doc/Makefile` is available, which automatically performs the substitution and then generates the documentation.

```
cd doc
make doc
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
