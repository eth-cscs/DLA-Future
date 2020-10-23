[![pipeline status](https://gitlab.com/cscs-ci/eth-cscs/DLA-Future/badges/master/pipeline.svg)](https://gitlab.com/cscs-ci/eth-cscs/DLA-Future/-/commits/master) [![codecov](https://codecov.io/gl/cscs-ci:eth-cscs/DLA-Future/branch/master/graph/badge.svg)](https://codecov.io/gl/cscs-ci:eth-cscs/DLA-Future)

# Distributed Linear Algebra with Futures.

## Getting started with DLAF

### Get DLA-Future

If you have `git` you can clone this repository with the command

```
git clone https://github.com/eth-cscs/DLA-Future.git
```

Otherwise you can download the archive of the latest `master` branch as a [zip](https://github.com/eth-cscs/DLA-Future/archive/master.zip) or [tar.gz](https://github.com/eth-cscs/DLA-Future/archive/master.tar.gz) archive.

### Dependencies

- MPI
- [HPX](https://github.com/STEllAR-GROUP/hpx)
- [blaspp](https://bitbucket.org/icl/blaspp/src/default/)
- [lapackpp](https://bitbucket.org/icl/lapackpp/src/default/)
- Intel MKL or other LAPACK implementation
- [cuBLAS](https://developer.nvidia.com/cublas) (optional)
- [GoogleTest](https://github.com/google/googletest) (optional; bundled) - unit testing
- Doxygen (optional) - documentation

#### Build using Spack

We provide a spack package `dla-future` that can be easily added to your own spack as follows:

`spack repo add $DLAF_ROOT/spack`

This will add a new repository with namespace `dlaf`.

Example installation:

`spack install dla-future ^intel-mkl`

Or you can go even further with a more detailed spec like this one, which builds dla-future in debug mode, using the clang compiler, specifying that the HPX on which it depends has to be built
in debug mode too, with APEX instrumentation enabled, and that we want to use MPICH as MPI implementation, without fortran support (because clang does not support it).

`spack install dla-future %clang build_type=Debug ^hpx build_type=Debug instrumentation=apex ^mpich ~fortran`

Notice that, for the package to work correctly, the HPX option `max_cpu_count` must be set accordingly to the architecture, as it represents the size of the bitmask to interface with hardware threads.

`spack install dla-future ^intel-mkl ^hpx max_cpu_count=256`

#### Build the old good way

You can build all the dependencies by yourself, but you have to ensure that:
- BLAS/LAPACK implementation is not multithreaded
- HPX: `HPX_WITH_NETWORKING=none` + `HPX_WITH_MAX_CPU_COUNT=n` (according to number of cores in the architecture, suggested the next closest power of 2)
- HPX and DLAF must have a compatible `CMAKE_BUILD_TYPE`: they must be built both in Debug, or with any combination of release types (Release, RelWithDebInfo or MinSizeRel)

And here the main CMake options for DLAF build customization:

CMake option | Values | Note
:---|:---|:---
`DLAF_ASSERT_ENABLE` | `{ON,OFF}` (default: `ON`) | to enable/disable cheap assertions
`DLAF_ASSERT_MODERATE_ENABLE` | `{ON,OFF}` (default: `ON` in Debug, `OFF` otherwise) | to enable/disable moderate assertions
`DLAF_ASSERT_HEAVY_ENABLE` | `{ON,OFF}` (default: `ON` in Debug, `OFF` otherwise) | to enable/disable heavy assertions
`DLAF_WITH_CUDA` | `{ON,OFF}` (default: `OFF`) | enable CUDA support
`HPX_DIR` | |
`blaspp_DIR` | |
`lapackpp_DIR` | |
`DLAF_WITH_MKL` | | if blaspp/lapackpp is built with MKL
`MKL_ROOT` | |
`DLAF_BUILD_MINIAPPS` | `{ON,OFF}` | to enable/disable building miniapps
`DLAF_WITH_TEST` | `{ON,OFF}` | to enable/disable building tests
`DLAF_INSTALL_TESTS` | `{ON,OFF}` | to enable/disable installing tests

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

### How to generate the documentation

The documentation can be built together with the project by enabling its generation with the flag `BUILD_DOC=on` and then use the `doc` target to eventually generate it.

```bash
# from the build folder, if you have already configured the CMake project
cmake -DBUILD_DOC=on .
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
