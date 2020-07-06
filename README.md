[![pipeline status](https://gitlab.com/cscs-ci/eth-cscs/DLA-Future/badges/master/pipeline.svg)](https://gitlab.com/cscs-ci/eth-cscs/DLA-Future/-/commits/master) [![codecov](https://codecov.io/gl/cscs-ci:eth-cscs/DLA-Future/branch/master/graph/badge.svg)](https://codecov.io/gl/cscs-ci:eth-cscs/DLA-Future)

# Distributed Linear Algebra with Futures.

## Dependencies

- MPI
- [HPX](https://github.com/STEllAR-GROUP/hpx)
- [blaspp](https://bitbucket.org/icl/blaspp/src/default/)
- [lapackpp](https://bitbucket.org/icl/lapackpp/src/default/)
- Intel MKL or other LAPACK implementation
- [cuBLAS](https://developer.nvidia.com/cublas) (optional)
- [GoogleTest](https://github.com/google/googletest) (optional; bundled) - unit testing
- Doxygen (optional) - documentation

## How to install DLA-Future with spack

We provide a spack package DLA-Future that can be easily added to your own spack as follows:

`spack repo add $DLAF_ROOT/spack`

This will add a new repository with namespace `dlaf`.

Example installation:

`spack install dla-future ^intel-mkl`

Notice that, for the package to work correctly, the HPX option `max_cpu_count` must be set accordingly to the platform,
as it represents the maximum number of OS-threads.

`spack install dla-future ^intel-mkl ^hpx max_cpu_count=256`

## How to use the library

Using DLAF in a CMake project is extremely easy!

Let's use the variable `$DLAF_ROOT` for referring to the install path of DLAF.

Configure your project with:

```bash
cmake -DDLAF_DIR="$DLAF_ROOT/lib/cmake" ..
```

Then, it is just as simple as:

```
find_package(DLAF)

# ...

target_link_libraries(<your_target> PRIVATE DLAF)
```

## How to generate the documentation

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
