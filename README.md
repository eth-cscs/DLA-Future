[![pipeline status](https://gitlab.com/cscs-ci/eth-cscs/DLA-Future/badges/master/pipeline.svg)](https://gitlab.com/cscs-ci/eth-cscs/DLA-Future/-/commits/master) [![codecov](https://codecov.io/gh/eth-cscs/DLA-Future/branch/master/graph/badge.svg)](https://codecov.io/gh/eth-cscs/DLA-Future)

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

## How to use the library

Using DLAF in a CMake project is extremly easy!

Let's use the variable `$DLAF_ROOT` for referring to the install path of DLAF.

Configure your project with:

`cmake -DDLAF_DIR="$DLAF_ROOT/lib/cmake" ..`

Then, it is just as simple as:

```
find_package(DLAF)

# ...

target_link_libraries(<your_target> PRIVATE DLAF)
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
