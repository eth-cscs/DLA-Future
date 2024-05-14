# Building DLAF

## Build using Spack

Minimal installation:

```
spack install dla-future ^[virtuals=blas,lapack] intel-oneapi-mkl
```

Minimal installation with (optional) C API compatible with ScaLAPACK:

```
spack install dla-future +scalapack ^[virtuals=blas,lapack,scalapack] intel-oneapi-mkl
```

GPU support can be enabled with the following variants:

Variant | Dependent variant | Note
:---|:---|:---
`+cuda` | `cuda_arch=<arch>` | enable CUDA GPU support and specify the GPU architecture
`+rocm` | `amdgpu_target=<arch>` | enable AMD GPU support and specify the GPU architecture
`+mpi-gpu-aware` | `+mpi-gpu-force-contiguous` | Enable GPU aware communication (recommended if supported by MPI), and force communication of contiguous GPU buffers (recommended)

You can examine all the available variants of the package using:

```
spack info dla-future
```

You can go even further with a more detailed spec like this one, which builds DLA-Future in debug mode, using the clang compiler, specifying that the pika on which it depends has to be built
in debug mode too, and that we want to use MPICH as MPI implementation, without Fortran support (because clang does not support it).

```
spack install dla-future %clang build_type=Debug ^pika build_type=Debug ^mpich ~fortran
```

See [Spack documentation](https://spack.readthedocs.io/en/latest/) for more information about the build system.

### master branch

If you want to test the newest available version,
you can use the spack package `dla-future` provided in the git repository, that can be easily added to your own spack as follows:

```
git clone https://github.com/eth-cscs/DLA-Future.git
spack repo add $DLAF_ROOT/spack
```

This will add a new repository with namespace `dlaf`.

You can then install the master version with:

```
spack install dla-future@master <variants>
```

## Build the good old way

### Dependencies

- MPI
- [pika](https://github.com/pika-org/pika)
- [umpire](https://github.com/LLNL/Umpire)
- [blaspp](https://bitbucket.org/icl/blaspp/src/default/)
- [lapackpp](https://bitbucket.org/icl/lapackpp/src/default/)
- Intel MKL or other LAPACK implementation
- A ScaLAPACK implementation (optional, ScaLAPACK-like C API only)
- [whip](https://github.com/eth-cscs/whip) (optional, GPU only)
- [CUDA](https://developer.nvidia.com/cuda) (optional, NVidia GPUs only)
- [HIP/ROCm](https://github.com/RadeonOpenCompute/ROCm) (optional, AMD GPUs only)
- [GoogleTest](https://github.com/google/googletest) (optional; bundled) - unit testing
- Doxygen (optional) - documentation

### Get DLA-Future

You can download the [releases](https://github.com/eth-cscs/DLA-Future/releases).

Otherwise, if you have `git`, you can clone this repository with

```
git clone https://github.com/eth-cscs/DLA-Future.git
```

### Build and install

You can build all the dependencies by yourself, but you have to ensure that:
- pika: `PIKA_WITH_CUDA=ON` (if building for CUDA) + `PIKA_WITH_MPI=ON`

The main CMake options for DLAF build customization are:

CMake option | Values | Note
:---|:---|:---
`pika_DIR` | CMAKE:PATH | Location of the pika CMake-config file
`blaspp_DIR` | CMAKE:PATH | Location of the blaspp CMake-config file
`lapackpp_DIR` | CMAKE:PATH | Location of the lapackpp CMake-config file
`DLAF_WITH_MKL` | `{ON,OFF}` (default: `OFF`) | if blaspp/lapackpp is built with oneMKL
`DLAF_WITH_LEGACY_MKL` | `{ON,OFF}` (default: `OFF`) | if blaspp/lapackpp is built with MKL (deprecated)
`DLAF_WITH_SCALAPACK` | `{ON,OFF}` (default: `OFF`) | Enable ScaLAPACK-like API.
`MKL_ROOT` | CMAKE:PATH | Location of the MKL library
`DLAF_ASSERT_ENABLE` | `{ON,OFF}` (default: `ON`) | enable/disable cheap assertions
`DLAF_ASSERT_MODERATE_ENABLE` | `{ON,OFF}` (default: `ON` in Debug, `OFF` otherwise) | enable/disable moderate assertions
`DLAF_ASSERT_HEAVY_ENABLE` | `{ON,OFF}` (default: `ON` in Debug, `OFF` otherwise) | enable/disable heavy assertions
`DLAF_WITH_CUDA` | `{ON,OFF}` (default: `OFF`) | enable CUDA support
`DLAF_WITH_HIP` | `{ON,OFF}` (default: `OFF`) | enable HIP support
`DLAF_WITH_MPI_GPU_AWARE` | `{ON,OFF}` (default: `OFF`) | enable GPU to GPU Communication
`DLAF_WITH_MPI_GPU_FORCE_CONTIGUOUS` | `{ON,OFF}` (default: `OFF`) | force the use of contiguous buffers for communication.
`DLAF_BUILD_MINIAPPS` | `{ON,OFF}` (default: `ON`) | enable/disable building miniapps
`DLAF_BUILD_TESTING` | `{ON,OFF}` (default: `ON`) | enable/disable building tests
`DLAF_INSTALL_TESTS` | `{ON,OFF}` (default: `OFF`) | enable/disable installing tests
`DLAF_MPI_PRESET` | `{plain-mpi, slurm, custom}` (default `plain-mpi`) | presets for MPI configuration for tests. See [CMake Doc](https://cmake.org/cmake/help/latest/module/FindMPI.html?highlight=mpiexec_executable#usage-of-mpiexec) for additional information
`DLAF_TEST_RUNALL_WITH_MPIEXEC` | `{ON, OFF}` (default: `OFF`) | Use mpi runner also for non-MPI based tests
`DLAF_PIKATEST_EXTRA_ARGS` | CMAKE:STRING | Additional pika command-line options for tests
`DLAF_BUILD_DOC` | `{ON,OFF}` (default: `OFF`) | enable/disable documentation generation
