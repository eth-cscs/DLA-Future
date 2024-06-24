# Changelog

## DLA-Future 0.6.0

### Changes

- Renamed ScaLAPACK-like generalized eigensolvers `pXsygvx`/`pXhegvx` to `pXsygvd`/`pXhegvd`. (#1168)
- Introduced generalized eigensolver where the matrix B is already factorized. (#1167)

### Performance improvements

- Local eigenvector permutations in the distributed tridiagonal eigensolver are executed directly in GPU memory. (#1118)

### Bug fixes

- Fixed ScaLAPACK detection in CMake for specific uenv cases. (#1159)

## DLA-Future 0.5.0

### Changes

- Introduced an option (*) for forcing contiguous GPU communication buffers. (#1096)
- Introduced an option (*) for enabling GPU aware MPI communication. (#1102)
- Removed special handling of Intel MKL, as it could lead to broken installations. (#1149)
    - Spack installations: spack will set the correct variables.
    - Manual installations: the user is responsible to correctly set variables (see [BUILD.md](BUILD.md)).

(*) These options are available as spack variants.

### Performance improvements

- Don't communicate in algorithms when using single rank communicators. (#1097)
- Fixed slow performance of local version of `bt_band_to_tridiagonal` (#1144)

### Bug fixes

- Implemented a workaround for `hipMemcpyDefault` 2D memcpys, due to bugs in HIP. (#1106)
- Miniapps initialize HIP before MPI, as on older Cray MPICH versions initializing HIP after MPI leads to HIP not seeing any devices. (#1090)

## DLA-Future 0.4.1

### Bug fixes

- Update project version and export it in CMake. (#1121)

## DLA-Future 0.4.0

### Changes

- Modified `CommunicatorGrid` to avoid blocking calls to `MPI_Comm_dup`. It now returns communicator pipelines. (#993)
- Added support for Intel oneMKL and the `intel-oneapi-mkl` spack package. (#1073) (*)

### Performance improvements

- Reduced the size of the matrix-matrix multiplications in the tridiagonal eigensolver to cover only the non deflated part of the eigenvectors. (#951, #967, #996, #997, #998)
- Introduced stackless threads where appropriate. (#1037)

### Bug fixes

- Use `drop_operation_state` to avoid stack overflows. (#1004)

### Notes

(*) At the time of the release the spack spec `blaspp~openmp ^intel-oneapi-mkl threads=openmp` doesn't build. If you rely on multithreaded BLAS we suggest to use `blaspp+openmp ^intel-oneapi-mkl threads=openmp` until https://github.com/spack/spack/pull/42087 gets merged.

## DLA-Future 0.3.1

### Bug fixes

- Fixed compilation with gcc 9.3. (#1043)
- Fixed compilation with CUDA 11.2. (#1045)
- Improved eigensolver tests. (#1039)

## DLA-Future 0.3.0

### Changes

- Added C and ScaLAPACK API (generalized eigensolver). (#992)
- Removed pika-algorithm dependency. (#945)

### Performance improvements

- Fixed Cholesky priorities. (#999)

## DLA-Future 0.2.1

### Bug fixes

- Fixed a problem in `reduction_to_band` that could have produced results filled with NaNs for certain corner cases. (E.g. input matrix with all off-band elements set to 0). (#980)

## DLA-Future 0.2.0

### Changes

- Renamed algorithms using snake case. (#942)
- Added C and ScaLAPACK API (cholesky and eigensolver). (#886)
- `Matrix` API:
  - Initial support for matrices with different tile/block-size. (#909)
  - Initial support for matrix subpipelines. (#898)
  - Initial support for submatrices. (#934)
  - Initial support for matrix redistribution. (#933)

### Bug fixes

- Fixed a problem in `tridiagonal_eigensolver` which produced wrong results for some classes of matrices. (#960)

### Performance improvements

- Introduced busy barriers in `reduction_to_band`. (#864)
- New `band_to_tridiagonal` algorithm implementation. (#938, #946)
- Improved the rank1 problem solution in `tridiagonal_eigensolver`. (#904, #936)

## DLA-Future 0.1.0

The first release of DLA-Future.
