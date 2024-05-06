# Changelog

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

-  Use `drop_operation_state` to avoid stack overflows. (#1004)

### Notes

(*) At the time of the release the spack spec `blaspp~openmp ^intel-oneapi-mkl threads=openmp` doesn't build. If you rely on multithreaded BLAS we suggest to use `blaspp+openmp ^intel-oneapi-mkl threads=openmp` until https://github.com/spack/spack/pull/42087 gets merged.

## DLA-Future 0.3.1

### Bug fixes

- Fixed compilation with gcc 9.3.
- Fixed compilation with CUDA 11.2.
- Improved eigensolver tests.

## DLA-Future 0.3.0

### Changes

- Added C and ScaLAPACK API (generalized eigensolver). (#992)
- Removed pika-algorithm dependency. (#945)

### Performance improvements

- Fixed Cholesky priorities. (#999)

## DLA-Future 0.2.1

### Bug fixes

- Fixed a problem in `reduction_to_band` that could have produced results filled with NaNs for certain corner cases. (E.g. input matrix with all off-band elements set to 0).

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
