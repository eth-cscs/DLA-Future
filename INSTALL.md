# Dependencies

- MPI
- OpenMP
- pika
- BLAS/LAPACK
- BLASPP & LAPACKPP

## MPI

## OpenMP

## pika

pika provides a CMake config script in `$pika_ROOT/lib/cmake/pika`. To make it available, the variable
`pika_DIR` has to be set to this path. Depending on the platform, the files may also be in `lib64` instead of `lib`.

e.g. `cmake -Dpika_DIR=${PIKA_ROOT}/lib/cmake/pika ..`

## BLAS/LAPACK

BLAS/LAPACK can be provided by:

- Compiler
- Custom
- MKL

Former two options are available if `DLAF_WITH_MKL=off`, allowing through `LAPACK_TYPE` to choose between:

- Compiler: choose this if a compiler wrapper that internally manages linking to BLAS/LAPACK is available
- Custom: choose this if you want to manually specify `LAPACK_INCLUDE_DIR` and `LAPACK_LIBRARY`

In particular:

- `LAPACK_INCLUDE_DIR`: a *;-list* of include paths (without `-I` option)
- `LAPACK_LIBRARY`: a *;-list* of
	- library names (e.g. BLAS)
	- library filepaths (e.g. /usr/local/lib/libblas.so)
	- `-L<library_folder>` (e.g. -L/usr/local/lib)

Otherwise, with `DLAF_WITH_MKL=on` it looks for BLAS/LAPACK provided by MKL. It uses `MKLROOT`
environment variable if set or `MKL_ROOT` CMake variable (which has priority if both are set).

MKL is fixed to "Sequential" and does not use threading.

## BLASPP & LAPACKPP

BLASPP and LAPACKPP dependencies can be satisfied by specifying installation directories into
`blaspp_DIR` and `lapackpp_DIR` variables, respectively.
