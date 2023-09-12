# Data

This directory contains `.h5`/`.hdf5` files that can be used for testing.

## Descriptions

* `issue-974.h5`: input matrix for the standard eigensolver obtained by running CP2K's regression test `QS/regtest-md-extrap/extrap-1-far.inp`. The matrix is extracted from DLA-Future in `numpy` format (single precision), and manually converted to DLA-Future-compatible HDF5 format using `pyh5`.
