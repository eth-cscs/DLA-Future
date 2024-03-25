//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <cmath>
#include <optional>
#include <vector>

#include <dlaf/blas/tile.h>
#include <dlaf/common/vector.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/band_to_tridiag.h>
#include <dlaf/eigensolver/bt_band_to_tridiag.h>
#include <dlaf/eigensolver/bt_reduction_to_band.h>
#include <dlaf/eigensolver/eigensolver/api.h>
#include <dlaf/eigensolver/internal/get_band_size.h>
#include <dlaf/eigensolver/reduction_to_band.h>
#include <dlaf/eigensolver/tridiag_solver.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/hdf5.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
void Eigensolver<B, D, T>::call(blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<BaseType<T>, D>& evals,
                                Matrix<T, D>& mat_e) {
  const SizeType band_size = getBandSize(mat_a.blockSize().rows());

  // need uplo check as reduction to band doesn't have the uplo argument yet.
  if (uplo != blas::Uplo::Lower)
    DLAF_UNIMPLEMENTED(uplo);

  auto mat_taus = reduction_to_band<B>(mat_a, band_size);
  auto ret = band_to_tridiagonal<Backend::MC>(uplo, band_size, mat_a);

  tridiagonal_eigensolver<B>(ret.tridiagonal, evals, mat_e);

  bt_band_to_tridiagonal<B>(band_size, mat_e, ret.hh_reflectors);
  bt_reduction_to_band<B>(band_size, mat_e, mat_a, mat_taus);
}

template <Backend B, Device D, class T>
void Eigensolver<B, D, T>::call(comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, D>& mat_a,
                                Matrix<BaseType<T>, D>& evals, Matrix<T, D>& mat_e) {
  const SizeType band_size = getBandSize(mat_a.blockSize().rows());

  // need uplo check as reduction to band doesn't have the uplo argument yet.
  if (uplo != blas::Uplo::Lower)
    DLAF_UNIMPLEMENTED(uplo);

#ifdef DLAF_WITH_HDF5
  std::optional<matrix::internal::FileHDF5> file_eigensolver;

  if (getTuneParameters().debug_dump_eigensolver_data) {
    file_eigensolver = matrix::internal::FileHDF5(grid.fullCommunicator(), "eigensolver.h5");
    file_eigensolver->write(mat_a, "/input");
  }

  std::optional<matrix::internal::FileHDF5> file_reduction_to_band;

  if (getTuneParameters().debug_dump_reduction_to_band_data) {
    file_reduction_to_band = matrix::internal::FileHDF5(grid.fullCommunicator(), "red_to_band.h5");
    file_reduction_to_band->write(mat_a, "/input");
  }
#endif

  auto mat_taus = reduction_to_band<B>(grid, mat_a, band_size);

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_reduction_to_band_data) {
    file_reduction_to_band->write(mat_a, "/band");
  }

  std::optional<matrix::internal::FileHDF5> file_band_to_tridiagonal;

  if (getTuneParameters().debug_dump_band_to_tridiagonal_data) {
    file_band_to_tridiagonal = matrix::internal::FileHDF5(grid.fullCommunicator(), "band_to_tridiag.h5");
    file_band_to_tridiagonal->write(mat_a, "/band");
  }
#endif

  auto ret = band_to_tridiagonal<Backend::MC>(grid, uplo, band_size, mat_a);

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_band_to_tridiagonal_data) {
    file_band_to_tridiagonal->write(ret.tridiagonal, "/tridiag");
  }

  std::optional<matrix::internal::FileHDF5> file_tridiag;

  if (getTuneParameters().debug_dump_trisolver_data) {
    file_tridiag = matrix::internal::FileHDF5(grid.fullCommunicator(), "tridiag.h5");
    file_tridiag->write(ret.tridiagonal, "/tridiag");
  }
#endif

  tridiagonal_eigensolver<B>(grid, ret.tridiagonal, evals, mat_e);

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_trisolver_data) {
    file_tridiag->write(evals, "/evals");
    file_tridiag->write(mat_e, "/evecs");
  }
#endif

  bt_band_to_tridiagonal<B>(grid, band_size, mat_e, ret.hh_reflectors);
  bt_reduction_to_band<B>(grid, band_size, mat_e, mat_a, mat_taus);

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_eigensolver_data) {
    file_eigensolver->write(evals, "/evals");
    file_eigensolver->write(mat_e, "/evecs");
  }
#endif
}
}
