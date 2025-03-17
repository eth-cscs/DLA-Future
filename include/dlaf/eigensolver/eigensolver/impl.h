//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <atomic>
#include <cmath>
#include <optional>
#include <sstream>

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
                                Matrix<T, D>& mat_e, const SizeType eigenvalues_index_begin,
                                const SizeType eigenvalues_index_end) {
  const SizeType band_size = getBandSize(mat_a.blockSize().rows());

  // need uplo check as reduction to band doesn't have the uplo argument yet.
  if (uplo != blas::Uplo::Lower)
    DLAF_UNIMPLEMENTED(uplo);

  auto mat_taus = reduction_to_band<B>(mat_a, band_size);
  auto ret = band_to_tridiagonal<Backend::MC>(uplo, band_size, mat_a);

  tridiagonal_eigensolver<B>(ret.tridiagonal, evals, mat_e);

  auto spec = matrix::util::internal::sub_matrix_spec_slice_cols(mat_e, eigenvalues_index_begin,
                                                                 eigenvalues_index_end);

  matrix::internal::MatrixRef mat_e_ref(mat_e, spec);
  bt_band_to_tridiagonal<B>(band_size, mat_e_ref, ret.hh_reflectors);
  bt_reduction_to_band<B>(band_size, mat_e_ref, mat_a, mat_taus);
}

template <Backend B, Device D, class T>
void Eigensolver<B, D, T>::call(comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, D>& mat_a,
                                Matrix<BaseType<T>, D>& evals, Matrix<T, D>& mat_e,
                                const SizeType eigenvalues_index_begin,
                                const SizeType eigenvalues_index_end) {
  const SizeType band_size = getBandSize(mat_a.blockSize().rows());

  // need uplo check as reduction to band doesn't have the uplo argument yet.
  if (uplo != blas::Uplo::Lower)
    DLAF_UNIMPLEMENTED(uplo);

#ifdef DLAF_WITH_HDF5
  static std::atomic<size_t> num_eigensolver_calls = 0;
  std::stringstream fname;
  fname << "eigensolver-" << matrix::internal::TypeToString_v<T> << "-"
        << std::to_string(num_eigensolver_calls) << ".h5";
  std::optional<matrix::internal::FileHDF5> file;

  if (getTuneParameters().debug_dump_eigensolver_data) {
    file = matrix::internal::FileHDF5(grid.fullCommunicator(), fname.str());
    file->write(mat_a, "/input");
  }
#endif

  auto mat_taus = reduction_to_band<B>(grid, mat_a, band_size);

  auto ret = band_to_tridiagonal<Backend::MC>(grid, uplo, band_size, mat_a);

  tridiagonal_eigensolver<B>(grid, ret.tridiagonal, evals, mat_e);

  auto spec = matrix::util::internal::sub_matrix_spec_slice_cols(mat_e, eigenvalues_index_begin,
                                                                 eigenvalues_index_end);
  matrix::internal::MatrixRef mat_e_ref(mat_e, spec);

  bt_band_to_tridiagonal<B>(grid, band_size, mat_e_ref, ret.hh_reflectors);
  bt_reduction_to_band<B>(grid, band_size, mat_e_ref, mat_a, mat_taus);

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_eigensolver_data) {
    file->write(evals, "/evals");
    file->write(mat_e, "/evecs");
  }

  num_eigensolver_calls++;
#endif
}
}
