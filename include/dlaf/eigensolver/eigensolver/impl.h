//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <cmath>
#include <vector>

#include "dlaf/eigensolver/eigensolver/api.h"

#include "dlaf/blas/tile.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/band_to_tridiag.h"
#include "dlaf/eigensolver/bt_band_to_tridiag.h"
#include "dlaf/eigensolver/bt_reduction_to_band.h"
#include "dlaf/eigensolver/get_band_size.h"
#include "dlaf/eigensolver/reduction_to_band.h"
#include "dlaf/eigensolver/tridiag_solver.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
EigensolverResult<T, D> Eigensolver<B, D, T>::call(blas::Uplo uplo, Matrix<T, D>& mat_a) {
  using common::internal::vector;

  const SizeType size = mat_a.size().rows();
  const SizeType band_size = getBandSize(mat_a.blockSize().rows());

  // need uplo check as reduction to band doesn't have the uplo argument yet.
  if (uplo != blas::Uplo::Lower)
    DLAF_UNIMPLEMENTED(uplo);

  auto taus = reductionToBand<B>(mat_a, band_size);
  auto ret = bandToTridiag<Backend::MC>(uplo, band_size, mat_a);

  // Note:
  // Since reduction from band to tridiagonal happens on MC for all backends, but eigensolver
  // requires tridiagonal matrix to be on CPU or GPU depending on the backend used, next snippet
  // ensures that tridiagonal matrix gets copied if needed (i.e. just for GPU backend).
  matrix::Matrix<BaseType<T>, D> tridiagonal = [&ret]() {
    if constexpr (B == Backend::MC) {
      return std::move(ret.tridiagonal);
    }
    else {
      matrix::Matrix<BaseType<T>, D> tridiagonal(ret.tridiagonal.distribution());
      copy(ret.tridiagonal, tridiagonal);
      return tridiagonal;
    }
  }();

  matrix::Matrix<BaseType<T>, D> evals(LocalElementSize(size, 1),
                                       TileElementSize(mat_a.blockSize().rows(), 1));
  matrix::Matrix<T, D> mat_e(LocalElementSize(size, size), mat_a.blockSize());

  eigensolver::tridiagSolver<B>(tridiagonal, evals, mat_e);

  backTransformationBandToTridiag<B>(band_size, mat_e, ret.hh_reflectors);
  backTransformationReductionToBand<B>(band_size, mat_e, mat_a, taus);

  return {std::move(evals), std::move(mat_e)};
}

template <Backend B, Device D, class T>
EigensolverResult<T, D> Eigensolver<B, D, T>::call(comm::CommunicatorGrid grid, blas::Uplo uplo,
                                                   Matrix<T, D>& mat_a) {
  using common::internal::vector;

  const SizeType size = mat_a.size().rows();
  const SizeType band_size = getBandSize(mat_a.blockSize().rows());

  // need uplo check as reduction to band doesn't have the uplo argument yet.
  if (uplo != blas::Uplo::Lower)
    DLAF_UNIMPLEMENTED(uplo);

  auto taus = reductionToBand<B>(grid, mat_a, band_size);
  auto ret = bandToTridiag<Backend::MC>(grid, uplo, band_size, mat_a);

  // Note:
  // Since reduction from band to tridiagonal happens on MC for all backends, but eigensolver
  // requires tridiagonal matrix to be on CPU or GPU depending on the backend used, next snippet
  // ensures that tridiagonal matrix gets copied if needed (i.e. just for GPU backend).
  matrix::Matrix<BaseType<T>, D> tridiagonal = [&ret]() {
    if constexpr (B == Backend::MC) {
      return std::move(ret.tridiagonal);
    }
    else {
      matrix::Matrix<BaseType<T>, D> tridiagonal(ret.tridiagonal.distribution());
      copy(ret.tridiagonal, tridiagonal);
      return tridiagonal;
    }
  }();

  matrix::Matrix<BaseType<T>, D> evals(LocalElementSize(size, 1),
                                       TileElementSize(mat_a.blockSize().rows(), 1));
  matrix::Matrix<T, D> mat_e(GlobalElementSize(size, size), mat_a.blockSize(), grid);

  eigensolver::tridiagSolver<B>(grid, tridiagonal, evals, mat_e);

  backTransformationBandToTridiag<B>(grid, band_size, mat_e, ret.hh_reflectors);
  backTransformationReductionToBand<B>(grid, band_size, mat_e, mat_a, taus);

  return {std::move(evals), std::move(mat_e)};
}

}
