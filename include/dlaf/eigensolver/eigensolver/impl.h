//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
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
#include "dlaf/eigensolver/band_to_tridiag.h"
#include "dlaf/eigensolver/bt_band_to_tridiag.h"
#include "dlaf/eigensolver/bt_reduction_to_band.h"
#include "dlaf/eigensolver/reduction_to_band.h"
#include "dlaf/eigensolver/tridiag_solver.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
EigensolverResult<T, D> Eigensolver<B, D, T>::call(blas::Uplo uplo, Matrix<T, D>& mat_a) {
  using common::internal::vector;

  const SizeType size = mat_a.size().rows();
  const SizeType band_size = mat_a.blockSize().rows();

  // need uplo check as reduction to band doesn't have the uplo argument yet.
  if (uplo != blas::Uplo::Lower)
    DLAF_UNIMPLEMENTED(uplo);

  auto taus = reductionToBand<B>(mat_a, band_size);
  auto ret = bandToTridiag<Backend::MC>(uplo, band_size, mat_a);

  matrix::Matrix<BaseType<T>, Device::CPU> evals(LocalElementSize(size, 1),
                                                 TileElementSize(mat_a.blockSize().rows(), 1));
  matrix::Matrix<T, Device::CPU> mat_e(LocalElementSize(size, size), mat_a.blockSize());

  eigensolver::tridiagSolver<Backend::MC>(ret.tridiagonal, evals, mat_e);

  // Note: This is just a temporary workaround. It will be removed as soon as we will have our
  // tridiagonal eigensolver implementation both on CPU and GPU.
  Matrix<BaseType<T>, D> evals_device = [&]() {
    if constexpr (D == Device::CPU)
      return std::move(evals);
    else {
      Matrix<BaseType<T>, D> evals_device(evals.distribution());
      dlaf::matrix::copy(evals, evals_device);
      return evals_device;
    }
  }();

  Matrix<T, D> mat_e_device = [&]() {
    if constexpr (D == Device::CPU)
      return std::move(mat_e);
    else {
      Matrix<T, D> e(mat_e.distribution());
      dlaf::matrix::copy(mat_e, e);
      return e;
    }
  }();

  backTransformationBandToTridiag<B>(band_size, mat_e_device, ret.hh_reflectors);
  backTransformationReductionToBand<B>(band_size, mat_e_device, mat_a, taus);

  return {std::move(evals_device), std::move(mat_e_device)};
}

}
