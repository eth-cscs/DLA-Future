//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <blas.hh>
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/band_to_tridiag/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// TODO
/// Implementation on local memory.
///
/// @param mat_a contains the Hermitian band matrix A (if A is real, the matrix is symmetric).
/// @pre mat_a has a square size,
/// @pre mat_a has a square block size,
/// @pre b is a divisor of mat_a.blockSize().cols(),
/// @pre mat_a is not distributed.
template <Backend backend, Device device, class T>
auto bandToTridiag(blas::Uplo uplo, SizeType band_size, Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);

  switch (uplo) {
    case blas::Uplo::Lower:
      return internal::BandToTridiag<backend, device, T>::call_L(band_size, mat_a);
      break;
    case blas::Uplo::Upper:
      DLAF_UNIMPLEMENTED(uplo);
      break;
    case blas::Uplo::General:
      DLAF_UNIMPLEMENTED(uplo);
      break;
  }

  using dlaf::common::internal::vector;
  using RetVec = vector<hpx::shared_future<vector<BaseType<T>>>>;
  return std::tuple<RetVec, RetVec>();
}

}
}
