//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <blas.hh>
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/solver/backtransformation/mc.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace solver {

  /// Eigenvalue back-transformation
  ///
  /// TODO: FIX THIS DESCRIPTION
  template <Backend backend, Device device, class T>
  void backTransformation(Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
                         Matrix<T, device>& mat_t){
  // TODO add preconditions
  DLAF_ASSERT(matrix::square_size(mat_c), mat_c);
  DLAF_ASSERT(matrix::square_blocksize(mat_c), mat_c);
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);
  DLAF_ASSERT(matrix::square_size(mat_v), mat_v);
  DLAF_ASSERT(matrix::local_matrix(mat_v), mat_v);
  DLAF_ASSERT(matrix::square_size(mat_t), mat_t);
  DLAF_ASSERT(matrix::local_matrix(mat_t), mat_t);

  internal::BackTransformation<backend, device, T>::call_FC(mat_c, mat_v, mat_t);
  }

}
}
