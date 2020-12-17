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

  /// Eigenvalue back-transformation implementation on local memory
  /// Solves the equation: C = C - V T V^H A
  /// where C is a square matrix, T an upper triangular matrix (T factors) and V a lower triangular matrix (reflectors).  
  ///
  /// @param mat_c contains the square matrix C,
  /// @param mat_v contains the lower triangular matrix of reflectors,
  /// @param mat_t contains the upper triangular matrix of T factors.
  template <Backend backend, Device device, class T>
  void backTransformation(Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
                         Matrix<T, device>& mat_t){
  // TODO preconditions are enough?
  // TODO blocksize? So far should be one
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
