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
#include "dlaf/matrix/matrix.h"
#include "dlaf/solver/backtransformation/mc.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace solver {

/// Eigenvalue back-transformation implementation on local memory
/// Solves the equation: C = C - V T V^H A
/// where C is a square matrix, T an upper triangular matrix (T factors) and V a lower triangular matrix
/// (reflectors).
///
/// @param mat_c contains the square matrix C,
/// @param mat_v contains the lower triangular matrix of reflectors,
/// @param taus is an array of taus, associated with the related elementary reflector.
template <Backend backend, Device device, class T>
void backTransformation(Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
                        common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  //// TODO preconditions are enough?
  //// TODO blocksize? So far should be one
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);
  DLAF_ASSERT(matrix::local_matrix(mat_v), mat_v);
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);

  internal::BackTransformation<backend, device, T>::call_FC(mat_c, mat_v, taus);
}

/// Eigenvalue back-transformation implementation on distributed memory
/// Solves the equation: C = C - V T V^H A
/// where C is a square matrix, T an upper triangular matrix (T factors) and V a lower triangular matrix
/// (reflectors).
///
/// @param mat_c contains the square matrix C,
/// @param mat_v contains the lower triangular matrix of reflectors,
/// @param taus is an array of taus, associated with the related elementary reflector.
template <Backend backend, Device device, class T>
void backTransformation(comm::CommunicatorGrid grid, Matrix<T, device>& mat_c,
                        Matrix<const T, device>& mat_v, common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  // TODO preconditions are enough?
  // TODO blocksize? So far should be one
  //    DLAF_ASSERT(matrix::square_size(mat_c), mat_c);
  //    DLAF_ASSERT(matrix::square_size(mat_v), mat_v);
  //    DLAF_ASSERT(matrix::square_size(mat_t), mat_t);
  //   DLAF_ASSERT(matrix::square_blocksize(mat_c), mat_c);
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);
  DLAF_ASSERT(matrix::equal_process_grid(mat_c, grid), mat_c, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_v, grid), mat_v, grid);

  internal::BackTransformation<backend, device, T>::call_FC(grid, mat_c, mat_v, taus);
}

}
}
