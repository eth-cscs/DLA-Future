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
#include "dlaf/eigensolver/backtransformation/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Eigenvalue back-transformation implementation on local memory.
///
/// It solves C = C - V T V^H C
/// where C is a generic matrix, T an upper triangular matrix (triangular factor T) and V a lower
/// triangular matrix (reflectors). Triangular factor T is computed from values of taus and block
/// reflector (in V) using @see computeTFactor() method.
///
/// @param mat_c contains on entry the generic matrix C, while on exit it contains the upper matrix
/// resulting from the eigenvalue back-transformation.
/// @param mat_v is a lower triangular matrix, containing Householder vectors (reflectors).
/// @param taus is a vectors of scalar, associated with the related elementary reflector.
/// The last two paramteres (@param mat_v and @param taus) are used to compute the T factor matrix
/// (compact WY representation of the Householder reflectors).
/// @pre mat_c is not distributed,
/// @pre mat_v is not distributed.
template <Backend backend, Device device, class T>
void backTransformation(Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
                        common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);
  DLAF_ASSERT(matrix::local_matrix(mat_v), mat_v);
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);

  internal::BackTransformation<backend, device, T>::call_FC(mat_c, mat_v, taus);
}

/// Eigenvalue back-transformation implementation on distributed memory.
///
/// It solves C = C - V T V^H C
/// where C is a generic matrix, T an upper triangular matrix (triangular factor T) and V a lower
/// triangular matrix (reflectors). Triangular factor T is computed from values of taus and block
/// reflector (in V) using @see computeTFactor() method.
///
/// @param mat_c contains on entry the generic matrix C, while on exit it contains the upper matrix
/// resulting from the eigenvalue back-transformation.
/// @param mat_v is a lower triangular matrix, containing Householder vectors (reflectors).
/// @param taus is a vectors of scalar, associated with the related elementary reflector.
/// The last two paramteres (@param mat_v and @param taus) are used to compute the T factor matrix
/// (compact WY representation of the Householder reflectors).
/// @pre mat_c is distributed,
/// @pre mat_v is distributed according to grid.
template <Backend backend, Device device, class T>
void backTransformation(comm::CommunicatorGrid grid, Matrix<T, device>& mat_c,
                        Matrix<const T, device>& mat_v,
                        common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);
  DLAF_ASSERT(matrix::equal_process_grid(mat_c, grid), mat_c, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_v, grid), mat_v, grid);

  internal::BackTransformation<backend, device, T>::call_FC(grid, mat_c, mat_v, taus);
}

}
}
