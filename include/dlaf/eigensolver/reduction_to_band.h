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

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf/eigensolver/reduction_to_band/mc.h"

namespace dlaf {
namespace eigensolver {

/// TODO Documentation
template <Backend backend, Device device, class T>
std::vector<hpx::shared_future<std::vector<T>>> reductionToBand(comm::CommunicatorGrid grid,
                                                                Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  // TODO fix for non-distributed
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);

  return internal::ReductionToBand<backend, device, T>::call(grid, mat_a);
}

}
}
