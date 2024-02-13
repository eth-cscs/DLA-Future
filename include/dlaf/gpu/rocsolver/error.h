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

#include <string>

#include <dlaf/gpu/lapack/api.h>
#include <dlaf/gpu/rocblas/error.h>

#ifdef DLAF_WITH_HIP

namespace dlaf::gpulapack::rocsolver::internal {

inline std::string getErrorString(rocblas_status st) {
  return dlaf::gpublas::rocblas::internal::getErrorString(st);
}

}
#endif
