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

#include <cuda_runtime.h>

/// @file

#define DLAF_DECLARE_CUSOLVER_ASSERT_INFO(func) void assertInfo##func(cudaStream_t stream, int* info)

namespace dlaf {
namespace cusolver {

DLAF_DECLARE_CUSOLVER_ASSERT_INFO(Potrf);
DLAF_DECLARE_CUSOLVER_ASSERT_INFO(Hegst);

}
}
