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

#ifdef DLAF_WITH_GPU

#include <whip.hpp>

/// @file

#define DLAF_DECLARE_CUSOLVER_ASSERT_INFO(func) void assertInfo##func(whip::stream_t stream, int* info)

namespace dlaf::gpulapack::internal {

DLAF_DECLARE_CUSOLVER_ASSERT_INFO(Potrf);
DLAF_DECLARE_CUSOLVER_ASSERT_INFO(Hegst);

}

#endif
