//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_GPU

#include <whip.hpp>

#define DLAF_DECLARE_CUSOLVER_ASSERT_INFO(func) void assert_info_##func(whip::stream_t stream, int* info)

namespace dlaf::gpulapack::internal {

DLAF_DECLARE_CUSOLVER_ASSERT_INFO(potrf);
DLAF_DECLARE_CUSOLVER_ASSERT_INFO(hegst);
DLAF_DECLARE_CUSOLVER_ASSERT_INFO(trtri);

}

#endif
