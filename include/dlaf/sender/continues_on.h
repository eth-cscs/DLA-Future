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

#include <pika/execution.hpp>

namespace dlaf::internal {
#if PIKA_VERSION_FULL < 0x001D00  // < 0.29.0
inline constexpr pika::execution::experimental::transfer_t continues_on{};
#else
using pika::execution::experimental::continues_on;
#endif
}
