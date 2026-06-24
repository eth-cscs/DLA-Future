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

/// @file

namespace dlaf::memory {

enum class AllocateOn { Construction, Demand };

struct AllocateOnDefault {};

using AllocateOnType = std::variant<AllocateOnDefault, AllocateOn>;

namespace internal {
enum class AllocationStatus { Empty, WaitAllocation, Allocated, ExternallyManaged };
}

}
