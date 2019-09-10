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

#include "dlaf/communicator/communicator.h"

namespace dlaf {
namespace comm {

/// It exposes the release method of Communicator that it is kept protected to not allow the user
/// to release a Communicator.
struct CommunicatorReleasable : Communicator {
  void release() noexcept(false) { Communicator::release(); }
};

/// internal helper function to release communicator
auto release_communicator = [](Communicator comm) {
  static_cast<CommunicatorReleasable*>(&comm)->release();
};

}
}
