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

/// @file

#include "dlaf/types.h"

namespace dlaf {
namespace comm {

/// Helper struct for determining the device to use for communication.
///
/// Contains a static value member, which will always be D if CUDA RDMA is
/// enabled. If CUDA RDMA is disabled, the value will always be Device::CPU.
template <Device D>
struct CommunicationDevice {
  static constexpr Device value = D;
};

#if defined(DLAF_WITH_CUDA) && !defined(DLAF_WITH_CUDA_RDMA)
template <>
struct CommunicationDevice<Device::GPU> {
  static constexpr Device value = Device::CPU;
};
#endif

}
}
