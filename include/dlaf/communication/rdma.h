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

/// @file

#include <pika/future.hpp>

#include "dlaf/communication/message.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/tile.h"
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

#if defined(DLAF_WITH_GPU) && !defined(DLAF_WITH_CUDA_RDMA)
template <>
struct CommunicationDevice<Device::GPU> {
  static constexpr Device value = Device::CPU;
};
#endif

template <Device D>
inline constexpr auto CommunicationDevice_v = CommunicationDevice<D>::value;

namespace internal {
/// Helper function for preparing a tile for sending.
///
/// Duplicates the tile to CPU memory if CUDA RDMA is not enabled for MPI.
/// Returns the tile unmodified otherwise.
template <Device D, typename T>
auto prepareSendTile(pika::shared_future<matrix::Tile<const T, D>> tile) {
  return matrix::duplicateIfNeeded<CommunicationDevice_v<D>>(std::move(tile));
}
}
}
}
