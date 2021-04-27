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

#include <hpx/local/future.hpp>

#include "dlaf/communication/message.h"
#include "dlaf/executors.h"
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

#if defined(DLAF_WITH_CUDA) && !defined(DLAF_WITH_CUDA_RDMA)
template <>
struct CommunicationDevice<Device::GPU> {
  static constexpr Device value = Device::CPU;
};
#endif

namespace internal {
/// Helper function for preparing a tile for sending.
///
/// Duplicates the tile to CPU memory if CUDA RDMA is not enabled for MPI.
/// Returns the tile unmodified otherwise.
template <Device D, typename T>
auto prepareSendTile(hpx::shared_future<matrix::Tile<const T, D>> tile) {
  return matrix::duplicateIfNeeded<CommunicationDevice<D>::value>(std::move(tile));
}

/// Helper function for handling a tile after receiving.
///
/// If CUDA RDMA is disabled, the tile returned from recvTile will always be on
/// the CPU. This helper duplicates to the GPU if the first template parameter
/// is a GPU device. The first template parameter must be given.
template <Device D, typename T>
auto handleRecvTile(hpx::future<matrix::Tile<const T, CommunicationDevice<D>::value>> tile) {
  return matrix::duplicateIfNeeded<D>(std::move(tile));
}
}
}
}
