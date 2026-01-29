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

#include <dlaf/communication/message.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>

namespace dlaf::comm {
/// Helper struct for determining the device to use for communication.
///
/// Contains a static value member, which will always be D if GPU-aware MPI
/// is enabled. If GPU-aware MPI is disabled, the value will always be
/// Device::CPU.
template <Device D>
struct CommunicationDeviceP2P {
  static constexpr Device value = D;
};

template <Device D>
struct CommunicationDeviceBroadcast {
  static constexpr Device value = D;
};

template <Device D>
struct CommunicationDeviceReduce {
  static constexpr Device value = D;
};

template <Device D>
struct CommunicationDeviceAllReduce {
  static constexpr Device value = D;
};

#if defined(DLAF_WITH_GPU) && !defined(DLAF_WITH_MPI_GPU_AWARE)
template <>
struct CommunicationDeviceP2P<Device::GPU> {
  static constexpr Device value = Device::CPU;
};

template <>
struct CommunicationDeviceBroadcast<Device::GPU> {
  static constexpr Device value = Device::CPU;
};
#endif

#if defined(DLAF_WITH_GPU) && (!defined(DLAF_WITH_MPI_GPU_AWARE) || defined(DLAF_WITH_MPI_GPU_AWARE_NO_REDUCE_OPS)
template <>
struct CommunicationDeviceReduce<Device::GPU> {
  static constexpr Device value = Device::CPU;
};

template <>
struct CommunicationDeviceAllReduce<Device::GPU> {
  static constexpr Device value = Device::CPU;
};
#endif

template <Device D>
inline constexpr auto CommunicationDeviceP2P_v = CommunicationDeviceP2P<D>::value;

template <Device D>
inline constexpr auto CommunicationDeviceBroadcast_v = CommunicationDeviceBroadcast<D>::value;

template <Device D>
inline constexpr auto CommunicationDeviceReduce_v = CommunicationDeviceReduce<D>::value;

template <Device D>
inline constexpr auto CommunicationDeviceAllReduce_v = CommunicationDeviceAllReduce<D>::value;
}
