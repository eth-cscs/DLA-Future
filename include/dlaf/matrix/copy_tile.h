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

#include <type_traits>

#if DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include "dlaf/common/callable_object.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/keep_if_shared_future.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"

namespace dlaf::matrix {

namespace internal {
template <Device Source, Device Destination>
struct CopyBackend;

template <>
struct CopyBackend<Device::CPU, Device::CPU> {
  static constexpr Backend value = Backend::MC;
};

#ifdef DLAF_WITH_GPU
template <>
struct CopyBackend<Device::CPU, Device::GPU> {
  static constexpr Backend value = Backend::GPU;
};

template <>
struct CopyBackend<Device::GPU, Device::CPU> {
  static constexpr Backend value = Backend::GPU;
};

template <>
struct CopyBackend<Device::GPU, Device::GPU> {
  static constexpr Backend value = Backend::GPU;
};
#endif

template <Device Source, Device Destination>
inline constexpr auto CopyBackend_v = CopyBackend<Source, Destination>::value;

template <typename T, Device Source, Device Destination>
struct CopyTile;

template <typename T>
struct CopyTile<T, Device::CPU, Device::CPU> {
  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::CPU>& destination) {
    if constexpr (IsFloatingPointOrComplex_v<T>) {
      dlaf::tile::lacpy<T>(source, destination);
    }
    else {
      // Fall back to a generic copy for non-floating point and non-complex
      // types (which lapack::lacpy doesn't support)
      common::copy(common::make_data(source), common::make_data(destination));
    }
  }
};

#if DLAF_WITH_GPU
template <typename T>
struct CopyTile<T, Device::CPU, Device::GPU> {
  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    whip::memcpy_2d(destination.ptr(), ld_destination * sizeof(T), source.ptr(), ld_source * sizeof(T),
                    m * sizeof(T), n, whip::memcpy_host_to_device);
  }

  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination, whip::stream_t stream) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    whip::memcpy_2d_async(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                          ld_source * sizeof(T), m * sizeof(T), n, whip::memcpy_host_to_device, stream);
  }
};

template <typename T>
struct CopyTile<T, Device::GPU, Device::CPU> {
  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::CPU>& destination) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    whip::memcpy_2d(destination.ptr(), ld_destination * sizeof(T), source.ptr(), ld_source * sizeof(T),
                    m * sizeof(T), n, whip::memcpy_device_to_host);
  }

  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::CPU>& destination, whip::stream_t stream) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    whip::memcpy_2d_async(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                          ld_source * sizeof(T), m * sizeof(T), n, whip::memcpy_device_to_host, stream);
  }
};

template <typename T>
struct CopyTile<T, Device::GPU, Device::GPU> {
  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    whip::memcpy_2d(destination.ptr(), ld_destination * sizeof(T), source.ptr(), ld_source * sizeof(T),
                    m * sizeof(T), n, whip::memcpy_device_to_device);
  }

  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination, whip::stream_t stream) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    whip::memcpy_2d_async(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                          ld_source * sizeof(T), m * sizeof(T), n, whip::memcpy_device_to_device,
                          stream);
  }
};
#endif

/// Copy a input tile to an output tile.
template <typename T, Device Source, Device Destination, typename... Ts>
void copy(const Tile<const T, Source>& source, const Tile<T, Destination>& destination, Ts&&... ts) {
  DLAF_ASSERT_HEAVY(source.size() == destination.size(), source.size(), destination.size());
  internal::CopyTile<T, Source, Destination>::call(source, destination, std::forward<Ts>(ts)...);
}

DLAF_MAKE_CALLABLE_OBJECT(copy);
}

DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Plain, copy,
                                     internal::copy_o)

/// Helper struct for copying a given tile to a tile on Destination.
///
/// Defines a call operator which allocates a tile of the same dimensions as the
/// input tile on Destination, and then copies the input tile to the output
/// tile.
/// The allocated tile on Destination will be of the same size of the input one, but it will be also
/// contiguous, i.e. whatever the input leading dimension is, the destination one will have the leading
/// dimension equal to the number of rows.
template <Device Destination>
struct Duplicate {
  template <typename T, Device Source, typename... Ts>
  Tile<T, Destination> operator()(const Tile<const T, Source>& source, Ts&&... ts) {
    auto source_size = source.size();
    dlaf::memory::MemoryView<T, Destination> mem_view(source_size.linear_size());
    Tile<T, Destination> destination(source_size, std::move(mem_view), source_size.rows());
    internal::copy(source, destination, std::forward<decltype(ts)>(ts)...);
    return Tile<T, Destination>(std::move(destination));
  }
};

/// Helper struct for duplicating a given tile to a tile on Destination without
/// copying the contents.
///
/// Defines a call operator which allocates a tile of the same dimensions as the
/// input tile on Destination.  The allocated tile on Destination will be of the
/// same size of the input one, but it will be also contiguous, i.e. whatever
/// the input leading dimension is, the destination one will have the leading
/// dimension equal to the number of rows.
template <Device Destination>
struct DuplicateNoCopy {
  template <typename T, Device Source>
  Tile<T, Destination> operator()(const Tile<const T, Source>& source) {
    auto source_size = source.size();
    dlaf::memory::MemoryView<T, Destination> mem_view(source_size.linear_size());
    Tile<T, Destination> destination(source_size, std::move(mem_view), source_size.rows());
    return Tile<T, Destination>(std::move(destination));
  }
};
}
