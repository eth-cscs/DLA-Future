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

#include <type_traits>

#if DLAF_WITH_CUDA
#include <cuda_runtime.h>

#include "dlaf/cuda/error.h"
#endif

#include "dlaf/common/callable_object.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/partial_transform.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"

namespace dlaf {
namespace matrix {
namespace internal {
template <Device Source, Device Destination>
struct CopyBackend;

template <>
struct CopyBackend<Device::CPU, Device::CPU> {
  static constexpr Backend value = Backend::MC;
};

#ifdef DLAF_WITH_CUDA
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

template <typename T, Device Source, Device Destination>
struct CopyTile;

template <typename T>
struct CopyTile<T, Device::CPU, Device::CPU> {
  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::CPU>& destination) {
    dlaf::tile::internal::lacpy<T>(source, destination);
  }
};

#if DLAF_WITH_CUDA
template <typename T>
struct CopyTile<T, Device::CPU, Device::GPU> {
  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    DLAF_CUDA_CALL(cudaMemcpy2D(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                                ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyHostToDevice));
  }

  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination, cudaStream_t stream) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    DLAF_CUDA_CALL(cudaMemcpy2DAsync(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                                     ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyHostToDevice,
                                     stream));
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

    DLAF_CUDA_CALL(cudaMemcpy2D(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                                ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDeviceToHost));
  }

  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::CPU>& destination, cudaStream_t stream) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    DLAF_CUDA_CALL(cudaMemcpy2DAsync(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                                     ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDeviceToHost,
                                     stream));
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

    DLAF_CUDA_CALL(cudaMemcpy2D(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                                ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDeviceToDevice));
  }

  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination, cudaStream_t stream) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    DLAF_CUDA_CALL(cudaMemcpy2DAsync(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                                     ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDeviceToDevice,
                                     stream));
  }
};
#endif
}

/// Copy a input tile to an output tile.
template <typename T, Device Source, Device Destination, typename... Ts>
void copy(const Tile<const T, Source>& source, const Tile<T, Destination>& destination, Ts&&... ts) {
  DLAF_ASSERT_HEAVY(source.size() == destination.size(), source.size(), destination.size());
  internal::CopyTile<T, Source, Destination>::call(source, destination, std::forward<Ts>(ts)...);
}

/// Copy a subregion of an input tile to an output tile.
template <class T>
void copy(TileElementSize region, TileElementIndex in_idx, const Tile<const T, Device::CPU>& in,
          TileElementIndex out_idx, const Tile<T, Device::CPU>& out) {
  dlaf::tile::internal::lacpy<T>(region, in_idx, in, out_idx, out);
}

DLAF_MAKE_CALLABLE_OBJECT(copy);

// copy overload taking a policy and a sender, returning a sender. This can be
// used in task graphs.
template <Backend B, typename Sender,
          typename = std::enable_if_t<hpx::execution::experimental::is_sender_v<Sender>>>
auto copy(const dlaf::internal::Policy<B> p, Sender&& s) {
  return dlaf::internal::transform<B>(p, copy_o, std::forward<Sender>(s));
}

// copy overload taking a policy, returning a partially applied algorithm. This
// can be used in task graphs with the | operator.
template <Backend B>
auto copy(const dlaf::internal::Policy<B> p) {
  return dlaf::internal::PartialTransform{p, copy_o};
}

// copy overload taking a policy and plain arguments. This is a blocking call.
template <Backend B, typename T1, typename T2>
void copy(const dlaf::internal::Policy<B> p, T1&& t1, T2&& t2) {
  hpx::execution::experimental::sync_wait(
      copy(p, hpx::execution::experimental::just(std::forward<T1>(t1), std::forward<T2>(t2))));
}

/// Helper struct for copying a given tile to an identical tile on Destination.
///
/// Defines a call operator which allocates a tile of the same dimensions as the
/// input tile on Destination, and then copies the input tile to the output
/// tile.
///
/// This is useful for use with dataflow, since the output tile is allocated
/// only when the input tile is ready.
template <Device Destination>
struct Duplicate {
  template <typename T, Device Source, typename... Ts>
  Tile<T, Destination> operator()(const Tile<T, Source>& source, Ts&&... ts) {
    auto source_size = source.size();
    dlaf::memory::MemoryView<std::remove_const_t<T>, Destination> mem_view(source_size.linear_size());
    Tile<std::remove_const_t<T>, Destination> destination(source_size, std::move(mem_view),
                                                          source_size.rows());
    copy(source, destination, std::forward<decltype(ts)>(ts)...);
    return Tile<T, Destination>(std::move(destination));
  }
};

/// Helper function for duplicating an input tile to Destination asynchronously,
/// but only if the destination device is different from the source device.
///
/// When Destination and Source are the same, returns the input tile unmodified.
template <Device Destination, typename T, Device Source>
auto duplicateIfNeeded(hpx::future<Tile<T, Source>> tile) {
  if constexpr (Source == Destination) {
    return tile;
  }
  else {
    return hpx::execution::experimental::make_future(
        dlaf::internal::transform(dlaf::internal::Policy<
                                      internal::CopyBackend<Source, Destination>::value>(
                                      hpx::threads::thread_priority::normal),
                                  dlaf::matrix::Duplicate<Destination>{}, std::move(tile)));
  }
}

template <Device Destination, typename T, Device Source>
auto duplicateIfNeeded(hpx::shared_future<Tile<T, Source>> tile) {
  if constexpr (Source == Destination) {
    return tile;
  }
  else {
    return hpx::execution::experimental::make_future(
        dlaf::internal::transform(dlaf::internal::Policy<
                                      internal::CopyBackend<Source, Destination>::value>(
                                      hpx::threads::thread_priority::normal),
                                  dlaf::matrix::Duplicate<Destination>{},
                                  hpx::execution::experimental::keep_future(std::move(tile))));
  }
}
}
}
