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

namespace dlaf {
namespace matrix {
namespace internal {
template <typename T, Device Source, Device Destination>
struct CopyTile;

template <typename T>
struct CopyTile<T, Device::CPU, Device::CPU> {
  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::CPU>& destination) {
    dlaf::tile::lacpy<T>(source, destination);
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
  dlaf::tile::lacpy<T>(region, in_idx, in, out_idx, out);
}

DLAF_MAKE_CALLABLE_OBJECT(copy);

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

namespace internal {
template <Device Destination, Device Source>
struct DuplicateIfNeeded {
  template <typename T, template <class> class Future>
  static auto call(Future<Tile<T, Source>> tile) {
    return getUnwrapReturnValue(hpx::dataflow(dlaf::getCopyExecutor<Source, Destination>(),
                                              unwrapExtendTiles(Duplicate<Destination>{}), tile));
  }
};

template <Device SourceDestination>
struct DuplicateIfNeeded<SourceDestination, SourceDestination> {
  template <typename T, template <class> class Future>
  static auto call(Future<Tile<T, SourceDestination>> tile) {
    return tile;
  }
};
}

/// Helper function for duplicating an input tile to Destination asynchronously,
/// but only if the destination device is different from the source device.
///
/// When Destination and Source are the same, returns the input tile unmodified.
template <Device Destination, typename T, Device Source, template <class> class Future>
auto duplicateIfNeeded(Future<Tile<T, Source>> tile) {
  return internal::DuplicateIfNeeded<Destination, Source>::call(std::move(tile));
}
}
}
