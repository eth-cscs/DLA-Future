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
#include "dlaf/lapack_tile.h"
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
                                ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDefault));
  }

  static void call(const matrix::Tile<const T, Device::CPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination, cudaStream_t stream) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    DLAF_CUDA_CALL(cudaMemcpy2DAsync(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                                     ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDefault,
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
                                ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDefault));
  }

  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::CPU>& destination, cudaStream_t stream) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    DLAF_CUDA_CALL(cudaMemcpy2DAsync(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                                     ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDefault,
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
                                ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDefault));
  }

  static void call(const matrix::Tile<const T, Device::GPU>& source,
                   const matrix::Tile<T, Device::GPU>& destination, cudaStream_t stream) {
    const std::size_t m = to_sizet(source.size().rows());
    const std::size_t n = to_sizet(source.size().cols());
    const std::size_t ld_source = to_sizet(source.ld());
    const std::size_t ld_destination = to_sizet(destination.ld());

    DLAF_CUDA_CALL(cudaMemcpy2DAsync(destination.ptr(), ld_destination * sizeof(T), source.ptr(),
                                     ld_source * sizeof(T), m * sizeof(T), n, cudaMemcpyDefault,
                                     stream));
  }
};
#endif
}

template <typename T, Device Source, Device Destination, typename... Ts>
void copy(const Tile<const T, Source>& source, const Tile<T, Destination>& destination, Ts&&... ts) {
  DLAF_ASSERT_HEAVY(source.size() == destination.size(), source.size(), destination.size());
  internal::CopyTile<T, Source, Destination>::call(source, destination, std::forward<Ts>(ts)...);
}

template <class T>
void copy(TileElementSize region, TileElementIndex in_idx, const matrix::Tile<const T, Device::CPU>& in,
          TileElementIndex out_idx, const matrix::Tile<T, Device::CPU>& out) {
  dlaf::tile::lacpy<T>(region, in_idx, in, out_idx, out);
}

DLAF_MAKE_CALLABLE_OBJECT(copy);
}
}
