//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#ifdef DLAF_WITH_GPU
#pragma once

#include <iostream>

#include "dlaf/gpu/api.h"
#include "dlaf/matrix/tile.h"

namespace dlaf::matrix {

/// Print a GPU tile in the given format
///
/// Copies the tile to CPU and output its elements.
/// It can be invoked in 4 different ways:
/// - when the output stream is omitted std::cout is used.
/// - when the cuda stream is omitted a new stream is created internally to execute the copy.
/// When this function is used in a GPU task it is recommended to invoke it with the task stream
/// (extracted from the cublas or cusolver handle if necessary).
/// Note: the cuda stream is synchronized internally if the tile is not empty.
template <class Format, class T>
void print(Format format, const Tile<const T, Device::GPU>& tile, std::ostream& os,
           cudaStream_t stream) {
  const auto size = tile.size();
  const auto ld = std::max<SizeType>(1, size.rows());
  Tile<T, Device::CPU> tile_h(size, memory::MemoryView<T, Device::CPU>(size.linear_size()), ld);

  if (!size.isEmpty()) {
    internal::copy_o(tile, tile_h, stream);
    DLAF_GPU_CHECK_ERROR(cudaStreamSynchronize(stream));
  }

  print(format, tile_h, os);
}

template <class Format, class T>
void print(Format format, const Tile<const T, Device::GPU>& tile, cudaStream_t stream) {
  print(format, tile, std::cout, stream);
}

template <class Format, class T>
void print(Format format, const Tile<const T, Device::GPU>& tile, std::ostream& os = std::cout) {
  cudaStream_t stream = NULL;
  if (!tile.size().isEmpty())
    DLAF_GPU_CHECK_ERROR(cudaStreamCreate(&stream));

  print(format, tile, os, stream);

  if (!tile.size().isEmpty())
    DLAF_GPU_CHECK_ERROR(cudaStreamDestroy(stream));
}

}
#endif
