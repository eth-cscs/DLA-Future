//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#ifdef DLAF_WITH_CUDA
#pragma once

#include <cuda_runtime.h>

#include "dlaf/matrix/tile.h"

namespace dlaf {

namespace matrix {

/// Print a tile in csv format to standard output
template <class Format, class T>
void print(Format format, const Tile<const T, Device::GPU>& tile, std::ostream& os = std::cout,
           cudaStream_t stream = NULL) {
  const auto size = tile.size();
  Tile<T, Device::CPU> tile_h(size, memory::MemoryView<T, Device::CPU>(size.rows() * size.cols()),
                              size.rows());

  internal::copy_o(tile, tile_h, stream);
  DLAF_CUDA_CALL(cudaStreamSynchronize(stream));
  print(format, tile_h, os);
}

}
}
#endif
