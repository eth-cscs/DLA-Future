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

#include <blas.hh>
#include <hpx/include/util.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

template <class T>
struct ExtraBuffers {
  ExtraBuffers(Matrix<T, Device::CPU>& mat, std::size_t num_extra_buffers)
      : num_extra_buffers_(num_extra_buffers), base_(mat),
        extra_(LocalElementSize(mat.blockSize().rows() * num_extra_buffers, mat.blockSize().cols()),
               mat.blockSize()) {
    DLAF_ASSERT(mat.nrTiles().rows() == 1, mat.nrTiles());
    DLAF_ASSERT(mat.nrTiles().cols() == 1, mat.nrTiles());
    clear();
  }

  auto get_buffer(const SizeType index) {
    const auto idx = num_extra_buffers_ != 0 ? index % num_extra_buffers_ : 0;
    if (idx == 0)
      return base_(LocalTileIndex(0, 0));
    else
      return extra_(LocalTileIndex(idx - 1, 0));
  }

  void reduce() {
    using hpx::util::unwrapping;
    using hpx::util::annotated_function;

    for (const auto& idx_buffer : iterate_range2d(extra_.nrTiles()))
      hpx::dataflow(unwrapping([](auto&& tile_result, auto&& tile_extra) {
                      for (SizeType j = 0; j < tile_result.size().cols(); ++j) {
                        // clang-format off
                        blas::axpy(
                            tile_result.size().rows(), 1,
                            tile_extra.ptr({0, j}), 1,
                            tile_result.ptr({0, j}), 1);
                        // clang-format on
                      }
                    }),
                    base_(LocalTileIndex(0, 0)), extra_.read(idx_buffer));
  }

  void clear() {
    dlaf::matrix::util::set(extra_, [](...) { return 0; });
  }

  std::size_t num_extra_buffers_;
  Matrix<T, Device::CPU>& base_;
  Matrix<T, Device::CPU> extra_;
};

}
}
