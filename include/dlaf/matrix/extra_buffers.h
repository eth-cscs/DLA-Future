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
#include <hpx/futures/future.hpp>
#include <hpx/include/util.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

template <class T>
class ExtraBuffers {
  using tile_t = matrix::Tile<T, Device::CPU>;
  using promise_t = hpx::lcos::local::promise<tile_t>;
  using future_t = hpx::lcos::future<tile_t>;

public:
  // TODO check if there is a way to get blocksize info from the tile without asking to the user
  ExtraBuffers(hpx::future<tile_t> tile, SizeType num_extra_buffers, TileElementSize tile_size)
      : num_extra_buffers_(num_extra_buffers), orig_base_tile_(std::move(tile)),
        extra_(LocalElementSize(tile_size.rows() * num_extra_buffers, tile_size.cols()), tile_size) {
    clear();
    setup();
  }

  ~ExtraBuffers() {
    DLAF_ASSERT(not orig_base_tile_.valid(),
                "extra buffer must be reduced to release the original tile");
  }

  future_t get_buffer(const SizeType index) {
    const SizeType idx = num_extra_buffers_ != 0 ? index % (num_extra_buffers_ + 1) : 0;
    if (idx == 0)
      return get_base();
    else
      return extra_(LocalTileIndex(idx - 1, 0));
  }

  future_t reduce() {
    using hpx::unwrapping;
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
                    get_base(), extra_.read(idx_buffer));
    return unlock_base();
  }

  // TODO clear also base one
  void clear() {
    dlaf::matrix::util::set(extra_, [](...) { return 0; });
  }

  // TODO this may be useful for extrabuffers re-usage
  // void set_base() {
  //  // TODO check tile size
  //  // unlock previous one, if set
  //  // setup again
  //}

protected:
  void setup() {
    promise_t p;
    base_tile_ = p.get_future();

    orig_base_tile_ =
        orig_base_tile_.then(hpx::launch::sync,
                             hpx::unwrapping([p = std::move(p)](auto original_tile) mutable {
                               auto memory_view_copy = original_tile.memory_view_;
                               tile_t tile(original_tile.size_, std::move(memory_view_copy),
                                           original_tile.ld_);
                               tile.setPromise(std::move(p));
                               // TODO exceptions: if I don't set promise values, I don't have to manage excpetions
                               return std::move(original_tile);
                             }));
  }

  future_t get_base() {
    promise_t p;
    future_t f = p.get_future();
    std::swap(f, base_tile_);
    return f.then(hpx::launch::sync, hpx::unwrapping([p = std::move(p)](auto tile) mutable {
                    tile.setPromise(std::move(p));
                    return std::move(tile);
                  }));
  }

  future_t unlock_base() {
    DLAF_ASSERT(orig_base_tile_.valid(), "");
    return hpx::dataflow(hpx::unwrapping([](auto tile, auto) { return std::move(tile); }), std::move(orig_base_tile_),
        std::move(base_tile_));
  }

  const SizeType num_extra_buffers_;
  future_t orig_base_tile_;
  Matrix<T, Device::CPU> extra_;

  future_t base_tile_;
};
}
}
