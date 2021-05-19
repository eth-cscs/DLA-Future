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

#include <hpx/tuple.hpp>

#include "dlaf/common/callable_object.h"
#include "dlaf/common/data_descriptor.h"
#include "dlaf/matrix/tile.h"

/// @file

namespace dlaf {
namespace common {

namespace internal {

// Note:
//
// A bag:
// - owns the temporary buffer (optionally allocated)
// - contains a data descriptor to the contiguous data to be used for communication (being it the
//    original tile or the temporary buffer)
//
// The bag can be create with the `makeItContiguous` helper function by passing the original tile
// and in case of a RW tile, it is possible to `copyBack` the memory from the temporary buffer to
// the original tile.

template <class T>
using Bag = hpx::tuple<common::Buffer<std::remove_const_t<T>>, common::DataDescriptor<T>>;

template <class T>
auto makeItContiguous(const matrix::Tile<T, Device::CPU>& tile) {
  common::Buffer<std::remove_const_t<T>> buffer;
  auto tile_data = common::make_data(tile);
  auto what_to_use = common::make_contiguous(tile_data, buffer);
  if (buffer)
    common::copy(tile_data, buffer);
  return Bag<T>(std::move(buffer), std::move(what_to_use));
}

DLAF_MAKE_CALLABLE_OBJECT(makeItContiguous);

template <class T>
auto copyBack(matrix::Tile<T, Device::CPU> tile, Bag<T> bag) {
  auto buffer_used = std::move(hpx::get<0>(bag));
  if (buffer_used)
    common::copy(buffer_used, common::make_data(tile));
  return std::move(tile);
}

DLAF_MAKE_CALLABLE_OBJECT(copyBack);

}

}
}
