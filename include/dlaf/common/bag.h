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

/// A Bag:
/// - owns the temporary buffer (optionally allocated)
/// - contains a data descriptor to the contiguous data to be used for communication
///   (being it the aforementioned temporary buffer if allocated, otherwise an externally managed memory,
///   e.g. a tile)
///
/// It is up to the user to guarantee the integrity of the Bag and its properties
template <class T>
using Bag = hpx::tuple<common::Buffer<std::remove_const_t<T>>, common::DataDescriptor<T>>;

/// It returns a Bag starting from a Tile, where the Bag will be either:
//
/// - if (data_iscontiguous(make_data(tile))) => Bag<NULL_BUFFER, DATA_DESCRIPTOR(TILE)>
/// - otherwise                               => Bag<TEMP_BUFFER, DATA_DESCRIPTOR(TEMP_BUFFER>>
//
/// so ensuring that the data_descriptor stored in the returned bag refers to a contiguous memory
/// chunk able to store all the data contained in @p tile.
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

/// It returns @p tile, ensuring that if the given @p bag owns a temporary buffer, it copies data from
/// this latter one to @p tile before returning it. Otherwise it is a no-op.
template <class T>
auto copyBack(matrix::Tile<T, Device::CPU> tile, Bag<T> bag) {
  auto buffer_used = std::move(hpx::get<0>(bag));
  if (buffer_used)
    common::copy(buffer_used, common::make_data(tile));
  return tile;
}

DLAF_MAKE_CALLABLE_OBJECT(copyBack);

}

}
}
