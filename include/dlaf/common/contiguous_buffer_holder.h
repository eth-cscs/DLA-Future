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

#include "dlaf/common/callable_object.h"
#include "dlaf/common/data_descriptor.h"
#include "dlaf/matrix/tile.h"

/// @file

namespace dlaf {
namespace common {

namespace internal {

/// A ContiguousBufferHolder:
/// - owns the temporary buffer (optionally allocated)
/// - contains a data descriptor to the contiguous data to be used for communication
///   (being it the aforementioned temporary buffer if allocated, otherwise an externally managed memory,
///   e.g. a tile)
///
/// It is up to the user to guarantee the integrity of the ContiguousBufferHolder and its properties
template <class T>
struct ContiguousBufferHolder {
  common::Buffer<std::remove_const_t<T>> buffer;
  common::DataDescriptor<T> descriptor;

  bool isAllocated() const {
    return static_cast<bool>(buffer);
  }
};

/// It returns a ContiguousBufferHolder starting from a Tile, where the ContiguousBufferHolder will be either:
//
/// - if (data_iscontiguous(make_data(tile))) => ContiguousBufferHolder<NULL_BUFFER, DATA_DESCRIPTOR(TILE)>
/// - otherwise                               => ContiguousBufferHolder<TEMP_BUFFER, DATA_DESCRIPTOR(TEMP_BUFFER>>
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
  return ContiguousBufferHolder<T>{std::move(buffer), std::move(what_to_use)};
}

DLAF_MAKE_CALLABLE_OBJECT(makeItContiguous);

/// Copy from bag to tile, just if the bag owns a temporary buffer. Otherwise it is a no-op.
template <class T>
void copyBack(const ContiguousBufferHolder<T>& bag, const matrix::Tile<T, Device::CPU>& tile) {
  if (bag.isAllocated())
    common::copy(bag.buffer, common::make_data(tile));
}

DLAF_MAKE_CALLABLE_OBJECT(copyBack);

}

}
}
