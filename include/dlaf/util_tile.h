//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <random>

#include "dlaf/tile.h"
#include "dlaf/traits.h"

namespace dlaf {
namespace tile {
namespace util {

/// @brief Sets the elements of the tile.
///
/// The (i, j)-element of the tile is set to el({i, j}).
/// @pre el argument is an index of type const TileElementIndex&.
/// @pre el return type should be T.
template <class T, class ElementGetter,
          enable_if_signature_t<ElementGetter, T(const TileElementIndex&), int> = 0>
void set(const Tile<T, Device::CPU>& tile, ElementGetter el) {
  for (SizeType j = 0; j < tile.size().cols(); ++j) {
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      TileElementIndex index(i, j);
      tile(index) = el(index);
    }
  }
}

/// @brief Sets all the elements of the tile with given value.
///
/// @pre el is an element of a type U that can be assigned to T
template <class T, class U, enable_if_convertible_t<U, T, int> = 0>
void set(const Tile<T, Device::CPU>& tile, U el) {
  for (SizeType j = 0; j < tile.size().cols(); ++j) {
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      tile(TileElementIndex{i, j}) = el;
    }
  }
}


/// @brief Initialize tile with random values
///
///
template <class T>
void set_random(const Tile<T, Device::CPU>& tile) {
  std::minstd_rand random_seed;
  std::uniform_real_distribution<T> random_sampler(-1, 1);

  dlaf::tile::util::set(tile, [random_sampler, random_seed](const TileElementIndex&) mutable {
    return random_sampler(random_seed);
  });
}

}
}
}
