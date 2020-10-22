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

/// @file

#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/tile.h"
#include "dlaf/traits.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/matrix/util_tile_blas.h"

namespace dlaf {
namespace matrix {
namespace test {
using namespace dlaf;

/// Create a read/write tile and fill with selected values
///
/// @pre val argument is an index of type const TileElementIndex&,
/// @pre val return type should be T;
/// @pre size is the dimension of the tile to be created (type: TileElementSize);
/// @pre ld is the leading dimension of the tile to be created.
template <class T, class ElementGetter>
Tile<T, Device::CPU> setup_tile(ElementGetter val, const TileElementSize& size, const SizeType& ld) {
  Tile<T, Device::CPU> support = create_tile<T>(size, ld);
  set(support, val);
  return support;
}

/// Create a read/write tile and fill with selected values
///
/// @pre val argument is an index of type const TileElementIndex&,
/// @pre val return type should be T;
/// @pre size is the dimension of the tile to be created (type: TileElementSize);
/// @pre ld is the leading dimension of the tile to be created.
/// @pre op is the blas::Op to be applied to the tile.
template <class T, class ElementGetter>
Tile<T, Device::CPU> setup_tile(ElementGetter val, const TileElementSize& size, const SizeType& ld,
                                const blas::Op& op) {
  Tile<T, Device::CPU> support = create_tile<T>(size, ld);
  set(support, val, op);
  return support;
}

/// Create a read-only tile and fill with selected values
///
/// @pre val argument is an index of type const TileElementIndex&,
/// @pre val return type should be T;
/// @pre size is the dimension of the tile to be created (type: TileElementSize);
/// @pre ld is the leading dimension of the tile to be created.
template <class T, class CT = const T, class ElementGetter>
Tile<CT, Device::CPU> setup_readonly_tile(ElementGetter val, const TileElementSize& size,
                                          const SizeType& ld) {
  Tile<T, Device::CPU> support = create_tile<T>(size, ld);
  set(support, val);
  Tile<CT, Device::CPU> readonly(std::move(support));
  return readonly;
}

/// Create a read-only tile and fill with selected values
///
/// @pre val argument is an index of type const TileElementIndex&,
/// @pre val return type should be T;
/// @pre size is the dimension of the tile to be created (type: TileElementSize);
/// @pre ld is the leading dimension of the tile to be created.
template <class T, class CT = const T, class ElementGetter>
Tile<CT, Device::CPU> setup_readonly_tile(ElementGetter val, const TileElementSize& size,
                                          const SizeType& ld, const blas::Op& op) {
  Tile<T, Device::CPU> support = create_tile<T>(size, ld);
  set(support, val, op);
  Tile<CT, Device::CPU> readonly(std::move(support));
  return readonly;
}
}
}
}
