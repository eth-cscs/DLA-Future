//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <utility>

#include <blas.hh>

#include <dlaf/matrix/tile.h>

#include <dlaf_test/matrix/util_generic_blas.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

namespace dlaf {
namespace matrix {
namespace test {

/// Sets the elements of the tile.
///
/// The (i, j)-element of the tile is set to val({i, j}) if op == NoTrans,
///                                          val({j, i}) if op == Trans,
///                                          conj(val({j, i})) if op == ConjTrans.
/// @pre el argument is an index of type const TileElementIndex& or TileElementIndex,
/// @pre el return type should be T.
template <class T, class ElementGetter>
void set(Tile<T, Device::CPU>& tile, ElementGetter val, const blas::Op op) {
  auto op_val = internal::opValFunc(val, op);
  set(tile, op_val);
}

/// Create a read-only tile and fill with selected values
///
/// @pre val argument is an index of type const TileElementIndex&,
/// @pre val return type should be T;
/// @pre size is the dimension of the tile to be created (type: TileElementSize);
/// @pre ld is the leading dimension of the tile to be created,
/// @pre op is the blas::Op to be applied to the tile.
template <class T, Device D = Device::CPU, class ElementGetter>
Tile<T, D> createTile(ElementGetter val, const TileElementSize size, const SizeType ld,
                      const blas::Op op) {
  auto op_val = internal::opValFunc(val, op);
  return createTile<T, D>(op_val, size, ld);
}

}
}
}
