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

/// @file

#include "blas.hh"
#include "dlaf/matrix/tile.h"
#include "dlaf_test/matrix/util_generic_blas.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace matrix {
namespace test {

/// Sets the elements of the tile.
///
/// The (i, j)-element of the tile is set to el({i, j}) if op == NoTrans,
///                                          el({j, i}) if op == Trans,
///                                          conj(el({j, i})) if op == ConjTrans.
/// @pre el argument is an index of type const TileElementIndex& or TileElementIndex,
/// @pre el return type should be T.
template <class T, class Func>
void set(Tile<T, Device::CPU>& tile, Func el, blas::Op op) {
  switch (op) {
    case blas::Op::NoTrans:
      set(tile, el);
      break;

    case blas::Op::Trans: {
      auto op_el = [&el](TileElementIndex i) {
        i.transpose();
        return el(i);
      };
      set(tile, op_el);
      break;
    }

    case blas::Op::ConjTrans: {
      auto op_el = [&el](TileElementIndex i) {
        i.transpose();
        return dlaf::test::TypeUtilities<T>::conj(el(i));
      };
      set(tile, op_el);
      break;
    }
  }
}

/// Create a read-only tile and fill with selected values
///
/// @pre val argument is an index of type const TileElementIndex&,
/// @pre val return type should be T;
/// @pre size is the dimension of the tile to be created (type: TileElementSize);
/// @pre ld is the leading dimension of the tile to be created,
/// @pre op is the blas::Op to be applied to the tile.
template <class T, class ElementGetter>
Tile<T, Device::CPU> createTile(ElementGetter val, const TileElementSize size, const SizeType ld,
                                const blas::Op op) {
  auto tile = createTile<std::remove_const_t<T>>(size, ld);
  set(tile, val, op);
  return Tile<T, Device::CPU>(std::move(tile));
}

}
}
}
