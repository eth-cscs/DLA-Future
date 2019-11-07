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

#include "blas.hh"
#include "dlaf/tile.h"
#include "dlaf_test/util_types.h"

namespace dlaf_test {
namespace tile_test {
using namespace dlaf;

/// @brief Sets the elements of the tile.
///
/// The (i, j)-element of the tile is set to el({i, j}) if op == NoTrans,
///                                          el({j, i}) if op == Trans,
///                                          conj(el({j, i})) if op == ConjTrans.
/// @pre el argument is an index of type const TileElementIndex&.
/// @pre el return type should be T.
template <class T, class Func>
void set(Tile<T, Device::CPU>& tile, Func el, blas::Op op) {
  switch (op) {
    case blas::Op::NoTrans:
      for (SizeType j = 0; j < tile.size().cols(); ++j) {
        for (SizeType i = 0; i < tile.size().rows(); ++i) {
          TileElementIndex index(i, j);
          tile(index) = el(index);
        }
      }
      break;
    case blas::Op::Trans:
      for (SizeType j = 0; j < tile.size().cols(); ++j) {
        for (SizeType i = 0; i < tile.size().rows(); ++i) {
          TileElementIndex index(i, j);
          TileElementIndex index_trans(j, i);
          tile(index) = el(index_trans);
        }
      }
      break;
    case blas::Op::ConjTrans:
      for (SizeType j = 0; j < tile.size().cols(); ++j) {
        for (SizeType i = 0; i < tile.size().rows(); ++i) {
          TileElementIndex index(i, j);
          TileElementIndex index_trans(j, i);
          tile(index) = TypeUtilities<T>::conj(el(index_trans));
        }
      }
      break;
  }
}
}
}
