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
#include "dlaf/matrix.h"
#include "dlaf_test/util_types.h"

namespace dlaf_test {
namespace matrix_test {
using namespace dlaf;

/// @brief Sets the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}) if op == NoTrans,
///                                          el({j, i}) if op == Trans,
///                                          conj(el({j, i})) if op == ConjTrans.
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre el return type should be T.
template <class T, class Func>
void set(Matrix<T, Device::CPU>& matrix, Func el, blas::Op op) {
  const matrix::Distribution& dist = matrix.distribution();
  switch (op) {
    case blas::Op::NoTrans:
      for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
        for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
          auto tile = matrix(LocalTileIndex(tile_i, tile_j)).get();
          for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
            SizeType j = dist.globalElementFromLocalTileAndTileElement<Coord::Col>(tile_j, jj);
            for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
              SizeType i = dist.globalElementFromLocalTileAndTileElement<Coord::Row>(tile_i, ii);
              tile({ii, jj}) = el({i, j});
            }
          }
        }
      }
      break;
    case blas::Op::Trans:
      for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
        for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
          auto tile = matrix(LocalTileIndex(tile_i, tile_j)).get();
          for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
            SizeType j = dist.globalElementFromLocalTileAndTileElement<Coord::Col>(tile_j, jj);
            for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
              SizeType i = dist.globalElementFromLocalTileAndTileElement<Coord::Row>(tile_i, ii);
              tile({ii, jj}) = el({j, i});
            }
          }
        }
      }
      break;
    case blas::Op::ConjTrans:
      for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
        for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
          auto tile = matrix(LocalTileIndex(tile_i, tile_j)).get();
          for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
            SizeType j = dist.globalElementFromLocalTileAndTileElement<Coord::Col>(tile_j, jj);
            for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
              SizeType i = dist.globalElementFromLocalTileAndTileElement<Coord::Row>(tile_i, ii);
              tile({ii, jj}) = TypeUtilities<T>::conj(el({j, i}));
            }
          }
        }
      }
      break;
  }
}
}
}
