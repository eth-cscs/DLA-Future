//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/blas/enum_output.h"

namespace dlaf {
namespace matrix {

template <class T, Device D>
template <template <class, Device> class MatrixType, class T2,
          std::enable_if_t<std::is_same_v<T, std::remove_const_t<T2>>, int>>
MatrixView<const T, D>::MatrixView(blas::Uplo uplo, MatrixType<T2, D>& matrix) : MatrixBase(matrix) {
  if (uplo != blas::Uplo::General)
    DLAF_UNIMPLEMENTED(uplo);
  setUpTiles(matrix);
}

template <class T, Device D>
pika::shared_future<Tile<const T, D>> MatrixView<const T, D>::read(const LocalTileIndex& index) noexcept {
  const auto i = tileLinearIndex(index);
  return tile_shared_futures_[i];
}

template <class T, Device D>
void MatrixView<const T, D>::done(const LocalTileIndex& index) noexcept {
  const auto i = tileLinearIndex(index);
  tile_shared_futures_[i] = {};
}

template <class T, Device D>
template <template <class, Device> class MatrixType, class T2,
          std::enable_if_t<std::is_same_v<T, std::remove_const_t<T2>>, int>>
void MatrixView<const T, D>::setUpTiles(MatrixType<T2, D>& matrix) noexcept {
  const auto& nr_tiles = matrix.distribution().localNrTiles();
  tile_shared_futures_.reserve(futureVectorSize(nr_tiles));

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      tile_shared_futures_.emplace_back(matrix.read(ind));
    }
  }
}

}
}
