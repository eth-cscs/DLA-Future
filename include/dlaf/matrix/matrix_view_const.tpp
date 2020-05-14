//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

template <class T, Device device>
template <template <class, Device> class MatrixType, class T2,
          std::enable_if_t<std::is_same<T, std::remove_const_t<T2>>::value, int>>
MatrixView<const T, device>::MatrixView(blas::Uplo uplo, MatrixType<T2, device>& matrix)
    : MatrixBase(matrix) {
  DLAF_ASSERT(uplo == blas::Uplo::General, "uplo != General not implemented yet.");
  setUpTiles(matrix);
}

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> MatrixView<const T, device>::read(
    const LocalTileIndex& index) noexcept {
  std::size_t i = tileLinearIndex(index);
  return tile_shared_futures_[i];
}

template <class T, Device device>
void MatrixView<const T, device>::done(const LocalTileIndex& index) noexcept {
  std::size_t i = tileLinearIndex(index);
  tile_shared_futures_[i] = {};
}

template <class T, Device device>
template <template <class, Device> class MatrixType, class T2,
          std::enable_if_t<std::is_same<T, std::remove_const_t<T2>>::value, int>>
void MatrixView<const T, device>::setUpTiles(MatrixType<T2, device>& matrix) noexcept {
  const auto& nr_tiles = matrix.distribution().localNrTiles();
  tile_shared_futures_.reserve(futureVectorSize(nr_tiles));

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      tile_shared_futures_.emplace_back(std::move(matrix.read(ind)));
    }
  }
}
