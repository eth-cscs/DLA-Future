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

#include <cmath>
#include <exception>
#include <random>
#include <string>

#ifndef M_PI
constexpr double M_PI = 3.141592;
#endif

#include <blas.hh>
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include "dlaf/blaspp/enums.h"
#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"

/// @file

namespace dlaf {
namespace matrix {

/// Returns true if the matrix is square.
template <class T, Device D>
bool square_size(const Matrix<const T, D>& m) noexcept {
  return m.size().rows() == m.size().cols();
}

/// Returns true if the matrix block size is square.
template <class T, Device D>
bool square_blocksize(const Matrix<const T, D>& m) noexcept {
  return m.blockSize().rows() == m.blockSize().cols();
}

/// Returns true if matrices have equal sizes.
template <class T, Device D>
bool equal_size(const Matrix<const T, D>& lhs, Matrix<const T, D>& rhs) noexcept {
  return lhs.size() == rhs.size();
}

/// Returns true if matrices have equal blocksizes.
template <class T, Device D>
bool equal_blocksize(const Matrix<const T, D>& lhs, Matrix<const T, D>& rhs) noexcept {
  return lhs.blockSize() == rhs.blockSize();
}

/// Returns true if the matrix is local to a process.
template <class T, Device D>
bool local_matrix(const Matrix<const T, D>& m) noexcept {
  return m.commGridSize() == comm::Size2D(1, 1);
}

/// Returns true if the matrix is distributed on the communication grid.
template <class T, Device D>
bool equal_process_grid(const Matrix<const T, D>& m, comm::CommunicatorGrid const& g) noexcept {
  return m.commGridSize() == g.size() && m.rankIndex() == g.rank();
}

/// Returns true if the matrices are distributed the same way.
template <class T, Device D>
bool equal_distributions(const Matrix<const T, D>& lhs, const Matrix<const T, D>& rhs) noexcept {
  return lhs.distribution() == rhs.distribution();
}

/// Returns true if the sizes are compatible for matrix multiplication.
template <class IndexT, class Tag>
bool multipliable_sizes(common::Size2D<IndexT, Tag> a, common::Size2D<IndexT, Tag> b,
                        common::Size2D<IndexT, Tag> c, const blas::Op opA, const blas::Op opB) noexcept {
  if (opA != blas::Op::NoTrans)
    a.transpose();
  if (opB != blas::Op::NoTrans)
    b.transpose();

  return a.rows() == c.rows() && a.cols() == b.rows() && b.cols() == c.cols();
}

/// Returns true if matrices `a`, `b` and `c` have matrix multipliable sizes and block sizes.
template <class T, Device D>
bool multipliable(const Matrix<const T, D>& a, const Matrix<const T, D>& b, const Matrix<const T, D>& c,
                  const blas::Op opA, const blas::Op opB) noexcept {
  return multipliable_sizes(a.size(), b.size(), c.size(), opA, opB) &&
         multipliable_sizes(a.blockSize(), b.blockSize(), c.blockSize(), opA, opB);
}

namespace util {
namespace internal {

/// Callable that returns random values in the range [-1, 1].
template <class T>
class getter_random {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T is not compatible with random generator used.");

public:
  getter_random(const unsigned long seed = std::minstd_rand::default_seed) : random_engine_(seed) {}

  T operator()() {
    return random_sampler_(random_engine_);
  }

private:
  std::mt19937_64 random_engine_;
  std::uniform_real_distribution<T> random_sampler_{-1, 1};
};

/// Callable that returns random complex numbers whose absolute values are less than 1.
template <class T>
class getter_random<std::complex<T>> : private getter_random<T> {
public:
  using getter_random<T>::getter_random;

  std::complex<T> operator()() {
    return std::polar<T>(std::abs(getter_random<T>::operator()()),
                         static_cast<T>(M_PI) * getter_random<T>::operator()());
  }
};

}

/// Set the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @param el a copy is given to each tile,
/// @pre el argument is an index of type const GlobalElementIndex&,
/// @pre el return type should be T.
template <class T, class ElementGetter>
void set(Matrix<T, Device::CPU>& matrix, const ElementGetter& el_f) {
  const Distribution& dist = matrix.distribution();
  for (auto tile_wrt_local : iterate_range2d(dist.localNrTiles())) {
    GlobalTileIndex tile_wrt_global = dist.globalTileIndex(tile_wrt_local);
    auto tl_index = dist.globalElementIndex(tile_wrt_global, {0, 0});
    auto set_f = hpx::util::unwrapping([tl_index, el_f = el_f](auto&& tile) {
      for (auto el_idx_l : iterate_range2d(tile.size())) {
        GlobalElementIndex el_idx_g(el_idx_l.row() + tl_index.row(), el_idx_l.col() + tl_index.col());
        tile(el_idx_l) = el_f(el_idx_g);
      }
    });
    hpx::dataflow(std::move(set_f), matrix(tile_wrt_local));
  }
}

/// Set the matrix with random values whose absolute values are less than 1.
///
/// Values will be random numbers in:
/// - real:     [-1, 1]
/// - complex:  a circle of radius 1 centered at origin.
///
/// Each tile creates its own random generator engine with a unique seed
/// which is computed as a function of the tile global index.
/// This means that a specific tile index, no matter on which rank it will be,
/// will have the same set of values.
template <class T>
void set_random(Matrix<T, Device::CPU>& matrix) {
  using namespace dlaf::util::size_t;
  const Distribution& dist = matrix.distribution();
  for (auto tile_wrt_local : iterate_range2d(dist.localNrTiles())) {
    GlobalTileIndex tile_wrt_global = dist.globalTileIndex(tile_wrt_local);
    auto tl_index = dist.globalElementIndex(tile_wrt_global, {0, 0});
    auto seed = sum(tl_index.col(), mul(tl_index.row(), matrix.size().cols()));
    auto rnd_f = hpx::util::unwrapping([seed](auto&& tile) {
      internal::getter_random<T> random_value(seed);
      for (auto el_idx : iterate_range2d(tile.size())) {
        tile(el_idx) = random_value();
      }
    });
    hpx::dataflow(std::move(rnd_f), matrix(tile_wrt_local));
  }
}

namespace internal {

template <class T>
void set_diagonal_tile(Tile<T, Device::CPU>& tile, internal::getter_random<T>& random_value,
                       std::size_t offset_value) {
  // DIAGONAL
  // for diagonal tiles get just lower matrix values and set value for both
  // straight and transposed indices
  for (SizeType j = 0; j < tile.size().cols(); ++j) {
    for (SizeType i = 0; i < j; ++i) {
      auto value = random_value();

      tile(TileElementIndex{i, j}) = value;
      tile(TileElementIndex{j, i}) = dlaf::conj(value);
    }
    tile(TileElementIndex{j, j}) = std::real(random_value()) + offset_value;
  }
}

template <class T>
void set_lower_and_upper_tile(Tile<T, Device::CPU>& tile, internal::getter_random<T>& random_value,
                              TileElementSize full_tile_size, GlobalTileIndex tile_wrt_global) {
  // LOWER or UPPER (except DIAGONAL)
  // random values are requested in the same order for both original and transposed
  for (SizeType j = 0; j < full_tile_size.cols(); ++j) {
    for (SizeType i = 0; i < full_tile_size.rows(); ++i) {
      auto value = random_value();

      // but they are set row-wise in the original tile and col-wise in the
      // transposed one
      if (tile_wrt_global.row() > tile_wrt_global.col()) {  // LOWER
        TileElementIndex index{i, j};
        if (index.isIn(tile.size()))
          tile(index) = value;
      }
      else {  // UPPER
        TileElementIndex index{j, i};
        if (index.isIn(tile.size()))
          tile(index) = dlaf::conj(value);
      }
    }
  }
}

/// Set a matrix with random values assuring it will be hermitian with an offset added to diagonal elements
///
/// Values on the diagonal are added offset_value to a random value in the range [-1, 1].
/// In case of complex values, the imaginary part is 0.
///
/// Values not on the diagonal will be random numbers in:
/// - real:     [-1, 1]
/// - complex:  a circle of radius 1 centered at origin.
///
/// Each tile creates its own random generator engine with a unique seed
/// which is computed as a function of the tile global index.
/// This means that the elements of a specific tile, no matter how the matrix is distributed,
/// will be set with the same set of values.
///
/// @pre @param matrix is a square matrix,
/// @pre @param matrix has a square blocksize.
template <class T>
void set_random_hermitian_with_offset(Matrix<T, Device::CPU>& matrix, const std::size_t offset_value) {
  // note:
  // By assuming square blocksizes, it is easier to locate elements. In fact:
  // - Elements on the diagonal are stored in the diagonal of the diagonal tiles
  // - Tiles under the diagonal store elements of the lower triangular matrix
  // - Tiles over the diagonal store elements of the upper triangular matrix

  using namespace dlaf::util::size_t;

  const Distribution& dist = matrix.distribution();

  DLAF_ASSERT(square_size(matrix), matrix);
  DLAF_ASSERT(square_blocksize(matrix), matrix);

  auto full_tile_size = matrix.blockSize();

  for (auto tile_wrt_local : iterate_range2d(dist.localNrTiles())) {
    GlobalTileIndex tile_wrt_global = dist.globalTileIndex(tile_wrt_local);

    auto tl_index = dist.globalElementIndex(tile_wrt_global, {0, 0});

    // compute the same seed for original and "transposed" tiles, so transposed ones will know the
    // values of the original one without the need of accessing real values (nor communication in case
    // of distributed matrices)
    size_t seed;
    if (tile_wrt_global.row() >= tile_wrt_global.col())  // LOWER or DIAGONAL
      seed = sum(tl_index.col(), mul(tl_index.row(), matrix.size().cols()));
    else
      seed = sum(tl_index.row(), mul(tl_index.col(), matrix.size().rows()));

    auto set_hp_f = hpx::util::unwrapping([=](auto&& tile) {
      internal::getter_random<T> random_value(seed);
      if (tile_wrt_global.row() == tile_wrt_global.col())
        internal::set_diagonal_tile(tile, random_value, offset_value);
      else
        internal::set_lower_and_upper_tile(tile, random_value, full_tile_size, tile_wrt_global);
    });

    hpx::dataflow(std::move(set_hp_f), matrix(tile_wrt_local));
  }
}

}

/// Set a matrix with random values assuring it will be hermitian
///
/// Values will be random numbers in:
/// - real:     [-1, 1]
/// - complex:  a circle of radius 1 centered at origin
///
/// Each tile creates its own random generator engine with a unique seed
/// which is computed as a function of the tile global index.
/// This means that the elements of a specific tile, no matter how the matrix is distributed,
/// will be set with the same set of values.
///
/// @pre @param matrix is a square matrix,
/// @pre @param matrix has a square blocksize.
template <class T>
void set_random_hermitian(Matrix<T, Device::CPU>& matrix) {
  internal::set_random_hermitian_with_offset(matrix, 0);
}

/// Set a matrix with random values assuring it will be hermitian and positive definite.
///
/// Values on the diagonal are 2*n added to a random value in the range [-1, 1], where
/// n is the matrix size. In case of complex values, the imaginary part is 0.
///
/// Values not on the diagonal will be random numbers in:
/// - real:     [-1, 1]
/// - complex:  a circle of radius 1 centered at origin
///
/// Each tile creates its own random generator engine with a unique seed
/// which is computed as a function of the tile global index.
/// This means that the elements of a specific tile, no matter how the matrix is distributed,
/// will be set with the same set of values.
///
/// @pre @param matrix is a square matrix,
/// @pre @param matrix has a square blocksize.
template <class T>
void set_random_hermitian_positive_definite(Matrix<T, Device::CPU>& matrix) {
  const auto offset_value = dlaf::util::size_t::mul(2, to_sizet(matrix.size().rows()));
  internal::set_random_hermitian_with_offset(matrix, offset_value);
}

}
}
}
