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

#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <utility>

#ifndef M_PI
constexpr double M_PI = 3.141592;
#endif

#include <blas.hh>

#include <pika/execution.hpp>

#include <dlaf/blas/enum_output.h>
#include <dlaf/common/assert.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/sender/transform.h>
#include <dlaf/types.h>

/// @file

namespace dlaf::matrix {

/// Returns true if the matrix is square.
template <class MatrixLike>
bool square_size(const MatrixLike& m) noexcept {
  return m.size().rows() == m.size().cols();
}

/// Returns true if the matrix block size is square.
template <class MatrixLike>
bool square_blocksize(const MatrixLike& m) noexcept {
  return m.block_size().rows() == m.block_size().cols();
}

/// Returns true if the matrix has a single tile per block.
template <class MatrixLike>
bool single_tile_per_block(const MatrixLike& m) noexcept {
  return m.block_size() == m.tile_size();
}

/// Returns true if matrices have equal sizes.
template <class MatrixLikeA, class MatrixLikeB>
bool equal_size(const MatrixLikeA& lhs, const MatrixLikeB& rhs) noexcept {
  return lhs.size() == rhs.size();
}

/// Returns true if matrices have equal blocksizes.
template <class T, Device D1, Device D2>
bool equal_blocksize(const Matrix<const T, D1>& lhs, Matrix<const T, D2>& rhs) noexcept {
  return lhs.block_size() == rhs.block_size();
}

/// Returns true if the matrix is local to a process.
template <template <class, Device> class MatrixLike, class T, Device D>
bool local_matrix(const MatrixLike<const T, D>& m) noexcept {
  return m.commGridSize() == comm::Size2D(1, 1);
}

/// Returns true if the matrix is distributed on the communication grid.
template <template <class, Device> class MatrixLike, class T, Device D>
bool equal_process_grid(const MatrixLike<const T, D>& m, const comm::CommunicatorGrid& g) noexcept {
  return m.commGridSize() == g.size() && m.rankIndex() == g.rank();
}

/// Returns true if the matrix is distributed on the grid of the pipeline.
template <template <class, Device> class MatrixLike, class T, Device D, comm::CommunicatorType CT>
bool equal_process_grid(const MatrixLike<const T, D>& m,
                        const comm::CommunicatorPipeline<CT>& p) noexcept {
  return m.commGridSize() == p.size_2d() && m.rankIndex() == p.rank_2d();
}

/// Returns true if the two communicator pipelines are distributed on the same grid.
template <comm::CommunicatorType CT1, comm::CommunicatorType CT2>
bool equal_process_grid(const comm::CommunicatorPipeline<CT1>& p1,
                        const comm::CommunicatorPipeline<CT2>& p2) noexcept {
  return p1.size_2d() == p2.size_2d() && p1.rank_2d() == p2.rank_2d();
}

/// Returns true if the two matrices are distributed on the same grid
template <template <class, Device> class MatrixLikeA, template <class, Device> class MatrixLikeB,
          class T, Device D1, Device D2>
bool same_process_grid(const MatrixLikeA<const T, D1>& a, const MatrixLikeB<const T, D2>& b) noexcept {
  return a.commGridSize() == b.commGridSize() && a.rankIndex() == b.rankIndex();
}

/// Returns true if the matrices are distributed the same way.
template <class T, Device D1, Device D2>
bool equal_distributions(const Matrix<const T, D1>& lhs, const Matrix<const T, D2>& rhs) noexcept {
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
template <template <class, Device> class MatrixLikeA, template <class, Device> class MatrixLikeB,
          template <class, Device> class MatrixLikeC, class T, Device D>
bool multipliable(const MatrixLikeA<const T, D>& a, const MatrixLikeB<const T, D>& b,
                  const MatrixLikeC<const T, D>& c, const blas::Op opA, const blas::Op opB) noexcept {
  const bool isSizeOk = multipliable_sizes(a.size(), b.size(), c.size(), opA, opB);

  if (a.size().isEmpty() || c.size().isEmpty())
    return isSizeOk;

  const bool allSameGrid = same_process_grid(a, b) && same_process_grid(b, c);
  const bool isTileSizeOk = multipliable_sizes(a.tile_size(), b.tile_size(), c.tile_size(), opA, opB);
  const bool isOffsetOk = multipliable_sizes(a.tile_size_of({0, 0}), b.tile_size_of({0, 0}),
                                             c.tile_size_of({0, 0}), opA, opB);

  if (local_matrix(c))
    return allSameGrid && isSizeOk && isTileSizeOk && isOffsetOk;

  // TODO distributed (fix offset)
  return allSameGrid && isSizeOk && isTileSizeOk && isOffsetOk;
}

namespace util {
namespace internal {

/// Callable that returns random values in the range [-1, 1].
template <class T>
class getter_random {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "T is not compatible with random generator used.");

public:
  getter_random() : random_engine_(std::minstd_rand::default_seed) {}
  getter_random(SizeType seed) : random_engine_(static_cast<std::size_t>(seed)) {
    DLAF_ASSERT(seed >= 0, "");
  }

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

/// Sets to zero the subset of local tiles of @p matrix in the 2D range starting at @p begin with size @p sz.
///
template <Backend backend, class T, Device D>
void set0(pika::execution::thread_priority priority, LocalTileIndex begin, LocalTileSize sz,
          Matrix<T, D>& matrix) {
  using dlaf::internal::Policy;
  using pika::execution::thread_stacksize;
  using pika::execution::experimental::start_detached;

  for (const auto& ij_lc : iterate_range2d(begin, sz))
    start_detached(matrix.readwrite(ij_lc) |
                   tile::set0(Policy<backend>(priority, thread_stacksize::nostack)));
}

/// \overload set0
///
/// This overload sets all tiles @p matrix to zero.
///
template <Backend backend, class T, Device D>
void set0(pika::execution::thread_priority priority, Matrix<T, D>& matrix) {
  set0<backend>(priority, LocalTileIndex(0, 0), matrix.distribution().local_nr_tiles(), matrix);
}

/// Sets all the elements of all the tiles in the active range to zero
template <Backend backend, class T, Coord axis, Device D, StoreTransposed storage>
void set0(pika::execution::thread_priority priority, Panel<axis, T, D, storage>& panel) {
  using dlaf::internal::Policy;
  using pika::execution::thread_stacksize;
  using pika::execution::experimental::start_detached;

  for (const auto& ij_lc : panel.iteratorLocal())
    start_detached(panel.readwrite(ij_lc) |
                   tile::set0(Policy<backend>(priority, thread_stacksize::nostack)));
}

/// Set the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @param el_f a copy is given to each tile,
/// @pre el_f argument is an index of type const GlobalElementIndex&,
/// @pre el_f return type should be T.
template <class T, class ElementGetter>
void set(Matrix<T, Device::CPU>& matrix, ElementGetter el_f) {
  using pika::execution::thread_stacksize;

  const Distribution& dist = matrix.distribution();
  for (auto ij_lc : iterate_range2d(dist.local_nr_tiles())) {
    GlobalTileIndex ij = dist.global_tile_index(ij_lc);
    auto tile_origin = dist.global_element_index(ij, {0, 0});

    using TileType = typename std::decay_t<decltype(matrix)>::TileType;
    auto set_f = [tile_origin, el_f = el_f](const TileType& tile) {
      for (auto ij_el_tl : iterate_range2d(tile.size())) {
        GlobalElementIndex ij_el_gl(tile_origin.row() + ij_el_tl.row(),
                                    tile_origin.col() + ij_el_tl.col());
        tile(ij_el_tl) = el_f(ij_el_gl);
      }
    };

    dlaf::internal::transformDetach(dlaf::internal::Policy<Backend::MC>(thread_stacksize::nostack),
                                    std::move(set_f), matrix.readwrite(ij_lc));
  }
}

/// Set the elements of the matrix according to transposition operator
///
/// The (i, j)-element of the matrix is set to op(el)(i, j).
/// i.e. `matrix = op(el_f)`
///
/// @param el_f a copy is given to each tile,
/// @param op transposition operator to apply to @p el_f before setting the value
/// @pre el_f argument is an index of type const GlobalElementIndex&,
/// @pre el_f return type should be T.
template <class T, class ElementGetter>
void set(Matrix<T, Device::CPU>& matrix, ElementGetter el_f, const blas::Op op) {
  auto el_op_f = [op, el_f](const GlobalElementIndex& index) -> T {
    using blas::Op;
    switch (op) {
      case Op::NoTrans:
        return el_f(index);
      case Op::Trans:
        return el_f(transposed(index));
      case Op::ConjTrans:
        return dlaf::conj(el_f(transposed(index)));
      default:
        DLAF_UNIMPLEMENTED(op);
        return T{};
    }
  };

  set(matrix, el_op_f);
}

/// Set the elements of the matrix.
///
/// The diagonal elements are set to 1 and the other elements to 0.
template <class T>
void set_identity(Matrix<T, Device::CPU>& matrix) {
  set(matrix, [](const GlobalElementIndex& ij_lc) {
    if (ij_lc.row() == ij_lc.col())
      return T{1};
    return T{0};
  });
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
  using pika::execution::thread_stacksize;

  const Distribution& dist = matrix.distribution();
  for (auto ij_lc : iterate_range2d(dist.local_nr_tiles())) {
    GlobalTileIndex ij = dist.global_tile_index(ij_lc);
    auto tile_origin = dist.global_element_index(ij, {0, 0});
    auto seed = tile_origin.col() + tile_origin.row() * matrix.size().cols();

    using TileType = typename std::decay_t<decltype(matrix)>::TileType;
    auto rnd_f = [seed](TileType&& tile) {
      internal::getter_random<T> random_value(seed);
      for (auto ij_el_tl : iterate_range2d(tile.size())) {
        tile(ij_el_tl) = random_value();
      }
    };

    dlaf::internal::transformDetach(dlaf::internal::Policy<Backend::MC>(thread_stacksize::nostack),
                                    std::move(rnd_f), matrix.readwrite(ij_lc));
  }
}

namespace internal {

template <class T>
void set_diagonal_tile(const Tile<T, Device::CPU>& tile, internal::getter_random<T>& random_value,
                       SizeType offset_value) {
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
void set_lower_and_upper_tile(const Tile<T, Device::CPU>& tile, internal::getter_random<T>& random_value,
                              TileElementSize full_tile_size, GlobalTileIndex ij,
                              dlaf::matrix::Distribution dist,
                              std::optional<SizeType> band_size = std::nullopt) {
  auto is_off_band = [](GlobalElementIndex ij_lc, std::optional<SizeType> band_size) {
    return band_size ? ij_lc.col() < ij_lc.row() - *band_size || ij_lc.col() > ij_lc.row() + *band_size
                     : false;
  };

  // LOWER or UPPER (except DIAGONAL)
  // random values are requested in the same order for both original and transposed
  for (SizeType j = 0; j < full_tile_size.cols(); ++j) {
    for (SizeType i = 0; i < full_tile_size.rows(); ++i) {
      auto value = random_value();

      // but they are set row-wise in the original tile and col-wise in the
      // transposed one
      if (ij.row() > ij.col()) {  // LOWER
        TileElementIndex index{i, j};
        if (index.isIn(tile.size())) {
          auto ij_lc = dist.globalElementIndex(ij, index);
          if (is_off_band(ij_lc, band_size))
            tile(index) = T(0);
          else
            tile(index) = value;
        }
      }
      else {  // UPPER
        TileElementIndex index{j, i};
        if (index.isIn(tile.size())) {
          auto ij_lc = dist.globalElementIndex(ij, index);
          if (is_off_band(ij_lc, band_size))
            tile(index) = T(0);
          else
            tile(index) = dlaf::conj(value);
        }
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
void set_random_hermitian_with_offset(Matrix<T, Device::CPU>& matrix, const SizeType offset_value,
                                      std::optional<SizeType> band_size = std::nullopt) {
  using pika::execution::thread_stacksize;

  // note:
  // By assuming square blocksizes, it is easier to locate elements. In fact:
  // - Elements on the diagonal are stored in the diagonal of the diagonal tiles
  // - Tiles under the diagonal store elements of the lower triangular matrix
  // - Tiles over the diagonal store elements of the upper triangular matrix

  const Distribution& dist = matrix.distribution();

  DLAF_ASSERT(square_size(matrix), matrix);
  DLAF_ASSERT(square_blocksize(matrix), matrix);

  auto full_tile_size = matrix.block_size();

  for (auto ij_lc : iterate_range2d(dist.local_nr_tiles())) {
    GlobalTileIndex ij = dist.global_tile_index(ij_lc);

    auto tile_origin = dist.global_element_index(ij, {0, 0});

    // compute the same seed for original and "transposed" tiles, so transposed ones will know the
    // values of the original one without the need of accessing real values (nor communication in case
    // of distributed matrices)
    SizeType seed;
    if (ij.row() >= ij.col())  // LOWER or DIAGONAL
      seed = tile_origin.col() + tile_origin.row() * matrix.size().cols();
    else
      seed = tile_origin.row() + tile_origin.col() * matrix.size().rows();

    using TileType = typename std::decay_t<decltype(matrix)>::TileType;
    auto set_hp_f = [=](const TileType& tile) {
      internal::getter_random<T> random_value(seed);
      if (ij.row() == ij.col())
        internal::set_diagonal_tile(tile, random_value, offset_value);
      else
        internal::set_lower_and_upper_tile(tile, random_value, full_tile_size, ij, dist, band_size);
    };

    dlaf::internal::transformDetach(dlaf::internal::Policy<Backend::MC>(thread_stacksize::nostack),
                                    std::move(set_hp_f), matrix.readwrite(ij_lc));
  }
}

template <class T, Device D>
dlaf::matrix::internal::SubMatrixSpec sub_matrix_spec_slice_cols(const Matrix<T, D>& matrix, SizeType fist_col_index,
                                SizeType last_col_index) {
  return dlaf::matrix::internal::SubMatrixSpec({{0, fist_col_index}, {matrix.size().rows(), last_col_index - fist_col_index + 1}});
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

/// Set a banded matrix with random values assuring it will be hermitian
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
void set_random_hermitian_banded(Matrix<T, Device::CPU>& matrix, const SizeType band_size) {
  internal::set_random_hermitian_with_offset(matrix, 0, band_size);
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
  internal::set_random_hermitian_with_offset(matrix, 2 * matrix.size().rows());
}

/// Set a matrix with random values assuring that the diagonal has non-zero elements.
///
/// Values not on the diagonal will be random numbers in:
/// - real:     [-1, 1]
/// - complex:  a circle of radius 1 centered at origin
/// Values on the diagonal will be random numbers in:
/// - real:     [-1, -0.1] or [0.1, 1]
/// - complex:  an annulus of inner radius 0.1 and outer radius 1 centered at origin
///
/// Each tile creates its own random generator engine with a unique seed
/// which is computed as a function of the tile global index.
/// This means that the elements of a specific tile, no matter how the matrix is distributed,
/// will be set with the same set of values.
///
/// @pre @param matrix is a square matrix,
/// @pre @param matrix has a square blocksize.
template <class T>
void set_random_non_zero_diagonal(Matrix<T, Device::CPU>& matrix) {
  using pika::execution::thread_stacksize;

  // note:
  // By assuming square blocksizes, it is easier to locate elements. In fact:
  // - Elements on the diagonal are stored in the diagonal of the diagonal tiles
  // - Tiles under the diagonal store elements of the lower triangular matrix
  // - Tiles over the diagonal store elements of the upper triangular matrix

  const Distribution& dist = matrix.distribution();

  DLAF_ASSERT(square_size(matrix), matrix);
  DLAF_ASSERT(square_blocksize(matrix), matrix);

  for (auto ij_lc : iterate_range2d(dist.local_nr_tiles())) {
    GlobalTileIndex ij = dist.global_tile_index(ij_lc);

    const SizeType seed = ij.col() + ij.row() * matrix.size().cols();
    const BaseType<T> lower_limit = static_cast<BaseType<T>>(0.1);

    using TileType = typename std::decay_t<decltype(matrix)>::TileType;
    auto set_hp_f = [=](const TileType& tile) {
      internal::getter_random<T> random_value(seed);
      if (ij.row() == ij.col()) {
        for (auto ij_el_tl : iterate_range2d(tile.size())) {
          auto value = random_value();
          if (ij_el_tl.row() == ij_el_tl.col() && std::abs(value) < lower_limit) {
            if (value == T{0})
              tile(ij_el_tl) = lower_limit;
            else
              tile(ij_el_tl) = lower_limit * value / std::abs(value);
          }
          else
            tile(ij_el_tl) = value;
        }
      }
      else
        for (auto ij_el_tl : iterate_range2d(tile.size())) {
          tile(ij_el_tl) = random_value();
        }
    };

    dlaf::internal::transformDetach(dlaf::internal::Policy<Backend::MC>(thread_stacksize::nostack),
                                    std::move(set_hp_f), matrix.readwrite(ij_lc));
  }
}

}
}
