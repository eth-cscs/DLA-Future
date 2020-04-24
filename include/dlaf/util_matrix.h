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
#include <hpx/hpx.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"

/// @file

namespace dlaf {
namespace matrix {
namespace util {
namespace internal {

/// @brief Assert that the @p matrix is square.
///
/// When the assertion is enabled, terminates the program with an error message if the matrix is not
/// square. This assertion is enabled when **DLAF_ASSERT_ENABLE** is ON.
#define DLAF_ASSERT_SIZE_SQUARE(matrix)                                                               \
  DLAF_ASSERT((matrix.size().rows() == matrix.size().cols()), "Matrix ", #matrix, " is not square (", \
              matrix.size().rows(), "x", matrix.size().cols(), ").")

/// @brief Assert that the @p matrix tiles are square.
///
/// When the assertion is enabled, terminates the program with an error message if the tiles of matrix
/// are not square. This assertion is enabled when **DLAF_ASSERT_ENABLE** is ON.
#define DLAF_ASSERT_BLOCKSIZE_SQUARE(matrix)                                                          \
  DLAF_ASSERT((matrix.blockSize().rows() == matrix.blockSize().cols()), "Block size in matrix ",      \
              #matrix, " is not square (", matrix.blockSize().rows(), "x", matrix.blockSize().cols(), \
              ").")

/// @brief Assert that the @p matrix is distributed on a (1x1) grid (i.e. if it is a local matrix).
///
/// When the assertion is enabled, terminates the program with an error message if matrix is not local.
/// This assertion is enabled when **DLAF_ASSERT_ENABLE** is ON.
#define DLAF_ASSERT_LOCALMATRIX(matrix)                                                         \
  DLAF_ASSERT((matrix.distribution().commGridSize() == comm::Size2D(1, 1)), "Matrix ", #matrix, \
              " is not local (grid size: ", matrix.distribution().commGridSize().rows(), "x",   \
              matrix.distribution().commGridSize().cols(), ").")

/// @brief Assert that the @p matrix is distributed according to the given communicator grid.
///
/// When the assertion is enabled, terminates the program with an error message if matrix is not on distributed
/// according to the given communicator grid. This assertion is enabled when **DLAF_ASSERT_ENABLE** is ON.
#define DLAF_ASSERT_DISTRIBUTED_ON_GRID(grid, matrix)                                                \
  DLAF_ASSERT(((matrix.distribution().commGridSize() == grid.size()) &&                              \
               (matrix.distribution().rankIndex() == grid.rank())),                                  \
              "The matrix ", #matrix, " (rank: ", matrix.distribution().rankIndex(),                 \
              ", grid size: ", matrix.distribution().commGridSize().rows(), "x",                     \
              matrix.distribution().commGridSize().cols(),                                           \
              ") is not distributed according to the communicator grid ", #grid,                     \
              " (rank: ", grid.rank(), ", grid size: ", grid.size().rows(), "x", grid.size().cols(), \
              ").")

template <class MatrixConst, class Matrix, class Mat, class Location>
void assertMultipliableMatrices(const MatrixConst& mat_a, const Matrix& mat_b, const Mat& mat_c,
                                const blas::Op opA, const blas::Op opB, const Location location,
                                std::string mat_a_name, std::string mat_b_name, std::string mat_c_name) {
  auto rows = [](const auto& size, const blas::Op op) -> decltype(size.rows()) {
    switch (op) {
      case blas::Op::NoTrans:
        return size.rows();
      case blas::Op::Trans:
      case blas::Op::ConjTrans:
        return size.cols();
      default:
        return {};
    }
  };
  auto cols = [](const auto& size, const blas::Op op) -> decltype(size.cols()) {
    switch (op) {
      case blas::Op::NoTrans:
        return size.cols();
      case blas::Op::Trans:
      case blas::Op::ConjTrans:
        return size.rows();
      default:
        return {};
    }
  };

  DLAF_ASSERT_WITH_ORIGIN(location,
                          rows(mat_a.size(), opA) == mat_c.size().rows() &&
                              cols(mat_a.size(), opA) == rows(mat_b.size(), opB) &&
                              cols(mat_b.size(), opB) == mat_c.size().cols(),
                          "Size mismatch: ", mat_a_name, " (", rows(mat_a.size(), opA), ", ",
                          cols(mat_a.size(), opA), ") x ", mat_b_name, " (", rows(mat_b.size(), opB),
                          ", ", cols(mat_b.size(), opB), ") --> ", mat_c_name, " ", mat_c.size(),
                          " cannot be performed.");

  DLAF_ASSERT_WITH_ORIGIN(location,
                          rows(mat_a.blockSize(), opA) == mat_c.blockSize().rows() &&
                              cols(mat_a.blockSize(), opA) == rows(mat_b.blockSize(), opB) &&
                              cols(mat_b.blockSize(), opB) == mat_c.blockSize().cols(),
                          "BlockSize mismatch: ", mat_a_name, " (", rows(mat_a.blockSize(), opA), ", ",
                          cols(mat_a.blockSize(), opA), ") x ", mat_b_name, " (",
                          rows(mat_b.blockSize(), opB), ", ", cols(mat_b.blockSize(), opB), ") --> ",
                          mat_c_name, " ", mat_c.blockSize(), " cannot be performed.");
}
/// @brief Assert that the matrices @p mat_a and @p mat_b are multipliable and that matrix @p mat_c can
/// store the result of this multiplication.
///
/// When the assertion is enabled, terminates the program with an error message if matrices @p mat_a and
/// @p mat_b are not multipliable or if the matrix @p mat_c can not store the result. This assertion is
/// enabled when **DLAF_ASSERT_ENABLE** is ON.
#define DLAF_ASSERT_MULTIPLIABLE_MATRICES(a, b, c, opA, opB)                                           \
  ::dlaf::matrix::util::internal::assertMultipliableMatrices(a, b, c, opA, opB, SOURCE_LOCATION(), #a, \
                                                             #b, #c);

/// Callable that returns random values in the range [-1, 1]
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

/// Callable that returns random complex numbers whose absolute values are less than 1
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

/// @brief Set the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @param el a copy is given to each tile.
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre el return type should be T.
template <class T, class ElementGetter>
void set(Matrix<T, Device::CPU>& matrix, const ElementGetter& el) {
  const Distribution& dist = matrix.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      LocalTileIndex tile_wrt_local{tile_i, tile_j};
      GlobalTileIndex tile_wrt_global = dist.globalTileIndex(tile_wrt_local);

      auto tl_index = dist.globalElementIndex(tile_wrt_global, {0, 0});

      hpx::dataflow(hpx::util::unwrapping([tl_index, el = el](auto&& tile) {
                      for (SizeType j = 0; j < tile.size().cols(); ++j)
                        for (SizeType i = 0; i < tile.size().rows(); ++i)
                          tile({i, j}) = el(GlobalElementIndex{i + tl_index.row(), j + tl_index.col()});
                    }),
                    matrix(tile_wrt_local));
    }
  }
}

/// Set the matrix with random values whose absolute values are less than 1.
///
/// Values will be random numbers in:
/// - real:     [-1, 1]
/// - complex:  a circle of radius 1 centered at origin
///
/// Each tile creates its own random generator engine with a unique seed
/// which is computed as a function of the tile global index.
/// This means that a specific tile index, no matter on which rank it will be,
/// will have the same set of values.
template <class T>
void set_random(Matrix<T, Device::CPU>& matrix) {
  using namespace dlaf::util::size_t;

  const Distribution& dist = matrix.distribution();

  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      LocalTileIndex tile_wrt_local{tile_i, tile_j};
      GlobalTileIndex tile_wrt_global = dist.globalTileIndex(tile_wrt_local);

      auto tl_index = dist.globalElementIndex(tile_wrt_global, {0, 0});
      auto seed = sum(tl_index.col(), mul(tl_index.row(), matrix.size().cols()));

      hpx::dataflow(hpx::util::unwrapping([seed](auto&& tile) {
                      internal::getter_random<T> random_value(seed);

                      for (SizeType j = 0; j < tile.size().cols(); ++j)
                        for (SizeType i = 0; i < tile.size().rows(); ++i)
                          tile(TileElementIndex{i, j}) = random_value();
                    }),
                    matrix(tile_wrt_local));
    }
  }
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
/// @pre @param matrix is a square matrix
/// @pre @param matrix has a square blocksize
template <class T>
void set_random_hermitian_positive_definite(Matrix<T, Device::CPU>& matrix) {
  // note:
  // By assuming square blocksizes, it is easier to locate elements. In fact:
  // - Elements on the diagonal are stored in the diagonal of the diagonal tiles
  // - Tiles under the diagonal store elements of the lower triangular matrix
  // - Tiles over the diagonal store elements of the upper triangular matrix

  using namespace dlaf::util::size_t;

  const Distribution& dist = matrix.distribution();

  // Check if matrix is square
  DLAF_ASSERT_SIZE_SQUARE(matrix);
  // Check if block matrix is square
  DLAF_ASSERT_BLOCKSIZE_SQUARE(matrix);

  auto offset_value = mul(2, to_sizet(matrix.size().rows()));
  auto full_tile_size = matrix.blockSize();

  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      LocalTileIndex tile_wrt_local{tile_i, tile_j};
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

      hpx::dataflow(hpx::util::unwrapping(
                        [tile_wrt_global, seed, offset_value, full_tile_size](auto&& tile) {
                          internal::getter_random<T> random_value(seed);

                          if (tile_wrt_global.row() == tile_wrt_global.col()) {  // DIAGONAL
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
                          else {  // LOWER or UPPER (except DIAGONAL)
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
                        }),
                    matrix(tile_wrt_local));
    }
  }
}

}
}
}
