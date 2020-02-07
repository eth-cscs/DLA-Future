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

#include <exception>
#include <random>
#include <string>

#include <hpx/util.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"

/// @file

namespace dlaf {
namespace util_matrix {

/// @brief Verify if dlaf::Matrix is square
///
/// @tparam Matrix refers to a dlaf::Matrix object
/// @throws std::invalid_argument if the matrix is not squared
template <class Matrix>
void assertSizeSquare(const Matrix& matrix, std::string function, std::string mat_name) {
  if (matrix.size().rows() != matrix.size().cols())
    throw std::invalid_argument(function + ": " + "Matrix " + mat_name + " is not square.");
}

/// @brief Verify if dlaf::Matrix tile is square
///
/// @tparam Matrix refers to a dlaf::Matrix object
/// @throws std::invalid_argument if the matrix block is not squared
template <class Matrix>
void assertBlocksizeSquare(const Matrix& matrix, std::string function, std::string mat_name) {
  if (matrix.blockSize().rows() != matrix.blockSize().cols())
    throw std::invalid_argument(function + ": " + "Block size in matrix " + mat_name +
                                " is not square.");
}

/// @brief Verify if dlaf::Matrix is distributed on a (1x1) grid (i.e. if it is a local matrix).
///
/// @tparam Matrix refers to a dlaf::Matrix object
/// @throws std::invalid_argument if the matrix is not local
template <class Matrix>
void assertLocalMatrix(const Matrix& matrix, std::string function, std::string mat_name) {
  if (matrix.distribution().commGridSize() != comm::Size2D{1, 1})
    throw std::invalid_argument(function + ": " + "Matrix " + mat_name + " is not local.");
}

/// @brief Verify that the matrix is distributed according to the given communicator grid.
///
/// @tparam Matrix refers to a dlaf::Matrix object
/// @throws std::invalid_argument if the matrix is not distributed correctly
template <class Matrix>
void assertMatrixDistributedOnGrid(const comm::CommunicatorGrid& grid, const Matrix& matrix,
                                   std::string function, std::string mat_name, std::string grid_name) {
  if ((matrix.distribution().commGridSize() != grid.size()) ||
      (matrix.distribution().rankIndex() != grid.rank()))
    throw std::invalid_argument(function + ": " + "The matrix " + mat_name +
                                " is not distributed according to the communicator grid " + grid_name +
                                ".");
}

/// @brief Verify that matrices A and B are multipliable,
///
/// @tparam A refers to a dlaf::Matrix object
/// @tparam B refers to a dlaf::Matrix object
/// @throws std::invalid_argument if matrices A and B are not multipliable, taking into account the Side
/// (Left/Right) and the Op (NoTrans/Trans/ConjTrans) of the multiplication itself
template <class MatrixConst, class Matrix>
void assertMultipliableMatrices(const MatrixConst& mat_a, const Matrix& mat_b, blas::Side side,
                                blas::Op op, std::string function, std::string mat_a_name,
                                std::string mat_b_name) {
  if (side == blas::Side::Left) {
    if (op == blas::Op::NoTrans) {
      if (mat_a.nrTiles().cols() != mat_b.nrTiles().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not left multipliable (cols of matrix A not equal to rows of matrix B).");
      }
      if (mat_a.size().cols() != mat_b.size().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not left multipliable (size of matrix A not equal to that of matrix B).");
      }
      if (mat_a.blockSize().cols() != mat_b.blockSize().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not left multipliable (blocksize of matrix A not equal to that of matrix B).");
      }
    }
    else if (op == blas::Op::Trans || op == blas::Op::ConjTrans) {
      if (mat_a.nrTiles().rows() != mat_b.nrTiles().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not left multipliable (cols of matrix A not equal to rows of matrix B).");
      }
      if (mat_a.size().rows() != mat_b.size().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not left multipliable (size of matrix A not equal to that of matrix B).");
      }
      if (mat_a.blockSize().rows() != mat_b.blockSize().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not left multipliable (blocksize of matrix A not equal to that of matrix B).");
      }
    }
  }
  else if (side == blas::Side::Right) {
    if (op == blas::Op::NoTrans) {
      if (mat_a.nrTiles().rows() != mat_b.nrTiles().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not right multipliable (rows of matrix A not equal to cols of matrix B).");
      }
      if (mat_a.size().rows() != mat_b.size().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not right multipliable (size of matrix A not equal to that of matrix B).");
      }
      if (mat_a.blockSize().rows() != mat_b.blockSize().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not right multipliable (blocksize of matrix A not equal to that of matrix B).");
      }
    }
    else if (op == blas::Op::Trans || op == blas::Op::ConjTrans) {
      if (mat_a.nrTiles().cols() != mat_b.nrTiles().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not right multipliable (rows of matrix A not equal to cols of matrix B).");
      }
      if (mat_a.size().cols() != mat_b.size().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not right multipliable (size of matrix A not equal to that of matrix B).");
      }
      if (mat_a.blockSize().cols() != mat_b.blockSize().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + mat_a_name + " and " + mat_b_name +
            " are not right multipliable (blocksize of matrix A not equal to that of matrix B).");
      }
    }
  }
}

}

namespace matrix {
namespace util {
namespace internal {

/// Callable that returns a random value between [-1, 1]
///
/// Return random values for any given index
template <class T>
class getter_random {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T is not compatible with random generator used.");

public:
  getter_random(const unsigned long seed = std::minstd_rand::default_seed) : random_engine_(seed) {}

  T operator()(const GlobalElementIndex&) {
    return random_sampler_(random_engine_);
  }

private:
  std::mt19937_64 random_engine_;
  std::uniform_real_distribution<T> random_sampler_{-1, 1};
};

template <class T>
class getter_random<std::complex<T>> : private getter_random<T> {
public:
  using getter_random<T>::getter_random;

  std::complex<T> operator()(const GlobalElementIndex& index) {
    return std::complex<T>(getter_random<T>::operator()(index), getter_random<T>::operator()(index));
  }
};

/// Helper function for random hermitian positive definite matrices
///
/// Return random values in range [-1, 1] for any given index and adds the specified offset on indexes on
/// the diagonal.
/// Moreover:
/// - Values on the diagonal are returned as real numbers (if type is std::complex with imag() == 0)
/// - Values that appears in the upper triangular part are conjugated
template <class T>
class getter_random_hermitian_positive_definite : private getter_random<T> {
public:
  getter_random_hermitian_positive_definite(const dlaf::BaseType<T> offset_value,
                                            const unsigned long seed = std::minstd_rand::default_seed)
      : getter_random<T>(seed), offset_value_(offset_value) {}

  T operator()(const GlobalElementIndex& index) {
    const auto element_position = dlaf::common::position(index);

    T random_value = getter_random<T>::operator()(index);

    if (element_position == dlaf::common::Position::DIAGONAL)
      return std::real(random_value) + offset_value_;

    if (element_position == dlaf::common::Position::LOWER)
      return random_value;
    else
      return dlaf::conj(random_value);
  }

private:
  dlaf::BaseType<T> offset_value_;
};

}

/// @brief Set the elements of the matrix
///
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @param el a copy is given to each tile
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre el return type should be T.
template <class T, class ElementGetter>
void set(Matrix<T, Device::CPU>& matrix, ElementGetter&& el) {
  const matrix::Distribution& dist = matrix.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      LocalTileIndex tile_wrt_local{tile_i, tile_j};
      GlobalTileIndex tile_wrt_global = dist.globalTileIndex(tile_wrt_local);

      auto tl_index = dist.globalElementIndex(tile_wrt_global, {0, 0});

      hpx::dataflow(hpx::util::unwrapping([tl_index, el](auto&& tile) {
                      for (SizeType j = 0; j < tile.size().cols(); ++j)
                        for (SizeType i = 0; i < tile.size().rows(); ++i)
                          tile({i, j}) = el(GlobalElementIndex{i + tl_index.row(), j + tl_index.col()});
                    }),
                    matrix(tile_wrt_local));
    }
  }
}

/// Set the matrix with random values in the range [-1, 1]
///
/// Each tile creates its own random generator engine with a unique seed
/// which is computed as a function of the tile global index.
/// This means that a specific tile index, no matter on which rank it will be,
/// will have the same set of values.
template <class T>
void set_random(Matrix<T, Device::CPU>& matrix) {
  const matrix::Distribution& dist = matrix.distribution();

  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      LocalTileIndex tile_wrt_local{tile_i, tile_j};
      GlobalTileIndex tile_wrt_global = dist.globalTileIndex(tile_wrt_local);

      auto tl_index = dist.globalElementIndex(tile_wrt_global, {0, 0});
      auto seed = tl_index.row() * matrix.size().cols() + tl_index.col();

      hpx::dataflow(hpx::util::unwrapping([tl_index, seed](auto&& tile) {
                      internal::getter_random<T> value_at(seed);

                      for (SizeType j = 0; j < tile.size().cols(); ++j)
                        for (SizeType i = 0; i < tile.size().rows(); ++i)
                          tile(TileElementIndex{i, j}) =
                              value_at(GlobalElementIndex{tl_index.row() + i, tl_index.col() + j});
                    }),
                    matrix(tile_wrt_local));
    }
  }
}

/// Set a matrix with random values in range [-1, 1] but assuring it will be positive definite.
///
/// Each tile creates its own random generator engine with a unique seed
/// which is computed as a function of the tile global index.
/// This means that a specific tile index, no matter on which rank it will be,
/// it will have the same set of values.
template <class T>
void set_random_hermitian_positive_definite(Matrix<T, Device::CPU>& matrix) {
  const matrix::Distribution& dist = matrix.distribution();

  // Check if matrix is square
  util_matrix::assertSizeSquare(matrix, "set_hermitian_random_positive_definite", "matrix");
  // Check if block matrix is square
  util_matrix::assertBlocksizeSquare(matrix, "set_hermitian_random_positive_definite", "matrix");

  std::size_t offset_value = 2 * std::max(matrix.size().rows(), matrix.size().cols());

  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      LocalTileIndex tile_wrt_local{tile_i, tile_j};
      GlobalTileIndex tile_wrt_global = dist.globalTileIndex(tile_wrt_local);

      auto tl_index = dist.globalElementIndex(tile_wrt_global, {0, 0});

      auto tile_position = dlaf::common::position(tile_wrt_global);

      // compute the same seed for original and "transposed" tiles, so transposed ones will know the
      // values of the original one without the need of accessing real values (nor communication in case
      // of distributed matrices)
      size_t seed;
      if (tile_position <= dlaf::common::Position::DIAGONAL)
        seed = tl_index.row() * matrix.size().cols() + tl_index.col();
      else
        seed = tl_index.col() * matrix.size().rows() + tl_index.row();

      hpx::dataflow(
          hpx::util::unwrapping([tile_position, tl_index, seed, offset_value](auto&& tile) {
            internal::getter_random_hermitian_positive_definite<T> value_at(offset_value, seed);

            if (tile_position == dlaf::common::Position::DIAGONAL) {
              // for diagonal tiles get just lower matrix values and set value for both straight and
              // transposed indices
              for (SizeType j = 0; j < tile.size().cols(); ++j) {
                for (SizeType i = 0; i <= j; ++i) {
                  auto value = value_at(GlobalElementIndex{tl_index.row() + i, tl_index.col() + j});

                  tile(TileElementIndex{i, j}) = value;
                  if (i != j)
                    tile(TileElementIndex{j, i}) = dlaf::conj(value);
                }
              }
            }
            else {
              // random values are requested in the same order for both original and transposed
              for (SizeType j = 0; j < tile.size().cols(); ++j) {
                for (SizeType i = 0; i < tile.size().rows(); ++i) {
                  auto value = value_at(GlobalElementIndex{tl_index.row() + i, tl_index.col() + j});

                  // but they are set row-wise in the original tile and col-wise in the transposed one
                  if (tile_position == dlaf::common::Position::LOWER)
                    tile(TileElementIndex{i, j}) = value;
                  else
                    tile(TileElementIndex{j, i}) = value;
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
