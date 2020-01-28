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
#include <string>
#include <random>

#include "dlaf/matrix.h"

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

/// @brief Sets the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre el return type should be T.
template <class T, class ElementGetter>
void set(Matrix<T, Device::CPU>& mat, ElementGetter el) {
  const matrix::Distribution& dist = mat.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      auto tile = mat(LocalTileIndex(tile_i, tile_j)).get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        SizeType j = dist.globalElementFromLocalTileAndTileElement<Coord::Col>(tile_j, jj);
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
          SizeType i = dist.globalElementFromLocalTileAndTileElement<Coord::Row>(tile_i, ii);
          tile({ii, jj}) = el({i, j});
        }
      }
    }
  }
}

template <class T>
void set_random(Matrix<T, Device::CPU>& matrix) {
  std::minstd_rand random_seed;
  std::uniform_real_distribution<T> random_sampler(-1, 1);

  dlaf::matrix::util::set(matrix, [random_sampler, random_seed](const GlobalElementIndex&) mutable {
      return random_sampler(random_seed);
  });
}

template <class T>
void set_random_positive_definite(Matrix<T, Device::CPU>& matrix) {
  std::minstd_rand random_seed;
  std::uniform_real_distribution<T> random_sampler(-1, 1);

  T offset_value = 2 * std::max(matrix.size().rows(), matrix.size().cols());

  dlaf::matrix::util::set(matrix, [random_sampler, random_seed, offset_value](const GlobalElementIndex& index) mutable {
      return random_sampler(random_seed) + (index.row() == index.col() ? 1 : offset_value);
  });
}

}
}

}
