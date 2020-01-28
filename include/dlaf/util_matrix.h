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

/// @brief Verify that matrices A and B are multipliable, i.e. the number of )
///
/// @tparam A refers to a dlaf::Matrix object
/// @tparam B refers to a dlaf::Matrix object
/// @throws std::invalid_argument if the matrix A column number is not equal to the matrix B row number
template <class Matrix>
void assertMultipliableMatrices(const Matrix& A, const Matrix& B, blas::Side side, blas::Op op,
                                std::string function, std::string A_name, std::string B_name) {
  if (side == blas::Side::Left) {
    if (op == blas::Op::NoTrans) {
      if (A.nrTiles().cols() != B.nrTiles().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not left multipliable (cols of matrix A not equal to rows of matrix B).");
      }
      if (A.size().cols() != B.size().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not left multipliable (size of matrix A not equal to that of matrix B).");
      }
      if (A.blockSize().cols() != B.blockSize().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not left multipliable (blocksize of matrix A not equal to that of matrix B).");
      }
    }
    else if (op == blas::Op::Trans || op == blas::Op::ConjTrans) {
      if (A.nrTiles().rows() != B.nrTiles().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not left multipliable (cols of matrix A not equal to rows of matrix B).");
      }
      if (A.size().rows() != B.size().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not left multipliable (size of matrix A not equal to that of matrix B).");
      }
      if (A.blockSize().rows() != B.blockSize().rows()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not left multipliable (blocksize of matrix A not equal to that of matrix B).");
      }
    }
  }
  else if (side == blas::Side::Right) {
    if (op == blas::Op::NoTrans) {
      if (A.nrTiles().rows() != B.nrTiles().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not right multipliable (rows of matrix A not equal to cols of matrix B).");
      }
      if (A.size().rows() != B.size().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not right multipliable (size of matrix A not equal to that of matrix B).");
      }
      if (A.blockSize().rows() != B.blockSize().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not right multipliable (blocksize of matrix A not equal to that of matrix B).");
      }
    }
    else if (op == blas::Op::Trans || op == blas::Op::ConjTrans) {
      if (A.nrTiles().cols() != B.nrTiles().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not right multipliable (rows of matrix A not equal to cols of matrix B).");
      }
      if (A.size().cols() != B.size().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not right multipliable (size of matrix A not equal to that of matrix B).");
      }
      if (A.blockSize().cols() != B.blockSize().cols()) {
        throw std::invalid_argument(
            function + ": " + "The matrices " + A_name + " and " + B_name +
            " are not right multipliable (blocksize of matrix A not equal to that of matrix B).");
      }
    }
  }
}

}
}
