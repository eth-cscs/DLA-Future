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
namespace matrix {

namespace util {

namespace details {

#define _DLAF_PRECONDITION_FUNCTION(condition) dlaf::matrix::util::details::assert##condition

#define _DLAF_PRECONDITION_1(condition, arg1) \
  _DLAF_PRECONDITION_FUNCTION(condition)(arg1, DLAF_SOURCE_LOCATION, #arg1)
#define _DLAF_PRECONDITION_2(condition, arg1, arg2) \
  _DLAF_PRECONDITION_FUNCTION(condition)(arg1, arg2, DLAF_SOURCE_LOCATION, #arg1, #arg2)

/// @brief Verify if dlaf::Matrix is square
///
/// @tparam Matrix refers to a dlaf::Matrix object
/// @throws std::invalid_argument if the matrix is not squared
template <class Matrix>
void assertSizeSquare(const Matrix& matrix, std::string function, std::string mat_name) {
  if (matrix.size().rows() != matrix.size().cols())
    throw std::invalid_argument(function + ": " + "Matrix " + mat_name + " is not square.");
}
#define DLAF_PRECONDITION_SIZE_SQUARE(matrix) _DLAF_PRECONDITION_1(SizeSquare, matrix)

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
#define DLAF_PRECONDITION_BLOCKSIZE_SQUARE(matrix) _DLAF_PRECONDITION_1(BlocksizeSquare, matrix)

/// @brief Verify if dlaf::Matrix is distributed on a (1x1) grid (i.e. if it is a local matrix).
///
/// @tparam Matrix refers to a dlaf::Matrix object
/// @throws std::invalid_argument if the matrix is not local
template <class Matrix>
void assertLocalMatrix(const Matrix& matrix, std::string function, std::string mat_name) {
  if (matrix.distribution().commGridSize() != comm::Size2D{1, 1})
    throw std::invalid_argument(function + ": " + "Matrix " + mat_name + " is not local.");
}
#define DLAF_PRECONDITION_LOCALMATRIX(matrix) _DLAF_PRECONDITION_1(LocalMatrix, matrix)

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
#define DLAF_PRECONDITION_IS_DISTRIBUTED_ON_GRID(grid, matrix) _DLAF_PRECONDITION_2(MatrixDistributedOnGrid, grid, matrix)

}
}
}
}
