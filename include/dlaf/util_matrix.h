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

#ifndef UTIL_MATRIX_H
#define UTIL_MATRIX_H

#include <exception>

/// @file

namespace dlaf {
namespace util_matrix {

/// @brief Verify if dlaf::Matrix is square
///
/// @tparam M refers to a dlaf::Matrix object
/// @throws std::invalid_argument if the matrix is not squared
/// @pre dlaf::Matrix sizes >= 0
template <class Matrix>
void check_size_square(const Matrix& matrix, std::string function, std::string mat_name) {
  assert(matrix.size().isValid());
  if (matrix.size().rows() != matrix.size().cols())
    throw std::invalid_argument(function + ": " + "Matrix " + mat_name + " is not square.");
}

/// @brief Verify if dlaf::Matrix tile is square
///
/// @tparam M refers to a dlaf::Matrix object
/// @throws std::invalid_argument if the matrix block is not squared
/// @pre dlaf::Matrix sizes > 0
template <class Matrix>
void check_blocksize_square(const Matrix& matrix, std::string function, std::string mat_name) {
  assert(matrix.size().isValid());
  if (matrix.blockSize().rows() != matrix.blockSize().cols())
    throw std::invalid_argument(function + ": " + "Block matrix in matrix " + mat_name +
                                " is not square.");
}

}
}
#endif
