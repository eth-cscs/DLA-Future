//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <utility>

#include <dlaf/matrix/col_major_layout.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/matrix.h>

namespace dlaf::matrix {

// Note: the templates of the following helper functions are inverted w.r.t. the Matrix templates
// to allow the user to only specify the device and let the compiler deduce the type T.

// Local versions

/// Create a non distributed matrix of size @p size and block size @p tile_size
/// which references elements
/// that are already allocated in the memory with a column major layout.
///
/// @param[in] ld the leading dimension of the matrix,
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre ld >= max(1, size.row()),
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> create_matrix_from_col_major(const GlobalElementSize& size,
                                          const TileElementSize& tile_size, SizeType ld, T* ptr) {
  Distribution dist(size, tile_size, {1, 1}, {0, 0}, {0, 0});
  ColMajorLayout layout{std::move(dist), ld};
  return Matrix<T, D>(layout, ptr);
}

// Distributed versions

/// Create a distributed matrix of size @p size and block size @p tile_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a column major layout.
///
/// @param[in] ld the leading dimension of the matrix,
/// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix,
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre @p ld >= max(1, size.row()),
/// @pre @p source_rank_index.isIn(grid_size),
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> create_matrix_from_col_major(const GlobalElementSize& size,
                                          const TileElementSize& tile_size, SizeType ld,
                                          const comm::CommunicatorGrid& comm,
                                          const comm::Index2D& source_rank_index, T* ptr) {
  Distribution dist(size, tile_size, comm.size(), comm.rank(), source_rank_index);
  ColMajorLayout layout{std::move(dist), ld};

  return Matrix<T, D>(layout, ptr);
}

/// Create a distributed matrix of size @p size and block size @p tile_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a column major layout.
///
/// This method assumes @p source_rank_index to be {0,0}.
/// @param[in] ld the leading dimension of the matrix,
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre @p ld >= max(1, size.row()),
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> create_matrix_from_col_major(const GlobalElementSize& size,
                                          const TileElementSize& tile_size, SizeType ld,
                                          const comm::CommunicatorGrid& comm, T* ptr) {
  return create_matrix_from_col_major<D>(size, tile_size, ld, comm, {0, 0}, ptr);
}

}
