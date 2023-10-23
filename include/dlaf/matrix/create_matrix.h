//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/layout_info.h>
#include <dlaf/matrix/matrix.h>

namespace dlaf::matrix {

// Note: the templates of the following helper functions are inverted w.r.t. the Matrix templates
// to allow the user to only specify the device and let the compiler deduce the type T.

// Local versions

/// Create a non distributed matrix of size @p size and block size @p block_size
/// which references elements
/// that are already allocated in the memory with a column major layout.
///
/// @param[in] ld the leading dimension of the matrix,
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre ld >= max(1, size.row()),
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> createMatrixFromColMajor(const LocalElementSize& size, const TileElementSize& block_size,
                                      SizeType ld, T* ptr) {
  return Matrix<T, D>(colMajorLayout(size, block_size, ld), ptr);
}

/// Create a non distributed matrix of size @p size and block size @p block_size
/// which references elements
/// that are already allocated in the memory with a tile layout.
///
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> createMatrixFromTile(const LocalElementSize& size, const TileElementSize& block_size,
                                  T* ptr) {
  return Matrix<T, D>(tileLayout(size, block_size), ptr);
}

/// Create a non distributed matrix of size @p size and block size @p block_size
/// which references elements
/// that are already allocated in the memory with a tile layout.
///
/// @param[in] ld_tile the leading dimension of the tiles,
/// @param[in] tiles_per_col the number of tiles stored for each column of tiles,
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre @p ld_tile >= max(1, min(block_size.row(), size.row())),
/// @pre @p tiles_per_col >= ceilDiv(size.row(), block_size.col()),
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> createMatrixFromTile(const LocalElementSize& size, const TileElementSize& block_size,
                                  SizeType ld_tile, SizeType tiles_per_col, T* ptr) {
  return Matrix<T, D>(tileLayout(size, block_size, ld_tile, tiles_per_col), ptr);
}

// Distributed versions

/// Create a distributed matrix of size @p size and block size @p block_size
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
Matrix<T, D> createMatrixFromColMajor(const GlobalElementSize& size, const TileElementSize& block_size,
                                      SizeType ld, const comm::CommunicatorGrid& comm,
                                      const comm::Index2D& source_rank_index, T* ptr) {
  Distribution distribution(size, block_size, comm.size(), comm.rank(), source_rank_index);
  auto layout = colMajorLayout(distribution.localSize(), block_size, ld);

  return Matrix<T, D>(std::move(distribution), layout, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
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
Matrix<T, D> createMatrixFromColMajor(const GlobalElementSize& size, const TileElementSize& block_size,
                                      SizeType ld, const comm::CommunicatorGrid& comm, T* ptr) {
  return createMatrixFromColMajor<D>(size, block_size, ld, comm, {0, 0}, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a tile layout.
///
/// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix,
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre @p source_rank_index.isIn(grid_size),
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                  const comm::CommunicatorGrid& comm,
                                  const comm::Index2D& source_rank_index, T* ptr) {
  Distribution distribution(size, block_size, comm.size(), comm.rank(), source_rank_index);
  auto layout = tileLayout(distribution.localSize(), block_size);

  return Matrix<T, D>(std::move(distribution), layout, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a tile layout.
///
/// This method assumes @p source_rank_index to be {0,0}.
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                  const comm::CommunicatorGrid& comm, T* ptr) {
  return createMatrixFromTile<D>(size, block_size, comm, {0, 0}, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a tile layout.
///
/// @param[in] ld_tile the leading dimension of the tiles,
/// @param[in] tiles_per_col the number of tiles stored for each column of tiles,
/// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix,
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre @p ld_tile >= max(1, min(block_size.row(), size.row())),
/// @pre @p tiles_per_col >= ceilDiv(size.row(), block_size.row()),
/// @pre @p source_rank_index.isIn(grid_size),
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                  SizeType ld_tile, SizeType tiles_per_col,
                                  const comm::CommunicatorGrid& comm,
                                  const comm::Index2D& source_rank_index, T* ptr) {
  Distribution distribution(size, block_size, comm.size(), comm.rank(), source_rank_index);
  auto layout = tileLayout(distribution.localSize(), block_size, ld_tile, tiles_per_col);

  return Matrix<T, D>(std::move(distribution), layout, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a tile layout.
///
/// This method assumes @p source_rank_index to be {0,0}.
/// @param[in] ld_tile the leading dimension of the tiles,
/// @param[in] tiles_per_col the number of tiles stored for each column of tiles,
/// @param[in] ptr is the pointer to the first element of the local part of the matrix,
/// @pre @p ld_tile >= max(1, min(block_size.row(), size.row()),
/// @pre @p tiles_per_col >= ceilDiv(size.row(), block_size.col()),
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device D, class T>
Matrix<T, D> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                  SizeType ld_tile, SizeType tiles_per_col,
                                  const comm::CommunicatorGrid& comm, T* ptr) {
  return createMatrixFromTile<D>(size, block_size, ld_tile, tiles_per_col, comm, {0, 0}, ptr);
}

}
