//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/tridiag_solver/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Finds the eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// Notation:
/// nb - the block/tile size of all matrices and vectors
/// n - the dimension of the full tridiagonal matrix
/// d - the diagonal of the full tridiagonal matrix
/// e - the subdiagonal of the full tridiagonal matrix
/// Q1 - (n1 x n1) the orthogonal matrix of the top subproblem
/// Q2 - (n2 x n2) the orthogonal matrix of the bottom subproblem
///      ┌───┬───┐
///      │Q1 │   │
/// Q := ├───┼───┤
///      │   │Q2 │
///      └───┴───┘
/// The following holds for each n1 and n2: `n2 = n1` or `n2 = n1 + nb`.
///
///                          ┌───┬──┬─┐
///                          │Q1'│  │ │
/// Q-U multiplication form: ├──┬┴──┤ │
///                          │  │Q2'│ │
///                          └──┴───┴─┘
///
/// @param mat_a  [in/out] `n x 2` matrix with the diagonal and off-diagonal of the symmetric tridiagonal
/// matrix in the first column and second columns respectively. The last entry of the second column is
/// not used. On exit the eigenvalues are saved in the first column.
/// @param mat_ev [out]    `n x n` matrix holding the eigenvectors of the the symmetric tridiagonal
/// matrix on exit.
///
///
/// @pre mat_a and mat_ev are local matrices
/// @pre mat_a has 2 columns
/// @pre mat_ev is a square matrix
/// @pre mat_ev has a square block size
template <Backend backend, Device device, class T>
void tridiagSolver(Matrix<T, device>& mat_trd, Matrix<T, device>& mat_ev) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "tridagSolver accepts only real values (float, double)!");

  DLAF_ASSERT(matrix::local_matrix(mat_trd), mat_trd);
  DLAF_ASSERT(mat_trd.distribution().size().cols() == 2, mat_trd);

  DLAF_ASSERT(matrix::local_matrix(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_size(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_blocksize(mat_ev), mat_ev);

  // Auxiliary matrix used for the D&C algorithm
  const matrix::Distribution& distr = mat_ev.distribution();
  // Extra workspace for Q1 and Q2
  Matrix<T, device> mat_qws(distr);
  // Extra workspace for U
  Matrix<T, device> mat_uws(distr);

  // Auxialiary vectors used for the D&C algorithm
  LocalElementSize vec_size(distr.size().rows(), 1);
  TileElementSize vec_tile_size(distr.blockSize().rows(), 1);
  // Holds the diagonal elements of the tridiagonal matrix
  Matrix<T, device> d(vec_size, vec_tile_size);
  // Holds the values of the deflated diagonal sorted in ascending order
  Matrix<T, device> d_defl(vec_size, vec_tile_size);
  // Holds the values of Cuppen's rank-1 vector
  Matrix<T, device> z(vec_size, vec_tile_size);
  // Holds the values of the rank-1 update vector sorted corresponding to `d_defl`
  Matrix<T, device> z_defl(vec_size, vec_tile_size);
  // Holds indices/permutations of elements of the diagonal sorted in ascending order.
  Matrix<SizeType, Device::CPU> perm_d(vec_size, vec_tile_size);
  // Holds indices/permutations of the rows of U that bring it in Q-U matrix multiplication form
  Matrix<SizeType, Device::CPU> perm_u(vec_size, vec_tile_size);
  // Holds indices/permutations of the columns of Q that bring it in Q-U matrix multiplication form
  Matrix<SizeType, Device::CPU> perm_q(vec_size, vec_tile_size);
  // Assigns a type to each column of Q which is used to calculate the permutation indices for Q and U
  // that bring them in matrix multiplication form.
  Matrix<internal::ColType, Device::CPU> coltypes(vec_size, vec_tile_size);

  // Tile indices of the first and last diagonal tiles
  SizeType i_begin = 0;
  SizeType i_end = SizeType(distr.nrTiles().rows() - 1);

  internal::TridiagSolver<backend, device, T>::call(i_begin, i_end, coltypes, d, d_defl, z, z_defl,
                                                    perm_d, perm_q, perm_u, mat_qws, mat_uws, mat_trd,
                                                    mat_ev);
}

/// TODO: more info on the distributed version
/// Finds the eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// @param mat_a  [in/out] `n x 2` matrix with the diagonal and off-diagonal of the symmetric tridiagonal
/// matrix in the first column and second columns respectively. The last entry of the second column is
/// not used. On exit the eigenvalues are saved in the first column.
/// @param mat_ev [out]    `n x n` matrix holding the eigenvectors of the the symmetric tridiagonal
/// matrix on exit.
///
/// @pre mat_a and mat_ev are local matrices
/// @pre mat_a has 2 columns
/// @pre mat_ev is a square matrix
/// @pre mat_ev has a square block size
template <Backend backend, Device device, class T>
void tridiagSolver(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a, Matrix<T, device>& mat_ev) {
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(mat_a.distribution().size().cols() == 2, mat_a);

  DLAF_ASSERT(matrix::local_matrix(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_size(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_blocksize(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_ev, grid), mat_ev, grid);

  internal::TridiagSolver<backend, device, T>::call(std::move(grid), mat_a, mat_ev);
}

}
}
