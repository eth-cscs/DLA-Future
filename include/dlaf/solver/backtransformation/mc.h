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

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include <blas.hh>

#include "dlaf/solver/backtransformation/api.h"

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/init.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix_output.h"

namespace dlaf {
namespace solver {
namespace internal {

  using namespace dlaf::matrix;
  //  using namespace dlaf::tile;

  template <class T>
  void print_mat(dlaf::Matrix<T, dlaf::Device::CPU>& matrix) {
    using dlaf::common::iterate_range2d;

    const auto& distribution = matrix.distribution();

    //std::cout << matrix << std::endl;

    for (const auto& index : iterate_range2d(distribution.localNrTiles())) {
      const auto index_global = distribution.globalTileIndex(index);
      std::cout << index_global << '\t' << matrix.read(index).get()({0, 0})  <<  " rank " << matrix.rankIndex() << std::endl;
    }
    std::cout << "finished" << std::endl;
  }
  
// Implementation based on:
// 1. Part of algorithm 6 "LAPACK Algorithm for the eigenvector back-transformation", page 15, PhD thesis "GPU
// Accelerated Implementations of a Generalized Eigenvalue Solver for Hermitian Matrices with Systematic
// Energy and Time to Solution Analysis" presented by Raffaele Solcà (2016)
// 2. Report "Gep + back-transformation", Alberto Invernizzi (2020)
// 3. Report "Reduction to band + back-transformation", Raffaele Solcà (2020)

template <class T>
struct BackTransformation<Backend::MC, Device::CPU, T> {
  static void call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
		      Matrix<T, Device::CPU>& mat_t);
  static void call_FC(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v, Matrix<T, Device::CPU>& mat_t);
 };

 
 template <class T>
   void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v, Matrix<T, Device::CPU>& mat_t)
   {
     constexpr auto Left = blas::Side::Left;
     constexpr auto Right = blas::Side::Right;
     constexpr auto Upper = blas::Uplo::Upper;
     constexpr auto Lower = blas::Uplo::Lower;
     constexpr auto NoTrans = blas::Op::NoTrans;
     constexpr auto ConjTrans = blas::Op::ConjTrans;
     constexpr auto NonUnit = blas::Diag::NonUnit;

     using hpx::threads::executors::pool_executor;
     using hpx::threads::thread_priority_high;
     using hpx::threads::thread_priority_default;
     using hpx::util::unwrapping;
     
     // Set up executor on the default queue with high priority.
     pool_executor executor_hp("default", thread_priority_high);
     // Set up executor on the default queue with default priority.
     pool_executor executor_normal("default", thread_priority_default);

     const SizeType m = mat_c.nrTiles().rows();
     const SizeType n = mat_c.nrTiles().cols();
     const SizeType mb = mat_c.blockSize().rows();
     const SizeType nb = mat_c.blockSize().cols();

     Matrix<T, Device::CPU> mat_w({mat_v.size().rows(), nb}, mat_v.blockSize());
     Matrix<T, Device::CPU> mat_w2({mb, mat_v.size().cols()}, mat_v.blockSize());

     SizeType last_nb;
     if (mat_v.blockSize().cols() == 1) {
       last_nb = 1;
     }
     else {
       if (mat_v.size().cols()%mat_v.blockSize().cols() == 0) 
	 last_nb = mat_v.blockSize().cols();
       else 
	 last_nb =mat_v.size().cols()%mat_v.blockSize().cols();
     }
     Matrix<T, Device::CPU> mat_w_last({mat_v.size().rows(), last_nb}, {mat_v.blockSize().rows(), last_nb});
     Matrix<T, Device::CPU> mat_w2_last({last_nb, mat_v.size().cols()}, {last_nb, mat_v.blockSize().cols()});

     const SizeType reflectors = (last_nb == 1) ? n-1 : n;
     
     for (SizeType k = 0; k < reflectors; ++k) {
       bool is_last = (k == n-1) ? true : false;
       
       void (&cpy)(const matrix::Tile<const T, Device::CPU>&, const matrix::Tile<T, Device::CPU>&) = copy<T>;
       // Copy V panel into WH
       for (SizeType i = 0; i < m; ++i) {
	 if (is_last == true) {
	   hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_v.read(LocalTileIndex(i, k)), mat_w_last(LocalTileIndex(i, 0)));
	 }
	 else {
	   hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_v.read(LocalTileIndex(i, k)), mat_w(LocalTileIndex(i, 0)));
	 }
       }

       // Reset W2 to zero
       if (is_last == true) {
	 matrix::util::set(mat_w2_last, [](auto&&){return 0;});
       }
       else {
	 matrix::util::set(mat_w2, [](auto&&){return 0;});
       }
	 
       for (SizeType i = k; i < n; ++i) {
	 // WH = V T
	 auto ik = LocalTileIndex{i, 0};
	 auto kk = LocalTileIndex{k, k};
	 if (is_last == true) {
	   hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, NoTrans, NonUnit, 1.0, mat_t.read(kk), std::move(mat_w_last(ik)));
	 }
	 else {
	   hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, NoTrans, NonUnit, 1.0, mat_t.read(kk), std::move(mat_w(ik)));
	 }
       }

       for (SizeType j = k; j < n; ++j) {
	 auto kj = LocalTileIndex{0, j};
	 for (SizeType i = k; i < m; ++i) {
	   auto ik = LocalTileIndex{i, 0};
	   auto ij = LocalTileIndex{i, j};
	   // W2 = W C
	   if (is_last == true) {
	     hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, 1.0, std::move(mat_w_last(ik)), mat_c.read(ij), 1.0, std::move(mat_w2_last(kj)));	     
	   }
	   else {
	     hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, 1.0, std::move(mat_w(ik)), mat_c.read(ij), 1.0, std::move(mat_w2(kj)));
	   }
	 }	 
       }
       
       for (SizeType i = k; i < m; ++i) {
	 auto ik = LocalTileIndex{i, k};
	 for (SizeType j = k; j < n; ++j) {
	   auto kj = LocalTileIndex{0, j};
	   auto ij = LocalTileIndex{i, j};
	   // C = C - V W2
	   if (is_last == true) {
	     hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans, NoTrans, -1.0, mat_v.read(ik), mat_w2_last.read(kj), 1.0, std::move(mat_c(ij)));
	   }
	   else {
	     hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans, NoTrans, -1.0, mat_v.read(ik), mat_w2.read(kj), 1.0, std::move(mat_c(ij)));
	   }
	 }
       }
       
     }
   }

 
 template <class T>
   void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v, Matrix<T, Device::CPU>& mat_t)
   {
     constexpr auto Left = blas::Side::Left;
     constexpr auto Right = blas::Side::Right;
     constexpr auto Upper = blas::Uplo::Upper;
     constexpr auto Lower = blas::Uplo::Lower;
     constexpr auto NoTrans = blas::Op::NoTrans;
     constexpr auto ConjTrans = blas::Op::ConjTrans;
     constexpr auto NonUnit = blas::Diag::NonUnit;

     using comm::IndexT_MPI;
     using comm::internal::mpi_pool_exists;
     using common::internal::vector;
     using dlaf::comm::Communicator;
     using dlaf::comm::CommunicatorGrid;
     using dlaf::common::make_data;

     using TileType = typename Matrix<T, Device::CPU>::TileType;     
     using ConstTileType = typename Matrix<const T, Device::CPU>::ConstTileType;     

     using hpx::threads::executors::pool_executor;
     using hpx::threads::thread_priority_high;
     using hpx::threads::thread_priority_default;
     using hpx::util::unwrapping;

     pool_executor executor_hp("default", thread_priority_high);
     pool_executor executor_normal("default", thread_priority_default);
     // Set up MPI
     auto executor_mpi = (mpi_pool_exists()) ? pool_executor("mpi", thread_priority_high) : executor_hp;
     common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

     SizeType m = mat_c.nrTiles().rows();
     SizeType n = mat_c.nrTiles().cols();
     SizeType mb = mat_c.blockSize().rows();
     SizeType nb = mat_c.blockSize().cols();

     auto distrib = mat_c.distribution();
     auto local_rows = distrib.localNrTiles().rows();
     auto local_cols = distrib.localNrTiles().cols();     
     
     LocalElementSize size_mat_w(mat_v.size().rows(), mat_v.blockSize().cols());
     TileElementSize blockSize_mat_w(mat_v.blockSize().rows(), mat_v.blockSize().cols());
     GlobalElementSize sz_mat_w(size_mat_w.rows(), size_mat_w.cols());
     Distribution distrib_mat_w(sz_mat_w, blockSize_mat_w, grid.size(), grid.rank(), mat_v.distribution().sourceRankIndex());
     Matrix<T, Device::CPU> mat_w(std::move(distrib_mat_w));

//     if (mat_v.rankIndex().row() == 0 && mat_v.rankIndex().col() == 0)
//       std::cout << "Mat W (beginning) " << mat_w << std::endl;


     LocalElementSize size_mat_w2(mat_v.blockSize().rows(), mat_v.size().cols());
     TileElementSize blockSize_mat_w2(mat_v.blockSize().rows(), mat_v.blockSize().cols());
//     GlobalElementSize sz_mat_w2(size_mat_w2.rows(), size_mat_w2.cols());
//     Distribution distrib_mat_w2(sz_mat_w2, blockSize_mat_w2, grid.size(), grid.rank(), mat_v.distribution().sourceRankIndex());
//     //Distributed
//     Matrix<T, Device::CPU> mat_w2(std::move(distrib_mat_w2));
     //Local
     Matrix<T, Device::CPU> mat_w2(size_mat_w2, blockSize_mat_w2);
     
//     if (mat_v.rankIndex().row() == 0 && mat_v.rankIndex().col() == 0)
//       std::cout << "Mat W2 (beginning) " << mat_w2 << std::endl;
     
     SizeType last_nb;
     if (mat_v.blockSize().cols() == 1) {
       last_nb = 1;
     }
     else {
       if (mat_v.size().cols()%mat_v.blockSize().cols() == 0) 
	 last_nb = mat_v.blockSize().cols();
       else 
	 last_nb =mat_v.size().cols()%mat_v.blockSize().cols();
     }

     LocalElementSize size_mat_w_last(mat_v.size().rows(), last_nb);
     TileElementSize blockSize_mat_w_last(mat_v.blockSize().rows(), last_nb);
     GlobalElementSize sz_mat_w_last(size_mat_w_last.rows(), size_mat_w_last.cols());
     Distribution distrib_mat_w_last(sz_mat_w_last, blockSize_mat_w_last, grid.size(), grid.rank(), mat_v.distribution().sourceRankIndex());
     Matrix<T, Device::CPU> mat_w_last(std::move(distrib_mat_w_last));

//     if (mat_v.rankIndex().row() == 0 && mat_v.rankIndex().col() == 0)
//       std::cout << "Mat W last (beginning) " << mat_w_last << std::endl;

     LocalElementSize size_mat_w2_last(last_nb, mat_v.size().cols());
     TileElementSize blockSize_mat_w2_last(last_nb, mat_v.blockSize().cols());
     GlobalElementSize sz_mat_w2_last(size_mat_w2_last.rows(), size_mat_w2_last.cols());
     Distribution distrib_mat_w2_last(sz_mat_w2_last, blockSize_mat_w2_last, grid.size(), grid.rank(), mat_v.distribution().sourceRankIndex());
     Matrix<T, Device::CPU> mat_w2_last(std::move(distrib_mat_w2_last));

//     if (mat_v.rankIndex().row() == 0 && mat_v.rankIndex().col() == 0)
//       std::cout << "Mat W2 last (beginning)" << mat_w2_last << std::endl;

     
     const SizeType reflectors = (last_nb == 1) ? n-1 : n;
          
     //     for (SizeType k = 0; k < refletors; ++k) {
     for (SizeType k = 0; k < 1; ++k) {
       bool is_last = (k == n-1) ? true : false;
       
       const IndexT_MPI rank_k_col = distrib.template rankGlobalTile<Coord::Col>(k); 
       const IndexT_MPI rank_k_row = distrib.template rankGlobalTile<Coord::Row>(k); 

//       if (mat_v.rankIndex().row() == rank_k_row && mat_v.rankIndex().col() == rank_k_col) {
//	 std::cout << "mat_v " << mat_v << std::endl;
//       }
	 
       const SizeType local_k_row = distrib.template localTileFromGlobalTile<Coord::Row>(k);
       const SizeType local_k_col = distrib.template localTileFromGlobalTile<Coord::Col>(k);

       // Copy V panel into WH
       void (&cpy)(const matrix::Tile<const T, Device::CPU>&, const matrix::Tile<T, Device::CPU>&) = copy<T>;
       
       for (SizeType i_local = distrib.template nextLocalTileFromGlobalTile<Coord::Row>(k); i_local < local_rows; ++i_local) {
	 auto i = distrib.template globalTileFromLocalTile<Coord::Row>(i_local);
	 auto ik = LocalTileIndex{i_local, local_k_col};
	   
	 if (mat_v.rankIndex().col() == rank_k_col) {
	   if (is_last == 1) {
	     //std::cout << " k " << k << " i_local " << i_local << " i " << ik << " ";
	     hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_v.read(ik), mat_w_last(LocalTileIndex(i_local, 0)));
	     //std::cout << "mat_w_last " << mat_w_last(LocalTileIndex(i_local, 0)).get()({0,0}) << std::endl;
	   }
	   else {
	     //std::cout << "copy: k " << k << " i " << i << " i_local " << i_local << " local_rows " << local_rows << " i " << ik << " rank (" << rank_k_row << ", " << rank_k_col << ") ";
	     hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_v.read(ik), mat_w(LocalTileIndex(i_local, 0)));
	     //std::cout << "copy: mat_w " << mat_w.read(LocalTileIndex(i_local, 0)).get()({0,0}) << " " << mat_w.read(LocalTileIndex(i_local, 0)).get().size() <<  " ------> mat_w " << mat_w << std::endl;
	   }
	 }

       }

       // Reset W2 to zero
       if (is_last == true) {
	 matrix::util::set(mat_w2_last, [](auto&&){return 0;});
       }
       else {
	 matrix::util::set(mat_w2, [](auto&&){return 0;});
       }


       hpx::shared_future<ConstTileType> matt_kk_tile; 
       auto kk = LocalTileIndex{local_k_row, local_k_col};
       
       // Broadcast Tkk column-wise
       if (mat_t.rankIndex().col() == rank_k_col) {
	 if (mat_t.rankIndex().row() == rank_k_row) {
	   matt_kk_tile = mat_t.read(kk);
	   comm::send_tile(executor_mpi, serial_comm, Coord::Col, mat_t.read(kk));
	   //std::cout << "send T" << kk << " (" << k << ") [" << mat_t.rankIndex().row() << ", " << mat_t.rankIndex().col() << "] = " << matt_kk_tile.get()({0,0}) << " " << matt_kk_tile.get().size() << std::endl;
	 }
	 else {
	   matt_kk_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Col, mat_t.tileSize(GlobalTileIndex(k, k)), rank_k_row);
	   //std::cout << "recv T" << kk << " (" << k << ") [" << mat_t.rankIndex().row() << ", " << mat_t.rankIndex().col() << "] = " << matt_kk_tile.get()({0,0}) << " " << matt_kk_tile.get().size() << std::endl;
	 }
       }

       std::vector<hpx::shared_future<ConstTileType>> wik_tile(mat_v.distribution().localNrTiles().rows());

       for (SizeType i_local = distrib.template nextLocalTileFromGlobalTile<Coord::Row>(k); i_local < local_rows; ++i_local) {
	 auto i = distrib.template globalTileFromLocalTile<Coord::Row>(i_local);
	 const IndexT_MPI rank_i_row = distrib.template rankGlobalTile<Coord::Row>(i); 	 
	 auto ik = LocalTileIndex{i_local, 0};

	 
	 if (mat_v.rankIndex().col() == rank_k_col) {
	   //std::cout << "i local " << i_local << " local_rows " << local_rows << " i " << i << " rank_i_row " << rank_i_row << " rank_k_col " << rank_k_col << " is_last " << is_last << std::endl;
	   
	   // Compute W = V T
	   if (is_last == 1) {
	     hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, NoTrans, NonUnit, 1.0, matt_kk_tile, std::move(mat_w_last(ik)));
	   // std::cout << " W (" << ik << ") [" << rank_i_row << ", "  << rank_k_col << "] " << " i,k " << i << " " << k << " " << mat_w_last(ik).get()({0,0}) << " Tkk " << matt_kk_tile.get()({0,0}) << std::endl;
	   }
	   else {
	     //std::cout << "TRMM: W (" << ik << ") [" << rank_i_row << ", "  << rank_k_col << "] " << " i,k " << i << ", " << k  << " START: Wik " << mat_w.read(ik).get()({0,0}) << " " << mat_w.read(ik).get().size() << ", Tkk " << matt_kk_tile.get()({0,0}) << " " << matt_kk_tile.get().size() << " --> END: ";
	     hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, NoTrans, NonUnit, 1.0, matt_kk_tile, std::move(mat_w(ik)));
	     //std::cout << mat_w.read(ik).get()({0,0}) << " " << mat_w.read(ik).get().size() << ", Tkk " << matt_kk_tile.get()({0,0}) << " " << matt_kk_tile.get().size() << std::endl;
	   }
	 }

	 // Broadcast Wik row-wise
	 if (mat_v.rankIndex().col() == rank_k_col) {
	   if (is_last == 1) {
	     comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_w_last.read(ik));
	     wik_tile[i_local] = mat_w_last.read(ik);
	     //std::cout << "Wik send (last) " << i << " (" << k << ") rank " << mat_v.rankIndex().row() << ", " << mat_v.rankIndex().col() << " Wik " << wik_tile[i_local].get()({0,0}) <<  std::endl;
	   }
	   else {
	     comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_w.read(ik));
	     wik_tile[i_local] = mat_w.read(ik);
	     //std::cout << "Wik send " << i << " (" << k << ") rank " << mat_v.rankIndex().row() << ", " << mat_v.rankIndex().col() << " Wik " << wik_tile[i_local].get()({0,0}) <<  std::endl;
	   }
	 }
	 else {
	   if (is_last == 1) {
	   wik_tile[i_local] = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row, mat_w_last.tileSize(GlobalTileIndex(i, 0)), rank_k_col);		   
	   //std::cout << "Wik recv (last) " << i  << " (" << k << ") rank " << mat_v.rankIndex().row() << ", " << mat_v.rankIndex().col() << " Wik " << wik_tile[i_local].get()({0,0})  << std::endl;
	   }
	   else {
	   wik_tile[i_local] = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row, mat_w.tileSize(GlobalTileIndex(i, 0)), rank_k_col);		   
	   //std::cout << "Wik recv " << i  << " (" << k << ") rank " << mat_v.rankIndex().row() << ", " << mat_v.rankIndex().col() << " Wik " << wik_tile[i_local].get()({0,0})  << std::endl;
	   }
	 }

       }
       
       std::vector<hpx::shared_future<ConstTileType>> w2kj_tile(mat_w2.distribution().localNrTiles().cols());

       for (SizeType j_local = distrib.template nextLocalTileFromGlobalTile<Coord::Col>(k); j_local < local_cols; ++j_local) {
	 auto j = distrib.template globalTileFromLocalTile<Coord::Col>(j_local);
	 const IndexT_MPI rank_j_col = distrib.template rankGlobalTile<Coord::Col>(j); 	 
	 auto kj = LocalTileIndex{0, j_local};
	 
	 for (SizeType i_local = distrib.template nextLocalTileFromGlobalTile<Coord::Row>(k); i_local < local_rows; ++i_local) {
	   auto i = distrib.template globalTileFromLocalTile<Coord::Row>(i_local);
	   const IndexT_MPI rank_i_row = distrib.template rankGlobalTile<Coord::Row>(i); 	 
	   auto ij = LocalTileIndex{i_local, j_local};
	   auto ik = LocalTileIndex{i_local, 0};
	   //auto kj = LocalTileIndex{0, j_local};
	   auto kj = LocalTileIndex{0, j};
	     
	   const T beta = (k == i) ? static_cast<T>(0) : static_cast<T>(1);
	   
	   //	   std::cout << "GEMM W2 rank " << rank_i_row << ", " << rank_j_col << " k " << k << ", i " << i << ", j " << j  << " i_local " << i_local << " j_local " << j_local << " W2  element " << mat_w2.read(kj).get()({0,0}) << " " << mat_w2.read(kj).get().size() << "   <--->  Wik " << wik_tile[i_local].get()({0,0}) << " " << wik_tile[i_local].get().size()  << "   <---> C " << mat_c.read(ij).get()({0,0}) << " " << mat_c.read(ij).get().size() << std::endl;
		 
	   // Compute Wki Ckj = W2kj, local on W2ki
	   //	   hpx::dataflow(executor_mpi, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, static_cast<T>(1), wik_tile[i_local], mat_c.read(ij), beta, std::move(w2_tile));
	   hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, static_cast<T>(1), wik_tile[i_local], mat_c.read(ij), beta, std::move(mat_w2(kj)));

	   //	   std::cout << "GEMM W2 (after) element " << mat_w2.read(kj).get()({0,0}) << " " << mat_w2.read(kj).get().size() << " rank " << rank_i_row << ", " << rank_j_col << " k " << k << ", i " << i << ", j " << j << "   <--->  Wik " << wik_tile[i_local].get()({0,0}) << " " << wik_tile[i_local].get().size()  << "   <---> C " << mat_c.read(ij).get()({0,0}) << " " << mat_c.read(ij).get().size() << " i_local " << i_local << " j_local " << j_local << std::endl;

	  
	 } // end loop on i_local (rows)
	 
       } // end loop on j_local (cols)
	 
       for (SizeType j_local = distrib.template nextLocalTileFromGlobalTile<Coord::Col>(k); j_local < local_cols; ++j_local) {
	 auto j = distrib.template globalTileFromLocalTile<Coord::Col>(j_local);
	 const IndexT_MPI rank_j_col = distrib.template rankGlobalTile<Coord::Col>(j); 	 

	 // auto kj = LocalTileIndex{0, j};
	   auto kj = LocalTileIndex{0, j_local};
	 //  auto kj = LocalTileIndex{0, 0};
       
       std::cout << "all reducing: rank " << rank_k_row << " " << rank_j_col << " j " << j  << " j_local " << j_local << " local_cols " << local_cols << " mat_w2 " << mat_w2.read(kj).get()({0,0}) << " " << mat_w2.read(kj).get().size() << std::endl;
	 
	   hpx::future<TileType> tile_w2 = mat_w2(kj);
	   auto all_reduce_w2_func = unwrapping([](auto&& tile_w2, auto&& comm_wrapper) {
	       comm::sync::all_reduce(comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(tile_w2), make_data(tile_w2));
	     });
	   
	   hpx::dataflow(executor_mpi, std::move(all_reduce_w2_func), std::move(tile_w2), serial_comm());

	   std::cout << "all reducing: after -- rank " << rank_k_row << " " << rank_j_col << " j " << j  << " j_local " << j_local << " mat_w2 " << mat_w2.read(kj).get()({0,0}) << " " << mat_w2.read(kj).get().size() << std::endl;
	   
       }

//       for (SizeType i_local = distrib.template nextLocalTileFromGlobalTile<Coord::Row>(k); i_local < local_rows; ++i_local) { 
//	 auto i = distrib.template globalTileFromLocalTile<Coord::Row>(i_local);
//	 const IndexT_MPI rank_i_row = distrib.template rankGlobalTile<Coord::Row>(i);
//	 
//	 hpx::shared_future<ConstTileType> vik_tile;
//
//	 auto ik = LocalTileIndex{i_local, local_k_col};
//	   
//	 // Broadcast Vki row-wise
//	 if (mat_v.rankIndex().col() == rank_k_col) {
//	   comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_v.read(ik));
//	   vik_tile = mat_v.read(ik);
//	   // std::cout << "send " << i << " " << k<< " rank " << mat_v.rankIndex().row() << " " << mat_v.rankIndex().col() << " Vik " << vik_tile.get()({0,0}) <<  std::endl;
//	 }
//	 else {
//	   vik_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row, mat_v.tileSize(GlobalTileIndex(i, k)), rank_k_col);
//	   //std::cout << "recv " << i << " " << k << " rank " << mat_v.rankIndex().row() << " " << mat_v.rankIndex().col() << " Vik " << vik_tile.get()({0,0})  << std::endl;
//	 }
//
//
//	 for (SizeType j_local = distrib.template nextLocalTileFromGlobalTile<Coord::Col>(k); j_local < local_cols; ++j_local) {
//
//	   auto ij = LocalTileIndex{i_local, j_local};
//
//	   // Compute C = C - V W2
//	   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans, NoTrans, -1.0, vik_tile, mat_w2(LocalTileIndex{0,j_local}), 1.0, std::move(mat_c(ij)));
//	   //	       std::cout << " ij " << i << " " << j << " Vik " << vik_tile.get()({0,0}) << " W2 " << mat_w2(LocalTileIndex{0,j_local}).get()({0,0}) <<  " C " << mat_c.read(ij).get()({0,0}) << std::endl;
//
//	 } // end loop on j_local (cols)
//	   
//       } // end loop on i_local (rows)

     } // loop on k

   }

 
 /// ---- ETI
#define DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(KWORD, DATATYPE)		\
 KWORD template struct BackTransformation<Backend::MC, Device::CPU, DATATYPE>;

 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, float)
 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, double)
 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<float>)
 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<double>)

}
}
}
