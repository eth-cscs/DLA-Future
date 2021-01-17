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

     Matrix<T, Device::CPU> mat_w({mat_c.size().rows(), nb}, mat_v.blockSize());
     Matrix<T, Device::CPU> mat_w2({mb, mat_c.size().cols()}, mat_c.blockSize());
     
     for (SizeType k = 0; k < n; ++k) {
       
       void (&cpy)(const matrix::Tile<const T, Device::CPU>&, const matrix::Tile<T, Device::CPU>&) = copy<T>;
       // Copy V panel into WH
       for (SizeType i = 0; i < m; ++i) {
	 hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_v.read(LocalTileIndex(i, k)), mat_w(LocalTileIndex(i, 0)));
       }

       // Reset W2 to zero
       matrix::util::set(mat_w2, [](auto&&){return 0;});
       
       for (SizeType i = k; i < n; ++i) {
	 auto ik = LocalTileIndex{i, 0};
	 auto kk = LocalTileIndex{k, k};
	 // WH = V T
	 hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, NoTrans,
		       NonUnit, 1.0, mat_t.read(kk), std::move(mat_w(ik)));
       }

       for (SizeType j = k; j < n; ++j) {
	 auto kj = LocalTileIndex{0, j};
	 for (SizeType i = k; i < n; ++i) {
	   auto ik = LocalTileIndex{i, 0};
	   auto ij = LocalTileIndex{i, j};
	   // W2 = W C
	   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
			 NoTrans, 1.0, std::move(mat_w(ik)), mat_c.read(ij), 1.0, std::move(mat_w2(kj)));
	 }
       }

       for (SizeType i = k; i < n; ++i) {
	 auto ik = LocalTileIndex{i, k};
	 for (SizeType j = k; j < n; ++j) {
	   auto kj = LocalTileIndex{0, j};
	   auto ij = LocalTileIndex{i, j};
	   // C = C - V W2
	   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
			 NoTrans, -1.0, mat_v.read(ik), mat_w2.read(kj), 1.0, std::move(mat_c(ij)));
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

//     // CHECK!!
//     // Distribution distributionC(szC, blockSizeC, comm_grid.size(), comm_grid.rank(), src_rank_index);
//     auto dist_w = mat_v.distribution();
//     Matrix<T, Device::CPU> mat_w(std::move(dist_w));
//     copy(mat_v, mat_w);
//
//     auto dist_w2 = mat_c.distribution();
//     Matrix<T, Device::CPU> mat_w2(std::move(dist_w2));
//     matrix::util::set(mat_w2, [](auto&&){return 0;});

     Matrix<T, Device::CPU> mat_w({mat_c.size().rows(), nb}, mat_v.blockSize());
     Matrix<T, Device::CPU> mat_w2({mb, mat_c.size().cols()}, mat_c.blockSize());
//     GlobalElementSize size_w = {mat_v.size().rows(), mat_v.blockSize().cols()};
//     Distribution dist_w(size_w, mat_v.blockSize(), grid.size(), grid.rank(), mat_v.distribution().sourceRankIndex());
//     Matrix<T, Device::CPU> mat_w(std::move(dist_w));
//     //     std::cout << "mat w " << mat_w << " src rank index " << mat_v.distribution().sourceRankIndex() << std::endl;
     
     for (SizeType k = 0; k < 1; ++k) {
     //     for (SizeType k = 0; k < n; ++k) {

       const IndexT_MPI rank_k_col = distrib.template rankGlobalTile<Coord::Col>(k); 
       const IndexT_MPI rank_k_row = distrib.template rankGlobalTile<Coord::Row>(k); 
	 
       const SizeType local_k_row = distrib.template localTileFromGlobalTile<Coord::Row>(k);
       const SizeType local_k_col = distrib.template localTileFromGlobalTile<Coord::Col>(k);

       // Copy V panel into WH
       void (&cpy)(const matrix::Tile<const T, Device::CPU>&, const matrix::Tile<T, Device::CPU>&) = copy<T>;
       for (SizeType i_local = distrib.template nextLocalTileFromGlobalTile<Coord::Row>(k); i_local < local_rows; ++i_local) {
	   auto i = distrib.template globalTileFromLocalTile<Coord::Row>(i_local);
	   const IndexT_MPI rank_i_row = distrib.template rankGlobalTile<Coord::Row>(i);
	   auto ik = LocalTileIndex{i_local, local_k_col};
	   hpx::shared_future<ConstTileType> tmp_tile;
	   
	   if (mat_v.rankIndex().col() == rank_k_col) {
	     if (mat_v.rankIndex().row() == rank_i_row) {
	       //std::cout << " k " << k << " i_local " << i_local << " i " << ik << ", rank_i_row " << rank_i_row << " rank_k_col " << rank_k_col << " ";
	       hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_v.read(ik), mat_w(LocalTileIndex(i_local, 0)));
	       //std::cout << "mat_w " << mat_w(LocalTileIndex(i_local, 0)).get()({0,0}) << std::endl;
	     }
	   }

       }
       
       // Reset W2 to zero
       matrix::util::set(mat_w2, [](auto&&){return 0;});
       
       hpx::shared_future<ConstTileType> matt_kk_tile; 
       auto kk = LocalTileIndex{local_k_row, local_k_col};
       
       // Broadcast Tkk column-wise
       if (mat_t.rankIndex().col() == rank_k_col) {
	 if (mat_t.rankIndex().row() == rank_k_row) {
	   matt_kk_tile = mat_t.read(kk);
	   comm::send_tile(executor_mpi, serial_comm, Coord::Col, mat_t.read(kk));
	   //std::cout << "send T" << kk << " (" << k << ") [" << mat_t.rankIndex().row() << ", " << mat_t.rankIndex().col() << "] = " << matt_kk_tile.get()({0,0}) << std::endl;
	 }
	 else {
	   matt_kk_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Col, mat_t.tileSize(GlobalTileIndex(k, k)), rank_k_row);
	   //std::cout << "recv T" << kk << " (" << k << ") [" << mat_t.rankIndex().row() << ", " << mat_t.rankIndex().col() << "] = " << matt_kk_tile.get()({0,0}) << std::endl;
	 }
       }

//	 for (SizeType k_local = distrib.template nextLocalTileFromGlobalTile<Coord::Row>(i); k_local < local_rows; ++k_local) {
//	   auto k = distrib.template globalTileFromLocalTile<Coord::Row>(k_local);
//	   const IndexT_MPI rank_k_row = distrib.template rankGlobalTile<Coord::Row>(k); 	 
//	   auto ki = LocalTileIndex{k_local, local_i_col};
//
//	   if (mat_w.rankIndex().col() == rank_i_col) {
//	     if (mat_w.rankIndex().row() == rank_k_row) {
//	       // Compute W = V T
//	       hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, ConjTrans,
//			     NonUnit, 1.0, matt_ii_tile, std::move(mat_w(ki)));
//	     }
//	   }
//
//	   // Broadcast Wki row-wise
//	   hpx::shared_future<ConstTileType> wki_tile;
//	   if (mat_w.rankIndex().row() == rank_k_row) {
//	     if (mat_w.rankIndex().col() == rank_i_col) {
//	       comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_w.read(ki));
//	       wki_tile = mat_w.read(ki);
//	       //std::cout << "send " << k << " (" << i << ") Wki " << wki_tile.get()({0,0}) <<  std::endl;
//	     }
//	     else {
//	       wki_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row, mat_w.tileSize(GlobalTileIndex(k, i)), rank_i_col);
//		   
//	       //std::cout << "receive " << k  << " (" << i << ") Wki " << wki_tile.get()({0,0})  << std::endl;
//	     }
//	   }
//
//
//	   for (SizeType j_local = distrib.template nextLocalTileFromGlobalTile<Coord::Col>(i); j_local < local_cols; ++j_local) {
//	     auto j = distrib.template globalTileFromLocalTile<Coord::Col>(j_local);
//	     const IndexT_MPI rank_j_col = distrib.template rankGlobalTile<Coord::Col>(j);
//
//	     //	     std::cout <<  "Loop k " << k << " j " << j  << " vs i " << i << std::endl;
//	     auto kj = LocalTileIndex{k_local, j_local};
//
//	     if (mat_w.rankIndex().row() == rank_k_row) {
//	       if (mat_w.rankIndex().col() == rank_j_col) {
//
//		 hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});
//		 const T beta = (k == i) ? 0 : 1;
//		 
//		 //// Compute Wki Ckj = W2kj, local on W2ki
//		 hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, static_cast<T>(1), wki_tile, std::move(mat_c.read(kj)), beta, std::move(tile_w2));
//		 std::cout << "w2 local: " << k << " " << j << " (" << i << ") Wki " << wki_tile.get()({0,0}) << " Ckj " << mat_c.read(kj).get()({0,0}) << " W2 " << mat_w2_local(LocalTileIndex{0,0}).get()({0,0}) << " rank i col " << rank_i_col << " rank k row " << rank_k_row << " rank j col " << rank_j_col << " beta " << beta << std::endl;
//	       }
//	     }
//
//
//	   } // end loop on j_local (cols)
//	   
//	 } // end loop on k_local (rows)
//
//	 hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});
//	 auto all_reduce_w2_func = unwrapping([](auto&& tile_w2, auto&& comm_wrapper) {
//	     comm::sync::all_reduce(comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(tile_w2), make_data(tile_w2));
//	   });
//
//	 hpx::dataflow(std::move(all_reduce_w2_func), std::move(tile_w2), serial_comm());
//	 
//	 //for (SizeType k_local = distrib.template nextLocalTileFromGlobalTile<Coord::Row>(i); k_local < local_rows; ++k_local) {
//	 //  auto k = distrib.template globalTileFromLocalTile<Coord::Row>(k_local);
//	 //  const IndexT_MPI rank_k_row = distrib.template rankGlobalTile<Coord::Row>(k); 	
//	 //
//	 //  for (SizeType j_local = distrib.template nextLocalTileFromGlobalTile<Coord::Col>(i); j_local < local_cols; ++j_local) {
//	 //    auto j = distrib.template globalTileFromLocalTile<Coord::Col>(j_local);
//	 //    const IndexT_MPI rank_j_col = distrib.template rankGlobalTile<Coord::Col>(j);
//	 //	   
//	 //    auto kj = LocalTileIndex{k_local, j_local};
//	 //
//	 //    std::cout << " W2 (" <<  k << ", " << j << ") " << mat_w2_local(LocalTileIndex{0,0}).get()({0,0}) << " [i = " << i << "]" << std::endl;
//	 //  }
//	 //}
//
//	 for (SizeType k_local = distrib.template nextLocalTileFromGlobalTile<Coord::Row>(i); k_local < local_rows; ++k_local) { 
//	   auto k = distrib.template globalTileFromLocalTile<Coord::Row>(k_local);
//	   const IndexT_MPI rank_k_row = distrib.template rankGlobalTile<Coord::Row>(k);
//	 
//	   hpx::shared_future<ConstTileType> vki_tile;
//
//	   auto ki = LocalTileIndex{k_local, local_i_col};
//	   
//	   // Broadcast Vki row-wise
//	   if (mat_v.rankIndex().row() == rank_k_row) {
//	     if (mat_v.rankIndex().col() == rank_i_col) {
//	       comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_v.read(ki));
//	       vki_tile = mat_v.read(ki);
//	       //std::cout << "send " << k << " " << i << " Vki " << vki_tile.get()({0,0}) <<  std::endl;
//	     }
//	     else {
//	       vki_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row, mat_v.tileSize(GlobalTileIndex(k, i)), rank_i_col);
//	       //		   std::cout << "receive " << k << " " << i << " Vkj " << vki_tile.get()({0,0})  << std::endl;
//	     }
//	   }
//
//
//
//	   for (SizeType j_local = distrib.template nextLocalTileFromGlobalTile<Coord::Col>(i); j_local < local_cols; ++j_local) {
//	     auto j = distrib.template globalTileFromLocalTile<Coord::Col>(j_local);
//	     const IndexT_MPI rank_j_col = distrib.template rankGlobalTile<Coord::Col>(j);
//
//	     auto kj = LocalTileIndex{k_local, j_local};
//
//	     if (mat_w.rankIndex().row() == rank_k_row) {
//	       if (mat_w.rankIndex().col() == rank_j_col) {
//		 
//		 // Compute C = C - V W2
//		 hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans, NoTrans, -1.0, vki_tile, mat_w2_local(LocalTileIndex{0,0}), 1.0, std::move(mat_c(kj)));
//		 //		 std::cout << " kj " << k << " " << j << " Vki " << vki_tile.get()({0,0}) << " W2 " << mat_w2_local(LocalTileIndex{0,0}).get()({0,0}) <<  " C " << mat_c.read(kj).get()({0,0}) << std::endl;
//	       }
//	     }
//
//	   } // end loop on j_local (cols)
//	   
//	 } // end loop on k_local (rows)

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



//hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, 1.0, panel_matwki[j], std::move(mat_c.read(kj)), 0.0, std::move(mat_w2_local(kj)));

//		 auto tmpw = panel_matwki[j];
//		 auto tmpc = mat_c(kj); 
//		 auto tmpt = tile_w2.get();
//		 std::cout << "Mao: send kj " << k << " " << j <<  " panel W " << tmpw.get()({0,0}) << " - Ckj " << tmpc.get()({0,0}) << " - W2 " << tmpt({0,0}) << std::endl;
		 


 
//	 hpx::future<TileType> tile_w2 = mat_w2(ki);
//	 vector<hpx::future<ConstTileType>> panel_colj(m-i);

//	 auto dist_w2_local = mat_v.distribution();
//	 auto layout_w2_local = tileLayout(dist_w2.localSize(), size);
//	 Matrix<T, Device::CPU> mat_w2_local(std::move(dist_w2_local), layout_w2_local);
////	 //	 Matrix<T, Device::CPU> mat_w2_local(std::move(dist_w2_local));
////	 matrix::util::set(mat_w2_local, [](auto&&){return 0;});
////	 hpx::future<TileType> tile_w2 = mat_w2(LocalTileIndex{0,0});

//	 for (SizeType j = i; j < m; ++j) {

       //	 auto reduce_w_func = hpx::util::unwrapping([rank_i_row](auto&& tile_w2, auto&& comm_wrapper) {
//	     dlaf::comm::sync::reduce(rank_i_row, comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(tile_w2), make_data(tile_w2));
//	     
//	 });

//       hpx::future<TileType> tile_w2 = mat_w2(LocalTileIndex{0,0});
//       hpx::dataflow(all_reduce_w_func, tile_w2, serial_comm());
       
       //       auto reduce_w_func = hpx::util::unwrapping([rank_i_row](auto&& tile_w2, auto&& comm_wrapper) {
//	   dlaf::comm::sync::reduce(rank_i_row, comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(tile_w2), make_data(tile_w2));
//	     
//	 });
//	 
//	 for (SizeType j = i; j < m; ++j) {
//	   const IndexT_MPI rank_j_col = distrib.template rankGlobalTile<Coord::Col>(j);
//	   const SizeType local_j_col = distrib.template localTileFromGlobalTile<Coord::Col>(j);
//	   auto ki = LocalTileIndex{local_k_row, local_i_col};
//	   hpx::future<TileType> tile_w2 = mat_w2(ki);
//
//	   if (mat_w.rankIndex().row() == rank_k_row) {
//	     if (rank_j_col == rank_i_col) {
//	       using dlaf::common::make_data;
//	       
//	       hpx::dataflow(executor_normal, hpx::util::unwrapping(dlaf::comm::sync::reduce<T, Device::CPU>), rank_i_col, grid.rowCommunicator(), MPI_SUM, make_data(tile_w2), make_data(tile_w2));
//	     }
//	   }
//	 }
