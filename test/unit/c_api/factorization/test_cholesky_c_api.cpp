//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <functional>
#include <tuple>

#include <pika/init.hpp>
#include <pika/runtime.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/factorization/cholesky.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf_c/grid.h>
#include <dlaf_c/init.h>

#include "test_cholesky_c_api_wrapper.h"

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_generic_lapack.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct CholeskyTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(CholeskyTestMC, RealMatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
struct CholeskyTestGPU : public TestWithCommGrids {};

TYPED_TEST_SUITE(CholeskyTestGPU, MatrixElementTypes);
#endif

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

const std::vector<std::tuple<SizeType, SizeType>> sizes = {
    // {0, 2},                              // m = 0
    // {5, 8}, {34, 34},                    // m <= mb
    // {4, 3},
    // {16, 10},
    // {34, 13},
    {4, 1},  // m > mb
             //    {32, 5}  // m > mb
};

template <class T, Backend B, Device D>
void testCholesky(comm::CommunicatorGrid grid, const blas::Uplo uplo, const SizeType m,
                  const SizeType mb) {
  const char* argv[] = {"test_c_api_", nullptr};
  dlaf_initialize(1, argv);

  // In normal use the runtime is resumed by the C API call
  // The pika runtime is suspended by dlaf_initialize
  // Here we need to resume it manually to build the matrices with DLA-Future
  pika::resume();

  int dlaf_context =
      dlaf_create_grid(grid.fullCommunicator(), grid.size().rows(), grid.size().cols(), 'R');

  const GlobalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);
  //Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));
  Index2D src_rank_index(0, 0); // TODO: Relax this assumption?

  Distribution distribution(size, block_size, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_h(std::move(distribution));

  auto [el, res] = getCholeskySetters<GlobalElementIndex, T>(uplo);
  set(mat_h, el);
  mat_h.waitLocalTiles();

  char dlaf_uplo = uplo == blas::Uplo::Upper ? 'U' : 'L';

  // Get top left local tiles
  auto toplefttile_a = pika::this_thread::experimental::sync_wait(mat_h.readwrite(LocalTileIndex(0, 0)));

  auto rank = grid.fullCommunicator().rank();
  std::cout << rank << " Local element (0, 0): " << *(toplefttile_a.ptr()) << "\n";

  int lld = mat_h.distribution().localSize().rows();
  DLAF_descriptor dlaf_desc = {(int) m, (int) m, (int) mb, (int) mb, 0, 0, 1, 1, lld};

  // Suspend pika to ensure it is resumed by the C API
  pika::suspend();

  if constexpr (std::is_same_v<T, double>) {
    C_dlaf_cholesky_d(dlaf_context, dlaf_uplo, toplefttile_a.ptr(), dlaf_desc);
  }
  else {
    C_dlaf_cholesky_s(dlaf_context, dlaf_uplo, toplefttile_a.ptr(), dlaf_desc);
  }

  CHECK_MATRIX_NEAR(res, mat_h, 4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error,
                    4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(CholeskyTestMC, CorrectnessDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (const auto& [m, mb] : sizes) {
        testCholesky<TypeParam, Backend::MC, Device::CPU>(comm_grid, uplo, m, mb);
        pika::threads::get_thread_manager().wait();
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(CholeskyTestGPU, CorrectnessDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (const auto& [m, mb] : sizes) {
        testCholesky<TypeParam, Backend::GPU, Device::GPU>(comm_grid, uplo, m, mb);
        pika::threads::get_thread_manager().wait();
      }
    }
  }
}
#endif

//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
//
// #include "test_cholesky_c_api_wrapper.h"
//
// #include "dlaf_c/grid.h"
// #include "dlaf_c/init.h"
// #include "dlaf_c/utils.h"
//
// #include <gtest/gtest.h>
// #include <pika/runtime.hpp>
//
// #include <mpi.h>
//
// #include <iostream>
//
// // BLACS
// DLAF_EXTERN_C void Cblacs_gridinit(int* ictxt, char* layout, int nprow, int npcol);
// DLAF_EXTERN_C void Cblacs_gridexit(int ictxt);
//
// // ScaLAPACK
// DLAF_EXTERN_C int numroc(const int* n, const int* nb, const int* iproc, const int* isrcproc,
//                          const int* nprocs);
// DLAF_EXTERN_C void descinit(int* desc, const int* m, const int* n, const int* mb, const int* nb,
//                             const int* irsrc, const int* icsrc, const int* ictxt, const int* lld,
//                             int* info);
// DLAF_EXTERN_C void pdgemr2d(int* m, int* n, double* A, int* ia, int* ja, int* desca, double* B, int* ib,
//                             int* jb, int* descb, int* ictxt);
//
// int izero = 0;
// int ione = 1;
//
// // TODO: Check double and float
// TEST(CholeskyCAPIScaLAPACKTest, CorrectnessDistributed) {
//   int rank;
//   int num_ranks;
//
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
//
//   EXPECT_EQ(num_ranks, 6);
//
//   int n = 3;
//   int m = 3;
//
//   int nprow = 2;  // Rows of process grid
//   int npcol = 3;  // Cols of process grid
//
//   EXPECT_EQ(nprow * npcol, num_ranks);
//
//   int nb = 1;
//   int mb = 1;
//
//   char order = 'C';
//   char uplo = 'L';
//
//   int contxt = 0;
//   int contxt_global = 0;
//
//   // Global matrix
//   Cblacs_get(0, 0, &contxt);
//   contxt_global = contxt;
//
//   Cblacs_gridinit(&contxt_global, &order, 1, 1);  // Global matrix: only on rank 0
//   Cblacs_gridinit(&contxt, &order, nprow, npcol);
//
//   int myprow, mypcol;
//   Cblacs_gridinfo(contxt, &nprow, &npcol, &myprow, &mypcol);
//
//   int izero = 0;
//   int m_local = numroc(&m, &mb, &myprow, &izero, &nprow);
//   int n_local = numroc(&n, &nb, &mypcol, &izero, &npcol);
//
//   // Global matrix (one copy on each rank)
//   double* A;
//   int descA[9] = {0, -1, 0, 0, 0, 0, 0, 0, 0};
//   if (rank == 0) {
//     A = new double[n * m];
//     A[0] = 4.0;
//     A[1] = 12.0;
//     A[2] = -16.0;
//     A[3] = 12.0;
//     A[4] = 37.0;
//     A[5] = -43.0;
//     A[6] = -16.0;
//     A[7] = -43.0;
//     A[8] = 98.0;
//
//     int info = -1;
//     int lldA = m;
//     descinit(descA, &m, &n, &m, &n, &izero, &izero, &contxt_global, &lldA, &info);
//     ASSERT_EQ(info, 0);
//     ASSERT_EQ(descA[0], 1);
//   }
//
//   auto a = new double[m_local * n_local];
//
//   int desca[9];
//   int info = -1;
//   int llda = m_local;
//   descinit(desca, &m, &n, &mb, &nb, &izero, &izero, &contxt, &llda, &info);
//   ASSERT_EQ(info, 0);
//   ASSERT_EQ(desca[0], 1);
//
//   // Distribute global matrix to local matrices
//   int ione = 1;
//   pdgemr2d(&m, &n, A, &ione, &ione, descA, a, &ione, &ione, desca, &contxt);
//
//   // Use EXPECT_EQ to avoid potential deadlocks!
//   if (myprow == 0 && mypcol == 0) {
//     EXPECT_EQ(m_local * n_local, 2);
//     EXPECT_DOUBLE_EQ(a[0], 4.0);
//     EXPECT_DOUBLE_EQ(a[1], -16.0);
//   }
//   if (myprow == 0 && mypcol == 1) {
//     EXPECT_EQ(m_local * n_local, 2);
//     EXPECT_DOUBLE_EQ(a[0], 12.0);
//     EXPECT_DOUBLE_EQ(a[1], -43.0);
//   }
//   if (myprow == 0 && mypcol == 2) {
//     EXPECT_EQ(m_local * n_local, 2);
//     EXPECT_DOUBLE_EQ(a[0], -16.0);
//     EXPECT_DOUBLE_EQ(a[1], 98.0);
//   }
//   if (myprow == 1 && mypcol == 0) {
//     EXPECT_EQ(m_local * n_local, 1);
//     EXPECT_DOUBLE_EQ(a[0], 12.0);
//   }
//   if (myprow == 1 && mypcol == 1) {
//     EXPECT_EQ(m_local * n_local, 1);
//     EXPECT_DOUBLE_EQ(a[0], 37.0);
//   }
//   if (myprow == 1 && mypcol == 2) {
//     EXPECT_EQ(m_local * n_local, 1);
//     EXPECT_DOUBLE_EQ(a[0], -43.0);
//   }
//
//   const char* argv[] = {"test_interface_", nullptr};
//   dlaf_initialize(1, argv);
//   dlaf_create_grid_from_blacs(contxt);
//
//   info = -1;
//   C_dlaf_pdpotrf(uplo, n, a, 1, 1, desca, &info);
//   ASSERT_EQ(info, 0);
//
//   // Gather local matrices into global one
//   pdgemr2d(&m, &n, a, &ione, &ione, desca, A, &ione, &ione, descA, &contxt);
//
//   if (rank == 0) {
//     EXPECT_DOUBLE_EQ(A[0], 2.0);
//     EXPECT_DOUBLE_EQ(A[1], 6.0);
//     EXPECT_DOUBLE_EQ(A[2], -8.0);
//     EXPECT_DOUBLE_EQ(A[3], 12.0);  // Upper: original value
//     EXPECT_DOUBLE_EQ(A[4], 1.0);
//     EXPECT_DOUBLE_EQ(A[5], 5.0);
//     EXPECT_DOUBLE_EQ(A[6], -16.0);  // Upper: original value
//     EXPECT_DOUBLE_EQ(A[7], -43.0);  // Upper: original value
//     EXPECT_DOUBLE_EQ(A[8], 3.0);
//
//     delete[] A;
//   }
//
//   dlaf_free_grid(contxt);
//   dlaf_finalize();
//
//   delete[] a;
//   Cblacs_gridexit(contxt);
// }
//
// TEST(CholeskyCAPITest, CorrectnessDistributed) {
//   int rank;
//   int num_ranks;
//
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
//
//   EXPECT_EQ(num_ranks, 6);
//
//   int n = 3;
//   int m = 3;
//
//   int nprow = 2;  // Rows of process grid
//   int npcol = 3;  // Cols of process grid
//
//   EXPECT_EQ(nprow * npcol, num_ranks);
//
//   int nb = 1;
//   int mb = 1;
//
//   char order = 'C';
//   char uplo = 'L';
//
//   int contxt = 0;
//   int contxt_global = 0;
//
//   // Global matrix
//   Cblacs_get(0, 0, &contxt);
//   contxt_global = contxt;
//
//   Cblacs_gridinit(&contxt_global, &order, 1, 1);  // Global matrix: only on rank 0
//   Cblacs_gridinit(&contxt, &order, nprow, npcol);
//
//   // Get MPI_Comm
//   // TODO: Re-use code from dlaf_create_grid_from_blacs?
//   int system_context;
//   int get_blacs_contxt = 10;
//   Cblacs_get(contxt, get_blacs_contxt, &system_context);
//   MPI_Comm comm = Cblacs2sys_handle(system_context);
//
//   int myprow, mypcol;
//   Cblacs_gridinfo(contxt, &nprow, &npcol, &myprow, &mypcol);
//
//   int izero = 0;
//   int m_local = numroc(&m, &mb, &myprow, &izero, &nprow);
//   int n_local = numroc(&n, &nb, &mypcol, &izero, &npcol);
//
//   // Global matrix (one copy on each rank)
//   double* A;
//   int descA[9] = {0, -1, 0, 0, 0, 0, 0, 0, 0};
//   if (rank == 0) {
//     A = new double[n * m];
//     A[0] = 4.0;
//     A[1] = 12.0;
//     A[2] = -16.0;
//     A[3] = 12.0;
//     A[4] = 37.0;
//     A[5] = -43.0;
//     A[6] = -16.0;
//     A[7] = -43.0;
//     A[8] = 98.0;
//
//     int info = -1;
//     int lldA = m;
//     descinit(descA, &m, &n, &m, &n, &izero, &izero, &contxt_global, &lldA, &info);
//     ASSERT_EQ(info, 0);
//     ASSERT_EQ(descA[0], 1);
//   }
//
//   auto a = new double[m_local * n_local];
//
//   int desca[9];
//   int info = -1;
//   int llda = m_local;
//   descinit(desca, &m, &n, &mb, &nb, &izero, &izero, &contxt, &llda, &info);
//   ASSERT_EQ(info, 0);
//   ASSERT_EQ(desca[0], 1);
//
//   // Distribute global matrix to local matrices
//   int ione = 1;
//   pdgemr2d(&m, &n, A, &ione, &ione, descA, a, &ione, &ione, desca, &contxt);
//
//   // Use EXPECT_EQ to avoid potential deadlocks!
//   if (myprow == 0 && mypcol == 0) {
//     EXPECT_EQ(m_local * n_local, 2);
//     EXPECT_DOUBLE_EQ(a[0], 4.0);
//     EXPECT_DOUBLE_EQ(a[1], -16.0);
//   }
//   if (myprow == 0 && mypcol == 1) {
//     EXPECT_EQ(m_local * n_local, 2);
//     EXPECT_DOUBLE_EQ(a[0], 12.0);
//     EXPECT_DOUBLE_EQ(a[1], -43.0);
//   }
//   if (myprow == 0 && mypcol == 2) {
//     EXPECT_EQ(m_local * n_local, 2);
//     EXPECT_DOUBLE_EQ(a[0], -16.0);
//     EXPECT_DOUBLE_EQ(a[1], 98.0);
//   }
//   if (myprow == 1 && mypcol == 0) {
//     EXPECT_EQ(m_local * n_local, 1);
//     EXPECT_DOUBLE_EQ(a[0], 12.0);
//   }
//   if (myprow == 1 && mypcol == 1) {
//     EXPECT_EQ(m_local * n_local, 1);
//     EXPECT_DOUBLE_EQ(a[0], 37.0);
//   }
//   if (myprow == 1 && mypcol == 2) {
//     EXPECT_EQ(m_local * n_local, 1);
//     EXPECT_DOUBLE_EQ(a[0], -43.0);
//   }
//
//   const char* argv[] = {"test_interface_", nullptr};
//   dlaf_initialize(1, argv);
//   // FIXME: Order 'C' insteaf of 'R'?
//   int dlaf_context = dlaf_create_grid(comm, nprow, npcol, 'R');
//
//   C_dlaf_cholesky_d(dlaf_context, uplo, a, {m, n, mb, nb, 0, 0, 1, 1, m_local});
//
//   // Gather local matrices into global one
//   pdgemr2d(&m, &n, a, &ione, &ione, desca, A, &ione, &ione, descA, &contxt);
//
//   if (rank == 0) {
//     EXPECT_DOUBLE_EQ(A[0], 2.0);
//     EXPECT_DOUBLE_EQ(A[1], 6.0);
//     EXPECT_DOUBLE_EQ(A[2], -8.0);
//     EXPECT_DOUBLE_EQ(A[3], 12.0);  // Upper: original value
//     EXPECT_DOUBLE_EQ(A[4], 1.0);
//     EXPECT_DOUBLE_EQ(A[5], 5.0);
//     EXPECT_DOUBLE_EQ(A[6], -16.0);  // Upper: original value
//     EXPECT_DOUBLE_EQ(A[7], -43.0);  // Upper: original value
//     EXPECT_DOUBLE_EQ(A[8], 3.0);
//
//     delete[] A;
//   }
//
//   dlaf_free_grid(dlaf_context);
//   dlaf_finalize();
//
//   delete[] a;
//   Cblacs_gridexit(contxt);
// }
