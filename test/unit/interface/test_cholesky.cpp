//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/interface/blacs_c.h"
#include "dlaf/interface/cholesky_c.h"
#include "dlaf/interface/utils.h"

#include <gtest/gtest.h>
#include <pika/runtime.hpp>

#include <mpi.h>

#include <iostream>

// BLACS
extern "C" void Cblacs_pinfo();
extern "C" void Cblacs_gridinit(int* contxt, const char* order, int nprow, int npcol);
extern "C" void Cblacs_gridexit(int contxt);
extern "C" void Cblacs_exit(int error);

// ScaLAPACK
extern "C" int numroc(const int* n, const int* nb, const int* iproc, const int* isrcproc,
                      const int* nprocs);
extern "C" void descinit(int* desc, const int* m, const int* n, const int* mb, const int* nb,
                         const int* irsrc, const int* icsrc, const int* ictxt, const int* lld,
                         int* info);
extern "C" void pdgemr2d(int* m, int* n, double* A, int* ia, int* ja, int* desca, double* B, int* ib,
                         int* jb, int* descb, int* ictxt);

// TODO: Check double and float
TEST(CholeskyInterfaceTest, CorrectnessDistributed) {
  int rank;
  int num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  EXPECT_EQ(num_ranks, 6);

  int n = 3;
  int m = 3;

  int nprow = 2;  // Rows of process grid
  int npcol = 3;  // Cols of process grid

  EXPECT_EQ(nprow * npcol, num_ranks);

  int nb = 1;
  int mb = 1;

  const char* order = "C";
  char uplo = 'L';

  int contxt = 0;
  int contxt_global = 0;

  // Global matrix
  Cblacs_get(0, 0, &contxt);
  contxt_global = contxt;

  Cblacs_gridinit(&contxt_global, order, 1, 1);  // Global matrix: only on rank 0
  Cblacs_gridinit(&contxt, order, nprow, npcol);

  int myprow, mypcol;
  Cblacs_gridinfo(contxt, &nprow, &npcol, &myprow, &mypcol);

  int izero = 0;
  int m_local = numroc(&m, &mb, &myprow, &izero, &nprow);
  int n_local = numroc(&n, &nb, &mypcol, &izero, &npcol);

  // Global matrix (one copy on each rank)
  double* A;
  int descA[9] = {0, -1, 0, 0, 0, 0, 0, 0, 0};
  if (rank == 0) {
    A = new double[n * m];
    A[0] = 4.0;
    A[1] = 12.0;
    A[2] = -16.0;
    A[3] = 12.0;
    A[4] = 37.0;
    A[5] = -43.0;
    A[6] = -16.0;
    A[7] = -43.0;
    A[8] = 98.0;

    int info = -1;
    int lldA = m;
    descinit(descA, &m, &n, &m, &n, &izero, &izero, &contxt_global, &lldA, &info);
    ASSERT_EQ(info, 0);
    ASSERT_EQ(descA[0], 1);
  }

  auto a = new double[m_local * n_local];

  int desca[9];
  int info = -1;
  int llda = m_local;
  descinit(desca, &m, &n, &mb, &nb, &izero, &izero, &contxt, &llda, &info);
  ASSERT_EQ(info, 0);
  ASSERT_EQ(desca[0], 1);

  // Distribute global matrix to local matrices
  int ione = 1;
  pdgemr2d(&m, &n, A, &ione, &ione, descA, a, &ione, &ione, desca, &contxt);

  // Use EXPECT_EQ to avoid potential deadlocks!
  if (myprow == 0 && mypcol == 0) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], 4.0);
    EXPECT_DOUBLE_EQ(a[1], -16.0);
  }
  if (myprow == 0 && mypcol == 1) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], 12.0);
    EXPECT_DOUBLE_EQ(a[1], -43.0);
  }
  if (myprow == 0 && mypcol == 2) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], -16.0);
    EXPECT_DOUBLE_EQ(a[1], 98.0);
  }
  if (myprow == 1 && mypcol == 0) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], 12.0);
  }
  if (myprow == 1 && mypcol == 1) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], 37.0);
  }
  if (myprow == 1 && mypcol == 2) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], -43.0);
  }

  const char* argv[] = {"test_interface_", nullptr};
  dlaf_initialize(1, argv);

  info = -1;
  dlaf_pdpotrf(uplo, n, a, 1, 1, desca, info);
  ASSERT_EQ(info, 0);

  // Gather local matrices into global one
  pdgemr2d(&m, &n, a, &ione, &ione, desca, A, &ione, &ione, descA, &contxt);

  if (rank == 0) {
    EXPECT_DOUBLE_EQ(A[0], 2.0);
    EXPECT_DOUBLE_EQ(A[1], 6.0);
    EXPECT_DOUBLE_EQ(A[2], -8.0);
    EXPECT_DOUBLE_EQ(A[3], 12.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[4], 1.0);
    EXPECT_DOUBLE_EQ(A[5], 5.0);
    EXPECT_DOUBLE_EQ(A[6], -16.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[7], -43.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[8], 3.0);

    delete[] A;
  }

  dlaf_finalize();

  delete[] a;
  Cblacs_gridexit(contxt);
}

TEST(CholeskyCInterfaceTest, CorrectnessDistributed) {
  int rank;
  int num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  EXPECT_EQ(num_ranks, 6);

  int n = 3;
  int m = 3;

  int nprow = 2;  // Rows of process grid
  int npcol = 3;  // Cols of process grid

  EXPECT_EQ(nprow * npcol, num_ranks);

  int nb = 1;
  int mb = 1;

  const char* order = "C";
  char uplo = 'L';

  int contxt = 0;
  int contxt_global = 0;

  // Global matrix
  Cblacs_get(0, 0, &contxt);
  contxt_global = contxt;

  Cblacs_gridinit(&contxt_global, order, 1, 1);  // Global matrix: only on rank 0
  Cblacs_gridinit(&contxt, order, nprow, npcol);

  MPI_Comm comm = blacs_communicator(contxt);

  int myprow, mypcol;
  Cblacs_gridinfo(contxt, &nprow, &npcol, &myprow, &mypcol);

  int izero = 0;
  int m_local = numroc(&m, &mb, &myprow, &izero, &nprow);
  int n_local = numroc(&n, &nb, &mypcol, &izero, &npcol);

  // Global matrix (one copy on each rank)
  double* A;
  int descA[9] = {0, -1, 0, 0, 0, 0, 0, 0, 0};
  if (rank == 0) {
    A = new double[n * m];
    A[0] = 4.0;
    A[1] = 12.0;
    A[2] = -16.0;
    A[3] = 12.0;
    A[4] = 37.0;
    A[5] = -43.0;
    A[6] = -16.0;
    A[7] = -43.0;
    A[8] = 98.0;

    int info = -1;
    int lldA = m;
    descinit(descA, &m, &n, &m, &n, &izero, &izero, &contxt_global, &lldA, &info);
    ASSERT_EQ(info, 0);
    ASSERT_EQ(descA[0], 1);
  }

  auto a = new double[m_local * n_local];

  int desca[9];
  int info = -1;
  int llda = m_local;
  descinit(desca, &m, &n, &mb, &nb, &izero, &izero, &contxt, &llda, &info);
  ASSERT_EQ(info, 0);
  ASSERT_EQ(desca[0], 1);

  // Distribute global matrix to local matrices
  int ione = 1;
  pdgemr2d(&m, &n, A, &ione, &ione, descA, a, &ione, &ione, desca, &contxt);

  // Use EXPECT_EQ to avoid potential deadlocks!
  if (myprow == 0 && mypcol == 0) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], 4.0);
    EXPECT_DOUBLE_EQ(a[1], -16.0);
  }
  if (myprow == 0 && mypcol == 1) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], 12.0);
    EXPECT_DOUBLE_EQ(a[1], -43.0);
  }
  if (myprow == 0 && mypcol == 2) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], -16.0);
    EXPECT_DOUBLE_EQ(a[1], 98.0);
  }
  if (myprow == 1 && mypcol == 0) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], 12.0);
  }
  if (myprow == 1 && mypcol == 1) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], 37.0);
  }
  if (myprow == 1 && mypcol == 2) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], -43.0);
  }

  const char* argv[] = {"test_interface_", nullptr};
  dlaf_initialize(1, argv);

  dlaf_cholesky_d(uplo, a, m, n, mb, nb, m_local, comm, nprow, npcol);

  // Gather local matrices into global one
  pdgemr2d(&m, &n, a, &ione, &ione, desca, A, &ione, &ione, descA, &contxt);

  if (rank == 0) {
    EXPECT_DOUBLE_EQ(A[0], 2.0);
    EXPECT_DOUBLE_EQ(A[1], 6.0);
    EXPECT_DOUBLE_EQ(A[2], -8.0);
    EXPECT_DOUBLE_EQ(A[3], 12.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[4], 1.0);
    EXPECT_DOUBLE_EQ(A[5], 5.0);
    EXPECT_DOUBLE_EQ(A[6], -16.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[7], -43.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[8], 3.0);

    delete[] A;
  }

  dlaf_finalize();

  delete[] a;
  Cblacs_gridexit(contxt);
}
