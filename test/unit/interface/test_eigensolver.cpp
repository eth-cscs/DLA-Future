//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/interface/blacs.h"
#include "dlaf/interface/eigensolver.h"
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
TEST(EigensolverInterfaceTest, CorrectnessDistributed) {
  int rank;
  int num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  EXPECT_EQ(num_ranks, 6);

  int m = 4;
  int n = 4;

  int nprow = 2;  // Rows of process grid
  int npcol = 2;  // Cols of process grid

  EXPECT_EQ(nprow * npcol, num_ranks);

  int mb = 3;
  int nb = 3;

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
  double* Z;
  int descA[9] = {0, -1, 0, 0, 0, 0, 0, 0, 0};
  int descZ[9] = {0, -1, 0, 0, 0, 0, 0, 0, 0};
  if (rank == 0) {
    A = new double[m * n];
    A[0] = 3.0;
    A[1] = 0.0;
    A[2] = 0.0;
    A[3] = 1.0;
    A[4] = 0.0;
    A[5] = 2.0;
    A[6] = -1.0;
    A[7] = 0.0;
    A[8] = 0.0;
    A[9] = -1.0;
    A[10] = 2.0;
    A[11] = 0.0;
    A[12] = 1.0;
    A[13] = 0.0;
    A[14] = 0.0;
    A[15] = 3.0;

    int info = -1;
    int lldA = m;
    descinit(descA, &m, &n, &m, &n, &izero, &izero, &contxt_global, &lldA, &info);
    std::cout << descA[0] << ' ' << descA[1] << ' ' << descA[2] << ' ' << descA[3] << ' ' << descA[4]
              << ' ' << descA[5] << ' ' << descA[6] << ' ' << descA[7] << ' ' << descA[8] << '\n';

    EXPECT_EQ(info, 0);
    EXPECT_EQ(descA[0], 1);

    Z = new double[m * n];

    info = -1;
    int lldZ = m;
    descinit(descZ, &m, &n, &m, &n, &izero, &izero, &contxt_global, &lldZ, &info);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(descZ[0], 1);
  }

  auto a = new double[m_local * n_local];  // Local matrix
  auto z = new double[m_local * n_local];  // Local eigenvectors
  auto w = new double[m];                  // Global eigenvalues
  w[0] = 0.0;
  w[1] = 0.0;
  w[2] = 0.0;
  w[3] = 0.0;

  int desca[9];
  int info = -1;
  int llda = m_local;
  descinit(desca, &m, &n, &mb, &nb, &izero, &izero, &contxt, &llda, &info);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(desca[0], 1);

  int descz[9];
  info = -1;
  int lldz = m_local;
  descinit(descz, &m, &n, &mb, &nb, &izero, &izero, &contxt, &lldz, &info);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(descz[0], 1);

  // Distribute global matrix to local matrices
  int ione = 1;
  pdgemr2d(&m, &n, A, &ione, &ione, descA, a, &ione, &ione, desca, &contxt);

  const char* argv[] = {"test_interface_", nullptr};
  dlaf_init(1, argv);

  std::cout << rank << ' ' << w[0] << ' ' << w[1] << ' ' << w[2] << ' ' << w[3] << '\n';

  std::cout << rank << " calling pdsyevd" << '\n';
  info = -1;
  pdsyevd(uplo, m, a, desca, w, z, descz, info);
  std::cout << rank << " pdsyevd done" << '\n';
  EXPECT_EQ(info, 0);

  // Gather local matrices into global one
  pdgemr2d(&m, &n, z, &ione, &ione, descz, Z, &ione, &ione, descZ, &contxt);
  std::cout << rank << ' ' << w[0] << ' ' << w[1] << ' ' << w[2] << ' ' << w[3] << '\n';

  // if (rank == 0) {
  //   // EXPECT_DOUBLE_EQ(Z[0], 1.0);
  //   // EXPECT_DOUBLE_EQ(Z[1], 0.0);
  //   // EXPECT_DOUBLE_EQ(Z[2], 0.0);
  //   // EXPECT_DOUBLE_EQ(Z[3], 0.0);
  //   // EXPECT_DOUBLE_EQ(Z[4], -2.0);
  //   // EXPECT_DOUBLE_EQ(Z[5], 1.0);
  //   // EXPECT_DOUBLE_EQ(Z[6], 0.0);
  //   // EXPECT_DOUBLE_EQ(Z[7], 1.0);
  //   // EXPECT_DOUBLE_EQ(Z[8], 2.0);
  //   std::cout << w[0] << ' ' << w[1] << ' ' << w[2] << ' ' << w[3] << '\n';
  //
  //   EXPECT_DOUBLE_EQ(w[0], 1.0);
  //   EXPECT_DOUBLE_EQ(w[1], 2.0);
  //   EXPECT_DOUBLE_EQ(w[2], 3.0);
  //   EXPECT_DOUBLE_EQ(w[3], 4.0);
  //
  //   delete[] A;
  //   delete[] Z;
  // }

  dlaf_finalize();

  delete[] a;
  delete[] z;
  delete[] w;
  Cblacs_gridexit(contxt);
}
