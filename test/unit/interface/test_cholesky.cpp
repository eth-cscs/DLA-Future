//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/interface/cholesky.h"
#include "dlaf/interface/blacs.h"
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
extern "C" int numroc(const int* n, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);
extern "C" void descinit(int* desc, const int* m, const int* n, const int* mb, const int* nb, const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);

// TODO: Check double and float
TEST(CholeskyInterfaceTest, CorrectnessDistributed) {
  int rank;
  int num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  EXPECT_EQ(num_ranks, 6);

  int n = 3;
  int m = 3;

  int nprow = 2; // Rows of process grid
  int npcol = 3; // Cols of process grid

  EXPECT_EQ(nprow * npcol, num_ranks);

  int nb = 1;
  int mb = 1;

  const char* order = "C";
  char uplo = 'L';

  int contxt = 0;
  
  dlaf::interface::blacs::Cblacs_get(0, 0, &contxt);
  Cblacs_gridinit(&contxt, order, nprow, npcol);
  
  int myprow, mypcol;
  dlaf::interface::blacs::Cblacs_gridinfo(contxt, &nprow, &npcol, &myprow, &mypcol);
  EXPECT_LT(myprow, nprow);
  EXPECT_LT(mypcol, npcol);

  int izero = 0;
  int m_local = numroc(&m, &mb, &myprow, &izero, &nprow);
  int n_local = numroc(&n, &nb, &mypcol, &izero, &npcol);

  auto A = new double[static_cast<unsigned long>(m_local * n_local)];

  // TODO:Fill A
  std::cout << myprow << ' ' << mypcol << ' ' << m_local << ' ' << n_local << '\n';
  if(myprow == 0 && mypcol == 0){
    assert(m_local * n_local == 2);
    A[0] = 4.0;
    A[1] = -16.0;
  }
  if(myprow == 1 && mypcol == 0){
    assert(m_local * n_local == 1);
    A[0] = 12;
  }
  if(myprow == 0 && mypcol == 1){
    assert(m_local * n_local == 2);
    A[0] = 12.0;
    A[1] = -43.0;
  }
  if(myprow == 1 && mypcol == 1){
    assert(m_local * n_local == 1);
    A[0] = 37;
  }
  if(myprow == 0 && mypcol == 2){
    assert(m_local * n_local == 2);
    A[0] = -16.0;
    A[1] = 98.0;
  }
  if(myprow == 1 && mypcol == 2){
    assert(m_local * n_local == 1);
    A[0] = -43;
  }

  int desca[9];
  int info = -1;
  int llda = m_local;
  descinit(desca, &m, &n, &mb, &nb, &izero, &izero, &contxt, &llda, &info);
  EXPECT_EQ(info, 0);

  const char* argv[] = {"test_interface_", nullptr};
  dlaf::interface::utils::dlafuture_init(1, argv);

  dlaf::interface::pdpotrf(uplo, n, A, 1, 1, desca, info);
  EXPECT_EQ(info, 0);

  // TODO: Check decomposition
  if(myprow == 0 && mypcol == 0){
    EXPECT_DOUBLE_EQ(A[0], 2.0);
    EXPECT_DOUBLE_EQ(A[1], -8.0);
  }
  if(myprow == 1 && mypcol == 0){
    EXPECT_DOUBLE_EQ(A[0], 6.0);
  }
  if(myprow == 0 && mypcol == 1){
    EXPECT_DOUBLE_EQ(A[0], 12.0); // Upper: contains original value
    EXPECT_DOUBLE_EQ(A[1], 5.0);
  }
  if(myprow == 1 && mypcol == 1){
    EXPECT_DOUBLE_EQ(A[0], 1.0);
  }
  if(myprow == 0 && mypcol == 2){
    EXPECT_DOUBLE_EQ(A[0], -16.0); // Upper: contains original value
    EXPECT_DOUBLE_EQ(A[1], 3.0);
  }
  if(myprow == 1 && mypcol == 2){
    EXPECT_DOUBLE_EQ(A[0], -43.0); // Upper: contains original value
  }

  dlaf::interface::utils::dlafuture_finalize();

  delete[] A;
  Cblacs_gridexit(contxt);
}
