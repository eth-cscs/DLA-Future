//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/interface/cholesky.h>

namespace dlaf::interface {

extern "C" void pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int& info) {
  pxpotrf<double>(uplo, n, a, ia, ja, desca, info);
}

extern "C" void pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int& info) {
  pxpotrf<float>(uplo, n, a, ia, ja, desca, info);
}

extern "C" void cholesky_d(char uplo, double* a, int m, int n, int mb, int nb, int lld, const MPI_Comm& communicator, int nprow, int npcol){
  pxpotrf<double>(uplo, a, m, n, mb, nb, lld, communicator, nprow, npcol);
}

extern "C" void cholesky_s(char uplo, float* a, int m, int n, int mb, int nb, int lld, const MPI_Comm& communicator, int nprow, int npcol){
  pxpotrf<float>(uplo, a, m, n, mb, nb, lld, communicator, nprow, npcol);
}

}
