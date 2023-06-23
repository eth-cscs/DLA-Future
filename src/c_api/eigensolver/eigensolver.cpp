//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "eigensolver.h"

#include <dlaf_c/eigensolver/eigensolver.h>
#include <dlaf_c/init.h>

void check_dlaf(char uplo, DLAF_descriptor desca, DLAF_descriptor descz) {
  if (uplo != 'L' and uplo != 'l') {
    std::cerr << "ERROR: The eigensolver currently supports only UPLO=='L'\n";
    exit(-1);
  }

  bool dims = (desca.m == desca.n and descz.m == descz.n and desca.m == descz.m);
  if (!dims) {
    std::cerr << "ERROR: Matrices A and Z need to have the same dimension.\n";
    exit(-1);
  }
}

void check_scalapack(char uplo, int* desca, int* descz) {
  if (uplo != 'L' or uplo != 'l') {
    std::cerr << "ERROR: The eigensolver currently supports only UPLO=='L'\n";
    exit(-1);
  }

  if (desca[0] != 1 or descz[0] != 1) {
    std::cerr << "ERROR: DLA-Future only supports dense matrices.\n";
    exit(-1);
  }

  bool dims = (desca[2] == desca[3] and descz[2] == descz[3] and desca[2] == descz[2]);
  if (!dims) {
    std::cerr << "ERROR: Matrices A and Z need to have the same dimension.\n";
    exit(-1);
  }
}

void dlaf_pssyevd(char uplo, int m, float* a, int* desca, float* w, float* z, int* descz, int* info) {
  pxsyevd<float>(uplo, m, a, desca, w, z, descz, *info);
}

void dlaf_pdsyevd(char uplo, int m, double* a, int* desca, double* w, double* z, int* descz, int* info) {
  pxsyevd<double>(uplo, m, a, desca, w, z, descz, *info);
}

void dlaf_eigensolver_s(int dlaf_context, char uplo, float* a, struct DLAF_descriptor dlaf_desca,
                        float* w, float* z, struct DLAF_descriptor dlaf_descz) {
  eigensolver<float>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz);
}

void dlaf_eigensolver_d(int dlaf_context, char uplo, double* a, struct DLAF_descriptor dlaf_desca,
                        double* w, double* z, struct DLAF_descriptor dlaf_descz) {
  eigensolver<double>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz);
}
