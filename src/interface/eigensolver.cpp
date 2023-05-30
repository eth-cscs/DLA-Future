//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/interface/eigensolver.h>

namespace dlaf::interface {

extern "C" void pssyevd(char uplo, int n, float* a, int* desca, float* w, float* z, int* descz,
                        int& info) {
  pxsyevd<float>(uplo, n, a, desca, w, z, descz, info);
}

extern "C" void pdsyevd(char uplo, int n, double* a, int* desca, double* w, double* z, int* descz,
                        int& info) {
  pxsyevd<double>(uplo, n, a, desca, w, z, descz, info);
}

}
