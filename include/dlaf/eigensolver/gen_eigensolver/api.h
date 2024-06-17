//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/eigensolver/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>

namespace dlaf::eigensolver::internal {

enum class Factorization { do_factorization, already_factorized };

template <Backend backend, Device device, class T>
struct GenEigensolver {
  static void call(blas::Uplo uplo, Matrix<T, device>& mat_a, Matrix<T, device>& mat_b,
                   Matrix<BaseType<T>, device>& eigenvalues, Matrix<T, device>& eigenvectors,
                   const Factorization factorization);
  static void call(comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, device>& mat_a,
                   Matrix<T, device>& mat_b, Matrix<BaseType<T>, device>& eigenvalues,
                   Matrix<T, device>& eigenvectors, const Factorization factorization);
};

// ETI
#define DLAF_EIGENSOLVER_GEN_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct GenEigensolver<BACKEND, DEVICE, DATATYPE>;

DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif

}
