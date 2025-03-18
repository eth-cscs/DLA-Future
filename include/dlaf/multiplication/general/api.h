//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <complex>

#include <blas.hh>

#include <dlaf/common/index2d.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/types.h>

namespace dlaf::multiplication::internal {
using dlaf::matrix::internal::MatrixRef;

template <Backend B, Device D, class T>
struct General {
  static void callNN(const T alpha, MatrixRef<const T, D>& mat_a, MatrixRef<const T, D>& mat_b,
                     const T beta, MatrixRef<T, D>& mat_c);
  static void callNN(comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_task_chain,
                     comm::CommunicatorPipeline<comm::CommunicatorType::Col>& col_task_chain,
                     const T alpha, MatrixRef<const T, D>& mat_a, MatrixRef<const T, D>& mat_b,
                     const T beta, MatrixRef<T, D>& mat_c);
};

#define DLAF_MULTIPLICATION_GENERAL_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct General<BACKEND, DEVICE, DATATYPE>;

DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif

}
