//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <blas.hh>

#include <dlaf/common/index2d.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/types.h>

namespace dlaf::multiplication {
namespace internal {
using dlaf::matrix::internal::MatrixRef;

template <Backend B, Device D, class T>
struct General {
  static void callNN(const T alpha, MatrixRef<const T, D>& mat_a, MatrixRef<const T, D>& mat_b,
                     const T beta, MatrixRef<T, D>& mat_c);
  static void callNN(common::Pipeline<comm::Communicator>& row_task_chain,
                     common::Pipeline<comm::Communicator>& col_task_chain, const T alpha,
                     MatrixRef<const T, D>& mat_a, MatrixRef<const T, D>& mat_b, const T beta,
                     MatrixRef<T, D>& mat_c);
};

template <Backend B, Device D, class T>
struct GeneralSub {
  static void callNN(const SizeType i_tile_from, const SizeType i_tile_to, const blas::Op opA,
                     const blas::Op opB, const T alpha, Matrix<const T, D>& mat_a,
                     Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c);
  static void callNN(comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_task_chain,
                     comm::CommunicatorPipeline<comm::CommunicatorType::Col>& col_task_chain,
                     const SizeType i_tile_from, const SizeType i_tile_to, const T alpha,
                     Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b, const T beta,
                     Matrix<T, D>& mat_c);
};

#define DLAF_MULTIPLICATION_GENERAL_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct General<BACKEND, DEVICE, DATATYPE>;               \
  KWORD template struct GeneralSub<BACKEND, DEVICE, DATATYPE>;

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
}
