//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/factorization/qr/mc.h"
#include "dlaf/factorization/qr/t_factor_impl.h"

namespace dlaf::factorization {

namespace internal {

// Forms the triangular factor T of a block reflector H of order n, which is defined as a product of k
// elementary reflectors.
//
// A Storage-Efficient WY Representation for Products of Householder Transformations.
// Schreiber, Robert & VanLoan, Charles. (1989)
// SIAM Journal on Scientific and Statistical Computing. 10. 10.1137/0910005.
template <Backend backend, Device device, class T>
void computeTFactor(const SizeType k, Matrix<const T, device>& v, const GlobalTileIndex v_start,
                    pika::shared_future<common::internal::vector<T>> taus,
                    pika::future<matrix::Tile<T, device>> t) {
  QR_Tfactor<backend, device, T>::call(k, v, v_start, taus, std::move(t));
}

// Forms the triangular factor T of a block reflector H of order n, which is defined as a product of k
// elementary reflectors.
//
// A Storage-Efficient WY Representation for Products of Householder Transformations.
// Schreiber, Robert & VanLoan, Charles. (1989)
// SIAM Journal on Scientific and Statistical Computing. 10. 10.1137/0910005.
template <Backend backend, Device device, class T>
void computeTFactor(const SizeType k, Matrix<const T, device>& v, const GlobalTileIndex v_start,
                    pika::shared_future<common::internal::vector<T>> taus,
                    pika::future<matrix::Tile<T, device>> t,
                    common::Pipeline<comm::Communicator>& mpi_col_task_chain) {
  QR_Tfactor<backend, device, T>::call(k, v, v_start, taus, std::move(t), mpi_col_task_chain);
}
/// ---- ETI
#define DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(KWORD, BACKEND, DEVICE, T)                 \
  KWORD template void                                                                      \
  computeTFactor<BACKEND, DEVICE, T>(const SizeType k, Matrix<const T, DEVICE>& v,         \
                                     const GlobalTileIndex v_start,                        \
                                     pika::shared_future<common::internal::vector<T>> taus, \
                                     pika::future<matrix::Tile<T, DEVICE>> t);

#define DLAF_FACTORIZATION_QR_TFACTOR_DISTR_ETI(KWORD, BACKEND, DEVICE, T)                 \
  KWORD template void                                                                      \
  computeTFactor<BACKEND, DEVICE, T>(const SizeType k, Matrix<const T, DEVICE>& v,         \
                                     const GlobalTileIndex v_start,                        \
                                     pika::shared_future<common::internal::vector<T>> taus, \
                                     pika::future<matrix::Tile<T, DEVICE>> t,               \
                                     common::Pipeline<comm::Communicator>& mpi_col_task_chain);

#define DLAF_FACTORIZATION_QR_TFACTOR_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  DLAF_FACTORIZATION_QR_TFACTOR_DISTR_ETI(KWORD, BACKEND, DEVICE, DATATYPE)

DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_CUDA
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}

}
