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

#include <pika/future.hpp>

#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/views.h"
#include "dlaf/types.h"

namespace dlaf::factorization::internal {

template <Backend backend, Device device, class T>
struct QR {};

template <Backend backend, Device device, class T>
struct QR_Tfactor;

template <class T>
struct QR_Tfactor<Backend::MC, Device::CPU, T> {
  static void call(matrix::Panel<Coord::Col, T, Device::CPU>& panel_view,
                   pika::shared_future<common::internal::vector<T>> taus,
                   pika::future<matrix::Tile<T, Device::CPU>> t);
};

#ifdef DLAF_WITH_GPU
template <class T>
struct QR_Tfactor<Backend::GPU, Device::GPU, T> {
  static void call(matrix::Panel<Coord::Col, T, Device::GPU>& panel_view,
                   pika::shared_future<common::internal::vector<T>> taus,
                   pika::future<matrix::Tile<T, Device::GPU>> t);
};
#endif

template <Backend backend, Device device, class T>
struct QR_TfactorDistributed {
  static void call(matrix::Panel<Coord::Col, T, device>& hh_panel,
                   pika::shared_future<common::internal::vector<T>> taus,
                   pika::future<matrix::Tile<T, device>> t,
                   common::Pipeline<comm::Communicator>& mpi_col_task_chain);
};

/// ---- ETI
#define DLAF_FACTORIZATION_QR_TFACTOR_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct QR_Tfactor<BACKEND, DEVICE, DATATYPE>;              \
  KWORD template struct QR_TfactorDistributed<BACKEND, DEVICE, DATATYPE>;

DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
