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

#include <complex>

#include <pika/execution.hpp>

#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/matrix/views.h>
#include <dlaf/types.h>

namespace dlaf::factorization::internal {

template <Backend backend, Device device, class T>
struct QR {};

template <Backend backend, Device device, class T>
struct QR_Tfactor {
  static void call(matrix::Panel<Coord::Col, T, device>& hh_panel,
                   matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                   matrix::ReadWriteTileSender<T, device> t,
                   matrix::Panel<Coord::Col, T, device>& workspaces);
  static void call(matrix::Panel<Coord::Col, T, device>& hh_panel,
                   matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                   matrix::ReadWriteTileSender<T, device> t,
                   matrix::Panel<Coord::Col, T, device>& workspaces,
                   comm::CommunicatorPipeline<comm::CommunicatorType::Col>& mpi_col_task_chain);
};

// ETI
#define DLAF_FACTORIZATION_QR_TFACTOR_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct QR_Tfactor<BACKEND, DEVICE, DATATYPE>;

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
