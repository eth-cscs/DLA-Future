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

#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>

namespace dlaf::permutations::internal {

template <Backend B, Device D, class T, Coord coord, CommunicatorType c2>
struct Permutations {
  static void call(SizeType i_begin, SizeType i_end, Matrix<const SizeType, D>& perms,
                   Matrix<const T, D>& mat_in, Matrix<T, D>& mat_out);
  static void call(comm::CommunicatorPipeline<c2>& sub_task_chain, SizeType i_begin,
                   SizeType i_end, Matrix<const SizeType, D>& perms, Matrix<const T, D>& mat_in,
                   Matrix<T, D>& mat_out);
};

// ETI
#define DLAF_PERMUTATIONS_GENERAL_ETI(KWORD, BACKEND, DEVICE, DATATYPE)      \
  KWORD template struct Permutations<BACKEND, DEVICE, DATATYPE, Coord::Col, CommunicatorType::Row>; \
  KWORD template struct Permutations<BACKEND, DEVICE, DATATYPE, Coord::Row, CommunicatorType::Col>;

DLAF_PERMUTATIONS_GENERAL_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_PERMUTATIONS_GENERAL_ETI(extern, Backend::MC, Device::CPU, double)

#ifdef DLAF_WITH_GPU
DLAF_PERMUTATIONS_GENERAL_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_PERMUTATIONS_GENERAL_ETI(extern, Backend::GPU, Device::GPU, double)
#endif
}
