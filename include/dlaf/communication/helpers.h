//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/include/util.hpp>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/functions_sync.h"

namespace dlaf {
namespace comm {

struct row_wise {};
struct col_wise {};

namespace sync {

inline auto broadcast_send_impl(row_wise) {
  using common::make_data;

  return [](const auto& source, auto&& comm_wrapper) {
    broadcast::send(comm_wrapper.ref().rowCommunicator(), make_data(source));
  };
}

inline auto broadcast_send_impl(col_wise) {
  using common::make_data;

  return [](const auto& source, auto&& comm_wrapper) {
    broadcast::send(comm_wrapper.ref().colCommunicator(), make_data(source));
  };
}

inline auto broadcast_recv_impl(row_wise, const IndexT_MPI rank) {
  using common::make_data;

  return [=](auto&& dest, auto&& comm_wrapper) {
    broadcast::receive_from(rank, comm_wrapper.ref().rowCommunicator(), make_data(dest));
  };
}

inline auto broadcast_recv_impl(col_wise, const IndexT_MPI rank) {
  using common::make_data;

  return [=](auto&& dest, auto&& comm_wrapper) {
    broadcast::receive_from(rank, comm_wrapper.ref().colCommunicator(), make_data(dest));
  };
}

template <class T>
auto broadcast_send(T row_or_col) {
  return hpx::util::unwrapping(broadcast_send_impl(row_or_col));
}

template <class T>
auto broadcast_recv(T row_or_col, const comm::IndexT_MPI rank) {
  return hpx::util::unwrapping(broadcast_recv_impl(row_or_col, rank));
}

}
}
}
