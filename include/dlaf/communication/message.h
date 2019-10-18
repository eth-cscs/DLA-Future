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

#include <type_traits>

#include "dlaf/common/buffer.h"
#include "dlaf/communication/type_handler.h"
#include "dlaf/mpi_header.h"

namespace dlaf {
namespace comm {

template <class Buffer>
struct message {
  using buffer_t = Buffer;
  using T = typename dlaf::common::buffer_traits<buffer_t>::element_t;

  message(buffer_t buffer) : buffer_(buffer) {
    if (get_num_blocks(buffer) == 1) {  // TODO contiguous
      classic_type_ = dlaf::comm::mpi_datatype<T>::type;
      return;
    }

    custom_type_ =
        type_handler<T>(get_pointer(buffer), get_num_blocks(buffer), get_blocksize(buffer), 0);
  }

  T* ptr() {
    return get_pointer(buffer_);
  }

  const T* ptr() const {
    return get_pointer(buffer_);
  }

  std::size_t count() const {
    if (custom_type_)
      return get_num_blocks(buffer_);
    return get_blocksize(buffer_);
    ;
  }

  MPI_Datatype mpi_type() const {
    if (custom_type_)
      return custom_type_();
    return classic_type_;
  }

  buffer_t buffer_;
  MPI_Datatype classic_type_;
  type_handler<T> custom_type_;
};

template <class Buffer>
auto make_message(Buffer buffer) -> decltype(message<Buffer>(buffer)) {
  return {buffer};
}

}
}
