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

    custom_type_ = internal::type_handler<T>(get_pointer(buffer), get_num_blocks(buffer),
                                             get_blocksize(buffer), get_stride(buffer));
  }

  message(message&&) = default;
  message& operator=(message&&) = default;

  message(const message&) = delete;
  message& operator=(const message&) = delete;

  T* ptr() noexcept {
    return get_pointer(buffer_);
  }

  const T* ptr() const noexcept {
    return get_pointer(buffer_);
  }

  std::size_t count() const noexcept {
    return custom_type_ ? 1 : get_blocksize(buffer_);
  }

  MPI_Datatype mpi_type() const noexcept {
    return custom_type_ ? custom_type_ : classic_type_;
  }

protected:
  buffer_t buffer_;
  MPI_Datatype classic_type_;
  internal::type_handler<T> custom_type_;
};

template <class T>
auto make_message(dlaf::common::Buffer<T>&& buffer) noexcept
    -> decltype(message<dlaf::common::Buffer<T>>{buffer}) {
  return {std::forward<dlaf::common::Buffer<T>>(buffer)};
}

template <class... Ts>
auto make_message(Ts&&... args) noexcept {
  return make_message(make_buffer(std::forward<Ts>(args)...));
}

}
}
