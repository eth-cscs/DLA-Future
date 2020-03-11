//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include <type_traits>

#include "dlaf/common/buffer.h"
#include "dlaf/communication/type_handler.h"
#include "dlaf/mpi_header.h"

namespace dlaf {
namespace comm {

/// @brief Message for MPI
///
/// Starting from a dlaf::common::Buffer it provides a suitable MPI_Datatype to be used for sending with
/// MPI all elements of the given buffer. It is movable but not copyable.
template <class Buffer>
class Message {
  /// Type of the dlaf::common::Buffer used by this Message
  using buffer_t = Buffer;

public:
  /// Type of the elements of the underlying buffer
  using element_t = typename dlaf::common::buffer_traits<buffer_t>::element_t;

  /// @brief Create a Message from a given dlaf::common::Buffer
  Message(buffer_t buffer) : buffer_(buffer) {
    if (get_num_blocks(buffer) == 1) {
      classic_type_ = dlaf::comm::mpi_datatype<element_t>::type;
      return;
    }

    custom_type_ = internal::type_handler<element_t>(get_num_blocks(buffer), get_blocksize(buffer),
                                                     get_stride(buffer));
  }

  // movable
  Message(Message&&) = default;
  Message& operator=(Message&&) = default;

  // not copyable
  Message(const Message&) = delete;
  Message& operator=(const Message&) = delete;

  /// @brief Return the pointer to the buffer containing the data
  element_t* data() noexcept {
    return get_pointer(buffer_);
  }

  const element_t* data() const noexcept {
    return get_pointer(buffer_);
  }

  /// @brief Return the number of Message::mpi_type() to send
  int count() const noexcept {
    return custom_type_ ? 1 : to_int(get_blocksize(buffer_));
  }

  /// @brief Return the MPI_Datatype to use during the MPI communication
  MPI_Datatype mpi_type() const noexcept {
    return custom_type_ ? custom_type_ : classic_type_;
  }

protected:
  buffer_t buffer_;  ///< The implementation of the Buffer concept used by this Message

  MPI_Datatype classic_type_;  ///< If a basic type can be used, it is stored here

  /// @brief Custom MPI_Datatype
  ///
  /// In case a basic MPI_Datatype is not suitable to represent the underlying buffer, this contains the
  /// custom MPI_Datatype
  internal::type_handler<element_t> custom_type_;
};

/// @brief helper function for creating a message from a buffer
template <class T>
auto make_message(dlaf::common::Buffer<T>&& buffer) noexcept
    -> decltype(Message<dlaf::common::Buffer<T>>{buffer}) {
  return {std::forward<dlaf::common::Buffer<T>>(buffer)};
}

/// @brief helper function for creating a message, given parameters for the buffer
template <class... Ts>
auto make_message(Ts&&... args) noexcept {
  return make_message(dlaf::common::make_buffer(std::forward<Ts>(args)...));
}

}
}
