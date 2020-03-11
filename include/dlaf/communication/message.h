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
  static_assert(dlaf::common::is_buffer<Buffer>::value,
                "Message works just with the Buffer concept (see dlaf/common/buffer.h)");

  /// Type of the elements of the underlying buffer
  using T = typename dlaf::common::buffer_traits<Buffer>::element_t;

public:
  /// @brief Create a Message from a given dlaf::common::Buffer
  Message(Buffer buffer) : buffer_(buffer) {
    if (buffer_iscontiguous(buffer) == 1)
      classic_type_ = dlaf::comm::mpi_datatype<T>::type;
    else
      custom_type_ = internal::type_handler<T>(buffer_nblocks(buffer), buffer_blocksize(buffer),
                                               buffer_stride(buffer));
  }

  // movable
  Message(Message&&) = default;
  Message& operator=(Message&&) = default;

  // not copyable
  Message(const Message&) = delete;
  Message& operator=(const Message&) = delete;

  /// @brief Return the pointer to the buffer containing the data
  T* data() const noexcept {
    return buffer_pointer(buffer_);
  }

  /// @brief Return the number of Message::mpi_type() to send
  int count() const noexcept {
    return custom_type_ ? 1 : to_int(buffer_blocksize(buffer_));
  }

  /// @brief Return the MPI_Datatype to use during the MPI communication
  MPI_Datatype mpi_type() const noexcept {
    return custom_type_ ? custom_type_ : classic_type_;
  }

protected:
  Buffer buffer_;  ///< The implementation of the Buffer concept used by this Message

  MPI_Datatype classic_type_;  ///< If a basic type can be used, it is stored here

  /// @brief Custom MPI_Datatype
  ///
  /// In case a basic MPI_Datatype is not suitable to represent the underlying buffer, this contains the
  /// custom MPI_Datatype
  internal::type_handler<T> custom_type_;
};

/// @brief helper function for creating a message from a buffer
template <class Buffer>
auto make_message(Buffer buffer) noexcept {
  return Message<Buffer>{buffer};
}

}
}
