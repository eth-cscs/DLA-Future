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

#include "dlaf/common/data.h"
#include "dlaf/communication/datatypes.h"
#include "dlaf/communication/type_handler.h"
#include "dlaf/mpi_header.h"

namespace dlaf {
namespace comm {

/// Message for MPI
///
/// Given a generic Data concept, it provides suitable MPI_Datatype to be used for sending
/// with MPI all elements of the given data.
/// It is movable but not copyable.
template <class Data>
class Message {
  static_assert(dlaf::common::is_data<Data>::value,
                "Message works just with the Data concept (see dlaf/common/data.h)");

  /// Type of the elements of the underlying data
  using T = typename dlaf::common::data_traits<Data>::element_t;

public:
  /// Create a Message from a given type implementing the Data concept
  Message(Data data) : data_(data) {
    if (data_iscontiguous(data) == 1)
      classic_type_ = dlaf::comm::mpi_datatype<T>::type;
    else
      custom_type_ =
          internal::type_handler<T>(data_nblocks(data), data_blocksize(data), data_stride(data));
  }

  // movable
  Message(Message&&) = default;
  Message& operator=(Message&&) = default;

  // not copyable
  Message(const Message&) = delete;
  Message& operator=(const Message&) = delete;

  /// Return the pointer to the data containing the data
  T* data() const noexcept {
    return data_pointer(data_);
  }

  /// Return the number of Message::mpi_type() to send
  int count() const noexcept {
    return custom_type_ ? 1 : to_int(data_blocksize(data_));
  }

  /// Return the MPI_Datatype to use during the MPI communication
  MPI_Datatype mpi_type() const noexcept {
    return custom_type_ ? custom_type_ : classic_type_;
  }

protected:
  Data data_;  ///< The implementation of the Data concept used by this Message

  MPI_Datatype classic_type_;  ///< If a basic type can be used, it is stored here

  /// Custom MPI_Datatype
  ///
  /// In case a basic MPI_Datatype is not suitable to represent the underlying data, this contains the
  /// custom MPI_Datatype
  internal::type_handler<T> custom_type_;
};

/// Helper function for creating a Message from a given type implementing the Data concept
template <class Data>
auto make_message(Data data) noexcept {
  return Message<Data>{data};
}
}
}
