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

#include "dlaf/communication/datatypes.h"
#include "dlaf/mpi_header.h"

namespace dlaf {
namespace comm {

template <typename T>
struct type_handler {
  type_handler() = default;

  type_handler(T* ptr, std::size_t nblocks, std::size_t block_size, std::size_t stride) {
    MPI_Datatype element_type = dlaf::comm::mpi_datatype<std::remove_pointer_t<T>>::type;
    MPI_Type_vector(nblocks, block_size, stride, element_type, &custom_type_);
    MPI_Type_commit(&custom_type_);
  }

  ~type_handler() {
    if (static_cast<bool>(*this))
      MPI_Type_free(&custom_type_);
  }

  type_handler(type_handler&& rhs) {
    custom_type_ = rhs.custom_type_;
    rhs.custom_type_ = MPI_DATATYPE_NULL;
  }

  type_handler& operator=(type_handler&& rhs) {
    custom_type_ = rhs.custom_type_;
    rhs.custom_type_ = MPI_DATATYPE_NULL;
    return *this;
  }

  type_handler(const type_handler&) = delete;
  type_handler& operator=(const type_handler&) = delete;

  operator bool() const {
    return MPI_DATATYPE_NULL != custom_type_;
  }

  operator MPI_Datatype() const {
    return custom_type_;
  }

  MPI_Datatype custom_type_ = MPI_DATATYPE_NULL;
};

}
}
