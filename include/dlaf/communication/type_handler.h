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

namespace internal {
void mpi_release_type(MPI_Datatype* d) {
  if (d == nullptr)
    return;
  MPI_Type_free(d);
}
}

template <typename T>
struct type_handler {
  using mpi_type_handler_t = std::unique_ptr<MPI_Datatype, decltype(internal::mpi_release_type)*>;

  type_handler() = default;

  type_handler(T* ptr, std::size_t nblocks, std::size_t block_size, std::size_t stride) {
    MPI_Datatype element_type = dlaf::comm::mpi_datatype<std::remove_pointer_t<T>>::type;
    MPI_Datatype new_type;
    MPI_Type_vector(nblocks, block_size, stride, element_type, &new_type);

    mpi_handler_ = mpi_type_handler_t(&new_type, internal::mpi_release_type);
  }

  operator bool() const {
    return static_cast<bool>(mpi_handler_);
  }

  MPI_Datatype operator()() const {
    return *mpi_handler_;
  }

  mpi_type_handler_t mpi_handler_{nullptr, internal::mpi_release_type};
};

}
}
