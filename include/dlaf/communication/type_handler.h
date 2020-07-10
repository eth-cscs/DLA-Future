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
#include "dlaf/communication/error.h"
#include "dlaf/types.h"

namespace dlaf {
namespace comm {
namespace internal {

/// MPI Type handler.
///
/// This class manages a custom MPI_Datatype with the RAII principle.
/// It is movable but not copyable.
template <typename T>
struct type_handler {
  /// Create a not valid custom MPI_Datatype.
  type_handler() noexcept = default;

  /// Create a custom MPI_Datatype for non-contiguous data.
  ///
  /// After creation, the MPI_Datatype is ready to be used.
  /// @param nblocks      number of blocks,
  /// @param block_size   number of contiguous elements of type @p T in each block,
  /// @param stride       stride (in elements) between starts of adjacent blocks.
  type_handler(std::size_t nblocks, std::size_t block_size, std::size_t stride) {
    MPI_Datatype element_type = dlaf::comm::mpi_datatype<std::remove_pointer_t<T>>::type;
    MPI_Type_vector(to_int(nblocks), to_int(block_size), to_int(stride), element_type, &custom_type_);
    MPI_Type_commit(&custom_type_);
  }

  /// Release the custom MPI_Datatype.
  ~type_handler() {
    if (static_cast<bool>(*this))
      MPI_Type_free(&custom_type_);
  }

  // movable
  type_handler(type_handler&& rhs) noexcept {
    custom_type_ = rhs.custom_type_;
    rhs.custom_type_ = MPI_DATATYPE_NULL;
  }

  type_handler& operator=(type_handler&& rhs) noexcept {
    custom_type_ = rhs.custom_type_;
    rhs.custom_type_ = MPI_DATATYPE_NULL;
    return *this;
  }

  // not copyable
  type_handler(const type_handler&) = delete;
  type_handler& operator=(const type_handler&) = delete;

  /// Implicit cast to bool.
  ///
  /// @return true if it is a valid custom MPI_Datatype.
  operator bool() const noexcept {
    return MPI_DATATYPE_NULL != custom_type_;
  }

  /// Implicit cast to MPI_Datatype.
  ///
  /// @return the underlying custom MPI_Datatype if valid, otherwise MPI_DATATYPE_NULL.
  operator MPI_Datatype() const noexcept {
    return custom_type_;
  }

protected:
  MPI_Datatype custom_type_ = MPI_DATATYPE_NULL;
};

}
}
}
