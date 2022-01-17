//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <type_traits>

namespace dlaf::internal {
/// Helper type to signal that a type should be formatted in its "short" form.
///
/// It is up to overloads of operator<< to define what "short" means for a
/// particular type.
template <typename T>
struct FormatShort {
  const T value;
};

template <typename T>
FormatShort(T &&) -> FormatShort<std::decay_t<T>>;
}
