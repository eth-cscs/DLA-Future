//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <utility>

/// @file

/// Given a function name @fname generates a constexpr object with name fname_o
/// with type fname_t. The generated type has one static operator() which
/// transparently forwards all arguments to a call to fname. This macro is
/// useful for creating wrappers of overloaded functions that would otherwise
/// require explicit selection of the overload when passing the function to
/// higher-order functions. With this wrapper the overload is selected inside
/// the wrapper's operator().
///
/// The function name is wrapped in parentheses to disable ADL. It is assumed
/// that all required overloads are found relative to the namespace where the
/// macro is used.

#define DLAF_MAKE_CALLABLE_OBJECT(fname)                                                              \
  constexpr struct fname##_t {                                                                        \
    template <typename... Ts>                                                                         \
    auto operator()(Ts&&... ts) const noexcept(                                                       \
        noexcept((fname) (std::forward<Ts>(ts)...))) -> decltype((fname) (std::forward<Ts>(ts)...)) { \
      return (fname) (std::forward<Ts>(ts)...);                                                       \
    }                                                                                                 \
  } fname##_o {                                                                                       \
  }
