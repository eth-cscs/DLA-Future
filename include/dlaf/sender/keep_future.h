//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <pika/execution.hpp>

#include <utility>

#include "dlaf/sender/traits.h"

namespace dlaf {
namespace internal {
/// Applies keep_future if the sender is a shared_future and fails to compile
/// otherwise.
///
/// Non-futures cannot be passed to keep_future at all. futures should not be
/// passed to keep_future in DLA-Future because of lifetime requirements.
/// Functions like transform use unwrapping internally and when passed a future
/// unwrapping will consume the value contained in the future. This breaks the
/// lifetime requirements of many algorithms. This helpers are intended to make
/// the right choice easier. keepFuture should be the first choice since it
/// fails to compile with futures. pika::keep_future should (almost) never be
/// used. Be very sure about what you are doing if you use pika::keep_future
/// directly.
template <typename S>
decltype(auto) keepFuture(S&& s) {
  if constexpr (is_shared_future_v<S>) {
    return pika::execution::experimental::keep_future(std::forward<S>(s));
  }
  else {
    static_assert(sizeof(S) == 0, "keepFuture should be used only on shared_futures");
  }
}
}
}
