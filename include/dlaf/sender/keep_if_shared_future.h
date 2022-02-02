//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
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
/// Applies keep_future if the sender is a shared_future and returns the sender
/// unmodified otherwise.
template <typename S>
decltype(auto) keepIfSharedFuture(S&& s) {
  static_assert(pika::execution::experimental::is_sender_v<S>,
                "liftNonSender should only be used with senders");

  if constexpr (is_shared_future_v<S>) {
    return pika::execution::experimental::keep_future(std::forward<S>(s));
  }
  else {
    return std::forward<S>(s);
  }
}
}
}
