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

#include <hpx/local/execution.hpp>

#include <type_traits>
#include <utility>

namespace dlaf {
namespace internal {
/// Makes a sender out of the input, if it is not already a sender.
template <typename S>
decltype(auto) liftNonSender(S&& s) {
  if constexpr (hpx::execution::experimental::is_sender_v<S>) {
    return std::forward<S>(s);
  }
  else {
    return hpx::execution::experimental::just(std::forward<S>(s));
  }
}
}
}
