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

#include <type_traits>
#include <utility>

#include <pika/execution.hpp>

#include "dlaf/sender/lift_non_sender.h"

namespace dlaf {
namespace internal {
/// Helper function which behaves similarly to when_all with the exception that
/// non-senders are first turned into senders using just.
template <typename... Ts>
auto whenAllLift(Ts&&... ts) {
  return pika::execution::experimental::when_all(liftNonSender<Ts>(std::forward<Ts>(ts))...);
}
}
}
