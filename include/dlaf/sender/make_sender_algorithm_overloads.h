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

#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"

/// Helper macro for generating overloads of algorithms that work with senders. The overloads will be
/// named fname. callable is the callable object that will be used internall for the transform. It
/// generates three overloads:
///
/// 1. One that takes a policy and a sender. The sender must send the required types of arguments for the
///    callable (e.g. tiles and constants for a tile algorithm).
/// 2. One that takes only a policy. This overload returns a partially applied algorithm that can be used
///    with operator|. fname(policy, sender) is equivalent to fname(policy)(sender) and sender |
///    fname(policy).
/// 3. One that takes a policy and the arguments required by the callable. This is almost equivalent to
///    calling the callable directly with the required arguments, with the difference that this overload
///    does the required synchronization before returning in cases when the callable is not blocking.
#define DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(tag, fname, callable)                                    \
  template <Backend B, typename Sender,                                                          \
            typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>     \
  auto fname(const dlaf::internal::Policy<B> p, Sender&& s) {                                    \
    return dlaf::internal::transform<B, tag>(p, callable, std::forward<Sender>(s));              \
  }                                                                                              \
                                                                                                 \
  template <Backend B>                                                                           \
  auto fname(const dlaf::internal::Policy<B> p) {                                                \
    return dlaf::internal::PartialTransform{p, callable};                                        \
  }                                                                                              \
                                                                                                 \
  template <Backend B, typename T1, typename T2, typename... Ts>                                 \
  void fname(const dlaf::internal::Policy<B> p, T1&& t1, T2&& t2, Ts&&... ts) {                  \
    pika::this_thread::experimental::sync_wait(                                                  \
        fname(p, pika::execution::experimental::just(std::forward<T1>(t1), std::forward<T2>(t2), \
                                                     std::forward<Ts>(ts)...)));                 \
  }
