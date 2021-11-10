//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/local/execution.hpp>

#include <type_traits>
#include <utility>

#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"

namespace dlaf {
namespace internal {
/// A partially applied transform, with the policy and callable object given,
/// but the predecessor sender missing. The predecessor sender is applied when
/// calling the operator| overload.
template <Backend B, typename F>
class PartialTransform {
  const Policy<B> policy_;
  std::decay_t<F> f_;

public:
  template <typename F_>
  PartialTransform(const Policy<B> policy, F_&& f) : policy_(policy), f_(std::forward<F_>(f)) {}
  PartialTransform(PartialTransform&&) = default;
  PartialTransform(PartialTransform const&) = default;
  PartialTransform& operator=(PartialTransform&&) = default;
  PartialTransform& operator=(PartialTransform const&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, const PartialTransform pa) {
    return transform<B>(pa.policy_, std::move(pa.f_), std::forward<Sender>(sender));
  }
};

template <Backend B, typename F>
PartialTransform(const Policy<B> policy, F&& f)->PartialTransform<B, std::decay_t<F>>;
}
}
