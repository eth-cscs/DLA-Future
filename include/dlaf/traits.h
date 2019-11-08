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

/// @file

#include <type_traits>

namespace dlaf {

template <class, class, class, class Enable = void>
struct enable_if_signature;

template <class T, class Func, class RetType, class ...Args>
struct enable_if_signature<
    Func, RetType(Args...), T,
    std::enable_if_t<std::is_convertible<Func, std::function<RetType(Args...)>>::value>> {
  using type = T;
};

template <class F, class Signature, class T = void>
using enable_if_signature_t = typename enable_if_signature<F, Signature, T>::type;

}
