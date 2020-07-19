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

#include <functional>
#include <type_traits>

namespace dlaf {

/// Check if a F is compatible with given Signature.
///
/// Types of the signatures are not required to match exactly, but they have to be compatible,
/// e.g.
///   float foo(double, double)  is compatible with the signature int(int, float),
///   float foo(double, double)  is compatible with the signature void(int, float),
///   void foo(double, double)   is NOT compatible with the signature int(int, float),
/// @returns type T (default = void) if F can be converted to a std::function with given Signature.
template <class F, class Signature, class T = void>
using enable_if_signature_t =
    typename std::enable_if_t<std::is_convertible<F, std::function<Signature>>::value, T>;

/// std::enable_if wrapper for std::is_convertible.
template <class U, class V, class T = void>
using enable_if_convertible_t = typename std::enable_if_t<std::is_convertible<U, V>::value, T>;
}
