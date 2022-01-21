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
#include <hpx/local/future.hpp>

namespace dlaf::internal {
template <typename...>
struct TypeList {};

template <typename ValueTypes>
struct SenderSingleValueTypeImpl {};

template <typename T>
struct SenderSingleValueTypeImpl<TypeList<TypeList<T>>> {
  using type = T;
};

// We are only interested in the types wrapped by future and shared_future since
// we will internally unwrap them.
template <typename T>
struct SenderSingleValueTypeImpl<TypeList<TypeList<hpx::future<T>>>> {
  using type = T;
};

template <typename T>
struct SenderSingleValueTypeImpl<TypeList<TypeList<hpx::shared_future<T>>>> {
  using type = T;
};

// The type sent by Sender, if Sender sends exactly one type.
template <typename Sender>
using SenderSingleValueType =
    typename SenderSingleValueTypeImpl<typename hpx::execution::experimental::sender_traits<
        Sender>::template value_types<TypeList, TypeList>>::type;

// The type of an embedded ElementType in the value_types of Sender, if it
// exists and Sender sends exactly one type.
template <typename Sender>
using SenderElementType = typename SenderSingleValueType<Sender>::ElementType;
}
