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

#include <functional>
#include <type_traits>

#include <pika/async_rw_mutex.hpp>
#include <pika/execution.hpp>
#include <pika/future.hpp>

#include <dlaf/types.h>

namespace dlaf::internal {
template <typename...>
struct TypeList {};

template <typename ValueTypes>
struct SenderSingleValueTypeImpl {};

template <typename T>
struct SenderSingleValueTypeImpl<TypeList<TypeList<T>>> {
  using type = T;
};

struct EmptyEnv {};

// We are only interested in the types wrapped by future and shared_future since
// we will internally unwrap them.
template <typename T>
struct SenderSingleValueTypeImpl<TypeList<TypeList<pika::future<T>>>> {
  using type = T;
};

template <typename T>
struct SenderSingleValueTypeImpl<TypeList<TypeList<pika::shared_future<T>>>> {
  using type = T;
};

template <typename T>
struct SenderSingleValueTypeImpl<TypeList<TypeList<std::reference_wrapper<T>>>> {
  using type = T;
};

template <typename RWType, typename RType>
struct SenderSingleValueTypeImpl<
    TypeList<TypeList<pika::execution::experimental::async_rw_mutex_access_wrapper<
        RWType, RType, pika::execution::experimental::async_rw_mutex_access_type::readwrite>>>> {
  using type = RWType;
};

template <typename RWType, typename RType>
struct SenderSingleValueTypeImpl<
    TypeList<TypeList<pika::execution::experimental::async_rw_mutex_access_wrapper<
        RWType, RType, pika::execution::experimental::async_rw_mutex_access_type::read>>>> {
  using type = RType;
};

template <typename... Ts>
using DecayedTypeList = TypeList<std::decay_t<Ts>...>;

// The type sent by Sender, if Sender sends exactly one type.
#if defined(PIKA_HAVE_STDEXEC)
// TODO: make it unique if several types have a identical decayed type
template <typename Sender>
using SenderSingleValueType =
    typename SenderSingleValueTypeImpl<pika::execution::experimental::value_types_of_t<
        std::decay_t<Sender>, EmptyEnv, DecayedTypeList, DecayedTypeList>>::type;
#else
template <typename Sender>
using SenderSingleValueType =
    typename SenderSingleValueTypeImpl<typename pika::execution::experimental::sender_traits<
        std::decay_t<Sender>>::template value_types<TypeList, TypeList>>::type;
#endif

// The type of an embedded ElementType in the value_types of Sender, if it
// exists and Sender sends exactly one type.
template <typename Sender>
using SenderElementType = typename SenderSingleValueType<Sender>::ElementType;

// The value of an embedded device in the value_types of Sender, if it
// exists and Sender sends exactly one type.
template <typename Sender>
inline constexpr Device sender_device = SenderSingleValueType<Sender>::device;

template <typename T>
struct IsSharedFuture : std::false_type {};

template <typename U>
struct IsSharedFuture<pika::shared_future<U>> : std::true_type {};

template <typename T>
inline constexpr bool is_shared_future_v = IsSharedFuture<std::decay_t<T>>::value;
}
