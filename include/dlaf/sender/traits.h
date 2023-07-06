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

#include <dlaf/types.h>

namespace dlaf::internal {
template <typename...>
struct TypeList {};

template <typename... Ts>
using DecayedTypeList = TypeList<std::decay_t<Ts>...>;

template <typename T>
struct IsFalse : std::integral_constant<bool, !(bool) T::value> {};

template <typename T>
inline constexpr bool IsFalseValue = IsFalse<T>::value;

template <typename... Ts>
struct AlwaysFalse : std::false_type {};

template <typename... Ts>
static std::true_type AnyOfImpl(...);

template <typename... Ts>
static auto AnyOfImpl(int) -> AlwaysFalse<std::enable_if_t<IsFalseValue<Ts>>...>;

template <typename... Ts>
struct AnyOf : decltype(AnyOfImpl<Ts...>(0)) {};

template <>
struct AnyOf<> : std::false_type {};

template <typename T, typename... Ts>
struct Contains : AnyOf<std::is_same<T, Ts>...> {};

template <typename PackUnique, typename PackRest>
struct UniqueHelper;

template <template <typename...> class Pack, typename... Ts>
struct UniqueHelper<Pack<Ts...>, Pack<>> {
  using type = Pack<Ts...>;
};

template <template <typename...> class Pack, typename... Ts, typename U, typename... Us>
struct UniqueHelper<Pack<Ts...>, Pack<U, Us...>>
    : std::conditional<Contains<U, Ts...>::value, UniqueHelper<Pack<Ts...>, Pack<Us...>>,
                       UniqueHelper<Pack<Ts..., U>, Pack<Us...>>>::type {};

template <typename Pack>
struct Unique;

template <template <typename...> class Pack, typename... Ts>
struct Unique<Pack<Ts...>> : UniqueHelper<Pack<>, Pack<Ts...>> {};

/// Remove duplicate types in the given pack.
template <typename Pack>
using UniqueType = typename Unique<Pack>::type;

struct EmptyEnv {};

template <typename ValueTypes>
struct SenderSingleValueTypeImpl {};

template <typename T>
struct SenderSingleValueTypeImpl<TypeList<TypeList<T>>> {
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

// The type sent by Sender, if Sender sends exactly one type.
#if defined(PIKA_HAVE_STDEXEC)
template <typename Sender>
using SenderSingleValueType =
    typename SenderSingleValueTypeImpl<UniqueType<pika::execution::experimental::value_types_of_t<
        std::decay_t<Sender>, EmptyEnv, DecayedTypeList, DecayedTypeList>>>::type;
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
}
